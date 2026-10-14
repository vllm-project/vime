"""DSpark draft model for vime Megatron backend.

Adapted from DeepSpec/deepspec/modeling/dspark/qwen3/modeling.py:Qwen3DSparkModel.

The DSpark draft model is a semi-autoregressive speculative decoding drafter:
    - Parallel backbone (5 decoder layers) produces draft hidden states
    - Markov head adds sequential bias to draft logits
    - Confidence head predicts per-position acceptance probability

Unlike Eagle3 (which uses ModelOpt's EagleModule with Megatron TP support),
DSpark's dual-input attention has no existing Megatron implementation, so
this module is replicated on every TP rank and uses plain ``nn.Linear`` + SDPA.

The model is attached as ``policy_chunk.draft_model`` before DDP wrapping,
following the NeMo RL Eagle3 pattern.
"""

import logging

import torch
import torch.nn.functional as F
from torch import nn

from .attention import DSparkParallelAttention
from .common import (
    AcceptRatePredictor,
    DSparkConfig,
    DSparkForwardOutput,
    build_eval_mask,
    create_dspark_attention_mask,
    create_noise_embed,
    create_position_ids,
    sample_anchor_positions,
)
from .markov_head import build_markov_head

logger = logging.getLogger(__name__)


def _all_gather_vocab_weight(weight: torch.Tensor) -> torch.Tensor:
    """All-gather a TP-sharded vocab weight along dim=0.

    Megatron's VocabParallelEmbedding shards the vocab dimension across TP
    ranks. Each rank holds [vocab_size // TP, hidden_size]. This function
    collects the full [padded_vocab_size, hidden_size] tensor on every rank.

    Returns the original weight if all-gather fails (e.g., TP group not
    accessible); callers should handle shape mismatch as a fallback.
    """
    import torch.distributed as dist

    if not dist.is_initialized():
        return weight

    tp_group = None
    for module_path, func_name in [
        ("megatron.core.tensor_parallel", "get_tensor_model_parallel_group"),
        ("megatron.core.parallel_state", "get_tensor_model_parallel_group"),
    ]:
        try:
            module = __import__(module_path, fromlist=[func_name])
            tp_group = getattr(module, func_name)()
            break
        except (ImportError, AttributeError, RuntimeError):
            continue

    if tp_group is None:
        return weight

    try:
        world_size = dist.get_world_size(group=tp_group)
    except (RuntimeError, ValueError):
        return weight

    if world_size <= 1:
        return weight

    tensor_list = [torch.empty_like(weight) for _ in range(world_size)]
    dist.all_gather(tensor_list, weight.contiguous().detach(), group=tp_group)
    return torch.cat(tensor_list, dim=0).detach()


class DSparkRMSNorm(nn.Module):
    """RMSNorm matching Qwen3's normalization."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return self.weight * hidden_states.to(input_dtype)


class DSparkMLP(nn.Module):
    """SwiGLU MLP matching Qwen3."""

    def __init__(self, hidden_size: int, intermediate_size: int):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class DSparkDecoderLayer(nn.Module):
    """One DSpark decoder layer with dual-input attention + SwiGLU MLP."""

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.self_attn = DSparkParallelAttention(
            hidden_size=config.hidden_size,
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            attention_bias=False,
            rms_norm_eps=config.rms_norm_eps,
            rotary_base=config.rotary_base,
        )
        self.mlp = DSparkMLP(
            hidden_size=config.hidden_size,
            intermediate_size=_compute_intermediate_size(config),
        )
        self.input_layernorm = DSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = DSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        target_hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden_states=target_hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


def _compute_intermediate_size(config: DSparkConfig) -> int:
    """Compute MLP intermediate size from config.

    For Qwen3 models, intermediate_size is typically ~2.75x hidden_size.
    This can be overridden by setting ``intermediate_size`` on the config.
    """
    if hasattr(config, "intermediate_size") and config.intermediate_size > 0:
        return config.intermediate_size
    # Qwen3-4B/8B: hidden=2560/4096, intermediate=6912/12288
    # Ratio ~2.7. Use a multiple of 256 for efficiency.
    raw = int(config.hidden_size * 2.75)
    return ((raw + 255) // 256) * 256


class DSparkModel(nn.Module):
    """DSpark draft model: parallel backbone + Markov head + confidence head.

    This is a plain ``nn.Module`` attached as ``policy_chunk.draft_model`` and
    replicated on every TP rank. DDP wrapping on the parent policy chunk still
    covers its parameters.

    Structure:
        - embed_tokens: shared from policy (frozen by default)
        - layers: ``num_draft_layers`` DSparkDecoderLayer
        - norm: final RMSNorm
        - fc: projection from [num_target_layers * hidden] -> hidden
        - hidden_norm: RMSNorm on target hidden states before fc
        - lm_head: shared from policy (frozen by default)
        - markov_head: vanilla/gated/rnn (default vanilla, rank=256)
        - confidence_head: AcceptRatePredictor
    """

    def __init__(self, config: DSparkConfig):
        super().__init__()
        self.config = config
        self.target_layer_ids = list(config.target_layer_ids)
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id
        self.num_anchors = config.num_anchors

        # Embedding and LM head — will be shared from policy via initialize_embeddings_and_head
        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # Backbone
        self.layers = nn.ModuleList([DSparkDecoderLayer(config) for _ in range(config.num_draft_layers)])
        self.norm = DSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # FC projection: [num_target_layers * hidden] -> hidden
        self.fc = nn.Linear(
            len(self.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = DSparkRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Markov head
        self.markov_head = build_markov_head(config)

        # Confidence head
        self.enable_confidence_head = config.enable_confidence_head
        self.confidence_head_with_markov = config.confidence_head_with_markov
        self.confidence_head = None
        if self.enable_confidence_head:
            input_dim = config.hidden_size
            if self.confidence_head_with_markov:
                input_dim += config.markov_rank
            self.confidence_head = AcceptRatePredictor(input_dim=input_dim)

    def initialize_embeddings_and_head(
        self,
        *,
        embed_tokens: nn.Module,
        lm_head: nn.Module,
        freeze: bool = True,
    ):
        """Copy policy's embedding and lm_head weights to DSpark (shared).

        When policy uses TP>1, Megatron's VocabParallelEmbedding shards
        the vocab dimension across TP ranks. We all-gather to reconstruct
        the full vocab embedding for the replicated DSpark model.
        """
        embed_weight = embed_tokens.weight.detach()
        lm_head_weight = lm_head.weight.detach()

        # All-gather if policy embedding is TP-sharded
        if embed_weight.shape != self.embed_tokens.weight.shape:
            embed_weight = _all_gather_vocab_weight(embed_weight)
        if lm_head_weight.shape != self.lm_head.weight.shape:
            lm_head_weight = _all_gather_vocab_weight(lm_head_weight)

        with torch.no_grad():
            # If all-gather succeeded, shapes match and we copy directly.
            # If all-gather failed (TP group not accessible), copy only the
            # matching portion; the rest stays zero-initialized and will be
            # overwritten by load_pretrained_weights() if a pretrained
            # checkpoint is provided.
            if embed_weight.shape == self.embed_tokens.weight.shape:
                self.embed_tokens.weight.copy_(embed_weight)
            else:
                min_rows = min(embed_weight.shape[0], self.embed_tokens.weight.shape[0])
                logger.warning(
                    "[DSpark] embed_tokens shape mismatch after all-gather: "
                    "dspark %s vs policy %s, copying first %d rows",
                    self.embed_tokens.weight.shape,
                    embed_weight.shape,
                    min_rows,
                )
                self.embed_tokens.weight.zero_()
                self.embed_tokens.weight[:min_rows].copy_(embed_weight[:min_rows])

            if lm_head_weight.shape == self.lm_head.weight.shape:
                self.lm_head.weight.copy_(lm_head_weight)
            else:
                min_rows = min(lm_head_weight.shape[0], self.lm_head.weight.shape[0])
                logger.warning(
                    "[DSpark] lm_head shape mismatch after all-gather: "
                    "dspark %s vs policy %s, copying first %d rows",
                    self.lm_head.weight.shape,
                    lm_head_weight.shape,
                    min_rows,
                )
                self.lm_head.weight.zero_()
                self.lm_head.weight[:min_rows].copy_(lm_head_weight[:min_rows])

        if freeze:
            self.set_embedding_head_trainable(False)

    def load_pretrained_weights(self, path: str):
        """Load pre-trained DSpark weights from safetensors file.

        Loads ALL parameters including embed_tokens and lm_head.
        The pre-trained DSpark checkpoint has untied embeddings
        (tie_word_embeddings=false), so its lm_head is very different
        from the policy's tied lm_head. Using the policy's lm_head
        would produce random predictions (CE loss ~11.0).
        """
        from safetensors.torch import load_file

        state_dict = load_file(path)
        loaded = 0
        missing = 0
        mismatched = 0
        for name, param in self.named_parameters():
            if name in state_dict:
                tensor = state_dict[name]
                if tensor.shape == param.data.shape:
                    param.data.copy_(tensor.to(param.dtype))
                    loaded += 1
                elif (
                    tensor.dim() == param.data.dim()
                    and tensor.shape[1:] == param.data.shape[1:]
                    and tensor.shape[0] <= param.data.shape[0]
                ):
                    # Vocab padding: checkpoint has fewer rows (original vocab),
                    # model has padded vocab for TP. Zero-pad and copy.
                    with torch.no_grad():
                        param.data.zero_()
                        param.data[: tensor.shape[0]].copy_(tensor.to(param.dtype))
                    loaded += 1
                    logger.info(
                        "[DSpark] Padded %s: checkpoint %s -> model %s (vocab padding)",
                        name,
                        tuple(tensor.shape),
                        tuple(param.data.shape),
                    )
                else:
                    logger.warning(
                        "[DSpark] Shape mismatch for %s: checkpoint %s vs model %s",
                        name,
                        tuple(tensor.shape),
                        tuple(param.data.shape),
                    )
                    mismatched += 1
            else:
                logger.warning("[DSpark] Missing key in pretrained: %s", name)
                missing += 1
        total = sum(1 for _, _ in self.named_parameters())
        logger.info(
            "[DSpark] Loaded %d/%d params from pretrained checkpoint " "(missing=%d, mismatched=%d)",
            loaded,
            total,
            missing,
            mismatched,
        )

    def set_embedding_head_trainable(self, trainable: bool):
        self.embed_tokens.requires_grad_(trainable)
        self.lm_head.requires_grad_(trainable)

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        return self.lm_head(hidden_states)

    def predict_confidence_step(
        self,
        hidden_states: torch.Tensor,
        prev_token_ids: torch.Tensor | None = None,
    ) -> torch.Tensor | None:
        """Compute confidence predictions for each position.

        Args:
            hidden_states: [bsz, num_blocks, block_size, hidden]
            prev_token_ids: [bsz, num_blocks, block_size] (needed if confidence_head_with_markov)
        Returns:
            [bsz, num_blocks, block_size] or None
        """
        if self.confidence_head is None:
            return None
        if self.confidence_head_with_markov:
            assert self.markov_head is not None
            assert prev_token_ids is not None
            prev_embeddings = self.markov_head.get_prev_embeddings(prev_token_ids).to(dtype=hidden_states.dtype)
            features = torch.cat([hidden_states, prev_embeddings], dim=-1)
            return self.confidence_head(features).float()
        return self.confidence_head(hidden_states).float()

    def forward(
        self,
        input_ids: torch.Tensor,
        target_hidden_states: torch.Tensor,
        loss_mask: torch.Tensor,
        target_last_hidden_states: torch.Tensor | None = None,
    ) -> DSparkForwardOutput:
        """DSpark training forward.

        Args:
            input_ids: [bsz, seq_len] token ids from policy
            target_hidden_states: [bsz, seq_len, num_target_layers * hidden]
                Concatenated hidden states from policy's target_layer_ids.
            loss_mask: [bsz, seq_len] loss mask (1 for supervised tokens)
            target_last_hidden_states: [bsz, seq_len, hidden] (optional)
                Policy's last layer hidden states, for L_tv/L_conf computation.
        Returns:
            DSparkForwardOutput
        """
        bsz, seq_len = input_ids.shape
        device = input_ids.device

        # 1. Sample anchor positions
        anchor_positions, block_keep_mask = sample_anchor_positions(
            seq_len=seq_len,
            loss_mask=loss_mask,
            num_anchors=self.num_anchors,
            device=device,
        )

        # 2. Create noise embedding (draft input)
        noise_embedding = create_noise_embed(
            self.embed_tokens,
            input_ids,
            anchor_positions,
            block_keep_mask,
            mask_token_id=self.mask_token_id,
            block_size=self.block_size,
        )

        # 3. Position ids: context + draft
        context_position_ids = torch.arange(seq_len, device=device).unsqueeze(0).expand(bsz, -1)
        draft_position_ids = create_position_ids(anchor_positions, self.block_size)
        full_position_ids = torch.cat([context_position_ids, draft_position_ids], dim=1)

        # 4. Project target hidden states
        # Detach: prevent DSpark loss gradient from flowing back through the
        # policy model. The draft model learns to predict the target, not
        # the other way around.
        target_hidden_projected = self.hidden_norm(self.fc(target_hidden_states.detach()))

        # 5. Build attention mask
        attn_mask = create_dspark_attention_mask(
            anchor_positions=anchor_positions,
            block_keep_mask=block_keep_mask,
            seq_len=seq_len,
            block_size=self.block_size,
            device=device,
            dtype=noise_embedding.dtype,
        )  # [bsz, 1, q_len, ctx_len + q_len]

        # 6. Backbone forward
        hidden_states = noise_embedding  # [bsz, q_len, hidden]
        # Position ids for rotary: need [bsz, ctx_len + q_len]
        # The rotary is applied to K which spans ctx_len + q_len
        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden_states=target_hidden_projected,
                position_ids=full_position_ids,
                attention_mask=attn_mask,
            )
        output_hidden = self.norm(hidden_states)  # [bsz, q_len, hidden]

        # 7. Reshape to [bsz, num_blocks, block_size, hidden]
        num_blocks = anchor_positions.size(1)
        output_hidden_4d = output_hidden.reshape(bsz, num_blocks, self.block_size, -1)

        # 8. Compute target ids (labels for CE loss)
        label_offsets = torch.arange(1, self.block_size + 1, device=device).view(1, 1, -1)  # [1, 1, block_size]
        label_indices = anchor_positions.unsqueeze(-1) + label_offsets  # [bsz, num_blocks, block_size]
        safe_label_indices = label_indices.clamp(max=seq_len - 1)
        safe_label_indices = torch.where(
            block_keep_mask.unsqueeze(-1),
            safe_label_indices,
            torch.zeros_like(safe_label_indices),
        )
        target_ids = torch.gather(
            input_ids.unsqueeze(1).expand(-1, anchor_positions.size(1), -1),
            2,
            safe_label_indices,
        )  # [bsz, num_blocks, block_size]

        # 9. Compute aligned target logits (for L_tv / L_conf)
        aligned_target_logits = None
        if target_last_hidden_states is not None:
            target_pred_indices = (safe_label_indices - 1).clamp(min=0)
            aligned_target_hidden = torch.gather(
                target_last_hidden_states.unsqueeze(1).expand(-1, anchor_positions.size(1), -1, -1),
                2,
                target_pred_indices.unsqueeze(-1).expand(-1, -1, -1, target_last_hidden_states.size(-1)),
            )  # [bsz, num_blocks, block_size, hidden]
            # Detach target logits: the l1_loss gradient must NOT flow
            # back through the policy model. The draft model is trained to
            # predict the target, not the other way around.
            aligned_target_logits = self.compute_logits(aligned_target_hidden).detach()

        # 10. Build eval mask
        eval_mask = build_eval_mask(
            seq_len=seq_len,
            loss_mask=loss_mask,
            label_indices=label_indices,
            safe_label_indices=safe_label_indices,
            block_keep_mask=block_keep_mask,
        )  # [bsz, num_blocks, block_size] bool

        # 11. Compute draft logits
        anchor_token_ids = torch.gather(input_ids, 1, anchor_positions)  # [bsz, num_blocks]
        prev_token_ids = torch.cat(
            [anchor_token_ids.unsqueeze(-1), target_ids[:, :, :-1]],
            dim=-1,
        )  # [bsz, num_blocks, block_size]

        draft_logits = self.compute_logits(output_hidden).reshape(
            bsz, num_blocks, self.block_size, -1
        )  # [bsz, num_blocks, block_size, vocab]

        # 12. Apply Markov head bias
        if self.markov_head is not None:
            draft_logits = self.markov_head.apply_block_logits(
                draft_logits,
                token_ids=prev_token_ids,
                hidden_states=output_hidden_4d,
            )

        # 13. Confidence prediction
        confidence_pred = None
        if self.confidence_head is not None:
            confidence_pred = self.predict_confidence_step(
                output_hidden_4d, prev_token_ids
            )  # [bsz, num_blocks, block_size]

        return DSparkForwardOutput(
            draft_logits=draft_logits,
            target_ids=target_ids,
            eval_mask=eval_mask,
            block_keep_mask=block_keep_mask,
            confidence_pred=confidence_pred,
            aligned_target_logits=aligned_target_logits,
        )


def build_dspark_model(
    dspark_config: DSparkConfig,
    policy_embed_tokens: nn.Module,
    policy_lm_head: nn.Module,
    pretrained_model_path: str | None = None,
) -> DSparkModel:
    """Build a DSpark draft model and initialize shared weights from policy.

    Args:
        dspark_config: DSparkConfig with model dims populated
        policy_embed_tokens: policy's embedding layer (for weight sharing)
        policy_lm_head: policy's lm_head layer (for weight sharing)
        pretrained_model_path: Optional path to pre-trained DSpark safetensors
            file. If provided, backbone weights are loaded from this file
            instead of random init. embed_tokens/lm_head are always copied
            from the policy model.
    Returns:
        DSparkModel (not yet attached to policy chunk)
    """
    model = DSparkModel(dspark_config)
    model.initialize_embeddings_and_head(
        embed_tokens=policy_embed_tokens,
        lm_head=policy_lm_head,
        freeze=True,
    )
    if pretrained_model_path is not None:
        model.load_pretrained_weights(pretrained_model_path)
    logger.info(
        "[DSpark] Built DSpark draft model: %d layers, block_size=%d, "
        "target_layer_ids=%s, markov_rank=%d, confidence=%s, "
        "intermediate_size=%d, pretrained=%s",
        dspark_config.num_draft_layers,
        dspark_config.block_size,
        dspark_config.target_layer_ids,
        dspark_config.markov_rank,
        dspark_config.enable_confidence_head,
        _compute_intermediate_size(dspark_config),
        pretrained_model_path is not None,
    )
    return model


def attach_dspark_model(model, args, config) -> None:
    dspark_config = DSparkConfig(
        block_size=args.dspark_block_size,
        num_draft_layers=args.dspark_num_draft_layers,
        target_layer_ids=args.dspark_target_layer_ids,
        mask_token_id=args.dspark_mask_token_id,
        num_anchors=args.dspark_num_anchors,
        markov_rank=args.dspark_markov_rank,
        markov_head_type=args.dspark_markov_head_type,
        enable_confidence_head=not args.dspark_disable_confidence_head,
        ce_loss_alpha=args.dspark_ce_loss_alpha,
        l1_loss_alpha=args.dspark_l1_loss_alpha,
        confidence_head_alpha=args.dspark_confidence_head_alpha,
        loss_decay_gamma=args.dspark_loss_decay_gamma,
        draft_loss_weight=args.dspark_draft_loss_weight,
        hidden_size=config.hidden_size,
        vocab_size=args.padded_vocab_size,
        org_vocab_size=args.vocab_size,
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=getattr(config, "num_query_groups", config.num_attention_heads),
        head_dim=getattr(config, "kv_channels", config.hidden_size // config.num_attention_heads),
        rms_norm_eps=getattr(config, "layernorm_epsilon", 1e-6),
        rotary_base=getattr(config, "rotary_base", 10000.0),
        intermediate_size=args.dspark_intermediate_size,
    )
    policy_embed = getattr(model.embedding, "word_embeddings", model.embedding)
    policy_lm_head = getattr(model, "output_layer", None)
    if policy_lm_head is None or getattr(policy_lm_head, "weight", None) is None:
        policy_lm_head = policy_embed
    model.draft_model = build_dspark_model(
        dspark_config,
        policy_embed,
        policy_lm_head,
        args.dspark_pretrained_model,
    )
    for param in model.draft_model.parameters():
        param.grad_norm_group = "dspark"


def restore_dspark_param_views(model_chunks):
    """Restore view relationship between DSpark draft params and DDP buffer after TMS resume.

    After torch_memory_saver.resume(), param.data tensors are no longer views into
    the DDP contiguous buffer. This function rebinds each draft parameter to its
    slice in the DDP buffer, so subsequent optimizer.step() updates are visible.

    This should be called once after TMS resume, not after every optimizer step.
    """
    from megatron.core.utils import unwrap_model

    for chunk in model_chunks:
        if not hasattr(chunk, "buffers"):
            continue
        unwrapped = unwrap_model(chunk)
        draft = getattr(unwrapped, "draft_model", None)
        if draft is None:
            continue
        draft_param_ids = {id(p) for p in draft.parameters()}
        for buffer in chunk.buffers:
            pim = buffer.param_index_map
            for param_obj, (_start, _end, bucket_id) in pim.items():
                if id(param_obj) not in draft_param_ids:
                    continue
                bucket = buffer.buckets[bucket_id]
                if hasattr(bucket, "param_to_index") and param_obj in bucket.param_to_index:
                    local_start, local_end = bucket.param_to_index[param_obj]
                    if isinstance(local_start, int):
                        # Restore view: rebind param.data to DDP buffer slice
                        param_obj.data = bucket.param_data.view(-1)[local_start:local_end].view(param_obj.data.shape)
