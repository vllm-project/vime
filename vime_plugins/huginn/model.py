"""Differentiable Huginn with replayed noise and current-weight prelude inputs."""

import math
from functools import partial
from typing import Literal

import torch
import torch.nn.functional as F
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.module import MegatronModule
from megatron.core.transformer.transformer_config import TransformerConfig
from torch.utils.checkpoint import checkpoint
from vllm_rlt.models.huginn import HuginnBlock, HuginnForCausalLM
from vllm_rlt.models.huginn_latents import HUGINN_LATENT_PROFILE, replay_huginn_latents

from vime.utils.types import RecurrentTrace


class HuginnMegatronModel(MegatronModule):
    def __init__(
        self,
        config: TransformerConfig,
        native: HuginnForCausalLM,
        *,
        role: Literal["actor", "critic"] = "actor",
        recompute: bool = False,
        model_revision: str | None = None,
    ):
        super().__init__(config)
        if (config.tensor_model_parallel_size, config.pipeline_model_parallel_size, config.context_parallel_size) != (
            1,
            1,
            1,
        ):
            raise ValueError("Huginn currently requires TP=PP=CP=1")
        if config.sequence_parallel or config.gradient_accumulation_fusion:
            raise ValueError("Huginn SDPA requires sequence parallel and grad fusion disabled")
        self.huginn_config = native.config
        self.transformer = native.transformer
        self.register_buffer("freqs_cis", native.freqs_cis)
        self.freqs_cis.tensor_model_parallel = False
        self.role, self.recompute, self.model_revision = role, recompute, model_revision
        if role == "critic":
            from vime.backends.megatron_utils.model_provider import LinearForLastLayer

            self.output_layer = LinearForLastLayer(native.config.n_embd, 1, config=config, bias=False)
        else:
            self.lm_head = native.lm_head
        self.requires_grad_(True)
        self.pre_process = self.post_process = True
        self.share_embeddings_and_output_weights = False
        self.vp_stage = None

    def _apply(self, fn, recurse=True):
        frequencies = self.freqs_cis
        super()._apply(fn, recurse=recurse)
        self.freqs_cis = frequencies.to(device=self.transformer.wte.weight.device, dtype=torch.float32)
        self.freqs_cis.tensor_model_parallel = False
        return self

    def set_input_tensor(self, input_tensor: list[torch.Tensor | None]) -> None:
        if any(value is not None for value in input_tensor):
            raise ValueError("Huginn does not accept pipeline input")

    def _block(self, block: HuginnBlock, hidden: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        attention = block.attn
        q, k, v = attention.Wqkv(block.norm_1(hidden)).chunk(3, dim=-1)
        shape = (len(hidden), attention.n_heads, attention.head_dim)
        q, k = q.reshape(shape) + attention.qk_bias[0], k.reshape(shape) + attention.qk_bias[1]
        pairs = torch.stack((q, k)).float().reshape(2, len(hidden), attention.n_heads, -1, 2)
        cos, sin = frequencies[..., 0].unsqueeze(0), frequencies[..., 1].unsqueeze(0)
        real = pairs[..., 0] * cos - pairs[..., 1] * sin
        imag = pairs[..., 1] * cos + pairs[..., 0] * sin
        q, k = torch.stack((real, imag), dim=-1).flatten(-2).to(hidden.dtype).unbind(0)
        attended = (
            F.scaled_dot_product_attention(
                q.transpose(0, 1).unsqueeze(0),
                k.transpose(0, 1).unsqueeze(0),
                v.reshape(shape).transpose(0, 1).unsqueeze(0),
                is_causal=True,
            )
            .squeeze(0)
            .transpose(0, 1)
            .reshape(len(hidden), -1)
        )
        hidden = block.norm_2(attention.proj(attended) + hidden)
        return block.norm_4(block.mlp(block.norm_3(hidden)) + hidden)

    def _blocks(self, blocks: torch.nn.ModuleList, hidden: torch.Tensor, frequencies: torch.Tensor) -> torch.Tensor:
        for block in blocks:
            forward = partial(self._block, block)
            # Drain each reused block's gradients into MCore before the next block.
            hidden = (
                checkpoint(forward, hidden, frequencies, use_reentrant=True)
                if (self.recompute and torch.is_grad_enabled())
                else forward(hidden, frequencies)
            )
        return hidden

    def _sequence(self, tokens: torch.Tensor, seed: int) -> torch.Tensor:
        width = self.huginn_config.n_embd
        frequencies = self.freqs_cis[0, : len(tokens)]
        injection = self._blocks(
            self.transformer.prelude,
            self.transformer.wte(tokens) * math.sqrt(width),
            frequencies,
        )
        state = replay_huginn_latents(
            width,
            [seed] * len(tokens),
            range(len(tokens)),
            dtype=injection.dtype,
            device=tokens.device,
        )
        for _ in range(self.huginn_config.mean_recurrence):
            state = self.transformer.adapter(torch.cat((state, injection), dim=-1))
            state = self._blocks(self.transformer.core_block, state, frequencies)
        state = self._blocks(self.transformer.coda, self.transformer.ln_f(state), frequencies)
        state = self.transformer.ln_f(state)
        if self.role == "critic":
            return self.output_layer(state)[0]
        # Drain the tied output gradient before recurrent backward reaches wte.
        return (
            checkpoint(self.lm_head, state, use_reentrant=True)
            if self.recompute and torch.is_grad_enabled()
            else self.lm_head(state)
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        position_ids=None,
        attention_mask=None,
        labels=None,
        packed_seq_params: PackedSeqParams | None = None,
        loss_mask=None,
        recurrent_inputs: list[RecurrentTrace] | None = None,
    ) -> torch.Tensor:
        if position_ids is not None or attention_mask is not None or labels is not None:
            raise ValueError("Use unmodified per-sequence positions and VIME's masked RL loss")
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or recurrent_inputs is None:
            raise ValueError("Huginn requires one packed token stream and its recurrent traces")
        boundaries = [0, input_ids.numel()] if packed_seq_params is None else packed_seq_params.cu_seqlens_q.tolist()
        if packed_seq_params is not None and packed_seq_params.qkv_format != "thd":
            raise ValueError("Huginn expects thd packed sequences")
        if len(recurrent_inputs) > len(boundaries) - 1:
            raise ValueError("Each real packed sequence requires one recurrent trace")
        seeds = []
        for trace in recurrent_inputs:
            if (
                trace.model_family != "huginn_raven"
                or trace.latent_profile != HUGINN_LATENT_PROFILE
                or trace.latent_seed is None
                or trace.prefill_depth != self.huginn_config.mean_recurrence
                or any(depth != self.huginn_config.mean_recurrence for depth in trace.decode_depths)
                or (self.model_revision is not None and trace.model_revision != self.model_revision)
            ):
                raise ValueError("Huginn requires the matching fixed-depth model and latent replay profile")
            seeds.append(trace.latent_seed)
        # A final padding sequence has no reward or likelihood contribution.
        if len(seeds) == len(boundaries) - 2:
            if loss_mask is None or bool(loss_mask[0, boundaries[-2] :].any()):
                raise ValueError("Only a fully masked padding sequence may omit its recurrent trace")
            seeds.append(0)
        if len(seeds) != len(boundaries) - 1:
            raise ValueError("Missing a recurrent trace for a real packed sequence")
        return torch.cat(
            [
                self._sequence(input_ids[0, start:end], seed)
                for start, end, seed in zip(boundaries[:-1], boundaries[1:], seeds, strict=True)
            ]
        ).unsqueeze(0)


def model_provider(
    pre_process: bool = True,
    post_process: bool = True,
    vp_stage: int | None = None,
    role: Literal["actor", "critic"] = "actor",
) -> HuginnMegatronModel:
    from megatron.training import get_args
    from megatron.training.arguments import core_transformer_config_from_args

    if not pre_process or not post_process or vp_stage is not None:
        raise ValueError("Huginn supports one non-pipelined model chunk")
    args = get_args()
    config = core_transformer_config_from_args(args)
    native = HuginnForCausalLM.from_pretrained(args.hf_checkpoint, dtype=config.params_dtype)
    return HuginnMegatronModel(
        config,
        native,
        role=role,
        recompute=args.recompute_granularity is not None,
        model_revision=args.rlt_model_revision,
    )
