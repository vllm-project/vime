"""DSpark 3-loss computation and DraftLossWrapper.

Adapted from:
    - DeepSpec/deepspec/modeling/dspark/loss.py (3-loss: CE + L1/TV + Confidence)
    - NeMo RL/nemo_rl/algorithms/loss/wrapper.py (DraftLossWrapper pattern)

The 3 losses are:
    L_ce:  cross-entropy of draft logits vs target token ids (weight 0.1)
    L_tv:  L1 distance between draft probs and target probs (weight 0.9)
    L_conf: BCE of confidence pred vs empirical accept rate (weight 1.0)

All losses use position decay: weight *= exp(-pos / gamma), gamma=4.0.

Loss denominators are all-reduced across the data-parallel group to normalize
correctly. The final backward loss is scaled by world_size to counteract
DDP's gradient averaging.
"""

import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint

from .common import DSparkConfig, DSparkForwardOutput

logger = logging.getLogger(__name__)

# Chunk size for vocab-dimension operations. Each chunk creates a temporary
# tensor of shape (batch, seq, block_size, chunk_size). With chunk_size=16384
# and typical batch*seq*block_size=~3500, each chunk is ~235 MB (float32).
_L1_VOCAB_CHUNK_SIZE = 16384


def _all_reduce_loss_denominators(
    loss_terms: dict[str, torch.Tensor],
    *,
    world_size: int,
) -> dict[str, torch.Tensor]:
    """All-reduce loss denominators across DP group for normalization."""
    denominators = {}
    for key in ("ce_loss_den", "l1_loss_den", "confidence_loss_den"):
        tensor = loss_terms[key].detach().clone()
        if world_size > 1:
            try:
                from megatron.core import mpu

                dp_group = mpu.get_data_parallel_group(with_context_parallel=True)
                if dp_group is not None:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM, group=dp_group)
                else:
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
            except Exception:
                # Fallback: global all_reduce
                if dist.is_initialized():
                    dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        denominators[key] = tensor
    return denominators


def _build_loss_weight_mask(
    *,
    eval_mask: torch.Tensor,
    block_size: int,
    device: torch.device,
    loss_decay_gamma: float | None,
) -> torch.Tensor:
    """Build per-position loss weight with exponential decay."""
    loss_weight_mask = eval_mask.to(torch.float32)
    if loss_decay_gamma is not None and loss_decay_gamma > 0:
        positions = torch.arange(block_size, device=device).view(1, 1, -1)
        decay_weights = torch.exp(-positions.float() / float(loss_decay_gamma))
        loss_weight_mask = loss_weight_mask * decay_weights
    return loss_weight_mask


def _compute_accept_rate_3d(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: torch.Tensor | None,
) -> torch.Tensor | None:
    """Compute per-position acceptance rate: 1 - 0.5 * L1(draft_probs, target_probs).

    Computed without gradients (only used as a detached target for the
    confidence head). Uses chunked logsumexp to avoid materializing the full
    vocab-dimension softmax or difference tensors.
    """
    if aligned_target_logits is None:
        return None
    with torch.no_grad():
        draft_logits = outputs.draft_logits.float()
        target_logits = aligned_target_logits.float()
        vocab_size = draft_logits.shape[-1]
        log_Z_draft = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
        log_Z_target = torch.logsumexp(target_logits, dim=-1, keepdim=True)
        l1_shape = draft_logits.shape[:-1]
        l1_dist = torch.zeros(l1_shape, device=draft_logits.device, dtype=torch.float32)
        for start in range(0, vocab_size, _L1_VOCAB_CHUNK_SIZE):
            end = min(start + _L1_VOCAB_CHUNK_SIZE, vocab_size)
            pa = torch.exp(draft_logits[..., start:end] - log_Z_draft)
            pb = torch.exp(target_logits[..., start:end] - log_Z_target)
            l1_dist += (pa - pb).abs().sum(dim=-1)
        accept_rate_3d = (1.0 - 0.5 * l1_dist).clamp_(0.0, 1.0)
    return accept_rate_3d


def _compute_local_l1_term(
    *,
    outputs: DSparkForwardOutput,
    aligned_target_logits: torch.Tensor | None,
    loss_weight_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute L1/TV loss numerator and denominator.

    Uses logsumexp + chunked exp + gradient checkpointing to avoid OOM.
    The full softmax tensors are never materialized; each vocab chunk is
    recomputed during backward via torch.utils.checkpoint.
    """
    zero = outputs.draft_logits.new_zeros((), dtype=torch.float32)
    if aligned_target_logits is None:
        return zero, zero
    draft_logits = outputs.draft_logits.float()
    target_logits = aligned_target_logits.float()
    vocab_size = draft_logits.shape[-1]

    # Log partition functions (small, no OOM risk)
    log_Z_draft = torch.logsumexp(draft_logits, dim=-1, keepdim=True)
    log_Z_target = torch.logsumexp(target_logits, dim=-1, keepdim=True)

    # Chunked L1: sum_i |exp(a_i - log_Z_a) - exp(b_i - log_Z_b)|
    # Each chunk creates a temporary of shape (..., chunk_size), well under
    # the torch_memory_saver margin. Gradient checkpointing ensures chunk
    # intermediates are NOT saved for backward (recomputed instead).
    l1_shape = draft_logits.shape[:-1]
    l1_dist = torch.zeros(l1_shape, device=draft_logits.device, dtype=torch.float32)

    def _chunk_l1(a_chunk, b_chunk, log_za, log_zb):
        pa = torch.exp(a_chunk - log_za)
        pb = torch.exp(b_chunk - log_zb)
        return (pa - pb).abs().sum(dim=-1)

    for start in range(0, vocab_size, _L1_VOCAB_CHUNK_SIZE):
        end = min(start + _L1_VOCAB_CHUNK_SIZE, vocab_size)
        a_slice = draft_logits[..., start:end]
        b_slice = target_logits[..., start:end]
        chunk_l1 = checkpoint.checkpoint(
            _chunk_l1,
            a_slice,
            b_slice,
            log_Z_draft,
            log_Z_target,
            use_reentrant=False,
        )
        l1_dist = l1_dist + chunk_l1

    l1_loss_num = (l1_dist * loss_weight_mask).sum()
    l1_loss_den = loss_weight_mask.sum()
    return l1_loss_num, l1_loss_den


def _collect_local_terms(
    *,
    outputs: DSparkForwardOutput,
    loss_decay_gamma: float | None,
    l1_loss_alpha: float,
) -> tuple[dict[str, torch.Tensor], bool]:
    """Collect local loss terms (numerators and denominators)."""
    draft_logits = outputs.draft_logits
    target_ids = outputs.target_ids
    eval_mask = outputs.eval_mask
    _, _, block_size, vocab_size = draft_logits.shape
    device = draft_logits.device

    loss_weight_mask = _build_loss_weight_mask(
        eval_mask=eval_mask,
        block_size=block_size,
        device=device,
        loss_decay_gamma=loss_decay_gamma,
    )
    flat_logits = draft_logits.reshape(-1, vocab_size)
    flat_targets = target_ids.reshape(-1)
    flat_weights = loss_weight_mask.reshape(-1)
    loss_per_token = F.cross_entropy(flat_logits, flat_targets, reduction="none")
    ce_loss_num = (loss_per_token * flat_weights).sum()
    ce_loss_den = flat_weights.sum()

    aligned_target_logits = outputs.aligned_target_logits
    accept_rate_3d = _compute_accept_rate_3d(
        outputs=outputs,
        aligned_target_logits=aligned_target_logits,
    )
    zero = ce_loss_num.new_zeros(())

    assert (
        l1_loss_alpha <= 0 or aligned_target_logits is not None
    ), "aligned_target_logits is required when l1_loss_alpha > 0."
    if l1_loss_alpha > 0:
        l1_loss_num, l1_loss_den = _compute_local_l1_term(
            outputs=outputs,
            aligned_target_logits=aligned_target_logits,
            loss_weight_mask=loss_weight_mask,
        )
    else:
        l1_loss_num = zero
        l1_loss_den = zero

    has_confidence = outputs.confidence_pred is not None
    confidence_loss_num = zero
    confidence_loss_den = zero
    if has_confidence:
        assert accept_rate_3d is not None, "aligned_target_logits is required when confidence head is enabled."
        confidence_targets = accept_rate_3d.detach()
        confidence_errors = (
            F.binary_cross_entropy_with_logits(
                outputs.confidence_pred.float(),
                confidence_targets,
                reduction="none",
            )
            * loss_weight_mask
        )
        confidence_loss_num = confidence_errors.sum()
        confidence_loss_den = loss_weight_mask.sum()

    loss_terms = {
        "ce_loss_num": ce_loss_num,
        "ce_loss_den": ce_loss_den,
        "l1_loss_num": l1_loss_num,
        "l1_loss_den": l1_loss_den,
        "confidence_loss_num": confidence_loss_num,
        "confidence_loss_den": confidence_loss_den,
    }
    return loss_terms, has_confidence


def _build_loss(
    *,
    loss_terms: dict[str, torch.Tensor],
    global_denominators: dict[str, torch.Tensor],
    ce_loss_alpha: float,
    l1_loss_alpha: float,
    confidence_head_alpha: float,
    has_confidence: bool,
    world_size: int,
) -> torch.Tensor:
    """Build the final backward loss from local terms and global denominators."""
    ce_loss = loss_terms["ce_loss_num"] / (global_denominators["ce_loss_den"] + 1e-6)
    l1_loss = ce_loss.new_zeros(())
    if global_denominators["l1_loss_den"].item() > 0:
        l1_loss = loss_terms["l1_loss_num"] / (global_denominators["l1_loss_den"] + 1e-6)
    confidence_loss = ce_loss.new_zeros(())
    if has_confidence:
        confidence_loss = loss_terms["confidence_loss_num"] / (global_denominators["confidence_loss_den"] + 1e-6)
    return (ce_loss_alpha * ce_loss + l1_loss_alpha * l1_loss + confidence_head_alpha * confidence_loss) * world_size


def compute_dspark_loss(
    *,
    outputs: DSparkForwardOutput,
    config: DSparkConfig,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Compute DSpark 3-loss.

    Args:
        outputs: DSparkForwardOutput from DSparkModel.forward
        config: DSparkConfig with loss weights and decay gamma
    Returns:
        (backward_loss, metrics_dict)
        - backward_loss: scalar tensor for backward()
        - metrics_dict: dict of float values for logging
    """
    loss_terms, has_confidence = _collect_local_terms(
        outputs=outputs,
        loss_decay_gamma=config.loss_decay_gamma,
        l1_loss_alpha=float(config.l1_loss_alpha),
    )

    world_size = dist.get_world_size() if dist.is_initialized() else 1
    global_denominators = _all_reduce_loss_denominators(loss_terms, world_size=world_size)

    # Local loss for logging
    local_ce_loss = loss_terms["ce_loss_num"] / (loss_terms["ce_loss_den"] + 1e-6)
    local_l1_loss = local_ce_loss.new_zeros(())
    if loss_terms["l1_loss_den"].item() > 0:
        local_l1_loss = loss_terms["l1_loss_num"] / (loss_terms["l1_loss_den"] + 1e-6)
    local_confidence_loss = local_ce_loss.new_zeros(())
    if has_confidence:
        local_confidence_loss = loss_terms["confidence_loss_num"] / (loss_terms["confidence_loss_den"] + 1e-6)

    backward_loss = _build_loss(
        loss_terms=loss_terms,
        global_denominators=global_denominators,
        ce_loss_alpha=float(config.ce_loss_alpha),
        l1_loss_alpha=float(config.l1_loss_alpha),
        confidence_head_alpha=float(config.confidence_head_alpha),
        has_confidence=has_confidence,
        world_size=world_size,
    )

    metrics = {
        "dspark/ce_loss": float(local_ce_loss.detach().item()),
        "dspark/l1_loss": float(local_l1_loss.detach().item()),
        "dspark/confidence_loss": float(local_confidence_loss.detach().item()),
        "dspark/total_loss": float(
            (
                config.ce_loss_alpha * local_ce_loss
                + config.l1_loss_alpha * local_l1_loss
                + config.confidence_head_alpha * local_confidence_loss
            )
            .detach()
            .item()
        ),
    }
    return backward_loss, metrics


class DraftLossWrapper:
    """Combine policy RL loss with DSpark draft loss.

    Following NeMo RL's DraftLossWrapper pattern:
        combined_loss = policy_loss + draft_loss_weight * draft_loss

    The draft loss is computed from DSparkForwardOutput and added to the
    policy loss. The policy loss function is called first, then the draft
    loss is computed and combined.
    """

    def __init__(self, config: DSparkConfig):
        self.config = config
        self.draft_loss_weight = config.draft_loss_weight

    def __call__(
        self,
        policy_loss: torch.Tensor,
        dspark_outputs: DSparkForwardOutput,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        """Compute combined loss.

        Args:
            policy_loss: scalar tensor, the RL policy loss (already computed)
            dspark_outputs: DSparkForwardOutput from DSparkModel.forward
        Returns:
            (combined_loss, dspark_metrics)
        """
        draft_loss, dspark_metrics = compute_dspark_loss(
            outputs=dspark_outputs,
            config=self.config,
        )
        combined_loss = policy_loss + self.draft_loss_weight * draft_loss
        return combined_loss, dspark_metrics


def build_combined_loss_fn(policy_loss_fn, args, batch, num_microbatches, global_batch_size, outputs, config):
    wrapper = DraftLossWrapper(config)

    def combined_loss_fn(logits):
        if args.dspark_freeze_policy:
            logits = logits.detach()
        policy_loss, num_elems, metrics = policy_loss_fn(
            args,
            batch,
            num_microbatches,
            global_batch_size,
            logits,
        )
        combined_loss, dspark_metrics = wrapper(policy_loss, outputs)
        keys = list(dspark_metrics)
        values = metrics["values"].new_tensor([dspark_metrics[key] for key in keys])
        metrics["keys"] += keys
        metrics["values"] = torch.cat((metrics["values"], values))
        return combined_loss, num_elems, metrics

    return combined_loss_fn
