"""DSpark common utilities: anchor sampling, mask construction, config.

Adapted from DeepSpec/deepspec/modeling/dspark/common.py for vime Megatron backend.

Key changes from DeepSpec:
- Removed ``add_metric`` calls (vime uses its own logging_utils).
- Replaced ``flex_attention.create_block_mask`` with an explicit SDPA-compatible
  boolean attention mask because flex_attention is not available in all
  Megatron/vLLM container images.
- Added ``DSparkConfig`` dataclass to bundle all DSpark hyperparameters.
"""

from dataclasses import dataclass

import torch
from torch import nn


@dataclass
class DSparkConfig:
    """Bundle of DSpark hyperparameters passed to ``build_dspark_model``.

    Defaults match DeepSpec's ``config/dspark/dspark_qwen3_4b.py``.
    """

    # Backbone
    block_size: int = 7
    num_draft_layers: int = 5
    target_layer_ids: tuple[int, ...] = (1, 9, 17, 25, 33)
    mask_token_id: int = 151669
    num_anchors: int = 512

    # Markov head
    markov_rank: int = 256
    markov_head_type: str = "vanilla"

    # Confidence head
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    # Loss weights
    ce_loss_alpha: float = 0.1
    l1_loss_alpha: float = 0.9
    confidence_head_alpha: float = 1.0
    loss_decay_gamma: float = 4.0

    # Draft loss combination weight (multiplies draft_loss added to policy_loss)
    draft_loss_weight: float = 1.0

    # Model dims (populated by build_dspark_model from policy config)
    hidden_size: int = 0
    vocab_size: int = 0
    org_vocab_size: int = 0  # original (unpadded) vocab size for vLLM export
    num_attention_heads: int = 0
    num_key_value_heads: int = 0
    head_dim: int = 0
    rms_norm_eps: float = 1e-6
    rotary_base: float = 10000.0
    rope_scaling: dict | None = None

    # MLP intermediate size (0 = auto-compute from hidden_size * 2.75)
    # Set to match pre-trained checkpoint (e.g. 9728 for Qwen3-4B)
    intermediate_size: int = 0


@dataclass
class DSparkForwardOutput:
    """Outputs for one DSpark training forward.

    Shapes:
        draft_logits:       [batch, num_anchors, block_size, vocab]
        target_ids:         [batch, num_anchors, block_size]
        eval_mask:          [batch, num_anchors, block_size] (bool)
        block_keep_mask:    [batch, num_anchors] (bool)
        confidence_pred:    [batch, num_anchors, block_size] (optional)
        aligned_target_logits: [batch, num_anchors, block_size, vocab] (optional)
    """

    draft_logits: torch.Tensor
    target_ids: torch.Tensor
    eval_mask: torch.Tensor
    block_keep_mask: torch.Tensor
    confidence_pred: torch.Tensor | None = None
    aligned_target_logits: torch.Tensor | None = None


class AcceptRatePredictor(nn.Module):
    """Confidence head: predicts P(token accepted | prev accepted) per position."""

    def __init__(self, input_dim: int):
        super().__init__()
        self.proj = nn.Linear(int(input_dim), 1)

    def forward(self, features):
        return self.proj(features).squeeze(-1)


def validate_target_layer_ids(layer_ids, num_target_layers: int):
    """Validate that target_layer_ids are strictly increasing and in range."""
    layer_ids = [int(layer_id) for layer_id in layer_ids]
    assert layer_ids, "target_layer_ids must not be empty."
    start = 0
    end = int(num_target_layers) - 1
    previous = None
    for layer_id in layer_ids:
        assert layer_id == -1 or start <= layer_id <= end, (
            f"target_layer_id {layer_id} is out of range {{-1}} U [{start}, {end}] "
            f"for num_target_layers={num_target_layers}. "
            "-1 denotes the embedding output."
        )
        assert previous is None or layer_id > previous, "target_layer_ids must be strictly increasing."
        previous = layer_id
    return layer_ids


def build_anchor_candidate_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
) -> torch.Tensor:
    """Build a boolean mask of valid anchor candidate positions.

    A position i is a valid anchor if both loss_mask[i] and loss_mask[i+1] are set
    (the anchor token and its first prediction target must be supervised).
    """
    num_candidates = max(seq_len - 1, 0)
    if num_candidates == 0:
        return loss_mask[:, :0].bool()

    anchor_valid = loss_mask[:, :num_candidates] > 0.5
    first_target_valid = loss_mask[:, 1 : num_candidates + 1] > 0.5
    return anchor_valid & first_target_valid


def sample_anchor_positions(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    num_anchors: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample up to ``num_anchors`` anchor positions per sequence.

    Returns:
        anchor_positions: [batch, num_anchors] long tensor (0 for invalid anchors)
        block_keep_mask: [batch, num_anchors] bool tensor (True for valid anchors)
    """
    valid = build_anchor_candidate_mask(
        seq_len=seq_len,
        loss_mask=loss_mask,
    )
    valid_counts = valid.sum(dim=1)
    bsz = loss_mask.shape[0]
    num_candidates = valid.shape[1]
    max_n = int(num_anchors)
    if num_candidates == 0:
        anchors = torch.zeros(bsz, max_n, dtype=torch.long, device=device)
        keep_mask = torch.zeros(bsz, max_n, dtype=torch.bool, device=device)
        return anchors, keep_mask

    indices = (
        torch.arange(num_candidates, device=device)
        .unsqueeze(0)
        .expand(
            bsz,
            -1,
        )
    )
    masked_indices = torch.where(
        valid,
        indices,
        torch.full_like(indices, seq_len + 1),
    )
    random_vals = torch.rand(bsz, num_candidates, device=device)
    random_vals = torch.where(valid, random_vals, torch.full_like(random_vals, 2.0))
    _, sorted_idx = random_vals.sort(dim=1)
    gathered = torch.gather(masked_indices, 1, sorted_idx)
    if num_candidates < max_n:
        pad = torch.full(
            (bsz, max_n - num_candidates),
            seq_len + 1,
            dtype=gathered.dtype,
            device=device,
        )
        gathered = torch.cat([gathered, pad], dim=1)
    anchors = gathered[:, :max_n].sort(dim=1).values
    keep_mask = torch.arange(max_n, device=device).unsqueeze(0) < (valid_counts.unsqueeze(1).clamp(max=max_n))
    anchors = torch.where(keep_mask, anchors, torch.zeros_like(anchors))
    return anchors, keep_mask


def build_eval_mask(
    *,
    seq_len: int,
    loss_mask: torch.Tensor,
    label_indices: torch.Tensor,
    safe_label_indices: torch.Tensor,
    block_keep_mask: torch.Tensor,
) -> torch.Tensor:
    """Build the per-position evaluation mask.

    A position is evaluated if:
    - its label index is within seq_len,
    - its label position is enabled by loss_mask,
    - its block is kept,
    - and all preceding positions in the block are also evaluated (cumprod).
    """
    target_valid = label_indices < seq_len
    target_loss_mask = torch.gather(
        loss_mask.unsqueeze(1).expand(-1, label_indices.size(1), -1),
        2,
        safe_label_indices,
    )
    eval_mask = target_valid & (target_loss_mask > 0.5)
    eval_mask = eval_mask & block_keep_mask.unsqueeze(-1)
    return eval_mask.to(torch.int32).cumprod(dim=-1).bool()


def create_position_ids(
    anchor_positions: torch.Tensor,
    block_size: int,
) -> torch.Tensor:
    """Create position ids for draft tokens: [batch, num_blocks * block_size]."""
    bsz, num_blocks = anchor_positions.shape
    device = anchor_positions.device
    offsets = torch.arange(block_size, device=device).view(1, 1, -1)
    return (anchor_positions.unsqueeze(-1) + offsets).view(
        bsz,
        num_blocks * block_size,
    )


def create_noise_embed(
    embed_tokens: nn.Module,
    input_ids: torch.Tensor,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    *,
    mask_token_id: int,
    block_size: int,
) -> torch.Tensor:
    """Create the noise embedding for DSpark draft tokens.

    Each block's first position is the anchor token; remaining positions are
    the mask token. This is the input to the DSpark backbone.
    """
    bsz = input_ids.shape[0]
    num_blocks = anchor_positions.shape[1]
    device = input_ids.device
    noise_ids = torch.full(
        (bsz, num_blocks * block_size),
        mask_token_id,
        dtype=torch.long,
        device=device,
    )
    block_starts = torch.arange(num_blocks, device=device) * block_size
    block_starts = block_starts.unsqueeze(0).expand(bsz, -1)
    anchor_tokens = torch.gather(input_ids, 1, anchor_positions)
    flat_batch_idx = (
        torch.arange(bsz, device=device)
        .unsqueeze(1)
        .expand(
            bsz,
            num_blocks,
        )
    )
    noise_ids[flat_batch_idx, block_starts] = torch.where(
        block_keep_mask,
        anchor_tokens,
        torch.tensor(mask_token_id, dtype=torch.long, device=device),
    )
    return embed_tokens(noise_ids)


def create_dspark_attention_mask(
    *,
    anchor_positions: torch.Tensor,
    block_keep_mask: torch.Tensor,
    seq_len: int,
    block_size: int,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build a dense boolean attention mask for DSpark.

    Replaces DeepSpec's flex_attention ``create_block_mask`` with an explicit
    [batch, 1, q_len, kv_len] mask suitable for ``scaled_dot_product_attention``.

    Layout:
        KV = [context (seq_len) | draft (num_blocks * block_size)]
        Q  = draft (num_blocks * block_size)

    For draft query in block b at position q_idx:
        - attend to context positions [0, anchor_pos)
        - attend to draft positions in the same block b
        - masked out if block is not kept
    """
    bsz, num_blocks = anchor_positions.shape
    q_len = num_blocks * block_size

    # Query block id for each query position: [q_len]
    q_block_id = torch.arange(q_len, device=device) // block_size

    # Anchor position per query position: [bsz, q_len]
    anchor_per_q = anchor_positions[:, q_block_id]  # [bsz, q_len]

    # Context mask: kv_idx < anchor_pos[bsz, q_idx]
    # kv_ctx_idx: [seq_len], anchor_per_q: [bsz, q_len]
    # Result: [bsz, q_len, seq_len]
    kv_ctx_idx = torch.arange(seq_len, device=device)  # [seq_len]
    ctx_mask = kv_ctx_idx.unsqueeze(0).unsqueeze(0) < anchor_per_q.unsqueeze(-1)

    # Draft mask: kv_idx >= seq_len AND same block as query
    kv_draft_idx = torch.arange(q_len, device=device)  # [q_len]
    kv_block_id = kv_draft_idx // block_size  # [q_len]
    # [q_len, q_len]
    draft_mask = q_block_id.unsqueeze(1) == kv_block_id.unsqueeze(0)
    # Expand to [bsz, q_len, q_len]
    draft_mask = draft_mask.unsqueeze(0).expand(bsz, -1, -1)

    # block keep mask: [bsz, num_blocks] -> [bsz, q_len]
    block_keep_per_q = block_keep_mask[:, q_block_id]  # [bsz, q_len]

    # Combine: [bsz, q_len, kv_len]
    full_mask = torch.cat([ctx_mask, draft_mask], dim=-1)  # [bsz, q_len, kv_len]
    full_mask = full_mask & block_keep_per_q.unsqueeze(-1)

    # Add head dim: [bsz, 1, q_len, kv_len]
    full_mask = full_mask.unsqueeze(1)
    return full_mask.to(dtype=dtype, device=device)
