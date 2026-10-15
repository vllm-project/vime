"""Flatten scheduled micro-batches into one padding-free stream for HF models.

The rollout side already decides which local samples form each micro-batch
(``micro_batch_indices``, see ``vime.utils.dp_schedule``). This module only
concatenates those samples into a single ``[1, T]`` row whose ``position_ids``
restart at every sample, which HF attention turns into per-sample causal masks
(``cu_seqlens`` for flash-attention, a block-diagonal mask otherwise), and
gathers the per-sample fields the shared loss functions consume.

The packing layout follows slime's FSDP ``data_packing.py``
(https://github.com/THUDM/slime, ``slime/backends/fsdp_utils/data_packing.py``
at 0104a9e9, Apache-2.0). Unlike that version, partitioning is left to the
shared scheduler and padding is tracked explicitly instead of being inferred
from trailing zero tokens, because token id 0 is a valid token.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass, field

import torch

PAD_TOKEN_ID = 0


@dataclass
class PackedMicroBatch:
    """One micro-batch as model inputs plus the per-sample loss fields."""

    input_ids: torch.Tensor  # [1, T]
    position_ids: torch.Tensor  # [1, T]
    cu_seqlens: torch.Tensor  # [num_segments + 1], int32; a padding tail is its own segment
    num_pad_tokens: int
    indices: list[int]
    batch: dict = field(default_factory=dict)


def pack_micro_batch(
    rollout_data: dict,
    indices: Sequence[int],
    keys: Sequence[str],
    *,
    device: torch.device | str | None = None,
    pad_to_multiple_of: int = 1,
) -> PackedMicroBatch:
    """Select ``indices`` from ``rollout_data`` and build the packed model inputs.

    ``batch`` maps every requested key to the selected per-sample values (or
    ``None`` when absent), mirroring the Megatron data iterator, and adds
    ``unconcat_tokens``, the selected token tensors before concatenation.
    """
    assert "tokens" in keys
    indices = list(indices)
    batch = {}
    for key in keys:
        values = rollout_data.get(key)
        batch[key] = None if values is None else [values[i] for i in indices]

    tokens = [torch.as_tensor(t, dtype=torch.long, device=device) for t in batch["tokens"]]
    batch["tokens"] = tokens
    batch["unconcat_tokens"] = tokens

    lengths = [t.numel() for t in tokens]
    total = sum(lengths)
    pad = (-total) % max(pad_to_multiple_of, 1)
    input_ids = torch.cat(tokens) if tokens else torch.empty(0, dtype=torch.long, device=device)
    position_ids = torch.cat([torch.arange(n, device=device) for n in lengths]) if lengths else input_ids
    cu_seqlens = [0]
    for n in lengths:
        cu_seqlens.append(cu_seqlens[-1] + n)
    if pad:
        # The tail restarts its positions, so it is masked off from the samples.
        input_ids = torch.cat([input_ids, input_ids.new_full((pad,), PAD_TOKEN_ID)])
        position_ids = torch.cat([position_ids, torch.arange(pad, device=device)])
        cu_seqlens.append(cu_seqlens[-1] + pad)

    return PackedMicroBatch(
        input_ids=input_ids.unsqueeze(0),
        position_ids=position_ids.unsqueeze(0),
        cu_seqlens=torch.tensor(cu_seqlens, dtype=torch.int32, device=device),
        num_pad_tokens=pad,
        indices=indices,
        batch=batch,
    )


def iter_step_micro_batches(rollout_data: dict) -> Iterator[tuple[int, list[list[int]]]]:
    """Yield ``(step_id, micro_batch_indices_of_step)`` following the shared schedule."""
    micro_batch_indices = rollout_data["micro_batch_indices"]
    offset = 0
    for step_id, num_microbatches in enumerate(rollout_data["num_microbatches"]):
        yield step_id, micro_batch_indices[offset : offset + num_microbatches]
        offset += num_microbatches
    assert offset == len(micro_batch_indices), f"schedule covers {offset} of {len(micro_batch_indices)} micro-batches"


def restore_sample_order(values: Sequence, order: Sequence[int]) -> list:
    """Place ``values[k]`` (produced for local sample ``order[k]``) back at its sample index."""
    assert len(values) == len(order), f"{len(values)} values for {len(order)} samples"
    restored = [None] * len(order)
    for value, index in zip(values, order, strict=True):
        assert restored[index] is None, f"sample {index} appears twice in the schedule"
        restored[index] = value
    return restored
