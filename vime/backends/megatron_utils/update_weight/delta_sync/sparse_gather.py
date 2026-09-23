"""Variable-length sparse gather, copied from verl's sharded delta path."""

from __future__ import annotations

import torch
import torch.distributed as dist

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


def shard_delta_indices(
    local_new: torch.Tensor,
    local_snap: torch.Tensor,
    offset: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Bit-exact local shard diff returning flattened positions and values."""
    int_dtype = _DTYPE_INT.get(local_new.element_size())
    if int_dtype is None:
        raise ValueError(f"unsupported element size {local_new.element_size()}")
    mask = local_new.view(int_dtype) != local_snap.view(int_dtype)
    local_idx = mask.nonzero(as_tuple=False).view(-1)
    values = local_new[local_idx]
    return local_idx.to(torch.int64) + offset, values


def gather_slot_entries_to_rank0(
    idx_concat: torch.Tensor,
    val_concat: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup | None = None,
    max_round_bytes: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Gather K variable-length slot payloads to process-group rank zero."""
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    dst = dist.get_global_rank(group, 0) if group is not None else 0
    device = idx_concat.device
    if device.type == "cpu" and hasattr(torch, "npu"):
        # Delta extraction may leave its compact buffers on CPU after the
        # trainer releases model memory.  The Megatron group is HCCL-only, so
        # normalize every collective tensor to the current NPU first.
        device = torch.device("npu", torch.npu.current_device())
        idx_concat = idx_concat.to(device)
        val_concat = val_concat.to(device)
    slot_count = int(counts.numel())

    # HCCL process groups only accept NPU tensors.  Build both the input and
    # receive buffers on the payload device; creating receive buffers from the
    # original CPU ``counts`` tensor makes all_gather select a CPU backend.
    counts_device = counts.to(device)
    counts_all = [torch.zeros_like(counts_device) for _ in range(world)]
    dist.all_gather(counts_all, counts_device, group=group)
    counts_cpu = torch.stack(counts_all).cpu().tolist()

    if max_round_bytes is not None and slot_count > 1:
        per_element = idx_concat.element_size() + val_concat.element_size()
        budget = max(int(max_round_bytes) // per_element, 1)
        cuts = [0]
        running = [0] * world
        for slot_index in range(slot_count):
            running = [running[r] + counts_cpu[r][slot_index] for r in range(world)]
            if max(running) > budget and cuts[-1] != slot_index:
                cuts.append(slot_index)
                running = [counts_cpu[r][slot_index] for r in range(world)]
        cuts.append(slot_count)
        if len(cuts) > 2:
            result: list[tuple[torch.Tensor, torch.Tensor]] = []
            local_offsets = [0]
            for slot_index in range(slot_count):
                local_offsets.append(local_offsets[-1] + counts_cpu[rank][slot_index])
            for lo, hi in zip(cuts[:-1], cuts[1:], strict=False):
                sub_counts = torch.tensor(counts_cpu[rank][lo:hi], dtype=torch.int64, device=device)
                sub_result = gather_slot_entries_to_rank0(
                    idx_concat[local_offsets[lo] : local_offsets[hi]],
                    val_concat[local_offsets[lo] : local_offsets[hi]],
                    sub_counts,
                    group=group,
                )
                if rank == 0:
                    assert sub_result is not None
                    result.extend(sub_result)
            return result if rank == 0 else None

    totals = [sum(row) for row in counts_cpu]
    max_elements = max(totals) if totals else 0
    if max_elements == 0:
        if rank != 0:
            return None
        empty_indices = torch.empty(0, dtype=idx_concat.dtype, device=device)
        empty_values = torch.empty(0, dtype=val_concat.dtype, device=device)
        return [(empty_indices, empty_values) for _ in range(slot_count)]

    padded_indices = torch.zeros(max_elements, dtype=idx_concat.dtype, device=device)
    padded_values = torch.zeros(max_elements, dtype=val_concat.dtype, device=device)
    local_elements = int(idx_concat.numel())
    padded_indices[:local_elements] = idx_concat
    padded_values[:local_elements] = val_concat

    index_list = (
        [torch.zeros_like(padded_indices) for _ in range(world)] if rank == 0 else None
    )
    value_list = (
        [torch.zeros_like(padded_values) for _ in range(world)] if rank == 0 else None
    )
    dist.gather(padded_indices, index_list, dst=dst, group=group)
    dist.gather(padded_values, value_list, dst=dst, group=group)
    if rank != 0:
        return None

    assert index_list is not None and value_list is not None
    offsets = [[0] * (slot_count + 1) for _ in range(world)]
    for process_rank in range(world):
        for slot_index in range(slot_count):
            offsets[process_rank][slot_index + 1] = (
                offsets[process_rank][slot_index] + counts_cpu[process_rank][slot_index]
            )

    result = []
    for slot_index in range(slot_count):
        index_parts = [
            index_list[r][offsets[r][slot_index] : offsets[r][slot_index + 1]]
            for r in range(world)
            if counts_cpu[r][slot_index]
        ]
        value_parts = [
            value_list[r][offsets[r][slot_index] : offsets[r][slot_index + 1]]
            for r in range(world)
            if counts_cpu[r][slot_index]
        ]
        if index_parts:
            result.append((torch.cat(index_parts), torch.cat(value_parts)))
        else:
            result.append(
                (
                    torch.empty(0, dtype=idx_concat.dtype, device=device),
                    torch.empty(0, dtype=val_concat.dtype, device=device),
                )
            )
    return result
