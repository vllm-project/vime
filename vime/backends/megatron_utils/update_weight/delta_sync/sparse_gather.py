"""Verl sharded delta semantics with HCCL placement and batched extraction."""

from __future__ import annotations

import torch
import torch.distributed as dist

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class GatherWorkspace:
    """Reusable collective scratch; returned slot payloads never alias it.

    Owned by one synchronous updater. Calls must use the same device stream.
    """

    def __init__(self):
        self.buffers = {}

    def tensor(self, key, size, reference):
        key = (key, reference.device, reference.dtype)
        buffer = self.buffers.get(key)
        if buffer is None or buffer.numel() < size:
            buffer = reference.new_empty(size)
            self.buffers[key] = buffer
        return buffer[:size]


def shard_delta_mask(local_new, local_snap):
    """The same bit comparison as verl, separated for batch profiling."""
    int_dtype = _DTYPE_INT.get(local_new.element_size())
    if int_dtype is None:
        raise ValueError(f"unsupported element size {local_new.element_size()}")
    return local_new.view(int_dtype) != local_snap.view(int_dtype)


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
    workspace: GatherWorkspace | None = None,
    _counts_cpu: list[list[int]] | None = None,
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
    workspace = workspace if workspace is not None else GatherWorkspace()
    if _counts_cpu is None:
        counts_device = counts.to(device)
        counts_buffer = workspace.tensor("counts", world * slot_count, counts_device)
        counts_all = list(counts_buffer.view(world, slot_count).unbind(0))
        dist.all_gather(counts_all, counts_device, group=group)
        counts_cpu = counts_buffer.view(world, slot_count).cpu().tolist()
    else:
        counts_cpu = _counts_cpu

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
                sub_counts = counts[lo:hi]
                sub_result = gather_slot_entries_to_rank0(
                    idx_concat[local_offsets[lo] : local_offsets[hi]],
                    val_concat[local_offsets[lo] : local_offsets[hi]],
                    sub_counts,
                    group=group,
                    workspace=workspace,
                    _counts_cpu=[row[lo:hi] for row in counts_cpu],
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

    padded_indices = workspace.tensor("send_indices", max_elements, idx_concat)
    padded_values = workspace.tensor("send_values", max_elements, val_concat)
    local_elements = int(idx_concat.numel())
    padded_indices[:local_elements].copy_(idx_concat)
    padded_values[:local_elements].copy_(val_concat)
    # Initialize only padding; all payload elements are overwritten above.
    padded_indices[local_elements:].zero_()
    padded_values[local_elements:].zero_()

    # torch-npu implements gather through all-gather, which replicates the
    # complete sparse payload on every training rank.  Keep the tiny counts
    # exchange collective, then move payloads directly to group rank zero.
    use_hccl_p2p = str(dist.get_backend(group)).lower() == "hccl"
    index_list = value_list = None
    if rank == 0:
        index_list = list(workspace.tensor("recv_indices", world * max_elements,
                                          idx_concat).view(world, max_elements).unbind(0))
        value_list = list(workspace.tensor("recv_values", world * max_elements,
                                          val_concat).view(world, max_elements).unbind(0))
    if use_hccl_p2p:
        root = dist.get_global_rank(group, 0) if group is not None else 0
        if rank == 0:
            assert index_list is not None and value_list is not None
            index_list[0].copy_(padded_indices)
            value_list[0].copy_(padded_values)
            operations = []
            for peer_rank in range(1, world):
                peer = dist.get_global_rank(group, peer_rank) if group is not None else peer_rank
                operations.extend((
                    dist.P2POp(dist.irecv, index_list[peer_rank], peer, group),
                    dist.P2POp(dist.irecv, value_list[peer_rank], peer, group),
                ))
        else:
            operations = [
                dist.P2POp(dist.isend, padded_indices, root, group),
                dist.P2POp(dist.isend, padded_values, root, group),
            ]
        for request in dist.batch_isend_irecv(operations):
            request.wait()
    else:
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


def compact_shard_masks(locals_, masks, *, with_midpoints=False):
    """One ordered nonzero per bounded batch, matching per-shard bit diffs.

    Masks remain bitwise comparisons, as in verl. Only the variable-length
    extraction is batched. The single boundary vector D2H replaces per-shard
    dynamic shape synchronization; no HF mapping or ownership changes here.
    """
    if not locals_:
        return []
    ends, midpoints, total = [], [], 0
    for local in locals_:
        midpoints.append(total + local.numel() // 2)
        total += local.numel()
        ends.append(total)
    positions = torch.cat(masks).nonzero(as_tuple=False).view(-1)
    boundaries = torch.tensor(ends + midpoints if with_midpoints else ends,
                              dtype=torch.int64, device=positions.device)
    splits = torch.searchsorted(positions, boundaries).cpu().tolist()
    result, start, base = [], 0, 0
    for slot, (local, end, boundary) in enumerate(zip(locals_, splits[:len(ends)], ends, strict=True)):
        indices = positions[start:end] - base
        row = (indices, local.index_select(0, indices))
        result.append((*row, splits[len(ends) + slot] - start) if with_midpoints else row)
        start, base = end, boundary
    return result
