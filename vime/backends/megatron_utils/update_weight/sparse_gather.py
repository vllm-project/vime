# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Variable-length sparse gather adapted from verl delta weight sync."""

from __future__ import annotations

import torch
import torch.distributed as dist


def gather_slot_entries_to_rank0(
    indices: torch.Tensor,
    values: torch.Tensor,
    counts: torch.Tensor,
    group: dist.ProcessGroup,
    max_round_bytes: int | None = None,
) -> list[tuple[torch.Tensor, torch.Tensor]] | None:
    """Gather variable-length sparse slots only to group rank 0.

    HCCL implements ``dist.gather`` with an all-gather fallback.  That copies every
    rank's sparse payload to every training rank and also requires padding all
    payloads to the largest rank.  Exchange the small per-slot counts collectively,
    then use exact-sized point-to-point transfers for the payload instead.
    """
    rank = dist.get_rank(group)
    world = dist.get_world_size(group)
    destination = dist.get_global_rank(group, 0)
    device = indices.device
    slot_count = int(counts.numel())

    all_counts = [torch.zeros_like(counts) for _ in range(world)]
    dist.all_gather(all_counts, counts.to(device), group=group)
    counts_cpu = torch.stack(all_counts).cpu().tolist()

    if max_round_bytes is not None and slot_count > 1:
        bytes_per_entry = indices.element_size() + values.element_size()
        budget = max(int(max_round_bytes) // bytes_per_entry, 1)
        cuts = [0]
        running = [0] * world
        for slot_index in range(slot_count):
            running = [
                running[r] + counts_cpu[r][slot_index]
                for r in range(world)
            ]
            if max(running) > budget and cuts[-1] != slot_index:
                cuts.append(slot_index)
                running = [
                    counts_cpu[r][slot_index] for r in range(world)
                ]
        cuts.append(slot_count)
        if len(cuts) > 2:
            offsets = [0]
            for count in counts_cpu[rank]:
                offsets.append(offsets[-1] + count)
            output = []
            for start, end in zip(cuts[:-1], cuts[1:], strict=True):
                sub_counts = torch.tensor(
                    counts_cpu[rank][start:end],
                    dtype=torch.int64,
                    device=device,
                )
                gathered = gather_slot_entries_to_rank0(
                    indices[offsets[start] : offsets[end]],
                    values[offsets[start] : offsets[end]],
                    sub_counts,
                    group,
                )
                if rank == 0:
                    output.extend(gathered)
            return output if rank == 0 else None

    totals = [sum(row) for row in counts_cpu]
    max_entries = max(totals) if totals else 0
    if max_entries == 0:
        if rank != 0:
            return None
        return [
            (
                torch.empty(0, dtype=indices.dtype, device=device),
                torch.empty(0, dtype=values.dtype, device=device),
            )
            for _ in range(slot_count)
        ]

    if rank == 0:
        index_list = [indices]
        value_list = [values]
        p2p_ops = []
        for group_rank in range(1, world):
            entry_count = totals[group_rank]
            if entry_count == 0:
                index_list.append(
                    torch.empty(0, dtype=indices.dtype, device=device)
                )
                value_list.append(
                    torch.empty(0, dtype=values.dtype, device=device)
                )
                continue
            index_buffer = torch.empty(
                entry_count, dtype=indices.dtype, device=device
            )
            value_buffer = torch.empty(
                entry_count, dtype=values.dtype, device=device
            )
            index_list.append(index_buffer)
            value_list.append(value_buffer)
            peer = dist.get_global_rank(group, group_rank)
            p2p_ops.extend(
                [
                    dist.P2POp(dist.irecv, index_buffer, peer, group),
                    dist.P2POp(dist.irecv, value_buffer, peer, group),
                ]
            )
    elif totals[rank] > 0:
        p2p_ops = [
            dist.P2POp(dist.isend, indices, destination, group),
            dist.P2POp(dist.isend, values, destination, group),
        ]
    else:
        p2p_ops = []

    if p2p_ops:
        for request in dist.batch_isend_irecv(p2p_ops):
            request.wait()
    if rank != 0:
        return None

    offsets = [[0] * (slot_count + 1) for _ in range(world)]
    for rank_index in range(world):
        for slot_index in range(slot_count):
            offsets[rank_index][slot_index + 1] = (
                offsets[rank_index][slot_index]
                + counts_cpu[rank_index][slot_index]
            )

    output = []
    for slot_index in range(slot_count):
        index_parts = [
            index_list[r][
                offsets[r][slot_index] : offsets[r][slot_index + 1]
            ]
            for r in range(world)
            if counts_cpu[r][slot_index]
        ]
        value_parts = [
            value_list[r][
                offsets[r][slot_index] : offsets[r][slot_index + 1]
            ]
            for r in range(world)
            if counts_cpu[r][slot_index]
        ]
        if index_parts:
            output.append((torch.cat(index_parts), torch.cat(value_parts)))
        else:
            output.append(
                (
                    torch.empty(0, dtype=indices.dtype, device=device),
                    torch.empty(0, dtype=values.dtype, device=device),
                )
            )
    return output
