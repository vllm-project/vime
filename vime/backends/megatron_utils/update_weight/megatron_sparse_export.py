# Copyright 2025 Bytedance Ltd. and/or its affiliates
# SPDX-License-Identifier: Apache-2.0
"""Shard-local sparse export through Megatron-Bridge parameter mappings.

This follows verl's Megatron delta exporter: each rank runs a communication-
free copy of the real Bridge mapping with its local shard inserted among NaN
placeholders. Non-NaN survivors are that rank's contribution in final HF
coordinates, including QKV and gate/up rearrangements.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu

from vime.utils.distributed_utils import get_gloo_group


class _ProbeGroup:
    """Process-group stand-in that preserves only size and rank."""

    def __init__(self, size: int, rank: int):
        self._size = int(size)
        self._rank = int(rank)

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def __getattr__(self, name):
        raise RuntimeError(
            f"Sparse Bridge probe attempted unstubbed communication {name!r}"
        )


_NAN_POOL: dict[tuple, torch.Tensor] = {}


def _nan_block(shape, dtype, device) -> torch.Tensor:
    key = (tuple(shape), dtype, str(device))
    tensor = _NAN_POOL.get(key)
    if tensor is None:
        tensor = torch.full(
            tuple(shape), float("nan"), dtype=dtype, device=device
        )
        _NAN_POOL[key] = tensor
    return tensor


def _warm_lazy_mapping(mapping, module) -> None:
    if (
        hasattr(mapping, "_detect_parallelism_type")
        and getattr(mapping, "_mapping", None) is None
    ):
        parallelism = mapping._detect_parallelism_type(module)
        mapping._mapping = mapping._get_or_create_mapping(parallelism)
        mapping._detected_type = parallelism


def make_probe(mapping, module):
    """Copy a Bridge mapping and replace its collectives with local synthesis."""
    from megatron.bridge.models.conversion.param_mapping import (
        MegatronParamMapping,
    )
    from megatron.core.utils import get_pg_rank, get_pg_size

    _warm_lazy_mapping(mapping, module)

    def _stub(mapping_copy):
        mapping_copy.pp_group = _ProbeGroup(1, 0)
        for attr in ("ep_group", "_tp_group", "_etp_group"):
            group = getattr(mapping_copy, attr, None)
            if group is None:
                setattr(mapping_copy, attr, _ProbeGroup(1, 0))
            else:
                setattr(
                    mapping_copy,
                    attr,
                    _ProbeGroup(get_pg_size(group), get_pg_rank(group)),
                )

        def _gather_tp(tensor, owner=mapping_copy):
            missing = _nan_block(tensor.shape, tensor.dtype, tensor.device)
            gathered = [missing] * owner.tp_size
            gathered[owner.tp_rank] = tensor
            return gathered

        mapping_copy.gather_from_tp_ranks = _gather_tp
        mapping_copy.gather_from_ep_ranks = (
            lambda weight, _module, name: {str(name): weight}
        )
        mapping_copy.gather_from_ep_ranks_scale = (
            lambda weight, _module, name: {
                str(name): weight.unsqueeze(0).squeeze().unsqueeze(-1)
            }
        )
        return mapping_copy

    def _inject(node):
        result = _stub(copy.copy(node))
        for attr, value in list(vars(result).items()):
            if isinstance(value, MegatronParamMapping):
                _warm_lazy_mapping(value, module)
                setattr(result, attr, _inject(value))
        return result

    return _inject(mapping)


@dataclass
class SparseExportRecord:
    megatron_name: str
    weight_key: str
    param: torch.Tensor
    gather_group: dist.ProcessGroup | None
    contributes: bool
    probe: Any
    mapping: Any = None
    module: Any = None
    slots: list[tuple[str, tuple[int, ...]]] | None = None


def build_sparse_export_index(
    bridge,
    model,
    local_weights: dict[str, torch.Tensor],
    slot_cache: dict[str, list[tuple[str, tuple[int, ...]]]],
) -> list[SparseExportRecord]:
    """Build the static Bridge directory and local probe for every parameter."""
    if mpu.get_pipeline_model_parallel_world_size() != 1:
        raise NotImplementedError(
            "Sparse shard export currently supports Megatron PP=1"
        )
    if mpu.get_expert_model_parallel_world_size() != 1:
        raise NotImplementedError(
            "Sparse shard export currently supports Megatron EP=1"
        )

    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = dist.get_world_size(group=tp_group)
    tp_rank = mpu.get_tensor_model_parallel_rank()
    dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
    records: list[SparseExportRecord] = []
    for task in bridge.get_conversion_tasks(model):
        if task.param_weight is None:
            continue
        weight_key = f"vp_stages.{task.vp_stage}.{task.param_name}"
        if weight_key not in local_weights:
            raise KeyError(
                f"Megatron sparse export weight {weight_key!r} is unavailable"
            )
        param = task.param_weight
        tp_sharded = (
            getattr(param, "tensor_model_parallel", False) and tp_world > 1
        )
        records.append(
            SparseExportRecord(
                megatron_name=task.global_param_name,
                weight_key=weight_key,
                param=param,
                gather_group=tp_group if tp_sharded else None,
                contributes=dp_rank == 0 and (tp_sharded or tp_rank == 0),
                probe=make_probe(task.mapping, task.megatron_module),
                mapping=task.mapping,
                module=task.megatron_module,
            )
        )

    _exchange_slot_tables(records, slot_cache)
    return records


def _exchange_slot_tables(
    records: list[SparseExportRecord],
    slot_cache: dict[str, list[tuple[str, tuple[int, ...]]]],
) -> None:
    """Make every directory row use an identical final-HF slot table."""
    local_rows = []
    for record in records:
        if record.megatron_name not in slot_cache:
            empty_idx = torch.empty(
                0, dtype=torch.int64, device=record.param.device
            )
            empty_val = torch.empty(
                0, dtype=record.param.dtype, device=record.param.device
            )
            sparse_hf_entry(record, empty_idx, empty_val, slot_cache)
        local_rows.append(slot_cache[record.megatron_name])

    gathered: list = [None] * dist.get_world_size()
    # The directory contains Python strings/shapes and belongs on the control
    # plane.  Do not let ``all_gather_object`` fall back to the HCCL world
    # group: besides staging a potentially large object through NPU memory,
    # some torch/HCCL combinations do not support object collectives at all.
    dist.all_gather_object(gathered, local_rows, group=get_gloo_group())
    if not all(len(rows) == len(local_rows) for rows in gathered):
        raise RuntimeError("Megatron sparse Bridge directories differ by rank")
    for row_index, record in enumerate(records):
        union: dict[tuple[str, tuple[int, ...]], None] = {}
        for rows in gathered:
            for name, shape in rows[row_index]:
                union[(name, tuple(shape))] = None
        record.slots = list(union)


def sparse_hf_entry(
    record: SparseExportRecord,
    local_indices: torch.Tensor,
    local_values: torch.Tensor,
    slot_cache: dict[str, list[tuple[str, tuple[int, ...]]]],
) -> tuple[
    list[tuple[str, tuple[int, ...]]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Map one local sparse shard patch to final HF names and coordinates."""
    slots = record.slots or slot_cache.get(record.megatron_name)
    if local_indices.numel() == 0 and slots is not None:
        return (
            slots,
            torch.zeros(len(slots), dtype=torch.int64),
            torch.empty(
                0, dtype=torch.int32, device=local_values.device
            ),
            torch.empty(
                0, dtype=local_values.dtype, device=local_values.device
            ),
        )

    if slots is not None:
        fast_entry = _sparse_hf_entry_fast(
            record, slots, local_indices, local_values
        )
        if fast_entry is not None:
            return fast_entry

    buffer = torch.full(
        tuple(record.param.shape),
        float("nan"),
        dtype=local_values.dtype,
        device=local_values.device,
    )
    if local_indices.numel():
        buffer.view(-1)[local_indices] = local_values
    outputs = record.probe.megatron_to_hf(buffer, record.module)

    if slots is None:
        slots = [
            (name, tuple(int(dim) for dim in tensor.shape))
            for name, tensor in outputs.items()
        ]
        slot_cache[record.megatron_name] = slots
    unknown = set(outputs) - {name for name, _ in slots}
    if unknown:
        raise RuntimeError(
            f"Bridge probe emitted unknown HF slots: {sorted(unknown)}"
        )

    counts = torch.zeros(len(slots), dtype=torch.int64)
    index_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for slot_index, (name, _shape) in enumerate(slots):
        output = outputs.get(name)
        if output is None:
            continue
        flat = output.reshape(-1)
        indices = (~torch.isnan(flat)).nonzero(as_tuple=False).view(-1)
        if indices.numel():
            counts[slot_index] = indices.numel()
            index_parts.append(indices.to(torch.int32))
            value_parts.append(flat[indices])

    if index_parts:
        return slots, counts, torch.cat(index_parts), torch.cat(value_parts)
    return (
        slots,
        counts,
        torch.empty(0, dtype=torch.int32, device=local_values.device),
        torch.empty(
            0, dtype=local_values.dtype, device=local_values.device
        ),
    )


def _sparse_hf_entry_fast(
    record: SparseExportRecord,
    slots: list[tuple[str, tuple[int, ...]]],
    local_indices: torch.Tensor,
    local_values: torch.Tensor,
) -> tuple[
    list[tuple[str, tuple[int, ...]]],
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
] | None:
    """Map common dense-Megatron layouts without materializing NaN tensors.

    Qwen3 uses only Auto(Column/Row/Replicated), QKV and GatedMLP mappings.
    These transforms rearrange coordinates, so changed entries can be routed
    directly in O(number of changes), instead of building and scanning dense
    full-HF probe outputs on every TP rank. Unknown Bridge mappings retain the
    generic probe fallback above.
    """
    mapping = getattr(record, "mapping", None)
    if mapping is None:
        return None
    mapping_name = type(mapping).__name__
    if mapping_name == "AutoMapping" or hasattr(mapping, "_mapping"):
        concrete = getattr(mapping, "_mapping", None)
        if concrete is not None:
            mapping = concrete
            mapping_name = type(mapping).__name__

    slot_by_name = {str(name): index for index, (name, _shape) in enumerate(slots)}
    tp_rank = int(getattr(mapping, "tp_rank", 0))
    contributions: dict[int, tuple[torch.Tensor, torch.Tensor]] = {}

    if mapping_name in {"DirectMapping", "ReplicatedMapping"}:
        slot = slot_by_name.get(str(mapping.hf_param))
        if slot is None:
            return None
        contributions[slot] = (local_indices.to(torch.int32), local_values)

    elif mapping_name == "ColumnParallelMapping":
        slot = slot_by_name.get(str(mapping.hf_param))
        if slot is None:
            return None
        global_indices = local_indices + tp_rank * record.param.numel()
        contributions[slot] = (global_indices.to(torch.int32), local_values)

    elif mapping_name == "RowParallelMapping":
        slot = slot_by_name.get(str(mapping.hf_param))
        if slot is None:
            return None
        if record.param.ndim <= 1:
            global_indices = local_indices
        else:
            local_width = record.param.shape[1]
            full_width = slots[slot][1][1]
            row = torch.div(local_indices, local_width, rounding_mode="floor")
            column = local_indices.remainder(local_width)
            global_indices = (
                row * full_width + tp_rank * local_width + column
            )
        contributions[slot] = (global_indices.to(torch.int32), local_values)

    elif mapping_name == "GatedMLPMapping":
        if record.param.shape[0] % 2:
            return None
        gate_slot = slot_by_name.get(str(mapping.hf_param["gate"]))
        up_slot = slot_by_name.get(str(mapping.hf_param["up"]))
        if gate_slot is None or up_slot is None:
            return None
        width = record.param.numel() // record.param.shape[0]
        half_rows = record.param.shape[0] // 2
        row = torch.div(local_indices, width, rounding_mode="floor")
        inner = local_indices.remainder(width)
        gate_mask = row < half_rows
        for slot, mask, local_row in (
            (gate_slot, gate_mask, row),
            (up_slot, ~gate_mask, row - half_rows),
        ):
            global_indices = (
                (tp_rank * half_rows + local_row[mask]) * width + inner[mask]
            )
            contributions[slot] = (
                global_indices.to(torch.int32),
                local_values[mask],
            )

    elif mapping_name == "QKVMapping":
        config = mapping._get_config(record.module)
        if getattr(config, "attention_output_gate", False):
            return None
        q_slot = slot_by_name.get(str(mapping.hf_param["q"]))
        k_slot = slot_by_name.get(str(mapping.hf_param["k"]))
        v_slot = slot_by_name.get(str(mapping.hf_param["v"]))
        if q_slot is None or k_slot is None or v_slot is None:
            return None
        head_count = int(config.num_attention_heads)
        group_count = int(config.num_query_groups)
        heads_per_group = head_count // group_count
        head_size = int(config.kv_channels or (config.hidden_size // head_count))
        width = record.param.numel() // record.param.shape[0]
        local_row = torch.div(local_indices, width, rounding_mode="floor")
        inner = local_indices.remainder(width)
        packed_row = tp_rank * record.param.shape[0] + local_row
        packed_head = torch.div(packed_row, head_size, rounding_mode="floor")
        head_inner = packed_row.remainder(head_size)
        group_width = heads_per_group + 2
        group = torch.div(packed_head, group_width, rounding_mode="floor")
        position = packed_head.remainder(group_width)
        q_mask = position < heads_per_group
        k_mask = position == heads_per_group
        v_mask = position == heads_per_group + 1
        q_row = (group * heads_per_group + position) * head_size + head_inner
        kv_row = group * head_size + head_inner
        for slot, mask, output_row in (
            (q_slot, q_mask, q_row),
            (k_slot, k_mask, kv_row),
            (v_slot, v_mask, kv_row),
        ):
            contributions[slot] = (
                (output_row[mask] * width + inner[mask]).to(torch.int32),
                local_values[mask],
            )
    else:
        return None

    counts = torch.zeros(len(slots), dtype=torch.int64)
    index_parts: list[torch.Tensor] = []
    value_parts: list[torch.Tensor] = []
    for slot in range(len(slots)):
        indices, values = contributions.get(
            slot,
            (
                torch.empty(0, dtype=torch.int32, device=local_indices.device),
                torch.empty(0, dtype=local_values.dtype, device=local_values.device),
            ),
        )
        counts[slot] = indices.numel()
        index_parts.append(indices)
        value_parts.append(values)
    return slots, counts, torch.cat(index_parts), torch.cat(value_parts)


_INTEGER_DTYPE = {
    1: torch.uint8,
    2: torch.int16,
    4: torch.int32,
    8: torch.int64,
}


def clone_cpu_snapshot(tensor: torch.Tensor) -> torch.Tensor:
    """Create an owning CPU snapshot that cannot alias a mutable weight backup."""
    return tensor.detach().cpu().contiguous().clone()


def local_bit_exact_diff(
    current: torch.Tensor, snapshot: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return changed local flat indices and current values, losslessly."""
    current = current.detach().contiguous().view(-1)
    snapshot = snapshot.detach().contiguous().view(-1)
    if current.shape != snapshot.shape or current.dtype != snapshot.dtype:
        raise RuntimeError("Megatron sparse local snapshot shape/dtype mismatch")
    integer_dtype = _INTEGER_DTYPE.get(current.element_size())
    if integer_dtype is None:
        raise NotImplementedError(
            f"Unsupported sparse local dtype {current.dtype}"
        )
    changed = current.view(integer_dtype) != snapshot.view(integer_dtype)
    indices = changed.nonzero(as_tuple=False).view(-1)
    return indices, current[indices]
