# Copyright 2025 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Export local shard deltas through VIME's ordinary HF conversion rules.

NaN placeholder shards preserve coordinate layout without dense inter-rank
weight gathers. Only floating, unquantized value-preserving layouts are supported.
Probe buffers and output scans remain dense and can grow with TP size.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


@dataclass
class McoreParamExport:
    """One native VIME parameter row: geometry, local converter and wire slots.
    ``param is None`` marks a parameter owned by another pipeline stage: the
    rank ships a zero-count lockstep row for it (probe/module unused)."""

    megatron_name: str
    param: torch.Tensor | None
    probe: Any
    module: Any  # reserved converter context; native VIME probes use None
    gather_group: ProcessGroup | None = None
    contributes: bool = True
    # GLOBAL slot table for this directory row (set by the index-build
    # exchange): identical on every rank of the row's merge group.
    slots: list | None = None
    hf_vocab_size: int | None = None
    # Owner dtype shared with PP placeholders to preserve gather queue order.
    dtype: torch.dtype | None = None


def trim_hf_vocab_padding(name: str, tensor: torch.Tensor, vocab_size: int | None):
    """Remove Megatron's TP vocabulary padding at the HF wire boundary."""
    if (
        vocab_size is not None
        and tensor.ndim >= 1
        and tensor.shape[0] > vocab_size
        and (name.endswith("embed_tokens.weight") or name.endswith("lm_head.weight"))
    ):
        return tensor[:vocab_size]
    return tensor


def build_export_index(iterator, megatron_model) -> list[McoreParamExport]:
    """Use VIME's global parameter directory; all ranks visit the same rows.

    src_rank chooses the PP owner within each TP lane. It is not a global
    unique sender: distinct TP shards and EP experts must all contribute.
    WORLD routing also includes rank zero for every PP/EP combination.
    """
    from megatron.core import parallel_state as mpu

    from .sparse_hf_export import SparseHFProbe

    del megatron_model  # The iterator owns the live model used by full export.
    local_weights = iterator.local_weights()
    rank = torch.distributed.get_rank()
    index = []
    for info in iterator.param_infos:
        if not info.dtype.is_floating_point:
            raise ValueError(f"Sparse conversion requires floating weights: {info.name}")
        param = local_weights.get(info.name)
        expert = ".experts." in info.name
        if expert:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
            tp_rank = mpu.get_expert_tensor_parallel_rank()
            replica = mpu.get_expert_data_parallel_rank()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()
            tp_rank = mpu.get_tensor_model_parallel_rank()
            replica = mpu.get_data_parallel_rank(with_context_parallel=True)
        sharded = info.attrs.get("tensor_model_parallel", False) and info.attrs.get("parallel_mode") != "duplicated"
        contributes = param is not None and info.src_rank == rank and replica == 0 and (sharded or tp_rank == 0)
        index.append(
            McoreParamExport(
                megatron_name=info.name,
                param=param,
                probe=(
                    SparseHFProbe(iterator.args, iterator.model_name, info, tp_rank, tp_size)
                    if param is not None
                    else None
                ),
                module=None,
                gather_group=torch.distributed.group.WORLD,
                contributes=contributes,
                hf_vocab_size=iterator.hf_vocab_size,
                dtype=info.dtype,
            )
        )
    _exchange_slot_tables(index)
    if any(record.slots is None for record in index):
        raise ValueError("VIME sparse parameter directory contains an unowned row")
    return index


def _exchange_slot_tables(index: list[McoreParamExport]) -> None:
    """Exchange native VIME output slots for the shared global parameter rows.

    PP placeholders inherit owner shapes/dtypes. The ordered union is identical
    on every rank, including rows with no local changes or only replicated data.
    """
    local_rows: list = []
    for rec in index:
        if rec.param is None:
            local_rows.append(None)
            continue
        if rec.slots is None:
            empty_idx = torch.empty(0, dtype=torch.int64, device=rec.param.device)
            empty_val = torch.empty(0, dtype=rec.param.dtype, device=rec.param.device)
            slots, *_ = mcore_hf_delta_entry(rec, empty_idx, empty_val)
            rec.slots = slots
        local_rows.append((rec.slots, rec.param.dtype))

    world = torch.distributed.get_world_size()
    gathered: list = [None] * world
    torch.distributed.all_gather_object(gathered, local_rows)
    n_rows = len(local_rows)
    assert all(len(rows) == n_rows for rows in gathered), (
        f"directory row counts diverge across ranks: {[len(r) for r in gathered]} -- "
        "the VIME's global enumeration is expected to be structurally parallel"
    )
    for k, rec in enumerate(index):
        union: dict = {}
        dtypes = set()
        for rows in gathered:  # rank order -> identical result on every rank
            row = rows[k]
            if row is None:
                continue
            slots, dtype = row
            dtypes.add(dtype)
            for slot in slots:
                union[(slot[0], tuple(slot[1]))] = None  # ordered-set semantics
        if len(dtypes) > 1:
            raise ValueError(f"directory row {k} has inconsistent owner dtypes: {dtypes}")
        rec.dtype = next(iter(dtypes), None)
        rec.slots = [(n, tuple(shape)) for (n, shape) in union.keys()] if union else None


def mcore_hf_delta_entry(
    rec: McoreParamExport,
    lidx: torch.Tensor,
    lval: torch.Tensor,
):
    """Convert one parameter's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, hf_idx, hf_val)``.

    Scatters the delta into a NaN buffer of the LOCAL shard shape (exactly what
    the native VIME converter receives), runs the comm-stubbed probe -- real
    group sizes, gathers synthesized locally, so the mapping executes its real
    TP>1 code paths -- and extracts each output slot's surviving positions.
    The slot list is cached after the first call (the converter's output names
    are deterministic, so every rank's cache agrees and the batched gather
    stays aligned)."""
    dtype_str = str(lval.dtype).replace("torch.", "")
    assert (
        lval.numel() == 0 or lval.is_floating_point()
    ), f"{rec.megatron_name}: NaN sentinels require a floating-point param, got {lval.dtype}"

    cached = rec.slots
    if rec.param is None:
        # owned by another pipeline stage: pure lockstep row. The slot table
        # was pre-seeded from the owner stage (fail loud if not -- an
        # unsized entry would silently misalign the batched gather).
        assert cached is not None, f"{rec.megatron_name}: slot table not pre-seeded for a non-owned PP param"
        assert lidx.numel() == 0, f"{rec.megatron_name}: delta reported for a param this rank does not own"
    if lidx.numel() == 0 and cached is not None:
        # empty delta: the slot table froze after the first probe, so the
        # zero-count lockstep entry needs no probe run at all -- skip the
        # buffer build, the transform and the full-output NaN scan.
        return (
            cached,
            dtype_str,
            torch.zeros(len(cached), dtype=torch.int64),
            torch.empty(0, dtype=torch.int32, device=lval.device),
            torch.empty(0, dtype=lval.dtype, device=lval.device),
        )

    buf = torch.full(tuple(rec.param.shape), float("nan"), dtype=lval.dtype, device=lval.device)
    if lidx.numel():
        buf.view(-1)[lidx] = lval

    outs = {
        name: trim_hf_vocab_padding(name, tensor, rec.hf_vocab_size)
        for name, tensor in rec.probe.megatron_to_hf(buf, rec.module).items()
    }

    key = rec.megatron_name
    slots = rec.slots
    if slots is None:
        # only reachable from the index-build exchange itself, which probes a
        # zero delta to reveal this rank's local list before the row unions
        # are installed on the recs; steady entries always find rec.slots.
        slots = [(n, tuple(int(x) for x in t.shape)) for n, t in outs.items()]
    unknown = set(outs) - {n for n, _ in slots}
    assert not unknown, (
        f"{key}: probe emitted slots missing from the global table {sorted(unknown)[:4]} -- "
        "non-deterministic converter naming would misalign the batched gather"
    )
    counts = torch.zeros(len(slots), dtype=torch.int64)
    idx_pieces: list[torch.Tensor] = []
    val_pieces: list[torch.Tensor] = []
    for s_i, (sname, _sshape) in enumerate(slots):
        out = outs.get(sname)
        if out is None:
            continue  # another rank's slot (e.g. its ep-local experts): zero count here
        fl = out.reshape(-1)
        p_ = (~torch.isnan(fl)).nonzero(as_tuple=False).view(-1)
        if p_.numel():
            counts[s_i] = p_.numel()
            idx_pieces.append(p_.to(torch.int32))
            val_pieces.append(fl[p_])
    if idx_pieces:
        hf_idx = torch.cat(idx_pieces)
        hf_val = torch.cat(val_pieces)
    else:
        hf_idx = torch.empty(0, dtype=torch.int32, device=lval.device)
        hf_val = torch.empty(0, dtype=lval.dtype, device=lval.device)
    return slots, dtype_str, counts, hf_idx, hf_val


def _device_shard(record: McoreParamExport) -> torch.Tensor:
    """Read current storage every sync; never diff a TensorBackuper CPU copy."""
    local = record.param.detach().contiguous().view(-1)
    if local.device.type == "cpu":
        raise RuntimeError(f"{record.megatron_name}: delta export requires a live accelerator shard")
    if not local.is_floating_point():
        raise TypeError(f"{record.megatron_name}: NaN probe requires floating point weights")
    return local


def prime_delta_snapshots(index, snapshots, *, pin=False):
    """Keep baseline snapshots on the host to leave accelerator memory for training."""
    for record in index:
        if record.param is None or not record.contributes:
            continue
        local = _device_shard(record)
        snapshot = torch.empty_like(local, device="cpu", pin_memory=pin)
        snapshot.copy_(local, non_blocking=True)
        snapshots[record.megatron_name] = snapshot


def validate_export_weights(index):
    """Reject NaN on every rank before payloads or snapshot changes.

    NaN is the probe/loader sentinel. Reduce device-side flags once per update,
    before ranks can enter different payload collectives; a local exception
    during streaming would strand peers. No per-parameter host synchronization.
    """
    flags = [torch.isnan(rec.param.detach()).any() for rec in index if rec.param is not None]
    invalid = (
        torch.stack(flags).any().to(torch.int32)
        if flags
        else torch.zeros((), dtype=torch.int32, device=torch.accelerator.current_device_index())
    )
    torch.distributed.all_reduce(invalid, op=torch.distributed.ReduceOp.MAX)
    if bool(invalid):
        raise ValueError("Sparse HCCL cannot export NaN weights; update rejected on all trainer ranks")


class _SnapshotRefreshQueue:
    """Bound sparse baseline refreshes by bytes and row count.

    Two blocking packed copies per batch establish host visibility before
    index_copy_. This avoids two device synchronizations for every parameter.
    The queue is drained before the generator completes and the next version.
    """

    def __init__(self, max_bytes=32 << 20, max_rows=32):
        self.max_bytes, self.max_rows = max_bytes, max_rows
        self.rows = []
        self.nbytes = 0

    def add(self, snapshot, indices, values):
        if not indices.numel():
            return
        size = indices.numel() * (indices.element_size() + values.element_size())
        if self.rows and (self.nbytes + size > self.max_bytes or self.rows[0][2].dtype != values.dtype):
            self.flush()
        self.rows.append((snapshot, indices, values))
        self.nbytes += size
        if self.nbytes >= self.max_bytes or len(self.rows) >= self.max_rows:
            self.flush()

    def flush(self):
        if not self.rows:
            return
        indices = torch.cat([row[1] for row in self.rows]).to("cpu", non_blocking=False)
        values = torch.cat([row[2] for row in self.rows]).to("cpu", non_blocking=False)
        offset = 0
        for snapshot, local_indices, _ in self.rows:
            end = offset + local_indices.numel()
            snapshot.index_copy_(0, indices[offset:end], values[offset:end])
            offset = end
        self.rows.clear()
        self.nbytes = 0


def shard_delta_mask(local_new, local_snap):
    """The same bit comparison as verl, used by batched extraction."""
    int_dtype = _DTYPE_INT.get(local_new.element_size())
    if int_dtype is None:
        raise ValueError(f"unsupported element size {local_new.element_size()}")
    return local_new.view(int_dtype) != local_snap.view(int_dtype)


def compact_shard_masks(locals_, masks):
    """One ordered nonzero per bounded batch, matching per-shard bit diffs.

    Masks remain bitwise comparisons, as in verl. Only the variable-length
    extraction is batched. The single boundary vector D2H replaces per-shard
    dynamic shape synchronization; no HF mapping or ownership changes here.
    """
    if not locals_:
        return []
    ends, total = [], 0
    for local in locals_:
        total += local.numel()
        ends.append(total)
    positions = torch.cat(masks).nonzero(as_tuple=False).view(-1)
    boundaries = torch.tensor(ends, dtype=torch.int64, device=positions.device)
    splits = torch.searchsorted(positions, boundaries).cpu().tolist()
    result, start, base = [], 0, 0
    for local, end, boundary in zip(locals_, splits, ends, strict=True):
        indices = positions[start:end] - base
        row = (indices, local.index_select(0, indices))
        result.append(row)
        start, base = end, boundary
    return result


def iter_delta_entries(index, snapshots, *, batch_size=32):
    """Batch dynamic diff extraction, preserving verl's record/slot ordering."""
    max_mask_bytes = 32 << 20
    pending, masks, locals_ = [], [], []
    staging = {}
    refresh = _SnapshotRefreshQueue()
    mask_bytes = 0
    batch_key = None

    def flush():
        deltas = iter(compact_shard_masks(locals_, masks))
        for record, local, snapshot in pending:
            if snapshot is not None:
                indices, values = next(deltas)
                if snapshot.device == local.device:
                    snapshot.copy_(local, non_blocking=True)
                elif indices.numel():
                    refresh.add(snapshot, indices, values)
            else:
                device = local.device if local is not None else torch.accelerator.current_device_index()
                dtype = local.dtype if local is not None else record.dtype
                indices = torch.empty(0, dtype=torch.int64, device=device)
                values = torch.empty(0, dtype=dtype, device=device)
            yield (
                *mcore_hf_delta_entry(record, indices, values),
                record.gather_group,
            )
        pending.clear()
        masks.clear()
        locals_.clear()

    for record in index:
        local = _device_shard(record) if record.param is not None else None
        active = local is not None and record.contributes
        key = (local.device, local.dtype) if active else batch_key
        size = local.numel() if active else 0
        if pending and (key != batch_key or mask_bytes + size > max_mask_bytes or len(pending) >= batch_size):
            yield from flush()
            mask_bytes = 0
        batch_key = key
        snapshot = None
        if active:
            snapshot = snapshots.get(record.megatron_name)
            if snapshot is None or snapshot.shape != local.shape or snapshot.dtype != local.dtype:
                raise RuntimeError(f"{record.megatron_name}: no matching seed snapshot for this shard")
            base = snapshot
            if snapshot.device != local.device:
                buffer = staging.get(key)
                if buffer is None or buffer.numel() < local.numel():
                    buffer = torch.empty_like(local)
                    staging[key] = buffer
                base = buffer[: local.numel()]
                base.copy_(snapshot, non_blocking=True)
            masks.append(shard_delta_mask(local, base))
            locals_.append(local)
            mask_bytes += size
        pending.append((record, local, snapshot))
    if pending:
        yield from flush()
    refresh.flush()
