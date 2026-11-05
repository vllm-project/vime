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
"""Export Megatron shard deltas in HF checkpoint coordinates via Megatron-Bridge.

Build an ordered export index, compare local shards with snapshots, and
convert changed entries using Bridge mappings with NaN sentinels. Probe
mappings preserve parallel sizes and ranks while replacing collectives
with local placeholders; unsupported communication raises an error.

Supports TP, EP and PP/VPP; LoRA is excluded. Mappings must preserve
individual values rather than blend contributions, so NaN sentinels remain
valid. Bridge upgrades require differential validation against full export.
"""

from __future__ import annotations

import copy
import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup

logger = logging.getLogger(__name__)

_DTYPE_INT = {1: torch.uint8, 2: torch.int16, 4: torch.int32, 8: torch.int64}


class _ProbeGroup:
    """Size/rank-faithful stand-in for a process group on probe copies: value
    math inside ``megatron_to_hf`` (e.g. Mamba's per-rank de-interleave dims)
    sees the REAL parallel sizes, while any attempt to actually communicate
    through the group fails loud (the probe stubs the bridge's comm helpers;
    anything else reaching a group is an unstubbed communication pattern)."""

    def __init__(self, size: int, rank: int):
        self._size = int(size)
        self._rank = int(rank)

    def size(self) -> int:
        return self._size

    def rank(self) -> int:
        return self._rank

    def __getattr__(self, name):
        raise RuntimeError(
            f"probe process group asked for {name!r}: this mapping communicates outside "
            "the stubbed helpers (gather_from_tp_ranks / gather_from_ep_ranks[_scale] / "
            "pp broadcasts) -- extend make_probe's comm stubs before trusting its export"
        )


def _nan_block(shape, dtype, device, pool) -> torch.Tensor:
    """Read-only all-NaN placeholder for another rank's gather chunk, pooled by
    (shape, dtype, device): the gather stub hands these out on every probe call
    for every param, and they are only ever READ (the transforms are functional),
    so one block per distinct shape serves the whole model for the run's
    lifetime instead of a fresh cudaMalloc+fill per param per sync."""
    key = (tuple(shape), dtype, str(device))
    t = pool.get(key)
    if t is None:
        t = torch.full(tuple(shape), float("nan"), dtype=dtype, device=device)
        pool[key] = t
    return t


def _warm_lazy_mappings(mapping, module) -> None:
    """Force AutoMapping's lazy concrete delegate into existence: AutoMapping
    resolves its Column/Row/Replicated delegate on first use, snapshotting
    whatever process groups exist at that moment -- if that first use happened
    inside the probe, the delegate would be born with the REAL groups and
    gather for real (the qkv-bias double-size bug). Self-only on purpose:
    ``_inject`` warms every child right before copying it, so each node of the
    mapping tree is warmed exactly once."""
    if hasattr(mapping, "_detect_parallelism_type") and getattr(mapping, "_mapping", None) is None:
        try:
            t = mapping._detect_parallelism_type(module)
            mapping._mapping = mapping._get_or_create_mapping(t)
            mapping._detected_type = t
        except Exception as e:  # pragma: no cover - defensive; probe falls back to real groups
            logger.warning("could not warm lazy mapping %s: %s", type(mapping).__name__, e)


def make_probe(mapping, module, nan_pool=None):
    """Copy a Megatron-Bridge param mapping tree and turn ``megatron_to_hf``
    into a communication-free LOCAL transform that still runs the REAL
    (tp_size > 1) code paths: every copy's groups become size-faithful
    :class:`_ProbeGroup` stand-ins and the bridge's comm helpers are stubbed
    with local synthesis. The copy is recursive (composite mappings delegate to
    inner mappings -- QKVMapping._tp_mapping is an AutoMapping which itself
    delegates to a lazily-created concrete mapping; none of these receive the
    outer stubbing on their own)."""
    from megatron.bridge.models.conversion.param_mapping import MegatronParamMapping
    from megatron.core.utils import get_pg_rank, get_pg_size

    _warm_lazy_mappings(mapping, module)
    nan_pool = {} if nan_pool is None else nan_pool

    def _stub(c):
        # size-faithful groups, sizes/ranks read from the ORIGINAL (real) groups
        # BEFORE replacement. The probe only ever runs on the stage that OWNS
        # the param, where a pp broadcast is the identity -- a (1, 0) group
        # keeps the pp helpers on that fast path regardless of the real pp size.
        c.pp_group = _ProbeGroup(1, 0)
        for attr in ("ep_group", "_tp_group", "_etp_group"):
            g = getattr(c, attr, None)
            setattr(c, attr, _ProbeGroup(get_pg_size(g), get_pg_rank(g)))

        def _gather_tp(tensor, _c=c):
            # what a real all_gather over the tp group would produce, minus the
            # other ranks' data: our shard rides at its true rank index, every
            # other chunk is a pooled read-only NaN block (those ranks export
            # their own contributions; transforms never write into their inputs,
            # which the TP>1 bitwise differential revalidates).
            nan = _nan_block(tensor.shape, tensor.dtype, tensor.device, nan_pool)
            out = [nan] * _c.tp_size
            out[_c.tp_rank] = tensor
            return out

        c.gather_from_tp_ranks = _gather_tp
        # each rank's probe emits only its own local experts, under the hf name
        # bound with the GLOBAL expert id at task construction; the engine
        # merges entries over the etp x ep group, so the ep fan-out reduces to
        # the bound name.
        c.gather_from_ep_ranks = lambda w, mod, name: {str(name): w}
        # mirrors the real helper's tail (unsqueeze(0) ... squeeze().unsqueeze(-1))
        c.gather_from_ep_ranks_scale = lambda w, mod, name: {str(name): w.unsqueeze(0).squeeze().unsqueeze(-1)}
        return c

    def _inject(m):
        c = _stub(copy.copy(m))
        for attr, value in list(vars(c).items()):
            if isinstance(value, MegatronParamMapping):
                # warm BEFORE copying the child: the lazy delegate must exist so
                # the child copy's vars() include it and the recursion reaches it.
                _warm_lazy_mappings(value, module)
                setattr(c, attr, _inject(value))
        return c

    return _inject(mapping)


@dataclass
class McoreParamExport:
    """One mcore parameter's export record: geometry + probe + module handle.
    ``param is None`` marks a parameter owned by another pipeline stage: the
    rank ships a zero-count lockstep row for it (probe/module unused)."""

    megatron_name: str
    param: torch.Tensor | None
    probe: Any
    module: Any  # module handle megatron_to_hf reads config from
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


def build_export_index(bridge, megatron_model) -> list[McoreParamExport]:
    """Enumerate every mcore parameter through the bridge's conversion tasks
    and precompute its probe + wire routing.

    No shard geometry is hand-computed here: the comm-stubbed probe emits final
    HF coordinates straight from the local shard, so the engine only needs to
    know WHICH group's entries to merge per parameter (the record's
    ``gather_group``) and which
    ranks contribute. The index is built once (parameter sets are static) and
    reused by both the shard export and the delta entry hook.

    Lockstep: ``get_conversion_tasks`` already enumerates the GLOBAL parameter
    list in an order identical on every rank (the bridge allgathers and sorts
    names across pp ranks, dedupes tied embeddings, and leaves
    ``param_weight=None`` placeholders for parameters owned by other pipeline
    stages). Under PP>1 those placeholders become zero-count lockstep rows and
    every param's entries merge over the WORLD group (owner-stage ranks
    contribute, dp/cp replicas and other stages stay empty); their slot tables
    are pre-seeded once via :func:`_preseed_slot_tables`. Under PP=1 the
    routing is unchanged (tp / etp x ep subgroups, replicated rank-0 direct).
    """
    from megatron.core import parallel_state as mpu

    pp_world = mpu.get_pipeline_model_parallel_world_size()
    tp_group = mpu.get_tensor_model_parallel_group()
    tp_world = torch.distributed.get_world_size(group=tp_group)
    ep_size = mpu.get_expert_model_parallel_world_size()
    world = torch.distributed.group.WORLD

    nan_pool = {}
    hf_config = bridge.hf_pretrained.config
    hf_config = getattr(hf_config, "text_config", hf_config)
    hf_vocab_size = getattr(hf_config, "vocab_size", None)
    tasks = bridge.get_conversion_tasks(megatron_model)
    index: list[McoreParamExport] = []
    for task in tasks:
        mapping = task.mapping
        param = task.param_weight
        name = task.global_param_name
        if param is None:
            if pp_world == 1:
                # not a pp placeholder (e.g. a skipped mapping); nothing to ship
                continue
            # owned by another pipeline stage: zero-count lockstep row -- the
            # rank walks the same global directory and contributes nothing.
            index.append(
                McoreParamExport(
                    megatron_name=name,
                    param=None,
                    gather_group=world,
                    contributes=False,
                    probe=None,
                    module=None,
                )
            )
            continue
        module = task.megatron_module

        is_expert = mapping.is_expert and (ep_size > 1 or tp_world > 1)
        is_tp_sharded = getattr(param, "tensor_model_parallel", False) and tp_world > 1

        if pp_world > 1:
            # single WORLD merge group for every param: the wire master (global
            # rank 0) sits in every gather regardless of which stage owns the
            # param, so no relay hop is needed. Owner-stage ranks contribute
            # their shard pieces; dp/cp replicas dedupe via ``contributes``
            # (identical copies -- exactly one replica set ships).
            if is_expert:
                contributes = mpu.get_expert_data_parallel_rank() == 0
            elif is_tp_sharded:
                contributes = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            else:
                contributes = (
                    mpu.get_tensor_model_parallel_rank() == 0
                    and mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                )
            gather_group = world
        elif is_expert:
            # every rank holding a piece of this expert set contributes; the
            # engine merges their probe entries over the joint etp x ep group.
            gather_group = mpu.get_expert_tensor_and_model_parallel_group()
            contributes = mpu.get_expert_data_parallel_rank() == 0
        elif is_tp_sharded:
            gather_group = tp_group
            contributes = mpu.get_data_parallel_rank(with_context_parallel=True) == 0
        else:
            # replicated: engine's pg=None path (rank 0 consumes its own entry
            # directly, replicas stay in lockstep via zero counts).
            gather_group = None
            contributes = torch.distributed.get_rank() == 0

        index.append(
            McoreParamExport(
                megatron_name=name,
                param=param,
                gather_group=gather_group,
                contributes=contributes,
                probe=make_probe(mapping, module, nan_pool),
                module=module,
                hf_vocab_size=hf_vocab_size,
            )
        )

    _exchange_slot_tables(index)
    # a row with an empty union has no owner on ANY rank: the bridge skipped
    # it everywhere (no mapping / missing HF key), so it is outside the
    # bridge's own export scope -- exactly what PP=1 does by skipping
    # param_weight=None tasks. Drop it (all ranks agree: the exchange result
    # is identical) instead of shipping unsized rows.
    unowned = [rec.megatron_name for rec in index if rec.slots is None]
    if unowned:
        logger.warning(
            "delta export drops %d row(s) the bridge skipped on every rank "
            "(no mapping or missing HF key; out of export scope): %s",
            len(unowned),
            unowned[:8],
        )
        index = [rec for rec in index if rec.slots is not None]
    return index


def _exchange_slot_tables(index: list[McoreParamExport]) -> None:
    """One-time GLOBAL slot-table exchange, run at every world size, keyed by
    DIRECTORY ROW (not by param name).

    The engine's batched gather merges entries BY SLOT POSITION within each
    directory row and rank 0 names the merged pieces from ITS OWN list, so
    the per-row list must be identical on every rank of the row's merge
    group. Name-keyed merging is not enough on mcore: expert param NAMES
    embed the expert ids (``...experts.linear_fc1.weight0`` on ep rank 0 vs
    ``weight1`` on ep rank 1), so the same row carries different names per
    rank while still gathering together. Each rank therefore probes a zero
    delta through its OWNED rows to reveal its local ``(hf_name, hf_shape)``
    list, one ``all_gather_object`` over WORLD exchanges the ordered row
    lists, and every row's table becomes the rank-order first-seen union --
    identical everywhere by construction. Dense/TP rows (same list on every
    rank) dedup to themselves; expert rows concatenate the per-ep-rank
    lists; PP placeholder rows inherit the owners' union and dtype."""
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
        "the bridge's global enumeration is expected to be structurally parallel"
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
    """Probe one mcore param's shard-local delta into its final HF-coordinate
    entry ``(slots, dtype_str, counts, hf_idx, hf_val)``.

    Scatters the delta into a NaN buffer of the LOCAL shard shape (exactly what
    the real ``megatron_to_hf`` receives), runs the comm-stubbed probe -- real
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
