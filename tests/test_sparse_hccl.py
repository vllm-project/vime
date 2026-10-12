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
"""Sparse HCCL regression checks and sparse/full weight comparison.

pytest tests/test_sparse_hccl.py runs loader, dtype, NaN, checksum and failure
recovery checks. torchrun runs the real Bridge differential; add
DELTA_ENTRY_ORACLE=1 to compare one sparse update with full exported weights.
Use --gather-regression for the four-rank payload length/buffer reuse checks.

Compare sparse reconstruction with an independent real Bridge full export.

Guard TP/PP/EP coordinate conversion, global parameter ordering, expert mappings
and snapshot reconstruction using tiny models and bitwise tensor comparisons.
DELTA_ENTRY_ORACLE=1 compares one sparse update with full export; MIXED_DTYPE=1 adds
FP32 norms beside BF16 matrices to exercise PP placeholder dtype alignment.
Run with torchrun in the Megatron-Bridge environment, not ordinary pytest.

Differential export oracle for the comm-stubbed mcore probe under TP/PP/EP.

For every conversion task of a tiny model the REAL ``megatron_to_hf`` (true
collectives, run in lockstep by all ranks) must equal the probe assembly:
each rank runs its comm-stubbed probe on its own local shard, the non-NaN
contributions are merged across ranks, and the merged tensor is compared to
the real output BITWISE. This is the regression oracle for the probe's two
assumptions (communication confined to the stubbed helpers; transforms
rearrange rather than blend) -- rerun it whenever Megatron-Bridge is upgraded.

Run under torchrun on >=2 GPUs, e.g.::

    MODEL_KIND=qwen2 torchrun --nproc_per_node=2 tests/test_sparse_hccl.py
    MODEL_KIND=nemotron_h torchrun --nproc_per_node=2 tests/test_sparse_hccl.py

``MODEL_KIND``: ``qwen2`` (Column/Row/QKV/GatedMLP/Replicated), ``qwen3_moe``
(adds fused-MoE expert mappings; uses etp=TP), ``nemotron_h`` (Mamba mixers --
the tp_size-dependent de-interleave the probe must reproduce).
"""

import os
import pathlib
import sys
from datetime import timedelta
from types import SimpleNamespace

import pytest
import torch
import torch.distributed as dist

# torchrun executes this file with its test directory as sys.path[0]. Prefer the checkout
# being validated over an editable installation of another Vime revision.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import _verify_dense_load
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import (
    SparseWeightPatch,
    apply_sparse_hf_patches_with_loader,
)

import vime.backends.megatron_utils  # noqa: F401 - Bootstrap NPU before importing Megatron.
from vime.backends.megatron_utils.update_weight import megatron_delta_export
from vime.backends.megatron_utils.update_weight import update_weight_from_sparse_hccl as updater_module
from vime.backends.megatron_utils.update_weight.delta_sync import checksum, shard_delta_indices
from vime.backends.megatron_utils.update_weight.delta_sync import sparse_gather as gather_module
from vime.backends.megatron_utils.update_weight.delta_sync.sparse_gather import (
    GatherWorkspace,
    gather_slot_entries_to_rank0,
)
from vime.backends.megatron_utils.update_weight.megatron_delta_export import _device_shard

MODEL_KIND = os.environ.get("MODEL_KIND", "qwen2")
TP = int(os.environ.get("TP_SIZE", "2"))
PP = int(os.environ.get("PP_SIZE", "1"))
EP = int(os.environ.get("EP_SIZE", "1"))


def _build_tiny_hf_dir(rank: int) -> str:
    from transformers import AutoConfig

    cfg_dir = f"/tmp/tiny_{MODEL_KIND}_probe_diff_v4"
    if MODEL_KIND in ("qwen2", "qwen2_tied"):
        hf_config = AutoConfig.for_model(
            "qwen2",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=(MODEL_KIND == "qwen2_tied"),
        )
        hf_config.architectures = ["Qwen2ForCausalLM"]
    elif MODEL_KIND == "qwen3":
        hf_config = AutoConfig.for_model(
            "qwen3",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["Qwen3ForCausalLM"]
    elif MODEL_KIND == "qwen3_moe":
        hf_config = AutoConfig.for_model(
            "qwen3_moe",
            hidden_size=64,
            intermediate_size=128,
            moe_intermediate_size=64,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            num_experts=4,
            num_experts_per_tok=2,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["Qwen3MoeForCausalLM"]
    elif MODEL_KIND == "nemotron_h":
        # hybrid mamba2+attention+mlp stack; sizes divisible by TP=2
        # (n_groups and mamba_num_heads both TP-divisible).
        hf_config = AutoConfig.for_model(
            "nemotron_h",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=4,
            hybrid_override_pattern="M*M-",
            num_attention_heads=4,
            num_key_value_heads=2,
            attention_head_dim=16,
            ssm_state_size=16,
            conv_kernel=4,
            expand=2,
            mamba_num_heads=8,
            mamba_head_dim=16,
            n_groups=2,
            chunk_size=32,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["NemotronHForCausalLM"]
    elif MODEL_KIND == "falcon_h1":
        # mamba2+attention parallel-hybrid; mamba_n_heads/n_groups TP=2 divisible
        hf_config = AutoConfig.for_model(
            "falcon_h1",
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            head_dim=16,
            mamba_d_ssm=64,
            mamba_n_heads=8,
            mamba_d_head=8,
            mamba_n_groups=2,
            mamba_d_state=16,
            mamba_d_conv=4,
            mamba_expand=2,
            mamba_chunk_size=32,
            mamba_norm_before_gate=False,
            vocab_size=512,
            max_position_embeddings=256,
            tie_word_embeddings=False,
        )
        hf_config.architectures = ["FalconH1ForCausalLM"]
    else:
        raise ValueError(f"unknown MODEL_KIND {MODEL_KIND!r}")

    if MODEL_KIND in ("nemotron_h", "falcon_h1"):
        # NemotronH's mixer pulls causal-conv1d / mamba-ssm kernels from the hub
        # at layer init; offline runs must fall back to the reference path.
        try:
            import transformers.integrations.hub_kernels as _hk

            _hk.lazy_load_kernel = lambda *a, **k: None
            import importlib

            _mnh = importlib.import_module(f"transformers.models.{MODEL_KIND}.modeling_{MODEL_KIND}")

            for _sym in ("lazy_load_kernel", "get_kernel"):
                if hasattr(_mnh, _sym):
                    setattr(_mnh, _sym, lambda *a, **k: None)
        except Exception as e:  # pragma: no cover
            print(f"kernel stub patch failed: {e}")

    if rank == 0 and not pathlib.Path(cfg_dir, "model.safetensors").exists():
        from transformers import AutoModelForCausalLM

        m = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16)
        m.save_pretrained(cfg_dir, safe_serialization=True)
        del m
        if MODEL_KIND in ("nemotron_h", "falcon_h1"):
            # newer transformers' serialization rename table emits
            # "backbone.embedding.weight" (singular) while the bridge mappings
            # (and the RELEASED checkpoints) use "backbone.embeddings.weight".
            # Normalize the stray key so the tiny checkpoint matches the
            # released format the bridge expects.
            from safetensors.torch import load_file, save_file

            f = str(pathlib.Path(cfg_dir, "model.safetensors"))
            sd = load_file(f)
            fixed = {k.replace("backbone.embedding.", "backbone.embeddings."): v for k, v in sd.items()}
            if fixed.keys() != sd.keys():
                save_file(fixed, f, metadata={"format": "pt"})
    dist.barrier()
    return cfg_dir


def _pp_differential(bridge, model, rank: int, world: int) -> None:
    """PP>1 differential: run the PRODUCTION export path (global directory +
    slot pre-seed + probe entries, a full-delta sync where every local element
    is 'changed') and compare the merged wire-view against
    ``bridge.export_hf_weights`` -- the bridge's own PP-broadcast full export,
    an independent chain. Bitwise, per HF tensor, with directory-lockstep and
    placeholder-row checks on the way."""
    from vime.backends.megatron_utils.update_weight.megatron_delta_export import (
        build_export_index,
        mcore_hf_delta_entry,
    )

    slot_cache: dict = {}
    index = build_export_index(bridge, model, slot_cache)

    # 1. directory lockstep: same ROW COUNT everywhere (mcore expert param
    # NAMES legitimately differ per ep rank -- the row is the unit).
    names = [r.megatron_name for r in index]
    all_names: list = [None] * world
    dist.all_gather_object(all_names, names)
    failures = []
    if rank == 0 and any(len(n) != len(names) for n in all_names):
        failures.append(f"directory row counts diverge: {[len(n) for n in all_names]}")

    # 1b. per-row slot-table identity: the batched gather merges BY SLOT
    # POSITION within a row and rank 0 names merged pieces from its own list,
    # so the row's table must be identical on every rank (the ep_size>1
    # collision this oracle now guards).
    row_tables = [r.slots for r in index]
    all_tables: list = [None] * world
    dist.all_gather_object(all_tables, row_tables)
    if rank == 0:
        for r_i, tbls in enumerate(all_tables):
            bad = [k for k in range(len(row_tables)) if tbls[k] != row_tables[k]][:3]
            if bad:
                failures.append(f"row slot tables diverge on rank {r_i} (rows {bad})")
                break

    # 2. entries: owned params ship a full delta (every local element), rows
    # owned by other stages must come back as zero-count lockstep entries.
    mine: dict = {}  # slot name -> (shape, positions int64 cpu, values bf16 cpu)
    for rec in index:
        if rec.param is None:
            e_idx = torch.empty(0, dtype=torch.int64, device="cuda")
            e_val = torch.empty(0, dtype=torch.bfloat16, device="cuda")
            slots, _dt, counts, hf_idx, _hf_val = mcore_hf_delta_entry(rec, 0, e_idx, e_val, slot_cache)
            if int(counts.sum()) != 0 or hf_idx.numel() != 0:
                failures.append(f"{rec.megatron_name}: placeholder row shipped a non-empty delta")
            continue
        if not rec.param.is_floating_point():
            continue
        local = rec.param.data.to(torch.bfloat16).reshape(-1)
        lidx = torch.arange(local.numel(), dtype=torch.int64, device=local.device)
        slots, _dt, counts, hf_idx, hf_val = mcore_hf_delta_entry(rec, 0, lidx, local, slot_cache)
        off = 0
        for (sname, sshape), c in zip(slots, counts.tolist(), strict=True):
            if c:
                pos = hf_idx[off : off + c].to(torch.int64).cpu()
                val = hf_val[off : off + c].cpu()
                off += c
                prev = mine.get(sname)
                if prev is None:
                    mine[sname] = (tuple(sshape), pos, val)
                else:
                    mine[sname] = (tuple(sshape), torch.cat([prev[1], pos]), torch.cat([prev[2], val]))

    # 3. independent reference: the bridge's own full export (PP broadcasts
    # inside; full tensors on every rank).
    exported = bridge.export_hf_weights(model, cpu=True, show_progress=False)
    ref = {entry[0]: entry[1].detach().to(torch.bfloat16).cpu() for entry in exported}

    all_pieces: list = [None] * world
    dist.all_gather_object(all_pieces, mine)
    n_checked = n_pass = 0
    if rank == 0:
        slot_names = {k for d in all_pieces for k in d}
        missing_ref = sorted(set(ref) - slot_names)
        if missing_ref:
            more = max(0, len(missing_ref) - 5)
            failures.append(f"HF tensors never covered by any entry: {missing_ref[:5]} (+{more})")
        for sname in sorted(slot_names):
            if sname not in ref:
                failures.append(f"entry slot {sname} absent from bridge export")
                continue
            full = ref[sname]
            merged = torch.full_like(full, float("nan"))
            covered = torch.zeros(full.numel(), dtype=torch.int32)
            mflat = merged.view(-1)
            for d in all_pieces:
                got = d.get(sname)
                if got is None:
                    continue
                sshape, pos, val = got
                if tuple(sshape) != tuple(full.shape):
                    failures.append(f"{sname}: slot shape {sshape} != export shape {tuple(full.shape)}")
                    continue
                both = covered[pos] > 0
                if both.any() and not torch.equal(mflat[pos[both]].view(torch.int16), val[both].view(torch.int16)):
                    failures.append(f"{sname}: conflicting rank contributions")
                mflat[pos] = val
                covered[pos] += 1
            n_checked += 1
            if torch.isnan(mflat).any():
                failures.append(f"{sname}: uncovered positions remain NaN")
            elif not torch.equal(mflat.view(torch.int16), full.view(-1).view(torch.int16)):
                failures.append(f"{sname}: bitwise mismatch")
            else:
                n_pass += 1
        print("=" * 60)
        for f in failures[:20]:
            print("FAIL:", f)
        print(f"PROBE DIFFERENTIAL [{MODEL_KIND} tp={TP} pp={PP}]: {n_pass}/{n_checked} bitwise-equal")
        print("PROBE DIFFERENTIAL PASSED" if (n_checked > 0 and not failures) else "PROBE DIFFERENTIAL FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and (failures or n_checked == 0):
        sys.exit(1)


def _delta_entry_differential(model, cfg_dir, rank):
    from types import SimpleNamespace

    from vime.backends.megatron_utils.update_weight.hf_weight_iterator_sparse_bridge import (
        HfWeightIteratorSparseBridge as HfWeightIteratorBridge,
    )
    from vime.backends.megatron_utils.update_weight.megatron_delta_export import (
        build_export_index,
        iter_delta_entries,
        prime_delta_snapshots,
        validate_export_weights,
    )
    from vime.backends.megatron_utils.update_weight.update_weight_from_sparse_hccl import _GatherQueue

    iterator = HfWeightIteratorBridge(SimpleNamespace(hf_checkpoint=cfg_dir, update_weight_buffer_size=4096), model)
    with iterator.model_context():
        index = build_export_index(iterator.bridge, model, {})
    assert all(record.param is None or record.param.device.type == "npu" for record in index)

    if os.environ.get("MIXED_DTYPE"):
        # Retain BF16 matrices but export FP32 norms on their owning PP stage.
        # Rebuild metadata so non-owner rows inherit the actual owner dtype.
        for record in index:
            if record.param is not None and record.param.ndim == 1:
                record.param.data = record.param.data.float()
        with iterator.model_context():
            index = build_export_index(iterator.bridge, model, {})
        assert {record.dtype for record in index} == {torch.bfloat16, torch.float32}

    def export():
        return {
            name: weight.detach().cpu().clone() for chunk in iterator.get_hf_weight_chunks() for name, weight in chunk
        }

    accumulated = export()
    snapshots = {}
    prime_delta_snapshots(index, snapshots, pin=False)
    torch.npu.synchronize()
    with torch.no_grad():
        for record in index:
            if record.param is not None:
                record.param.view(-1)[::97] += 0.25
    validate_export_weights(index)
    expected = export()
    updates = [0]

    def consume(name, dtype, shape, positions, values):
        assert tuple(accumulated[name].shape) == tuple(shape)
        accumulated[name].view(-1).index_copy_(0, positions.cpu().long(), values.cpu())
        updates[0] += values.numel()

    queue = _GatherQueue(4, 4096, rank == 0, consume)
    for slots, dtype, counts, positions, values, group in iter_delta_entries(index, snapshots):
        assert positions.device.type == values.device.type == "npu"
        queue.put(group, slots, dtype, counts, positions, values)
    queue.flush_all()
    if rank == 0:
        assert updates[0] > 0
        assert accumulated.keys() == expected.keys()
        for name, value in expected.items():
            actual = accumulated[name]
            assert actual.dtype == value.dtype and actual.shape == value.shape, name
            assert torch.equal(actual.view(torch.uint8), value.view(torch.uint8)), name
        print(f"SPARSE VS FULL PASS: tensors={len(expected)} updates={updates[0]} bitwise_equal=True", flush=True)
    dist.barrier()


def main():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group(backend="hccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    assert world >= 2, "differential needs TP>1 to be meaningful"

    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1 if EP > 1 else None,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)

    cfg_dir = _build_tiny_hf_dir(rank)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(cfg_dir, trust_remote_code=True)
    config = bridge.hf_pretrained.config
    rope = getattr(config, "rope_parameters", None) or {}
    if "rope_theta" in rope and not hasattr(config, "rope_theta"):
        config.rope_theta = rope["rope_theta"]
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
    if EP > 1:
        provider.expert_model_parallel_size = EP
        provider.expert_tensor_parallel_size = 1
    provider.bf16 = True
    provider.params_dtype = torch.bfloat16
    provider.finalize()
    model = provider.provide_distributed_model(wrap_with_ddp=False)
    model = [m.cuda().to(torch.bfloat16) for m in (model if isinstance(model, list) else [model])]
    # real HF weights, identically loaded on every rank: replicated params
    # (layernorms, routers) must be bit-identical across TP ranks or the
    # differential would flag test artifacts instead of probe bugs.
    bridge.load_hf_weights(model)

    if os.environ.get("DELTA_ENTRY_ORACLE"):
        _delta_entry_differential(model, cfg_dir, rank)
        dist.destroy_process_group()
        return

    if PP > 1 or EP > 1 or os.environ.get("ENGINE_DIFF"):
        _pp_differential(bridge, model, rank, world)
        return

    from vime.backends.megatron_utils.update_weight.megatron_delta_export import make_probe

    tasks = bridge.get_conversion_tasks(model)
    n_checked = n_pass = 0
    failures = []
    for t in tasks:
        p = t.param_weight
        if p is None or not p.is_floating_point():
            continue
        local = p.data.to(torch.bfloat16)

        # REAL path: true collectives, all ranks in lockstep; full tensors everywhere.
        real = {k: v.detach().clone() for k, v in t.mapping.megatron_to_hf(local, t.megatron_module).items()}

        # PROBE path: comm-stubbed, purely local. Feed the real local shard so
        # the surviving (non-NaN) elements carry real values.
        probe = make_probe(t.mapping, t.megatron_module)
        mine = {k: v.detach().to("cpu") for k, v in probe.megatron_to_hf(local.clone(), t.megatron_module).items()}

        # Assemble: union of every rank's non-NaN contributions.
        all_outs: list = [None] * world
        dist.all_gather_object(all_outs, mine)
        if rank == 0:
            names = {k for d in all_outs for k in d}
            for name in sorted(names):
                ref = real.get(name)
                if ref is None:
                    failures.append(f"{t.global_param_name}: probe slot {name} absent from real output")
                    continue
                merged = torch.full_like(ref.cpu(), float("nan"))
                covered = torch.zeros(ref.shape, dtype=torch.int32)
                for d in all_outs:
                    if name not in d:
                        continue
                    part = d[name]
                    mask = ~torch.isnan(part)
                    # replicated params legitimately arrive from every rank --
                    # overlap only counts as failure when the values CONFLICT.
                    both = mask & (covered > 0)
                    if both.any() and not torch.equal(merged[both].view(torch.int16), part[both].view(torch.int16)):
                        failures.append(f"{t.global_param_name}/{name}: conflicting rank contributions")
                    merged[mask] = part[mask]
                    covered[mask] += 1
                n_checked += 1
                if torch.isnan(merged).any():
                    failures.append(f"{t.global_param_name}/{name}: uncovered positions remain NaN")
                elif not torch.equal(merged.view(torch.int16), ref.cpu().view(torch.int16)):
                    failures.append(f"{t.global_param_name}/{name}: bitwise mismatch")
                else:
                    n_pass += 1
        dist.barrier()

    if rank == 0:
        print("=" * 60)
        for f in failures[:20]:
            print("FAIL:", f)
        print(f"PROBE DIFFERENTIAL [{MODEL_KIND} tp={TP}]: {n_pass}/{n_checked} bitwise-equal")
        print("PROBE DIFFERENTIAL PASSED" if (n_checked > 0 and not failures) else "PROBE DIFFERENTIAL FAILED")
    dist.barrier()
    dist.destroy_process_group()
    if rank == 0 and (failures or n_checked == 0):
        sys.exit(1)


class Model(torch.nn.Module):
    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.weight = torch.nn.Parameter(
            torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)[rank * 2 : rank * 2 + 2].clone()
        )
        self.calls = 0

    def load_weights(self, weights):
        self.calls += 1
        for name, weight in dict(weights).items():
            assert name == "weight"
            self.weight.data.copy_(weight[self.rank * 2 : self.rank * 2 + 2])


def test_cpu_backup_cannot_silently_become_diff_source():
    with pytest.raises(RuntimeError, match="live accelerator shard"):
        _device_shard(SimpleNamespace(param=torch.ones(4), megatron_name="weight"))


def test_snapshots_are_independent_host_copies(monkeypatch):
    local = torch.arange(4, dtype=torch.bfloat16)
    monkeypatch.setattr(megatron_delta_export, "_device_shard", lambda record: local)
    record = SimpleNamespace(param=local, spec=SimpleNamespace(contributes=True), megatron_name="weight")
    snapshots = {}
    megatron_delta_export.prime_delta_snapshots([record], snapshots)
    snapshot = snapshots["weight"]
    assert snapshot.device.type == "cpu"
    assert torch.equal(snapshot, local)
    local.add_(1)
    assert torch.equal(snapshot, torch.arange(4, dtype=torch.bfloat16))


@pytest.mark.parametrize("rank", [0, 1])
def test_three_real_deltas_split_and_empty_preserve_full_state(rank):
    model = Model(rank)
    expected = torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)
    for round_id in range(3):
        indices = torch.tensor([round_id, 7 - round_id], dtype=torch.int32)
        values = torch.tensor([20 + round_id, 40 + round_id], dtype=torch.bfloat16)
        # Same HF name appears in two wire pieces. Both must be applied.
        patches = [(SparseWeightPatch("weight", indices[i : i + 1], values[i : i + 1]), [4, 2]) for i in range(2)]
        apply_sparse_hf_patches_with_loader(model, patches, chunk_bytes=16)
        expected.view(-1).index_copy_(0, indices.long(), values)
        assert torch.equal(model.weight.view(torch.int16), expected[rank * 2 : rank * 2 + 2].view(torch.int16))
        _verify_dense_load(model, [("weight", expected)])
    apply_sparse_hf_patches_with_loader(model, [])


def test_bad_wire_bounds_fail_before_mutation():
    model = Model(0)
    before = model.weight.detach().clone()
    patches = [(SparseWeightPatch("weight", torch.tensor([8], dtype=torch.int32), torch.ones(1)), [4, 2])]
    with pytest.raises(IndexError):
        apply_sparse_hf_patches_with_loader(model, patches)
    assert torch.equal(model.weight, before)


def test_dense_verify_detects_missing_update():
    model = Model(0)
    with pytest.raises(RuntimeError, match="state verification failed"):
        _verify_dense_load(model, [("weight", torch.ones(4, 2, dtype=torch.bfloat16))])


@pytest.mark.parametrize("conflicting", [False, True])
def test_slot_exchange_inherits_dtype_or_rejects_disagreement(monkeypatch, conflicting):
    record = SimpleNamespace(param=None, megatron_name="remote")
    slots = [("weight", (4,))]
    rows = [[None], [(slots, torch.float32)], [(slots, torch.bfloat16 if conflicting else torch.float32)]]
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: len(rows))
    monkeypatch.setattr(
        torch.distributed, "all_gather_object", lambda output, local: output.__setitem__(slice(None), rows)
    )
    if conflicting:
        with pytest.raises(ValueError, match="inconsistent owner dtypes"):
            megatron_delta_export._exchange_slot_tables([record], {})
    else:
        megatron_delta_export._exchange_slot_tables([record], {})
        assert record.dtype == torch.float32
        assert record.slots == slots


@pytest.mark.parametrize("batch_size", [1, 32])
def test_fp32_placeholder_keeps_owner_dtype(monkeypatch, batch_size):
    record = SimpleNamespace(
        param=None,
        megatron_name="remote",
        dtype=torch.float32,
        slots=[("weight", (4,))],
        spec=SimpleNamespace(gather_group=None),
    )
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: "cpu")
    entries = list(megatron_delta_export.iter_delta_entries([record], {}, batch_size=batch_size))
    assert entries[0][1] == "float32"
    assert entries[0][4].dtype == torch.float32
    assert entries[0][4].numel() == 0


@pytest.mark.parametrize("local_nan,remote_nan", [(False, False), (True, False), (False, True)])
def test_nan_reduction_rejects_local_or_remote_failure(monkeypatch, local_nan, remote_nan):
    record = SimpleNamespace(param=torch.tensor([float("nan") if local_nan else 1.0]))
    calls = []

    def reduce(flag, op):
        calls.append(int(flag))
        if remote_nan:
            flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    if local_nan or remote_nan:
        with pytest.raises(ValueError, match="NaN weights"):
            megatron_delta_export.validate_export_weights([record])
    else:
        megatron_delta_export.validate_export_weights([record])
    assert calls == [int(local_nan)]


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_delta_indices_is_bit_exact(dtype):
    torch.manual_seed(0)
    previous = torch.randn(1000, dtype=dtype)
    current = previous.clone()
    changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64)
    current[changed] += 0.5

    indices, values = shard_delta_indices(current, previous, offset=4096)

    assert torch.equal(indices, changed + 4096)
    integer_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(values.view(integer_dtype), current[changed].view(integer_dtype))


@pytest.mark.unit
def test_shard_delta_indices_no_change_is_empty():
    previous = torch.randn(256, dtype=torch.bfloat16)
    indices, values = shard_delta_indices(previous.clone(), previous, offset=0)
    assert indices.numel() == 0
    assert values.numel() == 0


@pytest.mark.unit
def test_checksum_fits_uint64_wire_format(monkeypatch):
    hashes = iter([2**63 - 1, 2**63 - 1])
    monkeypatch.setattr(torch, "hash_tensor", lambda _: torch.tensor(next(hashes)))

    result = checksum(torch.ones(1, dtype=torch.uint8), torch.ones(1))

    assert 0 <= result <= 2**64 - 1


@pytest.mark.unit
def test_root_payload_lengths_and_returned_buffer_ownership(monkeypatch):
    # CPU-only transport mock: do not trigger the production NPU normalization.
    monkeypatch.setattr(gather_module, "torch", SimpleNamespace(empty=torch.empty, cat=torch.cat))
    calls = []
    counts = [[1, 0], [2, 3], [0, 1], [0, 0]]
    peer_indices = torch.tensor([10, 11, 20, 21, 22], dtype=torch.int32)
    peer_values = peer_indices.float()

    def batch(operations):
        for operation, tensor, peer, _group in operations:
            assert operation == "recv" and peer in (11, 12)
            assert tensor.numel() == (5 if peer == 11 else 1)
            source = peer_indices if tensor.dtype == torch.int32 else peer_values
            tensor.copy_(source if peer == 11 else torch.tensor([30], dtype=tensor.dtype))
        calls.append(operations)
        return [SimpleNamespace(wait=lambda: None)]

    monkeypatch.setattr(
        gather_module,
        "dist",
        SimpleNamespace(
            get_rank=lambda group: 0,
            get_world_size=lambda group: 4,
            get_global_rank=lambda group, rank: 10 + rank,
            irecv="recv",
            isend="send",
            P2POp=lambda *args: args,
            batch_isend_irecv=batch,
        ),
    )
    workspace = gather_module.GatherWorkspace()
    result = gather_module.gather_slot_entries_to_rank0(
        torch.tensor([1], dtype=torch.int32),
        torch.tensor([1.0]),
        torch.tensor([1, 0]),
        group=object(),
        workspace=workspace,
        _counts_cpu=counts,
    )
    assert result[0][0].tolist() == [1, 10, 11]
    assert result[1][0].tolist() == [20, 21, 22, 30]
    assert torch.equal(result[0][1], result[0][0].float())
    saved = [(indices.clone(), values.clone()) for indices, values in result]
    assert len(calls) == 1
    assert not any(key[0] == "send_indices" for key in workspace.buffers)
    # Reuse scratch; previously returned slot payloads must remain independent.
    for buffer in workspace.buffers.values():
        buffer.zero_()
    for actual, expected in zip(result, saved, strict=True):
        assert torch.equal(actual[0], expected[0])
        assert torch.equal(actual[1], expected[1])


@pytest.mark.unit
def test_empty_hccl_peer_does_not_enqueue_p2p(monkeypatch):
    monkeypatch.setattr(gather_module, "torch", SimpleNamespace(empty=torch.empty, cat=torch.cat))
    monkeypatch.setattr(
        gather_module,
        "dist",
        SimpleNamespace(
            get_rank=lambda group: 1,
            get_world_size=lambda group: 2,
            get_global_rank=lambda group, rank: rank,
            batch_isend_irecv=lambda operations: pytest.fail("empty peer must not send padding"),
        ),
    )
    assert (
        gather_module.gather_slot_entries_to_rank0(
            torch.empty(0, dtype=torch.int32),
            torch.empty(0),
            torch.tensor([0]),
            group=object(),
            _counts_cpu=[[3], [0]],
        )
        is None
    )


@pytest.fixture
def updater(monkeypatch):
    instance = updater_module.UpdateWeightFromSparseHCCL.__new__(updater_module.UpdateWeightFromSparseHCCL)
    instance.args = SimpleNamespace(update_weight_delta_verify_every=2)
    instance._client = object()
    instance._failed = False
    instance._seeded = True
    instance._steady_updates = 0
    instance.weight_version = 1
    instance._begin_update = lambda: None
    instance._ensure_export_index = lambda: None
    instance._index = []
    monkeypatch.setattr(updater_module, "validate_export_weights", lambda index: None)
    instance._sync_update_state = lambda error: (error is not None, instance._payload_started)
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(updater_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
    return instance


@pytest.mark.unit
def test_verify_schedule_checks_sparse_writes_without_dense_replay(updater):
    verification = []
    finishes = []
    updater._send_sparse_delta = lambda *, verify: (verification.append(verify) or (1, 24, 4))
    updater._send_dense = lambda **kwargs: pytest.fail("steady verification must not replay dense weights")
    updater._finish_update = lambda: finishes.append(updater.weight_version)
    for _ in range(3):
        updater.update_weights()
    assert verification == [False, True, False]
    assert finishes == [2, 3, 4]
    assert updater._steady_updates == 3


@pytest.mark.unit
def test_steady_update_adds_no_device_synchronization(updater, monkeypatch):
    monkeypatch.setattr(
        updater_module.torch.accelerator, "synchronize", lambda: pytest.fail("unexpected synchronization")
    )
    updater._send_sparse_delta = lambda **kwargs: (1, 24, 4)
    updater._finish_update = lambda: None
    updater.update_weights()
    assert not any(key.endswith("_seconds") for key in updater.pop_metrics())


@pytest.mark.parametrize("seeded", [False, True])
def test_nan_rejection_precedes_payload_and_snapshot_changes(updater, monkeypatch, seeded):
    updater._seeded = seeded
    recovered = []
    updater._recover_before_payload = lambda: recovered.append(True)
    updater._send_dense = lambda **kwargs: pytest.fail("NaN seed must not be sent")
    updater._send_sparse_delta = lambda **kwargs: pytest.fail("NaN delta must not be sent")
    monkeypatch.setattr(updater_module, "prime_delta_snapshots", lambda *a, **kw: pytest.fail("snapshot changed"))

    def reject(index):
        raise ValueError("NaN weights")

    monkeypatch.setattr(updater_module, "validate_export_weights", reject)
    with pytest.raises(ValueError, match="NaN weights"):
        updater.update_weights()
    assert recovered == [True]
    assert not updater._payload_started


@pytest.mark.unit
def test_sparse_delta_reuses_dtype_bucket_without_changing_payload(updater, monkeypatch):
    updater.args.update_weight_buffer_size = 1024
    updater._index = object()
    updater._snapshots = {}
    created = []
    original_bucket = updater_module._SparseFlushBucket

    def make_bucket(*args, **kwargs):
        bucket = original_bucket(*args, **kwargs)
        created.append(bucket)
        return bucket

    monkeypatch.setattr(updater_module, "_SparseFlushBucket", make_bucket)
    monkeypatch.setattr(updater_module, "checksum", lambda positions, values: 123)
    tensor = updater_module.torch.tensor
    entries = [
        (
            [(name, [4])],
            "float32",
            tensor([1]),
            tensor([index], dtype=updater_module.torch.int32),
            tensor([value]),
            None,
        )
        for name, index, value in [("a", 1, 10.0), ("b", 3, 20.0)]
    ]
    monkeypatch.setattr(updater_module, "iter_delta_entries", lambda *args, **kwargs: iter(entries))
    payloads = []
    updater._publish_flush = lambda flush, **kwargs: payloads.append(flush)
    assert updater._send_sparse_delta() == (1, 16, 2)
    assert len(created) == 1
    assert [param.name for param in payloads[0].params] == ["a", "b"]
    assert payloads[0].positions.view(updater_module.torch.int32).tolist() == [1, 3]
    assert payloads[0].values.tolist() == [10.0, 20.0]


@pytest.mark.unit
def test_gather_queue_reuses_workspace_between_flushes(monkeypatch):
    created, observed = [], []

    def make_workspace():
        workspace = object()
        created.append(workspace)
        return workspace

    def gather(indices, values, counts, **kwargs):
        observed.append(kwargs["workspace"])
        return [(indices, values)]

    monkeypatch.setattr(updater_module, "GatherWorkspace", make_workspace)
    monkeypatch.setattr(updater_module, "gather_slot_entries_to_rank0", gather)
    consumed = []
    queue = updater_module._GatherQueue(1, 1024, True, lambda *args: consumed.append(args))
    group = object()
    for name in ["a", "b"]:
        queue.put(
            group,
            [(name, [4])],
            "float32",
            updater_module.torch.tensor([1]),
            updater_module.torch.tensor([1], dtype=updater_module.torch.int32),
            updater_module.torch.tensor([10.0]),
        )
    assert len(created) == 1
    assert observed == [created[0], created[0]]
    assert [entry[0] for entry in consumed] == ["a", "b"]


@pytest.mark.unit
def test_partial_payload_failure_keeps_generation_paused_and_rejects_reuse(updater):
    def send(*, verify):
        updater._payload_started = True
        raise RuntimeError("receiver failed after partial apply")

    updater._send_sparse_delta = send
    updater._finish_update = lambda: pytest.fail("partial weights must not resume generation")
    updater._recover_before_payload = lambda: pytest.fail("partial apply has no rollback")
    with pytest.raises(RuntimeError, match="partial apply"):
        updater.update_weights()
    assert updater._failed
    with pytest.raises(RuntimeError, match="cannot be reused"):
        updater.update_weights()


@pytest.mark.unit
def test_pre_payload_failure_uses_recovery_instead_of_finish(updater):
    recovered = []

    def prepare():
        raise RuntimeError("export preparation failed")

    updater._ensure_export_index = prepare
    updater._recover_before_payload = lambda: recovered.append(True)
    updater._finish_update = lambda: pytest.fail("failed preparation must not finish an update")
    with pytest.raises(RuntimeError, match="preparation failed"):
        updater.update_weights()
    assert recovered == [True]
    assert not updater._failed


@pytest.mark.unit
def test_finish_failure_rejects_following_update(updater):
    updater._send_sparse_delta = lambda *, verify: (1, 24, 4)

    def finish():
        raise RuntimeError("finish RPC failed")

    updater._finish_update = finish
    with pytest.raises(RuntimeError, match="finish RPC failed"):
        updater.update_weights()
    assert updater._failed


def run_gather_regression():
    torch.npu.set_device(int(os.environ["LOCAL_RANK"]))
    dist.init_process_group("hccl", timeout=timedelta(seconds=120))
    if dist.get_world_size() != 4:
        dist.destroy_process_group()
        raise ValueError("This regression requires exactly four ranks (--nproc-per-node=4)")
    rank = dist.get_rank()
    workspace = GatherWorkspace()
    preserved = []
    cases = [
        [[0, 0, 0], [0, 0, 0], [2, 1, 0], [0, 0, 0]],
        [[1, 0, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 3, 0], [0, 2, 1], [2, 0, 4], [0, 0, 0]],
        [[0, 0, 0]] * 4,
        [[2, 1, 2], [1, 3, 1], [2, 0, 2], [1, 1, 0]],
    ]
    for dtype in (torch.float32, torch.bfloat16):
        for counts in cases:
            pieces = [
                [torch.arange(size, dtype=torch.int32) + r * 100 + slot * 10 for slot, size in enumerate(row)]
                for r, row in enumerate(counts)
            ]
            indices = torch.cat(pieces[rank]).npu()
            values = torch.cat([piece.float().to(dtype) for piece in pieces[rank]]).npu()
            # Cover noncontiguous caller buffers as well as empty participants.
            indices = torch.stack([indices, indices], dim=1)[:, 0]
            values = torch.stack([values, values], dim=1)[:, 0]
            for budget in (None, 12):
                result = gather_slot_entries_to_rank0(
                    indices,
                    values,
                    torch.tensor(counts[rank]),
                    max_round_bytes=budget,
                    workspace=workspace,
                )
                if rank == 0:
                    for slot, (actual_indices, actual_values) in enumerate(result):
                        expected = torch.cat([pieces[r][slot] for r in range(4)])
                        assert torch.equal(actual_indices.cpu(), expected)
                        assert torch.equal(actual_values.cpu(), expected.float().to(dtype))
                        preserved.append((actual_indices, actual_values, expected, dtype))
                else:
                    assert result is None
        dist.barrier()
    if rank == 0:
        for indices, values, expected, dtype in preserved:
            assert torch.equal(indices.cpu(), expected)
            assert torch.equal(values.cpu(), expected.float().to(dtype))
        print("NPU-SPARSE-GATHER PASS: 20 cases; empty/unequal/noncontiguous/split/reused; FP32/BF16", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    if "--gather-regression" in sys.argv:
        run_gather_regression()
    else:
        main()
