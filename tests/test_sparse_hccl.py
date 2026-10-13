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
"""Fixed Qwen3-MoE TP2/PP2/EP2 sparse HCCL regression (ETP1).

pytest runs focused loader, dtype, NaN, checksum and failure-state checks.
Four-rank torchrun compares one sparse update with complete Bridge full export,
including FP32 norms beside BF16 matrices. --gather-regression checks real HCCL
empty/unequal/noncontiguous/split/reused payloads.

Run manually in an Ascend/CANN environment with Megatron-Bridge:
    pytest -q tests/test_sparse_hccl.py
    torchrun --standalone --nproc-per-node=4 tests/test_sparse_hccl.py
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

TP = 2
PP = 2
EP = 2


def _build_tiny_qwen3_moe_checkpoint(rank: int) -> str:
    """Keep real Qwen3-MoE mappings while making four-rank tests inexpensive."""
    from transformers import AutoConfig

    cfg_dir = "/tmp/tiny_qwen3_moe_probe_diff_v4"
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
    if rank == 0 and not pathlib.Path(cfg_dir, "model.safetensors").exists():
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(hf_config).to(torch.bfloat16)
        model.save_pretrained(cfg_dir, safe_serialization=True)
        del model
    dist.barrier()
    return cfg_dir


def _compare_sparse_with_full_weights(model, cfg_dir, rank):
    """Require sparse reconstruction to equal full export, including untouched weights."""
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

    # Exercise PP placeholders with FP32 owner rows alongside BF16 matrices.
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
    assert world == TP * PP, "This regression requires four ranks for TP2/PP2/EP2"

    from megatron.core import parallel_state as mpu

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    )
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    model_parallel_cuda_manual_seed(1234)

    cfg_dir = _build_tiny_qwen3_moe_checkpoint(rank)

    from megatron.bridge import AutoBridge

    bridge = AutoBridge.from_hf_pretrained(cfg_dir, trust_remote_code=True)
    config = bridge.hf_pretrained.config
    rope = getattr(config, "rope_parameters", None) or {}
    if "rope_theta" in rope and not hasattr(config, "rope_theta"):
        config.rope_theta = rope["rope_theta"]
    provider = bridge.to_megatron_provider(load_weights=False)
    provider.tensor_model_parallel_size = TP
    provider.pipeline_model_parallel_size = PP
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

    _compare_sparse_with_full_weights(model, cfg_dir, rank)
    dist.destroy_process_group()


class _TPShardModel(torch.nn.Module):
    """Minimal loader fixture that owns one half of a full HF tensor."""

    def __init__(self, rank):
        super().__init__()
        self.rank = rank
        self.weight = torch.nn.Parameter(
            torch.arange(8, dtype=torch.bfloat16).reshape(4, 2)[rank * 2 : rank * 2 + 2].clone()
        )

    def load_weights(self, weights):
        for name, weight in dict(weights).items():
            assert name == "weight"
            self.weight.data.copy_(weight[self.rank * 2 : self.rank * 2 + 2])


def test_snapshot_is_independent_and_requires_live_weights(monkeypatch):
    """Prevent stale CPU backups or shared storage from hiding real weight changes."""
    with pytest.raises(RuntimeError, match="live accelerator shard"):
        _device_shard(SimpleNamespace(param=torch.ones(4), megatron_name="weight"))
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
def test_loader_preserves_full_state_across_split_updates(rank):
    """Check repeated/split/empty patches and detect a missed update on each TP shard."""
    model = _TPShardModel(rank)
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
    missing = expected.clone()
    missing[rank * 2, 0] += 1
    with pytest.raises(RuntimeError, match="state verification failed"):
        _verify_dense_load(model, [("weight", missing)])


def test_bad_wire_bounds_fail_before_mutation():
    """Reject an out-of-bounds patch before any rank-local weight is changed."""
    model = _TPShardModel(0)
    before = model.weight.detach().clone()
    patches = [(SparseWeightPatch("weight", torch.tensor([8], dtype=torch.int32), torch.ones(1)), [4, 2])]
    with pytest.raises(IndexError):
        apply_sparse_hf_patches_with_loader(model, patches)
    assert torch.equal(model.weight, before)


def test_pp_placeholders_share_owner_dtype(monkeypatch):
    """Keep FP32 empty rows aligned in both diff modes; reject conflicting owner dtypes."""
    slots = [("weight", (4,))]
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: 3)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: "cpu")
    for conflicting in (False, True):
        record = SimpleNamespace(param=None, megatron_name="remote", spec=SimpleNamespace(gather_group=None))
        rows = [[None], [(slots, torch.float32)], [(slots, torch.bfloat16 if conflicting else torch.float32)]]
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda output, local, rows=rows: output.__setitem__(slice(None), rows),
        )
        if conflicting:
            with pytest.raises(ValueError, match="inconsistent owner dtypes"):
                megatron_delta_export._exchange_slot_tables([record], {})
            continue
        megatron_delta_export._exchange_slot_tables([record], {})
        assert record.dtype == torch.float32 and record.slots == slots
        for batch_size in (1, 32):
            entries = list(megatron_delta_export.iter_delta_entries([record], {}, batch_size=batch_size))
            assert entries[0][1] == "float32"
            assert entries[0][4].dtype == torch.float32
            assert entries[0][4].numel() == 0


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_shard_delta_indices_is_bit_exact(dtype):
    """Preserve BF16/FP32 changed bits and offsets; unchanged shards must emit no payload."""
    torch.manual_seed(0)
    previous = torch.randn(1000, dtype=dtype)
    current = previous.clone()
    changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64)
    current[changed] += 0.5

    indices, values = shard_delta_indices(current, previous, offset=4096)

    assert torch.equal(indices, changed + 4096)
    integer_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(values.view(integer_dtype), current[changed].view(integer_dtype))
    indices, values = shard_delta_indices(current.clone(), current, offset=4096)
    assert indices.numel() == values.numel() == 0


@pytest.mark.unit
def test_checksum_fits_uint64_wire_format(monkeypatch):
    """Keep worst-case hashes representable by the JSON/msgpack control plane."""
    hashes = iter([2**63 - 1, 2**63 - 1])
    monkeypatch.setattr(torch, "hash_tensor", lambda _: torch.tensor(next(hashes)))

    result = checksum(torch.ones(1, dtype=torch.uint8), torch.ones(1))

    assert 0 <= result <= 2**64 - 1


@pytest.mark.unit
def test_root_payload_lengths_and_returned_buffer_ownership(monkeypatch):
    """Receive actual peer lengths; workspace reuse must not corrupt returned payloads."""
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
    """An empty sender must not enqueue a transfer that would leave peers waiting."""
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
    """Honor the local verify interval without introducing a full-weight replay."""
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


@pytest.mark.parametrize(
    "local_nan,remote_nan,seeded", [(False, False, True), (True, False, False), (False, True, True)]
)
def test_nan_guard_precedes_payload_and_snapshots(updater, monkeypatch, local_nan, remote_nan, seeded):
    """Exercise the real NaN guard with mocked reduction; fail before seed/delta writes."""
    updater._seeded = seeded
    updater._index = [SimpleNamespace(param=torch.tensor([float("nan") if local_nan else 1.0]))]
    monkeypatch.setattr(updater_module, "validate_export_weights", megatron_delta_export.validate_export_weights)
    calls = []

    def reduce(flag, op):
        assert op == torch.distributed.ReduceOp.MAX
        calls.append(int(flag))
        if remote_nan:
            flag.fill_(1)

    monkeypatch.setattr(torch.distributed, "all_reduce", reduce)
    recovered = []
    updater._recover_before_payload = lambda: recovered.append(True)
    if local_nan or remote_nan:
        updater._send_dense = lambda **kwargs: pytest.fail("NaN seed must not be sent")
        updater._send_sparse_delta = lambda **kwargs: pytest.fail("NaN delta must not be sent")
        monkeypatch.setattr(updater_module, "prime_delta_snapshots", lambda *a, **kw: pytest.fail("snapshot changed"))
        with pytest.raises(ValueError, match="NaN weights"):
            updater.update_weights()
        assert recovered == [True] and not updater._payload_started
    else:
        updater._send_sparse_delta = lambda **kwargs: (1, 24, 4)
        updater._finish_update = lambda: None
        updater.update_weights()
        assert not recovered
    assert calls == [int(local_nan)]


@pytest.mark.unit
@pytest.mark.parametrize("phase", ["prepare", "payload", "finish"])
def test_update_failure_preserves_safe_state(updater, phase):
    """Recover before transmission; partial apply or failed finish must reject reuse."""
    recovered = []

    def fail():
        raise RuntimeError(f"{phase} failed")

    def send(*, verify):
        if phase == "payload":
            updater._payload_started = True
            fail()
        return (1, 24, 4)

    updater._send_sparse_delta = send
    updater._ensure_export_index = fail if phase == "prepare" else lambda: None
    updater._recover_before_payload = lambda: recovered.append(True)
    updater._finish_update = (
        fail if phase == "finish" else lambda: pytest.fail("failed update must not resume generation")
    )
    with pytest.raises(RuntimeError, match=f"{phase} failed"):
        updater.update_weights()
    assert recovered == ([True] if phase == "prepare" else [])
    assert updater._failed == (phase != "prepare")
    if phase != "prepare":
        with pytest.raises(RuntimeError, match="cannot be reused"):
            updater.update_weights()


def run_gather_regression():
    """Check actual HCCL length/split/reuse behavior that CPU mocks cannot prove."""
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
