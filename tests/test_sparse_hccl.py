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

CPU coordinate tests run with `-k test_coordinates_` without serving/NPU packages.
The remaining tests run public loader, dtype, NaN and failure-state checks, then launch
four workers for real HCCL gather and one sparse/full weight comparison.
The distributed test requires four visible NPUs, CANN and Megatron Core;
without four available NPUs, pytest reports it as skipped.

Run:
    python -m pytest -q tests/test_sparse_hccl.py
"""

import importlib
import os
import pathlib
import sys
import time
import types
from datetime import timedelta
from types import SimpleNamespace
from unittest.mock import patch as mock_patch

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

# Keep spawned workers on the checkout under test, not another editable installation.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
# Coordinate tests must collect without the serving/NPU packages. Legacy engine
# tests load them only when executed; spawned workers initialize them explicitly.
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import _unit_stubs

CheckpointWeightPatch = load_checkpoint_weight_patches = None
megatron_delta_export = gather_module = updater_module = None
_device_shard = compact_shard_masks = shard_delta_mask = None
GatherWorkspace = gather_slot_entries_to_rank0 = None


def _load_sparse_dependencies():
    global CheckpointWeightPatch, load_checkpoint_weight_patches
    global megatron_delta_export, gather_module, updater_module
    global _device_shard, compact_shard_masks, shard_delta_mask
    global GatherWorkspace, gather_slot_entries_to_rank0
    if updater_module is not None:
        return
    pytest.importorskip("vllm", reason="Native sparse engine tests require vLLM")
    pytest.importorskip("vllm_ascend", reason="Native sparse engine tests require vLLM-Ascend")
    from vllm.model_executor.model_loader.checkpoint_weight_patch import (
        CheckpointWeightPatch,
        load_checkpoint_weight_patches,
    )

    # Adaptor must not parse pytest arguments, including inside spawned workers.
    with mock_patch.object(sys, "argv", [sys.argv[0]]):
        import vime.backends.megatron_utils  # noqa: F401

        if "megatron_adaptor" in sys.modules:
            from megatron_adaptor.utils.args_utils import get_mindspeed_args, set_full_args

            set_full_args(get_mindspeed_args(get_defaults=True))
    from vime.backends.megatron_utils.delta_sync import megatron_delta_export
    from vime.backends.megatron_utils.delta_sync import sparse_gather as gather_module
    from vime.backends.megatron_utils.delta_sync.megatron_delta_export import (
        _device_shard,
        compact_shard_masks,
        shard_delta_mask,
    )
    from vime.backends.megatron_utils.delta_sync.sparse_gather import GatherWorkspace, gather_slot_entries_to_rank0
    from vime.backends.megatron_utils.update_weight import update_weight_from_sparse_hccl as updater_module


@pytest.fixture(autouse=True)
def sparse_dependencies(request):
    if not {"coordinate_converter", "cpu_gather"}.intersection(request.fixturenames):
        _load_sparse_dependencies()


TP = 2
PP = 2
EP = 2


def _unpack_hf_experts(weights):
    """Keep the regression checkpoint in VIME's per-expert HF coordinates."""
    result = {}
    for name, weight in weights.items():
        if name.endswith(".experts.gate_up_proj"):
            prefix = name.removesuffix(".gate_up_proj")
            for expert, fused in enumerate(weight):
                gate, up = fused.chunk(2, dim=0)
                result[f"{prefix}.{expert}.gate_proj.weight"] = gate.clone()
                result[f"{prefix}.{expert}.up_proj.weight"] = up.clone()
        elif name.endswith(".experts.down_proj"):
            prefix = name.removesuffix(".down_proj")
            for expert, down in enumerate(weight):
                result[f"{prefix}.{expert}.down_proj.weight"] = down.clone()
        else:
            result[name] = weight
    return result


def _build_tiny_qwen3_moe_checkpoint(rank: int, cfg_dir: str) -> str:
    """Keep real Qwen3-MoE mappings while making four-rank tests inexpensive."""
    from transformers import AutoConfig

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
        model.save_pretrained(cfg_dir, state_dict=_unpack_hf_experts(model.state_dict()), safe_serialization=True)
        del model
    dist.barrier()
    return cfg_dir


def _compare_sparse_with_full_weights(model, cfg_dir, rank):
    """Require sparse reconstruction to equal full export, including untouched weights."""
    from types import SimpleNamespace

    from vime.backends.megatron_utils.delta_sync.megatron_delta_export import (
        build_export_index,
        iter_delta_entries,
        prime_delta_snapshots,
        validate_export_weights,
    )
    from vime.backends.megatron_utils.delta_sync.sparse_gather import _GatherQueue
    from vime.backends.megatron_utils.delta_sync.sparse_hf_export import SparseHFExporter

    args = SimpleNamespace(
        hf_checkpoint=cfg_dir,
        update_weight_buffer_size=4096,
        vocab_size=512,
        hidden_size=64,
        kv_channels=16,
        num_attention_heads=4,
        num_query_groups=2,
        num_experts=4,
        dspark_enabled=False,
    )
    iterator = SparseHFExporter(args, model, model_name="qwen3moe", quantization_config=None)
    index = build_export_index(iterator, model)
    assert all(record.param is None or record.param.device.type == "npu" for record in index)

    # Exercise PP placeholders with FP32 owner rows alongside BF16 matrices.
    for record in index:
        if record.param is not None and record.param.ndim == 1:
            record.param.data = record.param.data.float()
    iterator = SparseHFExporter(args, model, model_name="qwen3moe", quantization_config=None)
    index = build_export_index(iterator, model)
    assert {record.dtype for record in index} == {torch.bfloat16, torch.float32}

    def export():
        return {
            name: weight.detach().cpu().clone() for chunk in iterator.get_hf_weight_chunks() for name, weight in chunk
        }

    accumulated = export()
    from safetensors.torch import load_file

    checkpoint_weights = load_file(str(pathlib.Path(cfg_dir, "model.safetensors")))
    assert accumulated.keys() == checkpoint_weights.keys(), "Full VIME export must cover every checkpoint parameter"
    for name, value in checkpoint_weights.items():
        actual = accumulated[name].to(value.dtype).contiguous()
        assert torch.equal(actual.view(torch.uint8), value.contiguous().view(torch.uint8)), name
    snapshots = {}
    prime_delta_snapshots(index, snapshots, pin=False)
    torch.npu.synchronize()
    with torch.no_grad():
        for record in index:
            if record.param is not None:
                record.param.view(-1)[::97] += 0.25
    # Simulate the post-wake refresh against live shards while preserving the
    # pre-update baseline. The comparison below must still match full export.
    refreshed = _make_updater(offload_train=True)
    refreshed._iterator = iterator
    refreshed._index = index
    refreshed._snapshots = snapshots
    snapshot_ids = {name: id(value) for name, value in snapshots.items()}
    for record in index:
        record.gather_group = object()  # A retired process group must not survive.
    refreshed._ensure_export_index()
    assert refreshed._index is index
    assert all(record.gather_group is dist.group.WORLD for record in index)
    assert {name: id(value) for name, value in snapshots.items()} == snapshot_ids
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
        print("NPU-SPARSE-OFFLOAD-REFRESH PASS: retained baseline diff matches full export", flush=True)
    dist.barrier()


def _run_sparse_full_comparison(rank, cfg_dir):
    """Exercise real process groups using native VIME-loaded local parameter shards."""
    from megatron.core import parallel_state as mpu
    from megatron.core.transformer.transformer_config import TransformerConfig
    from transformers import AutoConfig

    from vime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
    from vime.backends.megatron_utils.hf_to_megatron.qwen import qwen_moe_hf_tensor

    mpu.initialize_model_parallel(
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    )
    cfg_dir = _build_tiny_qwen3_moe_checkpoint(rank, cfg_dir)
    hf_config = AutoConfig.from_pretrained(cfg_dir)
    reader = SafetensorReader(cfg_dir)
    config = TransformerConfig(
        num_layers=2,
        hidden_size=64,
        num_attention_heads=4,
        num_query_groups=2,
        ffn_hidden_size=128,
        num_moe_experts=4,
        moe_ffn_hidden_size=64,
        tensor_model_parallel_size=TP,
        pipeline_model_parallel_size=PP,
        pipeline_dtype=torch.bfloat16,
        expert_model_parallel_size=EP,
        expert_tensor_parallel_size=1,
    )
    root = torch.nn.Module()
    root.config = config

    def register(name, global_name, partition_dim=None, expert=False):
        full = qwen_moe_hf_tensor(global_name, reader, hf_config)
        size = mpu.get_expert_tensor_parallel_world_size() if expert else TP
        tp_rank = mpu.get_expert_tensor_parallel_rank() if expert else mpu.get_tensor_model_parallel_rank()
        value = full if partition_dim is None else full.chunk(size, dim=partition_dim)[tp_rank]
        parameter = torch.nn.Parameter(value.clone().npu().to(torch.bfloat16))
        parameter.tensor_model_parallel = partition_dim is not None
        parameter.partition_dim = partition_dim if partition_dim is not None else -1
        parameter.partition_stride = 1
        module = root
        parts = ("module.module." + name).split(".")
        for part in parts[:-1]:
            if part not in module._modules:
                module.add_module(part, torch.nn.Module())
            module = module._modules[part]
        module.register_parameter(parts[-1], parameter)

    stage = mpu.get_pipeline_model_parallel_rank()
    if stage == 0:
        register("embedding.word_embeddings.weight", "embedding.word_embeddings.weight", 0)
    else:
        register("output_layer.weight", "output_layer.weight", 0)
        register("decoder.final_layernorm.weight", "decoder.final_layernorm.weight")
    for suffix, dim in [
        ("self_attention.linear_qkv.weight", 0),
        ("self_attention.linear_proj.weight", 1),
        ("self_attention.linear_qkv.layer_norm_weight", None),
        ("self_attention.q_layernorm.weight", None),
        ("self_attention.k_layernorm.weight", None),
        ("pre_mlp_layernorm.weight", None),
        ("mlp.router.weight", None),
    ]:
        register(f"decoder.layers.0.{suffix}", f"decoder.layers.{stage}.{suffix}", dim)
    for local_expert in range(2):
        expert = mpu.get_expert_model_parallel_rank() * 2 + local_expert
        for projection, dim in [(1, 0), (2, 1)]:
            register(
                f"decoder.layers.0.mlp.experts.linear_fc{projection}.weight{local_expert}",
                f"decoder.layers.{stage}.mlp.experts.linear_fc{projection}.weight{expert}",
                dim,
                expert=True,
            )
    try:
        _compare_sparse_with_full_weights([root], cfg_dir, rank)
    finally:
        mpu.destroy_model_parallel()


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


def _make_updater(**args):
    instance = updater_module.UpdateWeightFromSparseHCCL.__new__(updater_module.UpdateWeightFromSparseHCCL)
    instance.args = SimpleNamespace(**args)
    instance._client = object()
    instance._failed = False
    instance._seeded = True
    instance._needs_reconnect = False
    instance._payload_started = False
    instance.weight_version = 1
    instance._index, instance._snapshots, instance._gather_workspaces = [], {}, {}
    return instance


@pytest.fixture
def updater(monkeypatch):
    instance = _make_updater()
    instance._begin_update = lambda: None
    instance._ensure_export_index = lambda: None
    monkeypatch.setattr(updater_module, "validate_export_weights", lambda index: None)
    instance._sync_update_state = lambda error: (error is not None, instance._payload_started)
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(updater_module.dist, "barrier", lambda **kwargs: None)
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
    return instance


@pytest.mark.unit
def test_offload_index_refresh_preserves_layout_and_snapshots(monkeypatch):
    instance = _make_updater(offload_train=True)
    instance._sync_update_state = lambda error: (error is not None, False)
    old = torch.nn.Parameter(torch.ones(2))
    restored = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    record = SimpleNamespace(megatron_name="weight", param=old, gather_group=object(), slots=[("weight", (2,))])
    instance._index = [record]
    instance._gather_workspaces = {("retired-group", "float32"): object()}
    instance._snapshots = {"weight": old.detach().clone()}
    snapshot = instance._snapshots["weight"]
    instance._iterator = SimpleNamespace(local_weights=lambda: {"weight": restored})
    monkeypatch.setattr(updater_module, "build_export_index", lambda *a: pytest.fail("static layout was rebuilt"))
    instance._ensure_export_index()
    assert record.param is restored
    assert record.gather_group is dist.group.WORLD
    assert instance._gather_workspaces == {}
    assert record.slots == [("weight", (2,))]
    assert instance._snapshots["weight"] is snapshot
    assert torch.equal(snapshot, torch.ones(2))
    assert instance._seeded


@pytest.mark.unit
@pytest.mark.parametrize(
    "scenario,expected",
    [
        ("same", True),
        ("replacement", False),
        ("version", False),
        ("query-failure", False),
        ("partial-failure", False),
        ("normal-disconnect", False),
    ],
)
def test_suspended_baseline_requires_same_receiver_version(monkeypatch, scenario, expected):
    instance = _make_updater()
    engine = SimpleNamespace(_actor_id="original", get_weight_version=SimpleNamespace(remote=lambda: "1"))
    instance._suspended_baseline = (("original",), 1)
    if scenario == "replacement":
        engine._actor_id = "replacement"
    elif scenario == "version":
        engine.get_weight_version.remote = lambda: "0"
    elif scenario == "partial-failure":
        instance._failed = True
    elif scenario == "normal-disconnect":
        instance._trainer = None
        instance.disconnect_rollout_engines()
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(updater_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(updater_module.dist, "broadcast_object_list", lambda *a, **kw: None)

    def get(refs):
        if scenario == "query-failure":
            raise RuntimeError("receiver unavailable")
        return refs

    monkeypatch.setattr(updater_module.ray, "get", get)
    assert instance._can_resume_baseline([engine]) is expected


@pytest.mark.unit
@pytest.mark.parametrize("restored", [None, torch.ones(3), torch.ones(2, dtype=torch.float64)])
def test_offload_layout_changes_cannot_reuse_cached_slots(restored):
    instance = _make_updater(offload_train=True)
    instance._sync_update_state = lambda error: (error is not None, False)
    record = SimpleNamespace(megatron_name="weight", param=torch.ones(2), gather_group=dist.group.WORLD)
    instance._index = [record]
    instance._iterator = SimpleNamespace(local_weights=lambda: {} if restored is None else {"weight": restored})
    with pytest.raises(ValueError, match="layout changed"):
        instance._ensure_export_index()


@pytest.mark.unit
@pytest.mark.parametrize("receiver_version,expected_send", [("1", "delta"), ("0", "seed")])
def test_suspend_connect_update_preserves_only_verified_baseline(
    updater, monkeypatch, receiver_version, expected_send
):
    updater.args = SimpleNamespace(rollout_num_gpus_per_engine=1)
    engine = SimpleNamespace(
        _actor_id="same-actor", get_weight_version=SimpleNamespace(remote=lambda: receiver_version)
    )
    updater.rollout_engines = [engine]
    updater._trainer = SimpleNamespace(shutdown=lambda: None)
    updater._snapshots = {"weight": torch.ones(2)}
    snapshots = updater._snapshots
    snapshot = snapshots["weight"]
    index = updater._index
    monkeypatch.setattr(updater_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(updater_module.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")
    monkeypatch.setattr(updater_module.dist, "broadcast_object_list", lambda *a, **kw: None)
    monkeypatch.setattr(
        updater_module.SparseHCCLTrainerWeightTransferEngine,
        "trainer_init",
        lambda *a, **kw: SimpleNamespace(shutdown=lambda: None),
    )
    monkeypatch.setattr(updater_module, "prime_delta_snapshots", lambda *a, **kw: None)
    updater.suspend_rollout_engines()
    assert updater._client is None
    updater.connect_rollout_engines([engine], None)
    if expected_send == "delta":
        assert updater._index is index
        assert updater._snapshots is snapshots and snapshots["weight"] is snapshot
    else:
        assert updater._index is None and not updater._snapshots
    sent = []
    updater._send_sparse_delta = lambda: sent.append("delta")
    updater._send_seed = lambda: sent.append("seed")
    updater._finish_update = lambda: None
    updater.update_weights()
    assert sent == [expected_send]


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode,transport,release", [("delta", "sparse_hccl", "suspend"), ("full", "nccl", "disconnect")]
)
def test_actor_offload_sleep_wake_order(monkeypatch, mode, transport, release):
    module = importlib.import_module("vime.backends.megatron_utils.actor")
    events = []
    for name in ("clear_memory", "print_memory", "reset_cuda_stack_size"):
        monkeypatch.setattr(module, name, lambda *a, **kw: None)
    monkeypatch.setattr(module, "destroy_process_groups", lambda: events.append("destroy"))
    monkeypatch.setattr(module, "reload_process_groups", lambda: events.append("reload"))
    monkeypatch.setattr(
        module,
        "torch_memory_saver",
        SimpleNamespace(
            pause=lambda: events.append("pause"),
            resume=lambda: events.append("resume"),
        ),
    )
    monkeypatch.setattr(module.mpu, "get_pipeline_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(module.accelerator, "current_device", lambda: 0)
    monkeypatch.setattr(module.dist, "barrier", lambda **kw: events.append("barrier"))
    actor = SimpleNamespace(
        args=SimpleNamespace(
            offload_train=True,
            use_critic=True,
            colocate=False,
            update_weight_mode=mode,
            update_weight_transport=transport,
        ),
        role="actor",
        _switch_model=lambda name: events.append(name),
        weight_updater=SimpleNamespace(
            suspend_rollout_engines=lambda: events.append("suspend"),
            disconnect_rollout_engines=lambda: events.append("disconnect"),
        ),
    )
    module.MegatronTrainRayActor.sleep.__wrapped__(actor)
    module.MegatronTrainRayActor.wake_up.__wrapped__(actor)
    assert events == [release, "destroy", "pause", "resume", "reload", "barrier", "actor"]


def _run_gather_regression():
    """Check actual HCCL length/split/reuse behavior on the existing process group."""
    rank = dist.get_rank()
    workspace = GatherWorkspace()
    preserved = []
    cases = [
        [[0, 0, 0], [0, 0, 0], [2, 1, 0], [0, 0, 0]],
        [[1, 0, 3], [0, 0, 0], [0, 0, 0], [0, 0, 0]],
        [[1, 3, 0], [0, 2, 1], [2, 0, 4], [0, 0, 0]],
        [[0, 0, 0]] * 4,
        [[2, 1, 2], [1, 3, 1], [2, 0, 2], [1, 1, 0]],
        [[0], [17], [0], [3]],
        [[2, 0], [0, 0], [31, 1], [0, 0]],
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

                class BoundedWorkspace(GatherWorkspace):
                    def tensor(self, key, size, reference, budget=budget, indices=indices, values=values):
                        if budget is not None and key in {"recv_indices", "recv_values"}:
                            assert size * (indices.element_size() + values.element_size()) <= budget
                        return super().tensor(key, size, reference)

                result = gather_slot_entries_to_rank0(
                    indices,
                    values,
                    torch.tensor(counts[rank]),
                    max_round_bytes=budget,
                    workspace=workspace if budget is None else BoundedWorkspace(),
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
        print("NPU-SPARSE-GATHER PASS: 28 cases; bounded/oversized-slot/unequal/split/reused; FP32/BF16", flush=True)


def _run_publish_failure_regression():
    """All ranks must stop at the failed batch, using real trainer collectives."""
    rank = dist.get_rank()
    device = torch.device("npu", torch.npu.current_device())
    for group in (dist.group.WORLD, None):
        for batch_size in (1, 4):
            for failure in (None, "validation", "send"):
                instance = _make_updater(
                    update_weight_delta_batch_gather=batch_size,
                    update_weight_delta_batch_diff=1,
                    update_weight_buffer_size=8,
                )
                instance._begin_update = instance._ensure_export_index = lambda: None
                recovered, finished, visited, sent = [], [], [], []
                instance._recover_before_payload = lambda recovered=recovered: recovered.append(True)
                instance._finish_update = lambda finished=finished: finished.append(True)

                def send(patches, sent=sent, failure=failure):
                    sent.extend(patches)
                    if failure == "send":
                        raise RuntimeError("injected sender failure")

                instance._trainer = SimpleNamespace(
                    model_update_group=SimpleNamespace(device=device), send_weight_chunk=send
                )

                def entries(*args, visited=visited, group=group, failure=failure, **kwargs):
                    for row in range(2):
                        visited.append(row)
                        count = int(group is not None or rank == 0)
                        values = torch.ones(count, device=device)
                        if failure == "validation" and rank == 0:
                            values.fill_(float("nan"))
                        yield (
                            [(f"weight{row}", (dist.get_world_size(),))],
                            "float32",
                            torch.tensor([count]),
                            torch.full((count,), rank, dtype=torch.int32, device=device),
                            values,
                            group,
                        )

                with pytest.MonkeyPatch.context() as patcher:
                    patcher.setattr(updater_module, "iter_delta_entries", entries)
                    patcher.setattr(updater_module, "validate_export_weights", lambda index: None)
                    if failure:
                        message = (
                            ("Sparse payload" if failure == "validation" else "injected sender failure")
                            if rank == 0
                            else "gather/publish"
                        )
                        with pytest.raises((ValueError, RuntimeError), match=message):
                            instance.update_weights()
                    else:
                        instance.update_weights()
                assert visited == ([0] if failure and batch_size == 1 else [0, 1])
                assert recovered == ([True] if failure == "validation" else [])
                assert instance._failed == (failure == "send")
                assert finished == ([] if failure else [True])
                if rank == 0:
                    expected = 2 * dist.get_world_size() if group is not None else 2
                    assert len(sent) == (0 if failure == "validation" else 1 if failure == "send" else expected)
                dist.barrier()
    if rank == 0:
        print(
            "NPU-SPARSE-PUBLISH PASS: 12 cases; batched/residual/direct; validation/send failures and success",
            flush=True,
        )


def _run_offload_refresh_failure_regression():
    rank = dist.get_rank()
    instance = _make_updater(offload_train=True)
    param = torch.ones(2, device="npu")
    record = SimpleNamespace(megatron_name="weight", param=param, gather_group=dist.group.WORLD)
    instance._index = [record]
    instance._iterator = SimpleNamespace(local_weights=lambda: {} if rank == 1 else {"weight": param})
    with pytest.raises(ValueError if rank == 1 else RuntimeError):
        instance._ensure_export_index()
    assert record.param is param
    # All ranks must reach this collective after only one rank detects a change.
    dist.barrier()
    if rank == 0:
        print("NPU-SPARSE-OFFLOAD-REFRESH-FAILURE PASS: single-rank rejection synchronized", flush=True)


def _distributed_worker(rank, port, cfg_dir):
    """Initialize one rank and guarantee cleanup for success or failed assertions."""
    os.environ.update(
        RANK=str(rank), LOCAL_RANK=str(rank), WORLD_SIZE=str(TP * PP), MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port)
    )
    _load_sparse_dependencies()
    torch.npu.set_device(rank)
    store = dist.TCPStore("127.0.0.1", port, is_master=False, timeout=timedelta(seconds=120))
    dist.init_process_group("hccl", store=store, rank=rank, world_size=TP * PP, timeout=timedelta(seconds=120))
    from vime.utils.distributed_utils import init_gloo_group

    init_gloo_group()
    try:
        _run_publish_failure_regression()
        _run_gather_regression()
        _run_offload_refresh_failure_regression()
        _run_sparse_full_comparison(rank, cfg_dir)
    finally:
        dist.destroy_process_group()


@pytest.fixture
def distributed_rendezvous(tmp_path):
    """Reserve a unique store and checkpoint directory for the spawned ranks."""
    if not hasattr(torch, "npu") or not torch.npu.is_available() or torch.npu.device_count() < TP * PP:
        pytest.skip("Sparse HCCL integration requires four visible NPUs")
    store = dist.TCPStore("127.0.0.1", 0, is_master=True, wait_for_workers=False, timeout=timedelta(seconds=120))
    yield store.port, str(tmp_path / "qwen3_moe")
    del store


def test_nonempty_delta_always_uses_vime_converter(monkeypatch):
    """Cached slots must not bypass the converter for nonempty deltas."""
    local = torch.tensor([[11.0, 0.0], [0.0, 13.0]])
    calls = []

    def convert(probe):
        calls.append(probe.clone())
        return {"weight": probe * 2}

    monkeypatch.setattr(megatron_delta_export, "_device_shard", lambda record: local.view(-1))
    record = SimpleNamespace(
        param=local,
        probe=SimpleNamespace(megatron_to_hf=convert),
        megatron_name="weight",
        slots=[("weight", (2, 2))],
        contributes=True,
        gather_group=None,
    )
    entries = list(
        megatron_delta_export.iter_delta_entries(
            [record],
            {"weight": torch.zeros_like(local).view(-1)},
            batch_size=32,
        )
    )
    assert len(calls) == len(entries) == 1
    assert torch.isnan(calls[0].reshape(-1)[[1, 2]]).all()
    _slots, _dtype, counts, indices, values, _group = entries[0]
    assert counts.tolist() == [2]
    assert indices.tolist() == [0, 3]
    assert values.tolist() == [22.0, 26.0]


def test_snapshot_is_independent_and_requires_live_weights(monkeypatch):
    """Prevent stale CPU backups or shared storage from hiding real weight changes."""
    with pytest.raises(RuntimeError, match="live accelerator shard"):
        _device_shard(SimpleNamespace(param=torch.ones(4), megatron_name="weight"))
    local = torch.arange(4, dtype=torch.bfloat16)
    monkeypatch.setattr(megatron_delta_export, "_device_shard", lambda record: local)
    record = SimpleNamespace(param=local, contributes=True, megatron_name="weight")
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
        patches = [
            CheckpointWeightPatch(
                name="weight",
                shape=(4, 2),
                dtype=values.dtype,
                indices=indices[i : i + 1],
                values=values[i : i + 1],
            )
            for i in range(2)
        ]
        load_checkpoint_weight_patches(model, patches)
        expected.view(-1).index_copy_(0, indices.long(), values)
        assert torch.equal(model.weight.view(torch.int16), expected[rank * 2 : rank * 2 + 2].view(torch.int16))
    load_checkpoint_weight_patches(model, [])


def test_bad_wire_bounds_fail_before_mutation():
    model = _TPShardModel(0)
    before = model.weight.detach().clone()
    patches = [
        CheckpointWeightPatch(
            name="weight",
            shape=(4, 2),
            dtype=torch.float32,
            indices=torch.tensor([8], dtype=torch.int32),
            values=torch.ones(1),
        )
    ]
    with pytest.raises((IndexError, ValueError)):
        load_checkpoint_weight_patches(model, patches)
    assert torch.equal(model.weight, before)


def test_pp_placeholders_share_owner_dtype(monkeypatch):
    """Keep FP32 empty rows aligned; reject conflicting owner dtypes."""
    slots = [("weight", (4,))]
    from vime.utils import distributed_utils

    group = object()
    monkeypatch.setattr(distributed_utils, "get_gloo_group", lambda: group)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda **kw: 3)
    monkeypatch.setattr(torch.accelerator, "current_device_index", lambda: "cpu")
    for conflicting in (False, True):
        record = SimpleNamespace(param=None, megatron_name="remote", gather_group=None)
        rows = [[None], [(slots, torch.float32)], [(slots, torch.bfloat16 if conflicting else torch.float32)]]
        monkeypatch.setattr(
            torch.distributed,
            "all_gather_object",
            lambda output, local, rows=rows, **kw: output.__setitem__(slice(None), rows),
        )
        if conflicting:
            with pytest.raises(ValueError, match="inconsistent owner dtypes"):
                megatron_delta_export._exchange_slot_tables([record])
            continue
        megatron_delta_export._exchange_slot_tables([record])
        assert record.dtype == torch.float32 and record.slots == slots
        entries = list(megatron_delta_export.iter_delta_entries([record], {}))
        assert entries[0][1] == "float32"
        assert entries[0][4].dtype == torch.float32
        assert entries[0][4].numel() == 0


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float32])
def test_compact_shard_masks_is_bit_exact(dtype):
    """Preserve BF16/FP32 changed bits; unchanged shards must emit no payload."""
    torch.manual_seed(0)
    previous = torch.randn(1000, dtype=dtype)
    current = previous.clone()
    changed = torch.tensor([3, 17, 500, 999], dtype=torch.int64)
    current[changed] += 0.5

    indices, values = compact_shard_masks([current], [shard_delta_mask(current, previous)])[0]

    assert torch.equal(indices, changed)
    integer_dtype = torch.int16 if dtype == torch.bfloat16 else torch.int32
    assert torch.equal(values.view(integer_dtype), current[changed].view(integer_dtype))
    indices, values = compact_shard_masks([current.clone()], [shard_delta_mask(current, current)])[0]
    assert indices.numel() == values.numel() == 0


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
    assert calls
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


@pytest.mark.unit
def test_sparse_bucket_preserves_checkpoint_coordinates_when_split():
    sent = []
    bucket = updater_module._SparseFlushBucket(16, lambda patches: sent.append(list(patches)))
    bucket.add("weight", (2, 3), torch.tensor([0, 2, 3, 5], dtype=torch.int32), torch.tensor([1.0, 2.0, 3.0, 4.0]))
    bucket.flush()
    assert len(sent) > 1
    assert all(sum(p.indices.numel() * (4 + p.values.element_size()) for p in chunk) <= 16 for chunk in sent)
    patches = [patch for chunk in sent for patch in chunk]
    assert all(patch.full_shape == (2, 3) and patch.indices.dtype == torch.int32 for patch in patches)
    assert torch.equal(torch.cat([patch.indices for patch in patches]), torch.tensor([0, 2, 3, 5], dtype=torch.int32))
    assert torch.equal(torch.cat([patch.values for patch in patches]), torch.tensor([1.0, 2.0, 3.0, 4.0]))


@pytest.mark.unit
@pytest.mark.parametrize("rank,capacity", [(0, 16), (0, 64), (1, 16)], ids=["split", "whole", "non-sender"])
def test_sparse_seed_uses_public_sparse_chunks(monkeypatch, rank, capacity):
    instance = _make_updater(update_weight_buffer_size=capacity)
    expected = {"weight": torch.arange(6, dtype=torch.float32).reshape(2, 3), "bias": torch.tensor([3.0, 5.0])}
    exported = []

    def export():
        for name, tensor in expected.items():
            exported.append(name)
            yield [(name, tensor)]

    instance._iterator = SimpleNamespace(get_hf_weight_chunks=export)
    sent = []
    instance._trainer = SimpleNamespace(
        model_update_group=SimpleNamespace(device=torch.device("cpu")),
        send_weight_chunk=lambda patches: sent.append(list(patches)),
    )
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: rank)
    assert instance._send_seed() is None
    assert exported == list(expected)
    assert instance._payload_started == (rank == 0)
    if rank != 0:
        assert not sent
        return
    assert sent
    actual = {name: torch.full_like(tensor, float("nan")).flatten() for name, tensor in expected.items()}
    for chunk in sent:
        assert sum(p.indices.numel() * (p.indices.element_size() + p.values.element_size()) for p in chunk) <= capacity
        for patch in chunk:
            assert patch.full_shape == tuple(expected[patch.name].shape)
            actual[patch.name].index_copy_(0, patch.indices.long(), patch.values)
    assert all(torch.equal(actual[name], tensor.flatten()) for name, tensor in expected.items())


@pytest.mark.unit
@pytest.mark.parametrize("indices,values", [([-1], [1.0]), ([4], [1.0]), ([1, 1], [1.0, 2.0]), ([0], [float("nan")])])
def test_sparse_payload_rejected_before_engine_send(monkeypatch, indices, values):
    instance = _make_updater()
    instance._trainer = SimpleNamespace(
        model_update_group=SimpleNamespace(device=torch.device("cpu")),
        send_weight_chunk=lambda patches: pytest.fail("invalid payload reached sender"),
    )
    monkeypatch.setattr(updater_module.dist, "get_rank", lambda: 0)
    patch = updater_module.SparseWeightPatch(
        "weight", torch.tensor(indices, dtype=torch.int32), torch.tensor(values), (4,)
    )
    with pytest.raises(ValueError, match="Sparse payload"):
        instance._publish_flush([patch])
    assert not instance._payload_started


@pytest.mark.unit
def test_sparse_payload_normalizes_noncontiguous_tensors():
    patches = [
        updater_module.SparseWeightPatch(
            name, torch.tensor([0, 99, 3, 99], dtype=torch.int32)[::2], torch.tensor([0.0, 99.0, 7.0, 99.0])[::2], (4,)
        )
        for name in ("first", "second")
    ]
    prepared = updater_module.prepare_sparse_patches(patches, torch.device("cpu"))
    assert all(p.indices.is_contiguous() and p.values.is_contiguous() for p in prepared)
    assert all(p.indices.tolist() == [0, 3] for p in prepared)
    assert all(p.values.tolist() == [0.0, 7.0] for p in prepared)


@pytest.mark.unit
def test_sparse_loader_large_shape_limit_is_explicit():
    patch = updater_module.SparseWeightPatch(
        "large", torch.tensor([0], dtype=torch.int32), torch.tensor([1.0]), (2**31,)
    )
    with pytest.raises(ValueError, match="smaller than 2\\*\\*31"):
        updater_module.prepare_sparse_patches([patch], torch.device("cpu"))


@pytest.mark.unit
@pytest.mark.parametrize(
    "config",
    [None, {"pp_size": 2}, {"pipeline_parallel_size": 2}, {"pcp_size": 2}, {"prefill_context_parallel_size": 2}],
    ids=["args-pp", "external-pp-short", "external-pp", "external-pcp-short", "external-pcp"],
)
def test_sparse_parallel_config_rejected_before_rendezvous(config):
    instance = _make_updater(vllm_pipeline_parallel_size=2 if config is None else 1)
    instance.disconnect_rollout_engines = lambda: pytest.fail("unsupported config reached connection setup")
    with pytest.raises(ValueError, match="requires inference"):
        instance.connect_rollout_engines([], None, engine_parallel_configs=None if config is None else [config])


@pytest.mark.unit
def test_sparse_updates_do_not_replay_seed(updater):
    finishes = []
    updater._send_sparse_delta = lambda: None
    updater._send_seed = lambda: pytest.fail("steady updates must not replay seed")
    updater._finish_update = lambda: finishes.append(updater.weight_version)
    for _ in range(3):
        updater.update_weights()
    assert finishes == [2, 3, 4]


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
        updater._send_seed = lambda **kwargs: pytest.fail("NaN seed must not be sent")
        updater._send_sparse_delta = lambda **kwargs: pytest.fail("NaN delta must not be sent")
        monkeypatch.setattr(updater_module, "prime_delta_snapshots", lambda *a, **kw: pytest.fail("snapshot changed"))
        with pytest.raises(ValueError, match="NaN weights"):
            updater.update_weights()
        assert recovered == [True] and not updater._payload_started
    else:
        updater._send_sparse_delta = lambda **kwargs: None
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

    def send():
        if phase == "payload":
            updater._payload_started = True
            fail()
        return None

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


@pytest.mark.unit
def test_tiny_checkpoint_unpacks_every_expert():
    prefix = "model.layers.0.mlp.experts"
    fused = torch.arange(48, dtype=torch.float32).reshape(2, 6, 4)
    down = torch.arange(24, dtype=torch.float32).reshape(2, 4, 3)
    other = torch.ones(4)
    expected_gate, expected_up = fused.chunk(2, dim=1)
    unpacked = _unpack_hf_experts(
        {f"{prefix}.gate_up_proj": fused, f"{prefix}.down_proj": down, "model.norm.weight": other}
    )
    assert set(unpacked) == {
        "model.norm.weight",
        *(
            f"{prefix}.{expert}.{projection}.weight"
            for expert in range(2)
            for projection in ("gate_proj", "up_proj", "down_proj")
        ),
    }
    for expert in range(2):
        assert torch.equal(unpacked[f"{prefix}.{expert}.gate_proj.weight"], expected_gate[expert])
        assert torch.equal(unpacked[f"{prefix}.{expert}.up_proj.weight"], expected_up[expert])
        assert torch.equal(unpacked[f"{prefix}.{expert}.down_proj.weight"], down[expert])
    assert unpacked["model.norm.weight"] is other
    assert _unpack_hf_experts(unpacked).keys() == unpacked.keys()
    fused.zero_()
    down.zero_()
    assert unpacked[f"{prefix}.1.gate_proj.weight"].sum() > 0
    assert unpacked[f"{prefix}.1.down_proj.weight"].sum() > 0


@pytest.mark.integration
def test_distributed_sparse_hccl_matches_full(distributed_rendezvous):
    """Run real HCCL boundaries and full-state equality; any rank failure fails pytest."""
    # NOTE: Spawn four ranks, check real HCCL gather boundaries, then build a
    # TP2/PP2/EP2 Qwen3-MoE model. Save its full HF baseline, modify sparse
    # positions and apply the gathered delta to that baseline. Compare every
    # reconstructed tensor byte-for-byte with the independent native full export
    # of the updated model, including untouched positions and BF16/FP32 dtypes.
    # This checks trainer-side HF reconstruction, not the live vLLM receiver.
    port, cfg_dir = distributed_rendezvous
    context = mp.spawn(_distributed_worker, args=(port, cfg_dir), nprocs=TP * PP, join=False)
    deadline = time.monotonic() + 600
    try:
        while not context.join(timeout=1):
            if time.monotonic() >= deadline:
                pytest.fail("Sparse HCCL workers did not finish within 600 seconds")
    finally:
        for process in context.processes:
            if process.is_alive():
                process.terminate()
        for process in context.processes:
            process.join(timeout=10)
            if process.is_alive():
                process.kill()
                process.join(timeout=5)


@pytest.mark.unit
@pytest.mark.parametrize("indices", [[3, 0, 2], [2, 0, 2]])
def test_unordered_sparse_payload_keeps_value_pairing_and_rejects_duplicates(indices):
    patch = updater_module.SparseWeightPatch(
        "weight", torch.tensor(indices, dtype=torch.int32), torch.tensor([9.0, 0.0, 7.0]), (4,)
    )
    if len(set(indices)) != len(indices):
        with pytest.raises(ValueError, match="duplicate"):
            updater_module.prepare_sparse_patches([patch], torch.device("cpu"))
    else:
        prepared = updater_module.prepare_sparse_patches([patch], torch.device("cpu"))[0]
        assert torch.equal(prepared.indices, patch.indices)
        assert torch.equal(prepared.values, patch.values)


# CPU-only coordinate conversion regressions.


@pytest.fixture
def coordinate_converter():
    names = (
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
        "ray",
        "ray.actor",
        "vime.utils.model_config",
    )
    saved = _unit_stubs.save_sys_modules(names)
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    config = types.ModuleType("vime.utils.model_config")
    config.load_hf_config = lambda *a, **k: pytest.fail("HF config loading is outside coordinate conversion")
    sys.modules[config.__name__] = config
    try:
        probe = importlib.import_module("vime.backends.megatron_utils.delta_sync.sparse_hf_export")
        common = importlib.import_module("vime.backends.megatron_utils.update_weight.common")
        converter = importlib.import_module("vime.backends.megatron_utils.megatron_to_hf")
        yield probe, common, converter
    finally:
        _unit_stubs.restore_sys_modules(saved)


CASES = [
    ("embedding.word_embeddings.weight", (8, 4), 0, None),
    ("output_layer.weight", (8, 4), 0, None),
    ("decoder.final_layernorm.weight", (4,), None, None),
    ("decoder.layers.3.self_attention.linear_qkv.weight", (16, 4), 0, None),
    ("decoder.layers.3.self_attention.linear_qkv.bias", (16,), 0, None),
    ("decoder.layers.3.self_attention.linear_proj.weight", (4, 8), 1, None),
    ("decoder.layers.3.mlp.linear_fc1.weight", (16, 4), 0, None),
    ("decoder.layers.3.mlp.experts.linear_fc1.weight7", (16, 4), 0, None),
    ("decoder.layers.3.mlp.experts.linear_fc2.weight7", (4, 8), 0, None),
    ("decoder.layers.3.self_attention.q_layernorm.weight", (2,), None, None),
    ("decoder.layers.3.mlp.linear_fc1.weight", (16, 4), 0, (8, 8)),
]


@pytest.mark.unit
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
@pytest.mark.parametrize("tp_size", [1, 2, 4])
@pytest.mark.parametrize("name,shape,partition_dim,sections", CASES)
def test_coordinates_match_full_native_conversion(
    coordinate_converter, dtype, tp_size, name, shape, partition_dim, sections
):
    probe_module, common, converter = coordinate_converter
    args = SimpleNamespace(vocab_size=7, kv_channels=2, hidden_size=4, num_attention_heads=4, num_query_groups=2)
    name = "module.module." + name
    # Grouped expert FC2 records dimension zero but is actually column-sharded.
    dim = 1 if "linear_fc2.weight" in name and partition_dim == 0 else partition_dim
    if dim is not None:
        assert shape[dim] % tp_size == 0
    complete = (torch.arange(torch.tensor(shape).prod().item()).reshape(shape).float() / 8).to(dtype)
    if dim is None:
        shards = [complete.clone()]
        effective_tp = 1
    elif "linear_fc1.weight" in name:
        gate, up = complete.chunk(2, dim=0)
        shards = [torch.cat(pair) for pair in zip(gate.chunk(tp_size), up.chunk(tp_size), strict=True)]
        effective_tp = tp_size
    else:
        shards = [part.clone() for part in complete.chunk(tp_size, dim=dim)]
        effective_tp = tp_size
    attrs = {
        "tensor_model_parallel": dim is not None,
        "partition_dim": partition_dim,
        "partition_stride": 1,
        "partition_sections": sections,
    }
    infos = [SimpleNamespace(name=name, attrs=attrs, shape=s.shape) for s in shards]
    expected_before = dict(converter.convert_to_hf(args, "qwen3moe", name, complete, None))
    actual = {key: value.clone() for key, value in expected_before.items()}
    changed_shards = []
    emitted = 0
    for rank, (shard, info) in enumerate(zip(shards, infos, strict=True)):
        # Include boundaries, zero values, negative zero and sparse interior changes.
        flat = shard.flatten().clone()
        positions = torch.unique(torch.tensor([0, flat.numel() // 2, flat.numel() - 1]))
        flat[positions] = -0.0
        if positions.numel() > 1:
            flat[positions[-1]] = -3.0
        changed_shards.append(flat.reshape(shard.shape))
        sparse = torch.full_like(shard, float("nan"))
        sparse.flatten()[positions] = flat[positions]
        before_probe = sparse.clone()
        probe = probe_module.SparseHFProbe(args, "qwen3moe", info, rank, effective_tp)
        for key, value in probe.megatron_to_hf(sparse).items():
            mask = ~torch.isnan(value)
            actual[key][mask] = value[mask]
            emitted += mask.sum().item()
        integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
        assert torch.equal(sparse.view(integer), before_probe.view(integer))
    rebuilt = (
        changed_shards[0]
        if dim is None
        else common._merge_param_partitions(name, changed_shards, partition_dim, sections)
    )
    expected = dict(converter.convert_to_hf(args, "qwen3moe", name, rebuilt, None))
    assert actual.keys() == expected.keys()
    assert emitted > 0
    integer = torch.int16 if dtype == torch.bfloat16 else torch.int32
    for key in expected:
        assert actual[key].shape == expected[key].shape
        assert torch.equal(actual[key].contiguous().view(integer), expected[key].contiguous().view(integer)), key


@pytest.mark.unit
def test_coordinates_need_no_bridge_or_collective(coordinate_converter, monkeypatch):
    probe_module, common, converter = coordinate_converter
    monkeypatch.setitem(sys.modules, "megatron.bridge", None)
    monkeypatch.setattr(torch.distributed, "all_gather", lambda *a, **k: pytest.fail("dense gather"))
    monkeypatch.setattr(torch.distributed, "broadcast", lambda *a, **k: pytest.fail("dense broadcast"))
    args = SimpleNamespace(vocab_size=4)
    info = SimpleNamespace(
        name="module.module.embedding.word_embeddings.weight",
        attrs={"tensor_model_parallel": True, "partition_dim": 0},
    )
    result = probe_module.SparseHFProbe(args, "qwen3moe", info, 1, 2).megatron_to_hf(torch.ones(2, 4))
    value = result["model.embed_tokens.weight"]
    assert torch.isnan(value[:2]).all() and torch.equal(value[2:], torch.ones(2, 4))
    empty = probe_module.SparseHFProbe(args, "qwen3moe", info, 1, 2).megatron_to_hf(torch.full((2, 4), float("nan")))
    assert torch.isnan(empty["model.embed_tokens.weight"]).all()


@pytest.mark.unit
@pytest.mark.parametrize("invalid", ["duplicate", "dtype"])
def test_coordinates_reject_ambiguous_or_dtype_changing_output(coordinate_converter, monkeypatch, invalid):
    probe_module, _, converter = coordinate_converter
    output = (
        [("hf", torch.ones(2)), ("hf", torch.zeros(2))] if invalid == "duplicate" else [("hf", torch.ones(2).half())]
    )
    monkeypatch.setattr(converter, "convert_to_hf", lambda *a, **k: output)
    info = SimpleNamespace(name="weight", attrs={})
    with pytest.raises(ValueError, match="Unsupported sparse conversion output"):
        probe_module.SparseHFProbe(SimpleNamespace(), "qwen3moe", info, 0, 1).megatron_to_hf(torch.ones(2))


@pytest.mark.unit
@pytest.mark.parametrize(
    "expert,sharded,tp_rank,tp_size,replica,rank,owner,local,expected",
    [
        (False, True, 1, 2, 0, 1, 1, True, True),
        (False, False, 1, 2, 0, 1, 1, True, False),
        (False, True, 0, 2, 1, 0, 0, True, False),
        (False, True, 0, 2, 0, 2, 2, False, False),
        (True, True, 0, 1, 0, 1, 1, True, True),
        (True, True, 1, 2, 0, 3, 3, True, True),
        (True, False, 1, 2, 0, 3, 3, True, False),
        (False, True, 0, 2, 0, 2, 0, True, False),
    ],
)
def test_coordinates_directory_retains_shards_and_deduplicates_replicas(
    coordinate_converter,
    monkeypatch,
    expert,
    sharded,
    tp_rank,
    tp_size,
    replica,
    rank,
    owner,
    local,
    expected,
):
    export = importlib.import_module("vime.backends.megatron_utils.delta_sync.megatron_delta_export")
    mpu = sys.modules["megatron.core.parallel_state"]
    monkeypatch.setattr(sys.modules["megatron.core"], "parallel_state", mpu)
    for prefix in ("tensor_model", "expert_tensor"):
        monkeypatch.setattr(mpu, f"get_{prefix}_parallel_rank", lambda: tp_rank, raising=False)
        monkeypatch.setattr(mpu, f"get_{prefix}_parallel_world_size", lambda: tp_size, raising=False)
    monkeypatch.setattr(mpu, "get_data_parallel_rank", lambda **kw: replica, raising=False)
    monkeypatch.setattr(mpu, "get_expert_data_parallel_rank", lambda: replica, raising=False)
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: rank)
    monkeypatch.setattr(
        export, "_exchange_slot_tables", lambda rows: [setattr(r, "slots", [("hf", (4,))]) for r in rows]
    )
    name = "mlp.experts.linear_fc1.weight7" if expert else "embedding.word_embeddings.weight"
    info = SimpleNamespace(
        name=name, dtype=torch.float32, src_rank=owner, attrs={"tensor_model_parallel": sharded, "partition_dim": 0}
    )
    iterator = SimpleNamespace(
        args=SimpleNamespace(),
        model_name="qwen3moe",
        hf_vocab_size=4,
        param_infos=[info],
        local_weights=lambda: {name: torch.zeros(4)} if local else {},
    )
    (record,) = export.build_export_index(iterator, [])
    assert record.contributes is expected
    assert (record.param is not None) is local


@pytest.mark.unit
@pytest.mark.parametrize(
    "options",
    [
        {"quantization_config": {"quant_method": "fp8"}},
        {"dspark_enabled": True},
        {"enable_mtp_training": True},
    ],
)
def test_coordinates_reject_unsupported_conversion_before_initialization(coordinate_converter, options):
    probe_module, _, _ = coordinate_converter
    options = options.copy()
    quantization = options.pop("quantization_config", None)
    with pytest.raises(ValueError):
        probe_module.SparseHFExporter(
            SimpleNamespace(**options), [], model_name="qwen3moe", quantization_config=quantization
        )


def _cpu_gather_worker(rank, rendezvous):
    path = pathlib.Path(__file__).resolve().parents[1] / "vime/backends/megatron_utils/delta_sync/sparse_gather.py"
    spec = importlib.util.spec_from_file_location("sparse_gather_cpu", path)
    gather = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gather)
    dist.init_process_group("gloo", init_method=rendezvous, rank=rank, world_size=2)
    try:
        for rows in ([[0], [31]], [[1, 13, 0], [3, 0, 2]], [[0, 0], [0, 0]]):
            for dtype in (torch.bfloat16, torch.float32):
                for budget in (12, 128):
                    pieces = [
                        [torch.arange(n, dtype=torch.int32) + r * 100 + s * 40 for s, n in enumerate(row)]
                        for r, row in enumerate(rows)
                    ]
                    indices = torch.cat(pieces[rank])
                    values = indices.to(dtype)

                    class Workspace(gather.GatherWorkspace):
                        def tensor(self, key, size, reference, budget=budget, indices=indices, values=values):
                            if key in {"recv_indices", "recv_values"}:
                                assert size * (indices.element_size() + values.element_size()) <= budget
                            return super().tensor(key, size, reference)

                    workspace = Workspace()
                    previous = None
                    for _ in range(2):
                        result = gather.gather_slot_entries_to_rank0(
                            indices,
                            values,
                            torch.tensor(rows[rank]),
                            max_round_bytes=budget,
                            workspace=workspace,
                        )
                        if rank == 0:
                            for s, (actual_indices, actual_values) in enumerate(result):
                                expected = torch.cat([pieces[r][s] for r in range(2)])
                                assert torch.equal(actual_indices, expected)
                                assert torch.equal(actual_values, expected.to(dtype))
                                if previous is not None:
                                    assert torch.equal(previous[s][0], expected)
                                    assert torch.equal(previous[s][1], expected.to(dtype))
                            previous = result
                        else:
                            assert result is None
    finally:
        dist.destroy_process_group()


@pytest.fixture
def cpu_gather():
    return _cpu_gather_worker


@pytest.mark.unit
def test_sparse_gather_cpu_oversized_slots_and_total_receive_budget(tmp_path, cpu_gather):
    mp.spawn(cpu_gather, args=(f"file://{tmp_path / 'rendezvous'}",), nprocs=2, join=True)
