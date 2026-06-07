"""Unit tests for colocated vLLM IPC weight sync (UpdateWeightFromTensor)."""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field
from unittest.mock import MagicMock, patch

import pytest
import torch

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_tensor"


def _install_stubs():
    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
    mpu_stub.get_tensor_model_parallel_group.return_value = "tp_group"
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu_stub
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.core = megatron_core
    sys.modules.setdefault("megatron", megatron_mod)
    sys.modules.setdefault("megatron.core", megatron_core)

    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_mod.ObjectRef = object
    ray_mod.actor = types.ModuleType("ray.actor")
    ray_mod.actor.ActorHandle = object
    sys.modules.setdefault("ray", ray_mod)
    sys.modules.setdefault("ray.actor", ray_mod.actor)

    import torch.distributed as _dist

    dist_stub = MagicMock()
    dist_stub.get_rank.return_value = 0
    dist_stub.get_world_size.return_value = 1
    dist_stub.get_process_group_ranks.return_value = [0, 1]
    dist_stub.barrier = MagicMock()
    dist_stub.all_gather_object = MagicMock()
    _dist.get_rank = dist_stub.get_rank
    _dist.get_world_size = dist_stub.get_world_size
    _dist.get_process_group_ranks = dist_stub.get_process_group_ranks
    _dist.barrier = dist_stub.barrier
    _dist.all_gather_object = dist_stub.all_gather_object

    vime_utils = types.ModuleType("vime.utils.distributed_utils")
    vime_utils.get_gloo_group = MagicMock(return_value="gloo")
    sys.modules.setdefault("vime.utils.distributed_utils", vime_utils)

    hf_iter_stub = MagicMock()
    hf_iter_stub.get_hf_weight_chunks.return_value = iter([])

    hf_base_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base")
    hf_base_mod.HfWeightIteratorBase = MagicMock()
    hf_base_mod.HfWeightIteratorBase.create.return_value = hf_iter_stub

    upw_dist_mod = types.ModuleType("vime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    upw_dist_mod.connect_rollout_engines_from_distributed = MagicMock(return_value="groups")
    upw_dist_mod.disconnect_rollout_engines_from_distributed = MagicMock()
    upw_dist_mod.post_process_weights = MagicMock()
    upw_dist_mod.update_weights_from_distributed = MagicMock(return_value=[])

    for key, mod in [
        ("vime.backends.megatron_utils.update_weight.hf_weight_iterator_base", hf_base_mod),
        ("vime.backends.megatron_utils.update_weight.update_weight_from_distributed", upw_dist_mod),
    ]:
        sys.modules.setdefault(key, mod)

    return hf_iter_stub, upw_dist_mod


_HF_ITER_STUB, _UPW_DIST_MOD = _install_stubs()


@pytest.fixture(scope="module")
def upw_vllm():
    sys.modules.pop(MODULE_PATH, None)
    return importlib.import_module(MODULE_PATH)


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self):
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return "ref"


@dataclass
class RecordingVLLMEngine:
    release_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    resume_memory_occupation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    init_weight_transfer_engine: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    finish_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights_from_tensor: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    pause_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    flush_cache: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    continue_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)


def _default_args(**kwargs) -> Namespace:
    base = dict(
        actor_num_nodes=1,
        actor_num_gpus_per_node=4,
        rollout_num_gpus_per_engine=2,
        megatron_to_hf_mode="raw",
        update_weight_buffer_size=1 << 30,
    )
    base.update(kwargs)
    return Namespace(**base)


def _make_instance(upw_vllm, args=None):
    obj = object.__new__(upw_vllm.UpdateWeightFromTensor)
    obj.args = args or _default_args()
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test"
    obj.quantization_config = None
    obj.weight_version = 0
    obj._hf_weight_iterator = _HF_ITER_STUB
    obj.rollout_engines = []
    obj.distributed_rollout_engines = []
    obj.use_distribute = False
    obj._ipc_engine = None
    obj._ipc_gather_group = None
    obj._ipc_gather_src = None
    obj._model_update_groups = None
    obj._is_distributed_src_rank = False
    obj._group_name = "vime"
    obj._ipc_initialized = False
    return obj


def _bind_single_slot(obj, engine, *, src=0):
    """Bind ``obj`` to one colocated engine forming a slot whose leader rank is ``src``."""
    obj.rollout_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_gather_group = "slot_group"
    obj._ipc_gather_src = src


def _chunks(n=1):
    return [[(f"p.{i}", torch.zeros(2, 2)) for i in range(2)] for _ in range(n)]


def _run_update(obj, *, chunks=None, rank=0, slot_size=1) -> dict:
    """Drive ``update_weights`` with controlled rank / slot size.

    ``slot_size`` is what ``dist.get_world_size(self._ipc_gather_group)`` returns,
    so slot_size==1 takes the direct IPC path and slot_size>1 the gather path.
    Returns counters for barriers and ipc_collect calls.
    """
    chunks = chunks or _chunks(1)
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(chunks)

    counters = {"barrier": 0, "ipc_collect": 0}

    def counting_barrier(*args, **kwargs):
        counters["barrier"] += 1

    def counting_ipc_collect(*args, **kwargs):
        counters["ipc_collect"] += 1

    with patch("torch.distributed.get_rank", return_value=rank), patch(
        "torch.distributed.get_world_size", return_value=slot_size
    ), patch("torch.distributed.barrier", side_effect=counting_barrier), patch(
        "torch.cuda.ipc_collect", side_effect=counting_ipc_collect
    ):
        obj.update_weights()
    return counters


@pytest.mark.unit
def test_colocated_lifecycle_uses_pause_flush_and_weight_transfer_apis(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]], "ipc_handles": [{"u": ("f", ())}]}
    with patch(f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors", return_value=(dummy_info, [])):
        counters = _run_update(obj, chunks=_chunks(2))

    # Colocate quiesce: pause_generation + flush_cache only, no /sleep round-trip;
    # continue_generation resumes. No release/resume_memory_occupation.
    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1
    assert len(engine.release_memory_occupation.calls) == 0
    assert len(engine.resume_memory_occupation.calls) == 0
    # vLLM #39212: init runs in connect_rollout_engines, not update_weights.
    assert len(engine.init_weight_transfer_engine.calls) == 0
    assert len(engine.start_weight_update.calls) == 1
    assert engine.start_weight_update.calls[0].kwargs.get("is_checkpoint_format") is True
    assert len(engine.finish_weight_update.calls) == 1
    assert len(engine.continue_generation.calls) == 1
    # ipc_collect: one per HF chunk + one after the loop.
    assert counters["ipc_collect"] == 2 + 1
    # lifecycle barriers (no per-chunk barrier).
    assert counters["barrier"] >= 4


@pytest.mark.unit
def test_send_via_ipc_dispatches_update_weights_from_tensor_with_version(upw_vllm):
    """slot_size=1: every HF chunk fires
    ``engine.update_weights_from_tensor.remote(**fields, weight_version=...)`` —
    same name, parameterized fields, version travels with data (no piggyback onto
    ``finish_weight_update``)."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": ["w"], "dtype_names": ["bfloat16"], "shapes": [[2, 2]], "ipc_handles": [{"u": ("f", ())}]}
    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ):
        _run_update(obj, chunks=_chunks(2))

    # 2 HF chunks → 2 IPC RPCs
    assert len(engine.update_weights_from_tensor.calls) == 2
    kwargs = engine.update_weights_from_tensor.calls[0].kwargs
    # fields are passed as explicit kwargs (** expanded from local_info)
    assert kwargs["names"] == dummy_info["names"]
    assert kwargs["dtype_names"] == dummy_info["dtype_names"]
    assert kwargs["shapes"] == dummy_info["shapes"]
    assert kwargs["ipc_handles"] is dummy_info["ipc_handles"]
    # weight_version is the trainer's post-increment version (0 + 1 = 1) as a str
    assert kwargs["weight_version"] == "1"
    # finish_weight_update is a stateless bookend now — no kwargs
    assert len(engine.finish_weight_update.calls) == 1
    assert engine.finish_weight_update.calls[0].kwargs == {}


@pytest.mark.unit
def test_send_via_ipc_dispatches_update_weights_from_tensor_coordinator_multi_gpu(upw_vllm):
    """slot_size > 1: the slot leader (rank == _ipc_gather_src) gathers payloads from
    all slot ranks, merges them, and fires a single update_weights_from_tensor RPC per chunk."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    _bind_single_slot(obj, engine, src=0)

    dummy_info_0 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu0": ("f", ())}],
    }
    dummy_info_1 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu1": ("f", ())}],
    }

    def fake_all_gather_object(gathered_payloads, payload, group=None):
        gathered_payloads[0] = "payload0"
        gathered_payloads[1] = "payload1"

    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info_0, []),
    ), patch(
        f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload0"
    ), patch(f"{MODULE_PATH}._deserialize_ipc_update_info", side_effect=[dummy_info_0, dummy_info_1] * 2), patch(
        "torch.distributed.all_gather_object", side_effect=fake_all_gather_object
    ):
        _run_update(obj, chunks=_chunks(2), rank=0, slot_size=2)

    assert len(engine.update_weights_from_tensor.calls) == 2
    kwargs = engine.update_weights_from_tensor.calls[0].kwargs
    assert kwargs["names"] == dummy_info_0["names"]
    assert kwargs["dtype_names"] == dummy_info_0["dtype_names"]
    assert kwargs["shapes"] == dummy_info_0["shapes"]
    assert len(kwargs["ipc_handles"]) == 1
    assert set(kwargs["ipc_handles"][0].keys()) == {"uuid-gpu0", "uuid-gpu1"}
    assert kwargs["weight_version"] == "1"


@pytest.mark.unit
def test_merge_ipc_update_infos_combines_gpu_uuids(upw_vllm):
    info0 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu0": ("f0", ())}],
    }
    info1 = {
        "names": ["w"],
        "dtype_names": ["bfloat16"],
        "shapes": [[2, 2]],
        "ipc_handles": [{"uuid-gpu1": ("f1", ())}],
    }
    merged = upw_vllm._merge_ipc_update_infos([info0, info1])
    assert set(merged["ipc_handles"][0].keys()) == {"uuid-gpu0", "uuid-gpu1"}


@pytest.mark.unit
def test_connect_binds_engine_and_slot_leader_per_gpu_slot(upw_vllm):
    """Each rank binds to its slot's engine; the slot leader (== _ipc_gather_src,
    the lowest trainer rank in the engine GPU range) is the start/finish coordinator."""
    engines = [RecordingVLLMEngine() for _ in range(4)]
    for rank, engine_idx, expected_src in [
        (0, 0, 0),
        (1, 0, 0),
        (2, 1, 2),
        (3, 1, 2),
    ]:
        obj = _make_instance(
            upw_vllm,
            args=_default_args(actor_num_gpus_per_node=8, rollout_num_gpus_per_engine=2),
        )
        with patch("torch.distributed.get_rank", return_value=rank), patch(
            "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=rank % 2
        ), patch("torch.distributed.new_group", return_value="slot_group"):
            obj.connect_rollout_engines(
                engines,
                rollout_engine_lock=MagicMock(),
                engine_gpu_counts=[2, 2, 2, 2],
                engine_gpu_offsets=[0, 2, 4, 6],
            )
        assert obj._ipc_engine is engines[engine_idx]
        assert obj._ipc_gather_src == expected_src
        is_coordinator = rank == obj._ipc_gather_src
        assert is_coordinator is (rank in (0, 2))
        assert obj.use_distribute is False
        assert obj.distributed_rollout_engines == []
        # vLLM #39212: init_weight_transfer_engine fires once during connect (rank 0 only).
        if rank == 0:
            assert len(engines[0].init_weight_transfer_engine.calls) == 1
            assert engines[0].init_weight_transfer_engine.calls[0].args[0] == {"init_info": {}}


@pytest.mark.unit
def test_non_leader_skips_start_finish_and_merged_rpc(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    # slot leader is rank 0; we drive update_weights as rank 1 (non-leader).
    _bind_single_slot(obj, engine, src=0)

    dummy_info = {"names": [], "dtype_names": [], "shapes": [], "ipc_handles": []}
    with patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=(dummy_info, []),
    ), patch(
        f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload"
    ), patch("torch.distributed.all_gather_object") as all_gather_obj:
        _run_update(obj, chunks=_chunks(1), rank=1, slot_size=2)

    all_gather_obj.assert_called_once()
    # non-leader: no start/finish, and no merged update_weights_from_tensor RPC
    assert len(engine.start_weight_update.calls) == 0
    assert len(engine.finish_weight_update.calls) == 0
    assert len(engine.update_weights_from_tensor.calls) == 0


@pytest.mark.unit
def test_ipc_init_runs_once_in_connect(upw_vllm):
    """init_weight_transfer_engine fires once in connect_rollout_engines (rank 0),
    not in update_weights. A second connect call does not re-init."""
    engines = [RecordingVLLMEngine() for _ in range(2)]
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=2),
    )
    with patch("torch.distributed.get_rank", return_value=0), patch(
        "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0
    ), patch("torch.distributed.new_group", return_value="slot_group"):
        obj.connect_rollout_engines(
            engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert obj._ipc_initialized is True
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(engines[1].init_weight_transfer_engine.calls) == 1

    # Second connect with _ipc_initialized=True does not re-init.
    engines2 = [RecordingVLLMEngine() for _ in range(2)]
    with patch("torch.distributed.get_rank", return_value=0), patch(
        "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=0
    ), patch("torch.distributed.new_group", return_value="slot_group"):
        obj.connect_rollout_engines(
            engines2,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )
    assert len(engines2[0].init_weight_transfer_engine.calls) == 0
    assert len(engines2[1].init_weight_transfer_engine.calls) == 0
