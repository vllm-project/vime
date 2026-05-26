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

MODULE_PATH = "slime.backends.megatron_utils.update_weight.update_weight_from_tensor"


def _install_stubs():
    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
    mpu_stub.get_tensor_model_parallel_group.return_value = "tp_group"
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu_stub
    sys.modules.setdefault("megatron", types.ModuleType("megatron"))
    sys.modules.setdefault("megatron.core", megatron_core)

    ray_mod = types.ModuleType("ray")
    ray_mod.get = lambda refs: refs
    ray_mod.actor = types.ModuleType("ray.actor")
    ray_mod.actor.ActorHandle = object
    sys.modules.setdefault("ray", ray_mod)
    sys.modules.setdefault("ray.actor", ray_mod.actor)

    import torch.distributed as _dist

    dist_stub = MagicMock()
    dist_stub.get_rank.return_value = 0
    dist_stub.get_world_size.return_value = 8
    dist_stub.get_process_group_ranks.return_value = [0, 1]
    dist_stub.barrier = MagicMock()
    dist_stub.all_gather_object = MagicMock()
    _dist.get_rank = dist_stub.get_rank
    _dist.get_world_size = dist_stub.get_world_size
    _dist.get_process_group_ranks = dist_stub.get_process_group_ranks
    _dist.barrier = dist_stub.barrier
    _dist.all_gather_object = dist_stub.all_gather_object

    slime_utils = types.ModuleType("slime.utils.distributed_utils")
    slime_utils.get_gloo_group = MagicMock(return_value="gloo")
    sys.modules.setdefault("slime.utils.distributed_utils", slime_utils)

    sglang_mod = types.ModuleType("slime.backends.megatron_utils.sglang")
    sglang_mod.monkey_patch_torch_reductions = MagicMock()
    sys.modules.setdefault("slime.backends.megatron_utils.sglang", sglang_mod)

    hf_iter_stub = MagicMock()
    hf_iter_stub.get_hf_weight_chunks.return_value = iter([])

    hf_base_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.hf_weight_iterator_base")
    hf_base_mod.HfWeightIteratorBase = MagicMock()
    hf_base_mod.HfWeightIteratorBase.create.return_value = hf_iter_stub

    upw_dist_mod = types.ModuleType("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")
    upw_dist_mod.connect_rollout_engines_from_distributed = MagicMock(return_value="groups")
    upw_dist_mod.disconnect_rollout_engines_from_distributed = MagicMock()
    upw_dist_mod.post_process_weights = MagicMock()
    upw_dist_mod.update_weights_from_distributed = MagicMock(return_value=[])

    for key, mod in [
        ("slime.backends.megatron_utils.update_weight.hf_weight_iterator_base", hf_base_mod),
        ("slime.backends.megatron_utils.update_weight.update_weight_from_distributed", upw_dist_mod),
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
    obj._colocated_engines = []
    obj._ipc_engine = None
    obj._ipc_engine_coordinator = False
    obj._ipc_engine_slot_start = None
    obj._ipc_engine_slot_end = None
    obj._distributed_engines = []
    obj._model_update_groups = None
    obj._is_distributed_src_rank = False
    obj._group_name = "slime"
    obj._ipc_initialized = False
    return obj


def _chunks(n=1):
    return [[(f"p.{i}", torch.zeros(2, 2)) for i in range(2)] for _ in range(n)]


def _run_update(obj, *, chunks=None, ipc_engine_cls=None, ipc_args_cls=None) -> int:
    chunks = chunks or _chunks(1)
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(chunks)

    ipc_engine_cls = ipc_engine_cls or MagicMock()
    ipc_args_cls = ipc_args_cls or MagicMock(side_effect=lambda **kw: kw)

    ipc_mod = types.SimpleNamespace(
        IPCWeightTransferEngine=ipc_engine_cls,
        IPCTrainerSendWeightsArgs=ipc_args_cls,
    )
    barrier_calls = {"n": 0}

    def counting_barrier(*args, **kwargs):
        barrier_calls["n"] += 1

    with patch.dict("sys.modules", {"vllm.distributed.weight_transfer.ipc_engine": ipc_mod}):
        with patch("torch.distributed.get_rank", return_value=0), patch(
            "torch.distributed.barrier", side_effect=counting_barrier
        ):
            with patch(f"{MODULE_PATH}._apply_monkey_patch_torch_reductions"):
                obj.update_weights()
    return barrier_calls["n"]


@pytest.mark.unit
def test_colocated_lifecycle_uses_vllm_sleep_and_weight_transfer_apis(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_engine_coordinator = True
    obj._ipc_engine_slot_start = 0
    obj._ipc_engine_slot_end = 1

    barrier_count = _run_update(obj, chunks=_chunks(2))

    assert len(engine.release_memory_occupation.calls) == 1
    assert engine.release_memory_occupation.calls[0].kwargs.get("level") == 0
    assert len(engine.init_weight_transfer_engine.calls) == 1
    assert engine.init_weight_transfer_engine.calls[0].args[0] == {"init_info": {}}
    assert len(engine.start_weight_update.calls) == 1
    assert engine.start_weight_update.calls[0].kwargs.get("is_checkpoint_format") is True
    assert len(engine.finish_weight_update.calls) == 1
    assert len(engine.resume_memory_occupation.calls) == 1
    # lifecycle barriers + one per HF chunk
    assert barrier_count >= 2 + 2
    assert engine.resume_memory_occupation.calls[0].kwargs.get("tags") == ["weights", "kv_cache"]


@pytest.mark.unit
def test_trainer_send_weights_uses_single_llm_handle_per_rank(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_engine_coordinator = True
    obj._ipc_engine_slot_start = 0
    obj._ipc_engine_slot_end = 1

    captured: list[dict] = []

    def fake_args(**kw):
        captured.append(kw)
        return kw

    ipc_engine = MagicMock()
    _run_update(obj, chunks=_chunks(2), ipc_engine_cls=ipc_engine, ipc_args_cls=fake_args)

    assert ipc_engine.trainer_send_weights.call_count == 2
    assert captured[0]["mode"] == "ray"
    assert captured[0]["llm_handle"] is engine


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
def test_connect_marks_one_coordinator_per_engine_gpu_slot(upw_vllm):
    """Only the first trainer rank in each engine GPU range may call start/finish."""
    engines = [RecordingVLLMEngine() for _ in range(4)]
    for rank, is_coordinator, engine_idx, tp_rank in [
        (0, True, 0, 0),
        (1, False, 0, 1),
        (2, True, 1, 0),
        (3, False, 1, 1),
    ]:
        obj = _make_instance(
            upw_vllm,
            args=_default_args(actor_num_gpus_per_node=8, rollout_num_gpus_per_engine=2),
        )
        with patch("torch.distributed.get_rank", return_value=rank), patch(
            "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=tp_rank
        ):
            obj.connect_rollout_engines(
                engines,
                rollout_engine_lock=MagicMock(),
                engine_gpu_counts=[2, 2, 2, 2],
                engine_gpu_offsets=[0, 2, 4, 6],
            )
        assert obj._ipc_engine is engines[engine_idx]
        assert obj._ipc_engine_coordinator is is_coordinator


@pytest.mark.unit
def test_non_coordinator_skips_start_finish(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_engine_coordinator = False
    obj._ipc_engine_slot_start = 0
    obj._ipc_engine_slot_end = 2

    dummy_info = {"names": [], "dtype_names": [], "shapes": [], "ipc_handles": []}

    with patch("torch.distributed.get_rank", return_value=1), patch(
        "megatron.core.mpu.get_tensor_model_parallel_rank", return_value=1
    ), patch(
        f"{MODULE_PATH}._build_ipc_update_info_from_named_tensors",
        return_value=dummy_info,
    ), patch(
        f"{MODULE_PATH}._serialize_ipc_update_info", return_value="payload"
    ), patch(
        "torch.distributed.all_gather_object"
    ) as all_gather_obj:
        _run_update(obj, chunks=_chunks(1))

    all_gather_obj.assert_called_once()

    assert len(engine.start_weight_update.calls) == 0
    assert len(engine.finish_weight_update.calls) == 0


@pytest.mark.unit
def test_ipc_init_runs_once(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]
    obj._ipc_engine = engine
    obj._ipc_engine_coordinator = True
    obj._ipc_engine_slot_start = 0
    obj._ipc_engine_slot_end = 1

    _run_update(obj)
    _run_update(obj)

    assert len(engine.init_weight_transfer_engine.calls) == 1
    assert obj._ipc_initialized is True
