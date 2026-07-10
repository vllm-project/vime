"""Unit tests for colocated vLLM IPC weight sync."""

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

_PURGE_PREFIXES = ("megatron", "mindspeed", "vime.backends.megatron_utils")


def _collect_subtree(prefix: str) -> list[str]:
    """Collect all modules in sys.modules that start with the given prefix."""
    return [k for k in sys.modules.keys() if k == prefix or k.startswith(prefix + ".")]


def _install_stubs():
    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_world_size.return_value = 2
    mpu_stub.get_tensor_model_parallel_group.return_value = "tp_group"
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0

    megatron_core = types.ModuleType("megatron.core")
    megatron_core.__path__ = []
    megatron_core.mpu = mpu_stub
    megatron_mod = types.ModuleType("megatron")
    megatron_mod.__path__ = []
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


_HF_ITER_STUB = MagicMock()
_HF_ITER_STUB.get_hf_weight_chunks.return_value = iter([])

_STUBBED_MODULES = (
    "megatron",
    "megatron.core",
    "ray",
    "ray.actor",
    "vime.utils.distributed_utils",
    "vime.backends.megatron_utils.update_weight.hf_weight_iterator_base",
    "vime.backends.megatron_utils.update_weight.update_weight_from_distributed",
)
_DIST_ATTRS = ("get_rank", "get_world_size", "get_process_group_ranks", "barrier", "all_gather_object")


@pytest.fixture(scope="module")
def upw_vllm():
    import torch.distributed as _dist

    purge_keys = set()
    for prefix in _PURGE_PREFIXES:
        purge_keys.update(_collect_subtree(prefix))
    for k in _STUBBED_MODULES:
        purge_keys.add(k)
    purge_keys.add(MODULE_PATH)

    saved_mods = {k: sys.modules.get(k) for k in purge_keys}
    saved_dist = {a: getattr(_dist, a, None) for a in _DIST_ATTRS}
    for k in purge_keys:
        sys.modules.pop(k, None)
    _install_stubs()
    sys.modules.pop(MODULE_PATH, None)

    with patch("vime.utils.common.is_npu", return_value=False):
        try:
            yield importlib.import_module(MODULE_PATH)
        finally:
            for k, original in saved_mods.items():
                if original is None:
                    sys.modules.pop(k, None)
                else:
                    sys.modules[k] = original
            for a, original in saved_dist.items():
                if original is not None:
                    setattr(_dist, a, original)


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
    update_weights: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
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
    obj._model_update_groups = None
    obj._is_distributed_src_rank = False
    obj._group_name = "vime"
    obj._ipc_initialized = False
    return obj


def _chunks(n=1):
    return [[(f"p.{i}", torch.zeros(2, 2)) for i in range(2)] for _ in range(n)]


def _run_update(obj, *, chunks=None, rank=0) -> dict[str, int]:
    chunks = chunks or _chunks(1)
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(chunks)

    counters = {"barrier": 0, "ipc_collect": 0}

    def counting_barrier(*args, **kwargs):
        counters["barrier"] += 1

    def counting_ipc_collect(*args, **kwargs):
        counters["ipc_collect"] += 1

    with patch("torch.distributed.get_rank", return_value=rank), patch(
        "torch.distributed.barrier", side_effect=counting_barrier
    ), patch("torch.cuda.ipc_collect", side_effect=counting_ipc_collect):
        obj.update_weights()
    return counters


@pytest.mark.unit
def test_colocated_lifecycle_uses_native_weight_transfer_session(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj.rollout_engines = [engine]

    with patch(f"{MODULE_PATH}._send_to_colocated_engine") as send_to_colocated:
        counters = _run_update(obj, chunks=_chunks(2))

    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1
    assert len(engine.release_memory_occupation.calls) == 0
    assert len(engine.resume_memory_occupation.calls) == 0
    assert len(engine.start_weight_update.calls) == 1
    assert engine.start_weight_update.calls[0].kwargs == {"is_checkpoint_format": True}
    assert len(engine.finish_weight_update.calls) == 1
    assert engine.finish_weight_update.calls[0].kwargs == {}
    assert len(engine.continue_generation.calls) == 1

    assert send_to_colocated.call_count == 2
    assert counters["ipc_collect"] == 3
    assert counters["barrier"] >= 4


@dataclass
class _FakeUpdateInfo:
    names: list[str]
    dtype_names: list[str]
    shapes: list[list[int]]
    ipc_handles: list[dict[str, tuple]]
    packed: bool = False


def _install_fake_npu_ipc_modules(monkeypatch, calls: list[dict]):
    root_mod = types.ModuleType("vllm_ascend")
    distributed_mod = types.ModuleType("vllm_ascend.distributed")
    weight_transfer_mod = types.ModuleType("vllm_ascend.distributed.weight_transfer")
    ipc_mod = types.ModuleType("vllm_ascend.distributed.weight_transfer.npu_ipc_engine")

    @dataclass
    class FakeNPUIPCTrainerSendWeightsArgs:
        send_mode: object
        packed: bool = False

    class FakeNPUIPCWeightTransferEngine:
        @staticmethod
        def trainer_send_weights(iterator, trainer_args):
            calls.append({"items": list(iterator), "trainer_args": trainer_args})
            trainer_args.send_mode(
                _FakeUpdateInfo(
                    names=["layer.weight"],
                    dtype_names=["float32"],
                    shapes=[[2, 2]],
                    ipc_handles=[{"uuid": ("handle", ())}],
                )
            )

    ipc_mod.NPUIPCTrainerSendWeightsArgs = FakeNPUIPCTrainerSendWeightsArgs
    ipc_mod.NPUIPCWeightTransferEngine = FakeNPUIPCWeightTransferEngine
    weight_transfer_mod.npu_ipc_engine = ipc_mod
    distributed_mod.weight_transfer = weight_transfer_mod
    root_mod.distributed = distributed_mod
    monkeypatch.setitem(sys.modules, "vllm_ascend", root_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed", distributed_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.weight_transfer", weight_transfer_mod)
    monkeypatch.setitem(sys.modules, "vllm_ascend.distributed.weight_transfer.npu_ipc_engine", ipc_mod)


@pytest.mark.unit
def test_send_to_colocated_engine_uses_native_npu_ipc_engine(upw_vllm, monkeypatch):
    engine = RecordingVLLMEngine()
    calls: list[dict] = []
    _install_fake_npu_ipc_modules(monkeypatch, calls)

    tensors = [("layer.weight", torch.zeros(2, 2))]
    with patch(f"{MODULE_PATH}.is_npu", return_value=True):
        upw_vllm._send_to_colocated_engine(tensors, rollout_engines=[engine], weight_version=42)

    assert calls[0]["items"] == tensors
    assert calls[0]["trainer_args"].packed is False
    assert len(engine.update_weights.calls) == 1
    call = engine.update_weights.calls[0]
    assert call.args[0]["update_info"]["names"] == ["layer.weight"]
    assert call.args[0]["update_info"]["packed"] is False
    assert call.kwargs == {"weight_version": "42"}


@pytest.mark.unit
def test_send_hf_params_returns_only_distributed_refs(upw_vllm):
    obj = _make_instance(upw_vllm)
    obj.rollout_engines = [RecordingVLLMEngine()]
    obj.distributed_rollout_engines = [RecordingVLLMEngine()]
    obj.use_distribute = True
    obj._is_distributed_src_rank = True
    obj._model_update_groups = "groups"
    tensors = _chunks(1)[0]

    with patch(f"{MODULE_PATH}._send_to_colocated_engine") as send_to_colocated, patch(
        f"{MODULE_PATH}.update_weights_from_distributed", return_value=["distributed-ref"]
    ) as send_distributed:
        refs = obj._send_hf_params(tensors)

    send_to_colocated.assert_called_once_with(
        tensors,
        rollout_engines=obj.rollout_engines,
        weight_version=obj.weight_version,
    )
    send_distributed.assert_called_once()
    assert refs == ["distributed-ref"]


@pytest.mark.unit
def test_connect_keeps_colocated_engines_and_initializes_once(upw_vllm):
    engines = [RecordingVLLMEngine() for _ in range(2)]
    obj = _make_instance(
        upw_vllm,
        args=_default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=2),
    )

    with patch("torch.distributed.get_rank", return_value=0):
        obj.connect_rollout_engines(
            engines,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )

    assert obj.rollout_engines == engines
    assert obj.distributed_rollout_engines == []
    assert obj.use_distribute is False
    assert obj._ipc_initialized is True
    assert len(engines[0].init_weight_transfer_engine.calls) == 1
    assert len(engines[1].init_weight_transfer_engine.calls) == 1

    engines2 = [RecordingVLLMEngine() for _ in range(2)]
    with patch("torch.distributed.get_rank", return_value=0):
        obj.connect_rollout_engines(
            engines2,
            rollout_engine_lock=MagicMock(),
            engine_gpu_counts=[2, 2],
            engine_gpu_offsets=[0, 2],
        )

    assert len(engines2[0].init_weight_transfer_engine.calls) == 0
    assert len(engines2[1].init_weight_transfer_engine.calls) == 0
