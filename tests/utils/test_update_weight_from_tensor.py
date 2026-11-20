"""CPU unit tests for native and rank-local colocated weight transfer."""

from __future__ import annotations

import importlib
import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from unittest.mock import MagicMock

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import torch

from vime.utils.types import ParamInfo

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_tensor"
COMMON_MODULE = "vime.backends.megatron_utils.update_weight.common"
HF_DIRECT_MODULE = "vime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
DISTRIBUTED_MODULE = "vime.backends.megatron_utils.update_weight.update_weight_from_distributed"


@pytest.fixture(scope="module")
def update_module():
    import torch.distributed as torch_dist

    module_names = (
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
        "ray",
        "ray.actor",
        "vime.utils.distributed_utils",
        COMMON_MODULE,
        HF_DIRECT_MODULE,
        DISTRIBUTED_MODULE,
        MODULE_PATH,
    )
    saved_modules = _unit_stubs.save_sys_modules(module_names)
    dist_attributes = ("get_rank", "get_world_size", "new_group", "barrier", "gather_object")
    saved_dist = {name: getattr(torch_dist, name) for name in dist_attributes}
    for name in module_names:
        sys.modules.pop(name, None)

    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()

    iterator = MagicMock()
    iterator.megatron_local_param_info_buckets = None
    hf_direct = types.ModuleType(HF_DIRECT_MODULE)
    hf_direct.HfWeightIteratorDirect = MagicMock(return_value=iterator)
    sys.modules[HF_DIRECT_MODULE] = hf_direct

    distributed = importlib.import_module(DISTRIBUTED_MODULE)
    distributed.post_process_weights = MagicMock()

    torch_dist.get_rank = MagicMock(return_value=0)
    torch_dist.get_world_size = MagicMock(return_value=1)
    torch_dist.new_group = MagicMock(return_value="slot-group")
    torch_dist.barrier = MagicMock()
    torch_dist.gather_object = MagicMock()

    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved_modules)
        for name, value in saved_dist.items():
            setattr(torch_dist, name, value)


@dataclass
class RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self):
        self.calls: list[RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(RemoteCall(args, kwargs))
        return "ref"


@dataclass
class RecordingEngine:
    init_weight_transfer_engine: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    destroy_weights_update_group: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_draft_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    finish_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    pause_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    flush_cache: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    continue_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)


class RecordingTrainer:
    def __init__(self, client, *, source=None, fail=False, backend="nccl", is_sender=True):
        self.client = client
        self.init_info_cls = types.SimpleNamespace(backend=backend)
        self.is_sender = is_sender
        self.source = source
        self.fail = fail
        self.draft_states = []
        self.source_draft_states = []
        self.shutdown_calls = 0

    def send_weights(self):
        self.draft_states.append(self.client.draft)
        self.source_draft_states.append(getattr(self.source, "draft", False))
        self.client.start_weight_update()
        if self.fail:
            raise RuntimeError("transfer failed")
        self.client.update_weights({"names": []})
        self.client.finish_weight_update()

    def shutdown(self):
        self.shutdown_calls += 1


def _args(**overrides):
    values = {
        "actor_num_nodes": 1,
        "actor_num_gpus_per_node": 2,
        "rollout_num_gpus_per_engine": 2,
        "update_weight_buffer_size": 1024,
        "flush_cache_interval": 1,
        "enable_mtp_training": False,
        "dspark_enabled": False,
        "dspark_pretrained_model": None,
        "vllm_speculative_config": None,
    }
    values.update(overrides)
    return Namespace(**values)


def _updater(update_module, **overrides):
    updater = object.__new__(update_module.UpdateWeightFromTensor)
    updater.args = _args(**overrides)
    updater.model = []
    updater.weights_getter = lambda: {}
    updater.rank = 0
    updater.model_name = "test"
    updater.quantization_config = None
    updater.weight_version = 0
    updater.update_weight_metrics = {}
    updater._hf_weight_iterator = MagicMock()
    updater._full_param_info_buckets = None
    updater._non_expert_param_info_buckets = None
    updater._source = types.SimpleNamespace(draft=False)
    updater._ipc_gather_group = None
    updater._ipc_gather_src = None
    updater._ipc_engine = None
    updater._expert_transfer_plan = []
    updater._native_trainers = []
    updater._all_rollout_engines = []
    updater.rollout_engines = []
    return updater


def _install_ipc_trainer_stubs(monkeypatch, created, platform_name="cuda"):
    factory_module = types.ModuleType("vllm.distributed.weight_transfer.factory")
    if platform_name == "npu":
        from vime.platforms import npu

        monkeypatch.setattr(npu, "_ensure_torch_npu", lambda: None)
        plugins = types.ModuleType("vllm.plugins")
        plugins.load_general_plugins = lambda: None
        monkeypatch.setitem(sys.modules, plugins.__name__, plugins)
        ipc_module = types.ModuleType("vllm_ascend.distributed.weight_transfer.npu_ipc_engine")
        ipc_backend = "npu_ipc"
    else:
        ipc_module = types.ModuleType("vllm.distributed.weight_transfer.ipc_engine")
        ipc_backend = "ipc"

    class IPCTrainerInitInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Factory:
        @staticmethod
        def trainer_init(init_info, *, client, source):
            created.append((init_info, client, source))
            return RecordingTrainer(client, backend=ipc_backend)

    factory_module.WeightTransferTrainerFactory = Factory
    ipc_module.IPCTrainerInitInfo = IPCTrainerInitInfo
    ipc_module.NPUIPCTrainerInitInfo = IPCTrainerInitInfo
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setitem(sys.modules, ipc_module.__name__, ipc_module)


@pytest.mark.unit
@pytest.mark.parametrize(
    "platform_name,backend,is_sender",
    [
        ("cuda", "nccl", True),
        ("cuda", "nccl", False),
        ("cuda", "ipc", True),
        ("npu", "hccl", True),
        ("npu", "hccl", False),
        ("npu", "npu_ipc", True),
    ],
)
def test_connect_uses_native_ipc_and_distributed_trainers(
    update_module, monkeypatch, platform_name, backend, is_sender
):
    from vime.platforms import get_platform

    monkeypatch.setattr(update_module, "current_platform", lambda: get_platform(platform_name))
    updater = _updater(update_module)
    old_trainer = RecordingTrainer(object(), backend=backend, is_sender=is_sender)
    updater._native_trainers = [old_trainer]
    engines = [RecordingEngine(), RecordingEngine()]
    ipc_created = []
    distributed_created = []
    _install_ipc_trainer_stubs(monkeypatch, ipc_created, platform_name)
    monkeypatch.setattr(update_module, "configure_expert_routing", lambda **kwargs: (None, []))

    def create_distributed(client, source, gpu_counts):
        distributed_created.append((client, source, gpu_counts))
        return RecordingTrainer(client, backend="hccl" if platform_name == "npu" else "nccl", is_sender=is_sender)

    monkeypatch.setattr(update_module, "connect_rollout_engines_from_distributed", create_distributed)
    updater.connect_rollout_engines(
        engines,
        object(),
        engine_gpu_counts=[2, 2],
        engine_gpu_offsets=[0, 2],
    )

    assert old_trainer.shutdown_calls == 1
    assert engines[0].destroy_weights_update_group.calls == []
    expected_remote_releases = int(backend in {"nccl", "hccl"} and is_sender)
    assert len(engines[1].destroy_weights_update_group.calls) == expected_remote_releases
    assert len(updater._native_trainers) == 2
    assert ipc_created[0][0].packed is True
    assert ipc_created[0][2] is updater._source
    assert distributed_created[0][1:] == (updater._source, [2])

    for _ in range(2):
        previous = list(updater._native_trainers)
        updater.connect_rollout_engines(engines, object(), engine_gpu_counts=[2, 2], engine_gpu_offsets=[0, 2])
        expected_remote_releases += int(is_sender)
        assert all(trainer.shutdown_calls == 1 for trainer in previous)
        assert all(trainer not in previous for trainer in updater._native_trainers)
        assert engines[0].destroy_weights_update_group.calls == []
        assert len(engines[1].destroy_weights_update_group.calls) == expected_remote_releases
    assert old_trainer.shutdown_calls == 1
    assert len(ipc_created) == len(distributed_created) == 3


@pytest.mark.unit
@pytest.mark.parametrize("platform_name,backend", [("cuda", "nccl"), ("npu", "hccl")])
@pytest.mark.parametrize("failure", [None, "local_shutdown", "remote_error", "remote_timeout"])
def test_reconnect_waits_for_distributed_release(update_module, monkeypatch, platform_name, backend, failure):
    from vime.platforms import get_platform

    monkeypatch.setattr(update_module, "current_platform", lambda: get_platform(platform_name))
    monkeypatch.setattr(update_module, "configure_expert_routing", lambda **kwargs: (None, []))
    updater = _updater(update_module)
    trainer = RecordingTrainer(object(), backend=backend)
    updater._native_trainers = [trainer]
    engines = [RecordingEngine(), RecordingEngine()]
    events = []
    refs = [object(), object()]

    def submit_release(index, group_name):
        assert group_name is None
        events.append(("remote", index))
        return refs[index]

    for index, engine in enumerate(engines):
        monkeypatch.setattr(
            engine.destroy_weights_update_group,
            "remote",
            lambda group_name, index=index: submit_release(index, group_name),
        )

    def shutdown():
        events.append("local")
        if failure == "local_shutdown":
            raise RuntimeError("local release failed")

    def wait(pending):
        assert pending == refs
        events.append("wait")
        if failure == "remote_error":
            raise RuntimeError("remote release failed")
        if failure == "remote_timeout":
            raise TimeoutError("remote release timed out")

    def connect(client, source, gpu_counts):
        events.append("connect")
        assert gpu_counts == [2, 2]
        return RecordingTrainer(client, backend=backend)

    monkeypatch.setattr(trainer, "shutdown", shutdown)
    disconnect = update_module.disconnect_rollout_engines_from_distributed
    monkeypatch.setattr(sys.modules[disconnect.__module__].ray, "get", wait)
    monkeypatch.setattr(update_module, "connect_rollout_engines_from_distributed", connect)

    def reconnect():
        updater.connect_rollout_engines(engines, object(), engine_gpu_counts=[2, 2], engine_gpu_offsets=[2, 4])

    if failure:
        error = TimeoutError if failure == "remote_timeout" else RuntimeError
        with pytest.raises(error, match="release"):
            reconnect()
        assert updater._native_trainers == [trainer]
    else:
        reconnect()
        assert updater._native_trainers[0] is not trainer

    expected = [("remote", 0), ("remote", 1), "local"]
    if failure != "local_shutdown":
        expected.append("wait")
    if failure is None:
        expected.append("connect")
    assert events == expected


@pytest.mark.unit
def test_connect_keeps_rank_local_expert_fallback(update_module, monkeypatch):
    updater = _updater(update_module)
    engine = RecordingEngine()
    plan = [object()]
    monkeypatch.setattr(update_module, "configure_expert_routing", lambda **kwargs: ([], plan))
    monkeypatch.setattr(update_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_module.dist, "new_group", lambda **kwargs: "slot-group")

    updater.connect_rollout_engines(
        [engine],
        object(),
        engine_gpu_counts=[2],
        engine_gpu_offsets=[0],
    )
    updater.connect_rollout_engines(
        [engine],
        object(),
        engine_gpu_counts=[2],
        engine_gpu_offsets=[0],
    )

    assert updater._native_trainers == []
    assert updater._ipc_engine is engine
    assert updater._ipc_gather_src == 0
    assert len(engine.init_weight_transfer_engine.calls) == 2


@pytest.mark.unit
def test_native_update_runs_main_and_draft_lifecycles(update_module, monkeypatch):
    updater = _updater(
        update_module,
        enable_mtp_training=True,
        vllm_speculative_config={"method": "mtp"},
    )
    engine = RecordingEngine()
    updater._all_rollout_engines = [engine]
    client = update_module.VimeRayWeightSyncClient([engine], lambda: updater.weight_version)
    trainer = RecordingTrainer(client)
    updater._native_trainers = [trainer]
    monkeypatch.setattr(update_module.dist, "barrier", lambda *args, **kwargs: None)

    updater.update_weights()

    assert trainer.draft_states == [False, True]
    assert len(engine.pause_generation.calls) == 1
    assert len(engine.start_weight_update.calls) == 1
    assert len(engine.start_draft_weight_update.calls) == 1
    assert len(engine.finish_weight_update.calls) == 2
    assert len(engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_native_dspark_update_uses_draft_source_and_lifecycle(update_module, monkeypatch):
    updater = _updater(
        update_module,
        dspark_enabled=True,
        vllm_speculative_config={"method": "dspark"},
    )
    engine = RecordingEngine()
    updater._all_rollout_engines = [engine]
    client = update_module.VimeRayWeightSyncClient([engine], lambda: updater.weight_version)
    trainer = RecordingTrainer(client, source=updater._source)
    updater._native_trainers = [trainer]
    monkeypatch.setattr(update_module.dist, "barrier", lambda *args, **kwargs: None)

    updater.update_weights()

    assert trainer.draft_states == [False, True]
    assert trainer.source_draft_states == [False, True]
    assert updater._source.draft is False
    assert len(engine.start_draft_weight_update.calls) == 1


@pytest.mark.unit
def test_rank_local_experts_keep_dspark_draft_update(update_module, monkeypatch):
    updater = _updater(update_module, dspark_enabled=True)
    engine = RecordingEngine()
    created = []
    _install_ipc_trainer_stubs(monkeypatch, created)
    monkeypatch.setattr(update_module, "configure_expert_routing", lambda **kwargs: ([], [object()]))
    updater.connect_rollout_engines([engine], object(), engine_gpu_counts=[2], engine_gpu_offsets=[0])

    assert len(updater._native_trainers) == 1
    trainer = updater._native_trainers[0]
    trainer.source = updater._source
    updater._update_rollout_weights = MagicMock()
    updater.update_weights()

    updater._update_rollout_weights.assert_called_once_with({}, draft=False)
    assert trainer.draft_states == [True]
    assert trainer.source_draft_states == [True]
    assert updater._source.draft is False
    assert len(engine.start_draft_weight_update.calls) == 1
    assert len(engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_failed_native_update_does_not_resume_generation(update_module, monkeypatch):
    updater = _updater(update_module)
    engine = RecordingEngine()
    updater._all_rollout_engines = [engine]
    client = update_module.VimeRayWeightSyncClient([engine], lambda: updater.weight_version)
    updater._native_trainers = [RecordingTrainer(client, fail=True)]
    monkeypatch.setattr(update_module.dist, "barrier", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="transfer failed"):
        updater.update_weights()

    assert engine.continue_generation.calls == []


@pytest.mark.unit
def test_native_ipc_buffer_covers_largest_reconstructed_tensor(update_module, monkeypatch):
    dense = ParamInfo("dense", torch.float16, (8,), {}, 16, 0)
    expert = ParamInfo("layers.0.experts.0.weight", torch.float16, (8,), {}, 16, 0)
    monkeypatch.setattr(update_module.mpu, "get_tensor_model_parallel_world_size", lambda: 4)
    monkeypatch.setattr(update_module.mpu, "get_expert_tensor_parallel_world_size", lambda: 8)

    size = update_module._native_ipc_buffer_size(_args(update_weight_buffer_size=32), [[dense, expert]])

    assert size == 128


@pytest.mark.unit
def test_build_packed_ipc_update_info_uses_vllm_wire_format(update_module, monkeypatch):
    packed_module = types.ModuleType("vllm.distributed.weight_transfer.packed_tensor")
    packed_tensor = torch.zeros(12, dtype=torch.uint8)
    packed_module.pack_tensors = lambda *args, **kwargs: types.SimpleNamespace(
        packed_tensor=packed_tensor,
        names=["a", "b"],
        dtypes=[torch.float16, torch.float32],
        shapes=[[2], [2]],
        tensor_sizes=[4, 8],
    )
    monkeypatch.setitem(sys.modules, packed_module.__name__, packed_module)
    monkeypatch.setattr(torch.multiprocessing.reductions, "reduce_tensor", lambda tensor: (None, ("ipc",)))
    monkeypatch.setattr(update_module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(
        update_module.torch.cuda,
        "get_device_properties",
        lambda device: types.SimpleNamespace(uuid="GPU-0"),
    )

    update_info, weight_ref = update_module._build_packed_ipc_update_info(
        [("a", torch.zeros(2, dtype=torch.float16)), ("b", torch.zeros(2))]
    )

    assert update_info == {
        "names": ["a", "b"],
        "dtype_names": ["float16", "float32"],
        "shapes": [[2], [2]],
        "tensor_sizes": [4, 8],
        "ipc_handles": {"GPU-0": ("ipc",)},
    }
    assert weight_ref is packed_tensor


@pytest.mark.unit
def test_single_worker_rank_local_payload(update_module, monkeypatch):
    engine = RecordingEngine()
    local_info = {"names": ["a"], "ipc_handles": {"GPU-0": ("ipc",)}}
    monkeypatch.setattr(update_module, "_build_packed_ipc_update_info", lambda tensors: (local_info, "weight-ref"))
    monkeypatch.setattr(update_module.dist, "get_world_size", lambda group: 1)

    refs, weight_ref = update_module._send_to_colocated_engine(
        [("a", torch.zeros(1))],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group="slot-group",
    )

    assert refs == ["ref"]
    assert weight_ref == "weight-ref"
    assert engine.update_weights.calls[0].args == (local_info,)


@pytest.mark.unit
def test_multi_worker_rank_local_payload(update_module, monkeypatch):
    engine = RecordingEngine()
    local_info = {"names": ["a"], "ipc_handles": {"GPU-0": ("ipc-0",)}}
    remote_info = {"names": ["b"], "ipc_handles": {"GPU-1": ("ipc-1",)}}
    monkeypatch.setattr(update_module, "_build_packed_ipc_update_info", lambda tensors: (local_info, "weight-ref"))
    monkeypatch.setattr(update_module.dist, "get_world_size", lambda group: 2)
    monkeypatch.setattr(update_module.dist, "get_rank", lambda: 0)

    def gather_object(local, object_gather_list, dst, group):
        object_gather_list[:] = [local, remote_info]

    monkeypatch.setattr(update_module.dist, "gather_object", gather_object)

    refs, _ = update_module._send_to_colocated_engine(
        [("a", torch.zeros(1))],
        ipc_engine=engine,
        ipc_gather_src=0,
        ipc_gather_group="slot-group",
    )

    assert refs == ["ref"]
    assert engine.update_weights.calls[0].args == ([local_info, remote_info],)
