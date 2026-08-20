"""CPU unit tests for the vLLM trainer-side weight-transfer adapter."""

from __future__ import annotations

import importlib
import sys
import types
from dataclasses import dataclass, field
from pathlib import Path

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import pytest
import torch

from vime.utils.types import ParamInfo

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_distributed"
COMMON_MODULE = "vime.backends.megatron_utils.update_weight.common"
DIRECT_MODULE = "vime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
CONVERTER_MODULE = "vime.backends.megatron_utils.megatron_to_hf"


@pytest.fixture(scope="module")
def update_module():
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
        MODULE_PATH,
    )
    saved = _unit_stubs.save_sys_modules(module_names)
    for name in module_names:
        sys.modules.pop(name, None)
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()
    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved)


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
    start_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    start_draft_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    update_weights: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    finish_weight_update: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    pause_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    flush_cache: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    continue_generation: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)
    post_process_weights: RecordingRemoteMethod = field(default_factory=RecordingRemoteMethod)


class RecordingTrainer:
    def __init__(self, client, *, fail=False):
        self.client = client
        self.fail = fail
        self.draft_states = []
        self.shutdown_calls = 0

    def send_weights(self):
        self.draft_states.append(self.client.draft)
        self.client.start_weight_update()
        if self.fail:
            raise RuntimeError("transfer failed")
        self.client.update_weights({"names": []})
        self.client.finish_weight_update()

    def shutdown(self):
        self.shutdown_calls += 1


@pytest.mark.unit
def test_ray_client_fans_out_and_offsets_nccl_ranks(update_module):
    engines = [RecordingEngine(), RecordingEngine()]
    client = update_module.VimeRayWeightSyncClient(engines, lambda: 7, [2, 4])

    client.init_weight_transfer_engine({"rank_offset": 1, "world_size": 7})
    client.start_weight_update()
    client.update_weights({"names": ["weight"]})
    client.finish_weight_update()

    assert engines[0].init_weight_transfer_engine.calls[0].args[0]["init_info"]["rank_offset"] == 1
    assert engines[1].init_weight_transfer_engine.calls[0].args[0]["init_info"]["rank_offset"] == 3
    assert len(engines[0].start_weight_update.calls) == 1
    assert len(engines[1].update_weights.calls) == 1
    assert engines[0].finish_weight_update.calls[0].kwargs == {"weight_version": "7"}


@pytest.mark.unit
def test_ray_client_selects_draft_lifecycle(update_module):
    engine = RecordingEngine()
    client = update_module.VimeRayWeightSyncClient([engine], lambda: 1)
    client.draft = True

    client.start_weight_update()

    assert engine.start_weight_update.calls == []
    assert len(engine.start_draft_weight_update.calls) == 1


@pytest.mark.unit
def test_weight_source_caches_metadata_and_reiterates(update_module, monkeypatch):
    class ParamMeta:
        def __init__(self, name, dtype, shape):
            self.name = name
            self.dtype = dtype
            self.shape = shape

    base_module = types.ModuleType("vllm.distributed.weight_transfer.base")
    base_module.ParamMeta = ParamMeta
    monkeypatch.setitem(sys.modules, "vllm.distributed.weight_transfer.base", base_module)

    calls = []

    class Iterator:
        def get_hf_weight_chunks(self, weights):
            calls.append(weights)
            yield [("a", torch.zeros(2)), ("b", torch.ones(3))]

    source = update_module.HfWeightSource(Iterator(), lambda: {"version": len(calls)})

    assert [item.name for item in source.metadata()] == ["a", "b"]
    assert [item.name for item in source.metadata()] == ["a", "b"]
    assert [name for name, _ in source] == ["a", "b"]
    assert len(calls) == 2


@pytest.mark.unit
def test_nccl_trainer_uses_single_packed_buffer(update_module, monkeypatch):
    adapter = sys.modules[update_module.create_nccl_trainer.__module__]
    created = []

    class NCCLTrainerInitInfo:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Factory:
        @staticmethod
        def trainer_init(init_info, *, client, source):
            created.append((init_info, client, source))
            return "trainer"

    factory_module = types.ModuleType("vllm.distributed.weight_transfer.factory")
    factory_module.WeightTransferTrainerFactory = Factory
    nccl_module = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    nccl_module.NCCLTrainerInitInfo = NCCLTrainerInitInfo
    monkeypatch.setitem(sys.modules, factory_module.__name__, factory_module)
    monkeypatch.setitem(sys.modules, nccl_module.__name__, nccl_module)
    monkeypatch.setattr(adapter.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(adapter.dist, "broadcast_object_list", lambda *args, **kwargs: None)
    monkeypatch.setattr(adapter, "get_gloo_group", lambda: None)
    monkeypatch.setattr(
        sys.modules["ray"],
        "_private",
        types.SimpleNamespace(services=types.SimpleNamespace(get_node_ip_address=lambda: "127.0.0.1")),
        raising=False,
    )

    assert adapter.create_nccl_trainer("client", "source", [2, 2]) == "trainer"
    init_info, client, source = created[0]
    assert init_info.packed_num_buffers == 1
    assert init_info.world_size == 5
    assert client == "client"
    assert source == "source"


@pytest.mark.unit
def test_connect_replaces_existing_trainer(update_module, monkeypatch):
    updater = object.__new__(update_module.UpdateWeightFromDistributed)
    updater.args = types.SimpleNamespace(rollout_num_gpus_per_engine=2)
    updater.weight_version = 0
    updater._source = object()
    old_trainer = RecordingTrainer(object())
    updater._trainer = old_trainer
    engine = RecordingEngine()
    created = []

    def create_trainer(client, source, gpu_counts):
        created.append((client, source, gpu_counts))
        return RecordingTrainer(client)

    monkeypatch.setattr(update_module, "create_nccl_trainer", create_trainer)
    updater.connect_rollout_engines([engine], object(), engine_gpu_counts=[4])

    assert old_trainer.shutdown_calls == 1
    assert created[0][1:] == (updater._source, [4])
    assert updater._trainer is not old_trainer


def _updater_for_transfer(update_module, *, mtp=False, fail=False):
    updater = object.__new__(update_module.UpdateWeightFromDistributed)
    updater.args = types.SimpleNamespace(
        enable_mtp_training=mtp,
        vllm_speculative_config={"method": "mtp"} if mtp else None,
    )
    updater.quantization_config = None
    updater.weight_version = 0
    updater.update_weight_metrics = {}
    updater.rollout_engines = [RecordingEngine()]
    client = update_module.VimeRayWeightSyncClient(updater.rollout_engines, lambda: updater.weight_version)
    updater._trainer = RecordingTrainer(client, fail=fail)
    return updater


@pytest.mark.unit
def test_update_uses_native_main_and_draft_lifecycles(update_module, monkeypatch):
    updater = _updater_for_transfer(update_module, mtp=True)
    monkeypatch.setattr(update_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_module.dist, "barrier", lambda *args, **kwargs: None)

    updater.update_weights()

    engine = updater.rollout_engines[0]
    assert updater._trainer.draft_states == [False, True]
    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1
    assert len(engine.start_weight_update.calls) == 1
    assert len(engine.start_draft_weight_update.calls) == 1
    assert len(engine.finish_weight_update.calls) == 2
    assert len(engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_failed_transfer_does_not_resume_generation(update_module, monkeypatch):
    updater = _updater_for_transfer(update_module, fail=True)
    monkeypatch.setattr(update_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(update_module.dist, "barrier", lambda *args, **kwargs: None)

    with pytest.raises(RuntimeError, match="transfer failed"):
        updater.update_weights()

    assert updater.rollout_engines[0].continue_generation.calls == []


@pytest.fixture
def weight_modules():
    module_names = (
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
        CONVERTER_MODULE,
        COMMON_MODULE,
        DIRECT_MODULE,
    )
    saved = _unit_stubs.save_sys_modules(module_names)
    for name in module_names:
        sys.modules.pop(name, None)
    _unit_stubs.install_megatron_mpu_stub()
    converter = types.ModuleType(CONVERTER_MODULE)
    converter.convert_to_hf = lambda *args, **kwargs: []
    sys.modules[CONVERTER_MODULE] = converter
    try:
        yield importlib.import_module(COMMON_MODULE), importlib.import_module(DIRECT_MODULE)
    finally:
        _unit_stubs.restore_sys_modules(saved)


class Handle:
    def wait(self):
        pass


def _param_info(name: str, param: torch.Tensor, src_rank: int = 0) -> ParamInfo:
    return ParamInfo(name, param.dtype, param.shape, {}, param.nbytes, src_rank)


def _tp_param(values, partition_dim: int) -> torch.nn.Parameter:
    parameter = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    parameter.tensor_model_parallel = True
    parameter.partition_dim = partition_dim
    parameter.partition_stride = 1
    return parameter


@pytest.mark.unit
def test_single_tp_returns_parameter_without_collective(monkeypatch, weight_modules):
    common, _ = weight_modules
    parameter = _tp_param([[1.0, 1.0]], partition_dim=0)
    calls = []
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common.dist, "all_gather", lambda *args, **kwargs: calls.append(args))

    gathered = common.all_gather_param("decoder.weight", parameter)

    assert gathered.data_ptr() == parameter.data_ptr()
    assert calls == []


@pytest.mark.unit
def test_all_gather_params_async_restores_layouts(monkeypatch, weight_modules):
    common, _ = weight_modules
    direct = torch.nn.Parameter(torch.tensor([99.0]))
    direct.tensor_model_parallel = False
    column = _tp_param([[1.0, 2.0], [3.0, 4.0]], partition_dim=0)
    glu = _tp_param([[1.0], [2.0], [10.0], [20.0]], partition_dim=0)
    glu.partition_stride = 2
    row = _tp_param([[1.0, 2.0], [3.0, 4.0]], partition_dim=0)
    entries = [
        (_param_info("dense", direct), direct),
        (_param_info("linear.weight", column), column),
        (_param_info("linear_fc1.weight", glu), glu),
        (_param_info("linear_fc2.weight", row), row),
    ]
    remote_parts = iter(
        [
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
            torch.tensor([[3.0], [4.0], [30.0], [40.0]]),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]),
        ]
    )
    calls = []

    def all_gather(partitions, local, group, async_op):
        calls.append((group, async_op))
        partitions[0].copy_(local)
        partitions[1].copy_(next(remote_parts))
        return Handle()

    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(common.dist, "all_gather", all_gather)

    gathered = common.all_gather_params_async(entries)

    assert calls == [("tp", True)] * 3
    assert gathered[0].data_ptr() == direct.data_ptr()
    assert torch.equal(gathered[1], torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]]))
    assert torch.equal(gathered[2], torch.tensor([[1.0], [2.0], [3.0], [4.0], [10.0], [20.0], [30.0], [40.0]]))
    assert torch.equal(gathered[3], torch.tensor([[1.0, 2.0, 5.0, 6.0], [3.0, 4.0, 7.0, 8.0]]))


@pytest.mark.unit
def test_broadcast_expert_params_coalesces_by_source(monkeypatch, weight_modules):
    _, direct = weight_modules
    params = [torch.tensor([float(index)]) for index in range(4)]
    infos = [
        _param_info("layers.0.experts.0.weight", params[0], src_rank=4),
        _param_info("layers.0.experts.1.weight", params[1], src_rank=5),
        _param_info("layers.0.dense.weight", params[2], src_rank=4),
        _param_info("layers.0.experts.2.weight", params[3], src_rank=9),
    ]
    calls = []
    monkeypatch.setattr(direct.mpu, "get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(
        direct.dist,
        "_broadcast_coalesced",
        lambda group, tensors, buffer_size, src: calls.append((group, tensors, buffer_size, src)),
    )

    direct._broadcast_expert_params(infos, params, 1024, {4: 0, 5: 1, 9: 0})

    assert calls == [
        ("ep", [params[0], params[3]], 1024, 0),
        ("ep", [params[1]], 1024, 1),
    ]


@pytest.mark.unit
def test_ep_broadcast_source_map_tracks_pp_groups(monkeypatch, weight_modules):
    _, direct = weight_modules
    monkeypatch.setattr(direct.mpu, "get_expert_model_parallel_group", lambda: "ep")
    monkeypatch.setattr(direct.mpu, "get_expert_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(direct.mpu, "get_pipeline_model_parallel_group", lambda: "pp")
    monkeypatch.setattr(direct.dist, "get_process_group_ranks", lambda group: [0, 2] if group == "pp" else [0, 1])

    def all_gather_object(output, local_pp_group, group):
        assert local_pp_group == [0, 2]
        assert group == "ep"
        output[:] = [[0, 2], [1, 3]]

    monkeypatch.setattr(direct.dist, "all_gather_object", all_gather_object)

    assert direct._get_ep_broadcast_src_rank_map() == {0: 0, 2: 0, 1: 1, 3: 1}
