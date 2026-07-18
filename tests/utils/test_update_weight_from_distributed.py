"""CPU unit tests for ``vime.backends.megatron_utils.update_weight.update_weight_from_distributed``."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
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

MODULE_PATH = "vime.backends.megatron_utils.update_weight.update_weight_from_distributed"
COMMON_MODULE = "vime.backends.megatron_utils.update_weight.common"
DIRECT_MODULE = "vime.backends.megatron_utils.update_weight.hf_weight_iterator_direct"
CONVERTER_MODULE = "vime.backends.megatron_utils.megatron_to_hf"

NUM_GPUS = 0


# Modules stubbed by _install_stubs(). These are installed ONLY for the duration of this
# module's tests (inside the fixture) and restored on teardown. Installing them at import
# time (top level) left a fake ``vllm`` (with no ``.engine``) in sys.modules, which broke
# COLLECTION of sibling test modules (e.g. test_vllm_engine.py -> ModuleNotFoundError
# 'vllm.engine'). pytest imports all test modules in one process before running fixtures,
# so the stub leak must be confined to test runtime, not collection.
_STUBBED_MODULES = (
    "megatron",
    "megatron.core",
    "megatron.core.parallel_state",
    "megatron.core.transformer",
    "megatron.core.transformer.transformer_layer",
    "ray",
    "ray.actor",
    "vime.utils.distributed_utils",
    "vllm",
    "vllm.utils",
    "vllm.utils.deep_gemm",
    "vllm.third_party",
    "vllm.third_party.deep_gemm",
    "vllm.third_party.deep_gemm.utils",
    "vllm.third_party.deep_gemm.utils.layout",
    "vllm.distributed",
    "vllm.distributed.weight_transfer",
    "vllm.distributed.weight_transfer.nccl_engine",
    "triton",
    "triton.language",
)


@pytest.fixture(scope="module")
def upw():
    saved = _unit_stubs.save_sys_modules((*_STUBBED_MODULES, MODULE_PATH))
    # Pop first so _install_stubs()'s setdefault() actually installs the stubs (hermetic),
    # then drop the module-under-test so it re-imports against the stubs.
    for k in _STUBBED_MODULES:
        sys.modules.pop(k, None)
    _install_stubs()
    sys.modules.pop(MODULE_PATH, None)
    try:
        yield importlib.import_module(MODULE_PATH)
    finally:
        _unit_stubs.restore_sys_modules(saved)


def _install_stubs():
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()
    _unit_stubs.install_triton_stub()

    nccl_mod = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")

    class DummyNCCLTrainerSendWeightsArgs:
        def __init__(self, *, group, packed):
            self.group = group
            self.packed = packed

    class DummyNCCLWeightTransferEngine:
        @staticmethod
        def trainer_send_weights(*args, **kwargs):
            return None

        @staticmethod
        def trainer_init(*args, **kwargs):
            return object()

    nccl_mod.NCCLTrainerSendWeightsArgs = DummyNCCLTrainerSendWeightsArgs
    nccl_mod.NCCLWeightTransferEngine = DummyNCCLWeightTransferEngine
    vllm_mod = types.ModuleType("vllm")
    vllm_mod.__path__ = []
    distributed_mod = types.ModuleType("vllm.distributed")
    distributed_mod.__path__ = []
    weight_transfer_mod = types.ModuleType("vllm.distributed.weight_transfer")
    weight_transfer_mod.__path__ = []
    vllm_mod.distributed = distributed_mod
    distributed_mod.weight_transfer = weight_transfer_mod
    weight_transfer_mod.nccl_engine = nccl_mod
    sys.modules.setdefault("vllm", vllm_mod)
    sys.modules.setdefault("vllm.distributed", distributed_mod)
    sys.modules.setdefault("vllm.distributed.weight_transfer", weight_transfer_mod)
    sys.modules.setdefault("vllm.distributed.weight_transfer.nccl_engine", nccl_mod)


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    def __init__(self, return_value: str = "ref"):
        self._return_value = return_value
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return self._return_value


@dataclass
class RecordingEngine:
    update_weights_from_distributed: RecordingRemoteMethod = field(
        default_factory=lambda: RecordingRemoteMethod("ref")
    )
    init_weights_update_group: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("init_ref"))
    destroy_weights_update_group: RecordingRemoteMethod = field(
        default_factory=lambda: RecordingRemoteMethod("destroy_ref")
    )
    start_weight_update: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("start_ref"))
    finish_weight_update: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("finish_ref"))


@dataclass
class RecordingLock:
    acquire: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("acquired"))
    release: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod("released"))


@dataclass
class DummyGroup:
    token: str = "dummy"


def _real_tensors(n: int = 2):
    return [(f"layer.{i}.weight", torch.zeros(2, 2)) for i in range(n)]


def _make_dummy_nccl_engine(*, send_seen: list[dict] | None = None, init_seen: list[dict] | None = None):
    """Build dummy NCCL types; patch on *upw* module (top-level import, not sys.modules)."""

    class DummyNCCLTrainerSendWeightsArgs:
        def __init__(self, *, group, packed):
            self.group = group
            self.packed = packed

    class DummyNCCLWeightTransferEngine:
        @staticmethod
        def trainer_send_weights(iterator, trainer_args):
            if send_seen is not None:
                send_seen.append(
                    {
                        "items": list(iterator),
                        "group": trainer_args.group,
                        "packed": trainer_args.packed,
                    }
                )

        @staticmethod
        def trainer_init(cfg):
            if init_seen is not None:
                init_seen.append(cfg)
            return DummyGroup("group-from-trainer-init")

    return DummyNCCLWeightTransferEngine, DummyNCCLTrainerSendWeightsArgs


def _patch_nccl_on_module(
    monkeypatch, upw, *, send_seen: list[dict] | None = None, init_seen: list[dict] | None = None
):
    dummy_engine, dummy_args = _make_dummy_nccl_engine(send_seen=send_seen, init_seen=init_seen)
    monkeypatch.setattr(upw, "NCCLWeightTransferEngine", dummy_engine)
    monkeypatch.setattr(upw, "NCCLTrainerSendWeightsArgs", dummy_args)


def _patch_trainer_send(monkeypatch, upw, seen: list[dict]) -> None:
    _patch_nccl_on_module(monkeypatch, upw, send_seen=seen)
    monkeypatch.setattr(upw.torch.cuda, "synchronize", lambda: None)


def _make_instance(upw):
    obj = object.__new__(upw.UpdateWeightFromDistributed)
    obj.args = type("Args", (), {"update_weight_buffer_size": 1 << 30})()
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test"
    obj.quantization_config = None
    obj.weight_version = 0
    obj._model_update_groups = DummyGroup()
    obj._hf_weight_iterator = None
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj.rollout_engines = []
    obj.rollout_engine_lock = RecordingLock()
    return obj


@pytest.mark.unit
def test_signature_no_use_vllm(upw):
    sig = inspect.signature(upw.update_weights_from_distributed)
    params = sig.parameters
    assert "use_vllm" not in params
    assert "packed" not in params
    for p in ("group", "weight_version", "rollout_engines", "converted_named_tensors"):
        assert p in params


@pytest.mark.unit
def test_signature_rejects_legacy_use_vllm_call(upw):
    with pytest.raises(TypeError, match="use_vllm"):
        upw.update_weights_from_distributed(
            DummyGroup(),
            1,
            [RecordingEngine()],
            _real_tensors(),
            use_vllm=True,
        )


@pytest.mark.unit
def test_uses_packed_vllm_trainer_send_weights(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    refs = upw.update_weights_from_distributed(group, 7, [engine], tensors)

    assert len(seen) == 1
    sent = seen[0]["items"]
    assert [n for n, _ in sent] == [n for n, _ in tensors]
    assert seen[0]["group"] is group
    assert seen[0]["packed"] is True
    assert refs == ["ref"]


@pytest.mark.unit
def test_no_dist_broadcast_fallback(upw, monkeypatch):
    import torch.distributed as dist

    seen_broadcast = []
    seen_send = []

    def fake_broadcast(*a, **k):
        seen_broadcast.append((a, k))

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)
    _patch_trainer_send(monkeypatch, upw, seen_send)

    group = DummyGroup()
    engine = RecordingEngine()
    upw.update_weights_from_distributed(group, 1, [engine], _real_tensors())

    assert seen_broadcast == []
    assert len(seen_send) == 1


@pytest.mark.unit
def test_remote_kwargs_are_always_packed(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors(n=1)
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed(group, 42, [engine], tensors)

    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is True
    assert len(engine.update_weights_from_distributed.calls) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert "packed" not in kw
    assert "group_name" not in kw
    assert kw["weight_version"] == "42"
    assert kw["names"] == ["layer.0.weight"]
    assert kw["shapes"] == [torch.Size([2, 2])]
    assert kw["dtypes"] == [torch.float32]


@pytest.mark.unit
def test_remote_kwargs_no_use_vllm(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed(group, 1, [engine], _real_tensors())

    assert len(seen_send) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert "use_vllm" not in kw


@pytest.mark.unit
def test_multiple_engines_each_get_call(upw, monkeypatch):
    group = DummyGroup()
    engines = [RecordingEngine() for _ in range(3)]
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed(group, 1, engines, _real_tensors())
    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is True
    for e in engines:
        assert len(e.update_weights_from_distributed.calls) == 1


@pytest.mark.unit
def test_empty_tensor_list_still_dispatches(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    refs = upw.update_weights_from_distributed(group, 1, [engine], [])

    assert refs == ["ref"]
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["names"] == []
    assert kw["shapes"] == []
    assert len(seen_send) == 1
    assert seen_send[0]["items"] == []
    assert seen_send[0]["packed"] is True


@pytest.mark.unit
def test_raw_path_sends_dense_then_expert(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj._hf_weight_iterator = None
    obj._iter_non_expert_chunks = lambda: iter([[("dense.0", torch.zeros(1))], [("dense.1", torch.zeros(1))]])
    obj._iter_expert_chunks = lambda: iter([[("expert.0", torch.zeros(1))]])

    seen: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        upw.UpdateWeightFromDistributed,
        "_update_bucket_weights_from_distributed",
        lambda self, converted_named_tensors, pbar=None: seen.append(
            ([name for name, _ in converted_named_tensors], pbar)
        ),
    )
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")

    upw.UpdateWeightFromDistributed._send_weights(obj, pbar="pbar")

    assert seen == [
        (["dense.0"], "pbar"),
        (["dense.1"], "pbar"),
        (["expert.0"], "pbar"),
    ]


@pytest.mark.unit
def test_source_has_no_packed_mode_switch(upw):
    src = inspect.getsource(upw)
    assert "vllm_weight_sync_packed" not in src
    assert "packed=False" not in src


@pytest.mark.unit
def test_single_ep_converts_without_collective(upw, monkeypatch):
    obj = _make_instance(upw)
    tensors = [("expert.0", torch.ones(2)), ("expert.1", torch.ones(3))]
    collectives = []

    monkeypatch.setattr(upw.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(upw.dist, "all_gather", lambda *args, **kwargs: collectives.append(args))
    monkeypatch.setattr(
        upw,
        "convert_to_hf",
        lambda args, model_name, name, tensor, quantization_config: [(f"hf.{name}", tensor)],
    )

    converted = upw.UpdateWeightFromDistributed._ep_gather_and_convert(obj, tensors)

    assert [name for name, _ in converted] == ["hf.expert.0", "hf.expert.1"]
    assert tensors == []
    assert collectives == []


@pytest.mark.unit
def test_expert_chunks_keep_each_layer_together(upw, monkeypatch):
    obj = _make_instance(upw)
    obj.args.update_weight_buffer_size = 24
    params = [
        ("decoder.layers.0.mlp.experts.linear_fc1.weight0", torch.ones(2)),
        ("decoder.layers.1.mlp.experts.linear_fc1.weight0", torch.ones(2)),
        ("decoder.layers.0.mlp.experts.linear_fc2.weight0", torch.ones(2)),
        ("decoder.layers.1.mlp.experts.linear_fc2.weight0", torch.ones(2)),
    ]

    monkeypatch.setattr(upw, "all_gather_param", lambda name, param: param)
    monkeypatch.setattr(upw.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(
        upw,
        "convert_to_hf",
        lambda args, model_name, name, tensor, quantization_config: [(name, tensor)],
    )

    chunks = list(upw.UpdateWeightFromDistributed._iter_expert_chunks(obj, iter(params)))

    assert [[name for name, _ in chunk] for chunk in chunks] == [
        [
            "decoder.layers.0.mlp.experts.linear_fc1.weight0",
            "decoder.layers.0.mlp.experts.linear_fc2.weight0",
        ],
        [
            "decoder.layers.1.mlp.experts.linear_fc1.weight0",
            "decoder.layers.1.mlp.experts.linear_fc2.weight0",
        ],
    ]


@pytest.mark.unit
def test_bridge_path_listifies_chunks(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._is_pp_src_rank = True
    obj._group_name = "g"
    obj.weights_getter = lambda: {"actor": torch.zeros(1)}
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(
        ((("bridge.0", torch.zeros(1)),), (("bridge.1", torch.zeros(1)),))
    )

    seen: list[tuple[list[str], str]] = []
    monkeypatch.setattr(
        upw.UpdateWeightFromDistributed,
        "_update_bucket_weights_from_distributed",
        lambda self, converted_named_tensors, pbar=None: seen.append(
            ([name for name, _ in converted_named_tensors], pbar)
        ),
    )
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")

    upw.UpdateWeightFromDistributed._sync_bridge_weights_to_rollout_engines(obj, pbar="pbar")

    assert seen == [
        (["bridge.0"], "pbar"),
        (["bridge.1"], "pbar"),
    ]


@pytest.mark.unit
def test_source_no_standalone_use_vllm_param(upw):
    src = inspect.getsource(upw)
    lines = [line.strip() for line in src.splitlines() if "use_vllm=" in line]
    assert lines == []


@pytest.mark.unit
def test_source_no_dist_broadcast_fallback(upw):
    src = inspect.getsource(upw)
    assert "dist.broadcast(" not in src


@pytest.mark.unit
def test_source_no_materialized_named_gpu_list(upw):
    src = inspect.getsource(upw.update_weights_from_distributed)
    assert "named_gpu = []" not in src
    assert "named_gpu_iter =" in src


@pytest.mark.unit
def test_connect_rollout_engines_always_uses_vllm_trainer_init(upw, monkeypatch):
    args = type("Args", (), {"rollout_num_gpus_per_engine": 1})()
    engines = [RecordingEngine(), RecordingEngine()]
    seen: list[dict] = []

    _patch_nccl_on_module(monkeypatch, upw, init_seen=seen)
    monkeypatch.setattr(upw.torch.cuda, "synchronize", lambda: None)
    monkeypatch.setattr(upw.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(upw.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(upw.ray, "get", lambda refs: refs)
    monkeypatch.setattr(upw.ray._private.services, "get_node_ip_address", lambda: "127.0.0.1")

    group = upw.connect_rollout_engines_from_distributed(args, "g", engines, engine_gpu_counts=[1, 2])

    assert isinstance(group, DummyGroup)
    assert len(seen) == 1
    assert seen[0]["master_address"] == "127.0.0.1"
    assert seen[0]["world_size"] == 4  # 1 + (1 + 2)
    assert len(engines[0].init_weights_update_group.calls) == 1
    assert len(engines[1].init_weights_update_group.calls) == 1


@pytest.mark.unit
def test_connect_rollout_engines_defers_vllm_group_init_for_multi_pp(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._model_update_groups = None
    engines = [RecordingEngine()]
    connect_calls: list[str] = []

    monkeypatch.setattr(upw.mpu, "get_data_parallel_rank", lambda **kwargs: 0)
    monkeypatch.setattr(upw.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_rank", lambda: 1)
    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        upw,
        "connect_rollout_engines_from_distributed",
        lambda *args, **kwargs: connect_calls.append(args[1]) or DummyGroup("unexpected"),
    )

    upw.UpdateWeightFromDistributed.connect_rollout_engines(
        obj,
        engines,
        RecordingLock(),
        engine_gpu_counts=[1],
    )

    assert obj._is_pp_src_rank is True
    assert obj._pp_world_size == 2
    assert obj._group_name == "vime-pp_1"
    assert obj._model_update_groups is None
    assert connect_calls == []


@pytest.mark.unit
@pytest.mark.parametrize(("pp_rank", "is_src", "expected_connect_calls"), [(0, True, ["vime-pp_0"]), (1, False, [])])
def test_bridge_multi_pp_connects_only_pp0(upw, monkeypatch, pp_rank, is_src, expected_connect_calls):
    obj = _make_instance(upw)
    obj._model_update_groups = None
    obj._hf_weight_iterator = MagicMock()
    actual_connect_calls: list[str] = []

    monkeypatch.setattr(upw.mpu, "get_data_parallel_rank", lambda **kwargs: 0)
    monkeypatch.setattr(upw.mpu, "get_tensor_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_rank", lambda: pp_rank)
    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(
        upw,
        "connect_rollout_engines_from_distributed",
        lambda *args, **kwargs: actual_connect_calls.append(args[1]) or DummyGroup(args[1]),
    )

    upw.UpdateWeightFromDistributed.connect_rollout_engines(
        obj,
        [RecordingEngine()],
        RecordingLock(),
        engine_gpu_counts=[1],
    )

    assert obj._is_pp_src_rank is is_src
    assert actual_connect_calls == expected_connect_calls


@pytest.mark.unit
def test_multi_pp_weight_sync_connects_only_active_pp_stage(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._model_update_groups = None
    obj._pp_world_size = 2
    obj._group_name = "vime-pp_0"
    obj._engine_gpu_counts = [1]
    obj.rollout_engines = [RecordingEngine()]
    send_calls: list[tuple[int, bool, str, bool, object]] = []
    connect_calls: list[str] = []
    barriers: list[object] = []

    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: barriers.append(kwargs.get("group")))
    monkeypatch.setattr(upw.torch.cuda, "synchronize", lambda: None)

    def fake_connect(args, group_name, rollout_engines, engine_gpu_counts=None):
        connect_calls.append(group_name)
        return DummyGroup(group_name)

    def fake_send(self, pbar):
        send_calls.append(
            (
                self._active_weight_sync_pp_rank,
                self._is_active_weight_sync_pp_stage(),
                self._group_name,
                pbar is not None,
                self._model_update_groups,
            )
        )

    monkeypatch.setattr(upw, "connect_rollout_engines_from_distributed", fake_connect)
    monkeypatch.setattr(upw.UpdateWeightFromDistributed, "_send_weights", fake_send)

    upw.UpdateWeightFromDistributed._send_weights_to_rollout_engines(obj)

    assert connect_calls == ["vime-pp_0"]
    assert send_calls == [
        (0, True, "vime-pp_0", True, DummyGroup("vime-pp_0")),
        (1, False, "vime-pp_0", False, DummyGroup("vime-pp_0")),
    ]
    assert barriers == ["gloo", "gloo", "gloo", "gloo"]
    assert obj._active_weight_sync_pp_rank is None
    assert obj._is_pp_src_rank is True
    assert obj._group_name == "vime-pp_0"


@pytest.mark.unit
def test_inactive_pp_stage_joins_raw_send_barriers_without_iterating(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._active_weight_sync_pp_rank = 1
    barriers: list[object] = []

    monkeypatch.setattr(upw.mpu, "get_pipeline_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: barriers.append(kwargs.get("group")))
    obj._iter_non_expert_chunks = lambda: (_ for _ in ()).throw(AssertionError("inactive stage must not iterate"))
    obj._iter_expert_chunks = lambda: (_ for _ in ()).throw(AssertionError("inactive stage must not iterate"))

    upw.UpdateWeightFromDistributed._send_weights(obj, pbar=None)

    assert barriers == ["gloo", "gloo"]


@pytest.mark.unit
def test_bridge_export_is_not_staged_by_pp(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._pp_world_size = 2
    obj._is_pp_src_rank = False
    obj._hf_weight_iterator = MagicMock()
    send_calls: list[tuple[object, object]] = []

    monkeypatch.setattr(
        upw.UpdateWeightFromDistributed,
        "_send_weights",
        lambda self, pbar: send_calls.append((getattr(self, "_active_weight_sync_pp_rank", None), pbar)),
    )

    upw.UpdateWeightFromDistributed._send_weights_to_rollout_engines(obj)

    assert send_calls == [(None, None)]


@pytest.mark.unit
def test_bridge_export_runs_on_non_source_pp_stage(upw, monkeypatch):
    obj = _make_instance(upw)
    obj._is_pp_src_rank = False
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = []
    barriers: list[object] = []

    monkeypatch.setattr(upw, "get_gloo_group", lambda: "gloo")
    monkeypatch.setattr(upw.dist, "barrier", lambda *args, **kwargs: barriers.append(kwargs.get("group")))

    upw.UpdateWeightFromDistributed._send_weights(obj, pbar=None)

    obj._hf_weight_iterator.get_hf_weight_chunks.assert_called_once_with({})
    assert barriers == ["gloo"]


@pytest.mark.unit
def test_weight_update_session_calls_start_and_finish(upw, monkeypatch):
    import torch.distributed as dist

    engines = [RecordingEngine(), RecordingEngine()]
    ray_refs = []
    barrier_calls: list[object] = []

    def fake_barrier(*, group=None, **kwargs):
        barrier_calls.append(group)

    monkeypatch.setattr(dist, "get_rank", lambda: 0)
    monkeypatch.setattr(dist, "barrier", fake_barrier)
    monkeypatch.setattr(upw, "get_gloo_group", lambda: "dummy-gloo-group")
    monkeypatch.setattr(upw.ray, "get", lambda refs: ray_refs.extend(refs) or refs)

    upw._begin_vllm_weight_update_session(engines)
    upw._end_vllm_weight_update_session(engines)

    assert len(engines[0].start_weight_update.calls) == 1
    assert engines[0].start_weight_update.calls[0].kwargs["is_checkpoint_format"] is True
    assert len(engines[1].start_weight_update.calls) == 1
    assert len(engines[0].finish_weight_update.calls) == 1
    assert len(engines[1].finish_weight_update.calls) == 1
    assert barrier_calls == ["dummy-gloo-group", "dummy-gloo-group"]


@pytest.mark.unit
def test_source_wraps_sync_with_weight_update_session(upw):
    src = inspect.getsource(upw.UpdateWeightFromDistributed.update_weights)
    assert "_begin_vllm_weight_update_session" in src
    assert "start_draft_weight_update" in src
    assert "_end_vllm_weight_update_session" in src
    assert src.count("_send_weights_to_rollout_engines") == 2


@pytest.mark.unit
def test_source_uses_nccl_trainer_send_weights_args(upw):
    src = inspect.getsource(upw.update_weights_from_distributed)
    assert "NCCLTrainerSendWeightsArgs" in src
    assert "weight_transfer_compat" not in src


@pytest.mark.unit
def test_cuda_sync_once_after_all_buckets_not_per_bucket(upw):
    send_src = inspect.getsource(upw.update_weights_from_distributed)
    sync_src = inspect.getsource(upw.UpdateWeightFromDistributed._send_weights_to_rollout_engines)
    assert "torch.cuda.synchronize" not in send_src
    assert "torch.cuda.synchronize" in sync_src


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


class _Handle:
    def wait(self) -> None:
        pass


def _param_info(name: str, param: torch.Tensor, src_rank: int = 0) -> ParamInfo:
    return ParamInfo(name, param.dtype, param.shape, {}, param.nbytes, src_rank)


def _tp_param(values, partition_dim: int) -> torch.nn.Parameter:
    param = torch.nn.Parameter(torch.tensor(values, dtype=torch.float32))
    param.tensor_model_parallel = True
    param.partition_dim = partition_dim
    param.partition_stride = 1
    return param


@pytest.mark.unit
def test_single_tp_returns_parameter_without_collective(monkeypatch, weight_modules):
    common, _ = weight_modules
    parameter = _tp_param([[1.0, 1.0]], partition_dim=0)
    calls = []
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(common.dist, "all_gather", lambda *args, **kwargs: calls.append(args))

    gathered = common.all_gather_param("decoder.weight", parameter)

    assert gathered.data_ptr() == parameter.data_ptr()
    assert calls == []


@pytest.mark.unit
def test_all_gather_params_coalesces_and_restores_layouts(monkeypatch, weight_modules):
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
    remote_flat = torch.cat(
        [
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]).flatten(),
            torch.tensor([[3.0], [4.0], [30.0], [40.0]]).flatten(),
            torch.tensor([[5.0, 6.0], [7.0, 8.0]]).flatten(),
        ]
    )
    calls = []

    def all_gather_into_tensor(output, local, group, async_op):
        calls.append((group, async_op))
        output[: local.numel()].copy_(local)
        output[local.numel() :].copy_(remote_flat)
        return _Handle()

    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_world_size", lambda: 2)
    monkeypatch.setattr(common.mpu, "get_tensor_model_parallel_group", lambda: "tp")
    monkeypatch.setattr(common.dist, "all_gather_into_tensor", all_gather_into_tensor)

    gathered = common.all_gather_params_async(entries)

    assert calls == [("tp", True)]
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


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
