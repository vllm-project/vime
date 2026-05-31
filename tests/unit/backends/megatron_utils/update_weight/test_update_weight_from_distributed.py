"""Unit tests for slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py."""

from __future__ import annotations

import importlib
import inspect
from dataclasses import dataclass, field

import pytest
import torch


@pytest.fixture(scope="module")
def upw():
    return importlib.import_module("slime.backends.megatron_utils.update_weight.update_weight_from_distributed")


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


@pytest.mark.unit
def test_signature_no_use_vllm(upw):
    sig = inspect.signature(upw.update_weights_from_distributed)
    params = sig.parameters
    assert "use_vllm" not in params
    for p in ("group_name", "group", "weight_version", "rollout_engines", "converted_named_tensors", "packed"):
        assert p in params


@pytest.mark.unit
def test_signature_rejects_legacy_use_vllm_call(upw):
    with pytest.raises(TypeError, match="use_vllm"):
        upw.update_weights_from_distributed(
            "g",
            DummyGroup(),
            1,
            [RecordingEngine()],
            _real_tensors(),
            use_vllm=True,
            packed=False,
        )


@pytest.mark.unit
def test_packed_true_uses_vllm_trainer_send_weights(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    refs = upw.update_weights_from_distributed("groupA", group, 7, [engine], tensors, packed=True)

    assert len(seen) == 1
    sent = seen[0]["items"]
    assert [n for n, _ in sent] == [n for n, _ in tensors]
    assert seen[0]["group"] is group
    assert seen[0]["packed"] is True
    assert refs == ["ref"]


@pytest.mark.unit
def test_packed_false_still_uses_vllm_trainer_send_weights(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    refs = upw.update_weights_from_distributed("groupB", group, 7, [engine], tensors, packed=False)

    assert len(seen) == 1
    assert len(seen[0]["items"]) == len(tensors)
    assert seen[0]["packed"] is False
    assert refs == ["ref"]


@pytest.mark.unit
def test_default_packed_is_false(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen = []
    _patch_trainer_send(monkeypatch, upw, seen)

    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors())

    assert len(seen) == 1
    assert seen[0]["packed"] is False


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
    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors(), packed=False)

    assert seen_broadcast == []
    assert len(seen_send) == 1


@pytest.mark.unit
def test_remote_kwargs_include_packed_true(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors(n=1)
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("myg", group, 42, [engine], tensors, packed=True)

    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is True
    assert len(engine.update_weights_from_distributed.calls) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is True
    assert kw["group_name"] == "myg"
    assert kw["weight_version"] == "42"
    assert kw["names"] == ["layer.0.weight"]
    assert kw["shapes"] == [torch.Size([2, 2])]
    assert kw["dtypes"] == [torch.float32]


@pytest.mark.unit
def test_remote_kwargs_include_packed_false(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    tensors = _real_tensors(n=2)
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("g", group, 99, [engine], tensors, packed=False)

    assert len(seen_send) == 1
    assert seen_send[0]["packed"] is False
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is False
    assert kw["weight_version"] == "99"
    assert kw["names"] == ["layer.0.weight", "layer.1.weight"]


@pytest.mark.unit
def test_remote_kwargs_no_use_vllm(upw, monkeypatch):
    group = DummyGroup()
    engine = RecordingEngine()
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors(), packed=False)

    assert len(seen_send) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert "use_vllm" not in kw


@pytest.mark.unit
def test_multiple_engines_each_get_call(upw, monkeypatch):
    group = DummyGroup()
    engines = [RecordingEngine() for _ in range(3)]
    seen_send = []
    _patch_trainer_send(monkeypatch, upw, seen_send)

    upw.update_weights_from_distributed("g", group, 1, engines, _real_tensors(), packed=True)
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

    refs = upw.update_weights_from_distributed("g", group, 1, [engine], [], packed=False)

    assert refs == ["ref"]
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["names"] == []
    assert kw["shapes"] == []
    assert len(seen_send) == 1
    assert seen_send[0]["items"] == []


@pytest.mark.unit
def test_source_no_standalone_use_vllm_param(upw):
    src = inspect.getsource(upw)
    lines = [line.strip() for line in src.splitlines() if "use_vllm=" in line and "use_vllm_packed" not in line]
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
    assert "_end_vllm_weight_update_session" in src
    assert "_sync_weights_to_rollout_engines" in src


@pytest.mark.unit
def test_source_uses_nccl_trainer_send_weights_args(upw):
    src = inspect.getsource(upw.update_weights_from_distributed)
    assert "NCCLTrainerSendWeightsArgs" in src
    assert "weight_transfer_compat" not in src


@pytest.mark.unit
def test_cuda_sync_once_after_all_buckets_not_per_bucket(upw):
    send_src = inspect.getsource(upw.update_weights_from_distributed)
    sync_src = inspect.getsource(upw.UpdateWeightFromDistributed._sync_weights_to_rollout_engines)
    assert "torch.cuda.synchronize" not in send_src
    assert "torch.cuda.synchronize" in sync_src
