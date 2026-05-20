"""Unit tests for slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py."""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
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


@dataclass
class RecordingNcclBridge:
    broadcast_calls: list[list[torch.Tensor]] = field(default_factory=list)
    packed_calls: list[list[tuple[str, torch.Tensor]]] = field(default_factory=list)

    def broadcast_tensors(self, tensors: Iterable[torch.Tensor]) -> None:
        self.broadcast_calls.append(list(tensors))

    def send_weights_packed(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.packed_calls.append(list(named_tensors))


def _real_tensors(n: int = 2):
    return [(f"layer.{i}.weight", torch.zeros(2, 2)) for i in range(n)]


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
            RecordingNcclBridge(),
            1,
            [RecordingEngine()],
            _real_tensors(),
            use_vllm=True,
            packed=False,
        )


@pytest.mark.unit
def test_packed_true_dispatches_send_weights_packed(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()
    tensors = _real_tensors()

    refs = upw.update_weights_from_distributed("groupA", group, 7, [engine], tensors, packed=True)

    assert len(group.packed_calls) == 1
    assert len(group.broadcast_calls) == 0
    sent = group.packed_calls[0]
    assert [n for n, _ in sent] == [n for n, _ in tensors]
    assert refs == ["ref"]


@pytest.mark.unit
def test_packed_false_dispatches_broadcast_tensors(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()
    tensors = _real_tensors()

    refs = upw.update_weights_from_distributed("groupB", group, 7, [engine], tensors, packed=False)

    assert len(group.broadcast_calls) == 1
    assert len(group.packed_calls) == 0
    assert len(group.broadcast_calls[0]) == len(tensors)
    assert refs == ["ref"]


@pytest.mark.unit
def test_default_packed_is_false(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()

    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors())

    assert len(group.broadcast_calls) == 1
    assert len(group.packed_calls) == 0


@pytest.mark.unit
def test_no_dist_broadcast_fallback(upw, monkeypatch):
    import torch.distributed as dist

    seen_broadcast = []

    def fake_broadcast(*a, **k):
        seen_broadcast.append((a, k))

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)

    group = RecordingNcclBridge()
    engine = RecordingEngine()
    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors(), packed=False)

    assert seen_broadcast == []
    assert group.broadcast_calls


@pytest.mark.unit
def test_remote_kwargs_include_packed_true(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()
    tensors = _real_tensors(n=1)

    upw.update_weights_from_distributed("myg", group, 42, [engine], tensors, packed=True)

    assert len(engine.update_weights_from_distributed.calls) == 1
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is True
    assert kw["group_name"] == "myg"
    assert kw["weight_version"] == "42"
    assert kw["names"] == ["layer.0.weight"]
    assert kw["shapes"] == [torch.Size([2, 2])]
    assert kw["dtypes"] == [torch.float32]


@pytest.mark.unit
def test_remote_kwargs_include_packed_false(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()
    tensors = _real_tensors(n=2)

    upw.update_weights_from_distributed("g", group, 99, [engine], tensors, packed=False)

    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is False
    assert kw["weight_version"] == "99"
    assert kw["names"] == ["layer.0.weight", "layer.1.weight"]


@pytest.mark.unit
def test_remote_kwargs_no_use_vllm(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()

    upw.update_weights_from_distributed("g", group, 1, [engine], _real_tensors(), packed=False)

    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert "use_vllm" not in kw


@pytest.mark.unit
def test_multiple_engines_each_get_call(upw):
    group = RecordingNcclBridge()
    engines = [RecordingEngine() for _ in range(3)]
    upw.update_weights_from_distributed("g", group, 1, engines, _real_tensors(), packed=True)
    for e in engines:
        assert len(e.update_weights_from_distributed.calls) == 1


@pytest.mark.unit
def test_empty_tensor_list_still_dispatches(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()

    refs = upw.update_weights_from_distributed("g", group, 1, [engine], [], packed=False)

    assert refs == ["ref"]
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["names"] == []
    assert kw["shapes"] == []
    assert len(group.broadcast_calls) == 1
    assert group.broadcast_calls[0] == []


@pytest.mark.unit
def test_source_no_standalone_use_vllm_param(upw):
    src = inspect.getsource(upw)
    lines = [line.strip() for line in src.splitlines() if "use_vllm=" in line and "use_vllm_packed" not in line]
    assert lines == []


@pytest.mark.unit
def test_source_no_sglang_dist_broadcast_fallback(upw):
    fn_src = inspect.getsource(upw.update_weights_from_distributed)
    assert "dist.broadcast(" not in fn_src
