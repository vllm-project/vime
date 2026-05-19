"""Unit tests for slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py.

Test philosophy: minimize mocks. Use real torch tensors, real function calls.
The only test doubles are lightweight recording classes (not MagicMock) that
stand in for the ray ActorHandle / _NcclBridge surfaces — these can't be
instantiated standalone (Ray requires a cluster; _NcclBridge spawns a
subprocess with NCCL), so we substitute named classes that record calls.

Real megatron + ray + torch are already installed in the docker image, so the
module loads without any sys.modules stubbing.
"""

from __future__ import annotations

import importlib
import inspect
from collections.abc import Iterable
from dataclasses import dataclass, field

import pytest
import torch


@pytest.fixture(scope="module")
def upw():
    return importlib.import_module(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )


# ============================================================================
# Test doubles: lightweight recording classes (NOT MagicMock).
# Each method records its call args so tests can assert on dispatch + payload.
# ============================================================================


@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    """Mimics ``handle.method.remote(...)`` — records each call, returns a stub ref."""

    def __init__(self, return_value: str = "ref"):
        self._return_value = return_value
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return self._return_value


@dataclass
class RecordingEngine:
    """Stands in for a ray ActorHandle of a rollout engine."""

    update_weights_from_distributed: RecordingRemoteMethod = field(
        default_factory=lambda: RecordingRemoteMethod("ref")
    )


@dataclass
class RecordingNcclBridge:
    """Stands in for ``_NcclBridge`` — records broadcast / send_packed calls."""

    broadcast_calls: list[list[torch.Tensor]] = field(default_factory=list)
    packed_calls: list[list[tuple[str, torch.Tensor]]] = field(default_factory=list)

    def broadcast_tensors(self, tensors: Iterable[torch.Tensor]) -> None:
        self.broadcast_calls.append(list(tensors))

    def send_weights_packed(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> None:
        self.packed_calls.append(list(named_tensors))


def _real_tensors(n: int = 2):
    return [(f"layer.{i}.weight", torch.zeros(2, 2)) for i in range(n)]


# ============================================================================
# Signature changes: `use_vllm` parameter removed; only `packed` remains.
# ============================================================================


@pytest.mark.unit
def test_signature_no_use_vllm(upw):
    sig = inspect.signature(upw.update_weights_from_distributed)
    params = sig.parameters
    assert "use_vllm" not in params, "the dead `use_vllm` parameter must not be reintroduced"
    # All other expected params present
    for p in ("group_name", "group", "weight_version", "rollout_engines",
              "converted_named_tensors", "packed"):
        assert p in params, f"missing expected param: {p}"


@pytest.mark.unit
def test_signature_rejects_legacy_use_vllm_call(upw):
    """Old callsite ``update_weights_from_distributed(..., use_vllm=True, ...)``
    must now raise TypeError."""
    with pytest.raises(TypeError, match="use_vllm"):
        upw.update_weights_from_distributed(
            "g", RecordingNcclBridge(), 1, [RecordingEngine()],
            _real_tensors(), use_vllm=True, packed=False,
        )


# ============================================================================
# Behavior: packed=True → group.send_weights_packed; packed=False →
# group.broadcast_tensors. No sglang `dist.broadcast` fallback.
# ============================================================================


@pytest.mark.unit
def test_packed_true_dispatches_send_weights_packed(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()
    tensors = _real_tensors()

    refs = upw.update_weights_from_distributed(
        "groupA", group, 7, [engine], tensors, packed=True
    )

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

    refs = upw.update_weights_from_distributed(
        "groupB", group, 7, [engine], tensors, packed=False
    )

    assert len(group.broadcast_calls) == 1
    assert len(group.packed_calls) == 0
    assert len(group.broadcast_calls[0]) == len(tensors)
    assert refs == ["ref"]


@pytest.mark.unit
def test_default_packed_is_false(upw):
    """Default is unpacked (broadcast_tensors), matching the function default."""
    group = RecordingNcclBridge()
    engine = RecordingEngine()

    upw.update_weights_from_distributed(
        "g", group, 1, [engine], _real_tensors()
    )

    assert len(group.broadcast_calls) == 1
    assert len(group.packed_calls) == 0


@pytest.mark.unit
def test_no_dist_broadcast_fallback(upw, monkeypatch):
    """The old sglang `else: dist.broadcast(...)` branch is removed. Patch
    `dist.broadcast` and confirm it's never invoked, even on packed=False."""
    import torch.distributed as dist

    seen_broadcast = []

    def fake_broadcast(*a, **k):
        seen_broadcast.append((a, k))

    monkeypatch.setattr(dist, "broadcast", fake_broadcast)

    group = RecordingNcclBridge()
    engine = RecordingEngine()
    upw.update_weights_from_distributed(
        "g", group, 1, [engine], _real_tensors(), packed=False
    )

    assert seen_broadcast == [], "dist.broadcast must not be called (sglang path removed)"
    assert group.broadcast_calls, "broadcast went via _NcclBridge.broadcast_tensors"


# ============================================================================
# Remote engine RPC payload: kwargs forwarded to each rollout engine.
# ============================================================================


@pytest.mark.unit
def test_remote_kwargs_include_packed_true(upw):
    group = RecordingNcclBridge()
    engine = RecordingEngine()
    tensors = _real_tensors(n=1)

    upw.update_weights_from_distributed(
        "myg", group, 42, [engine], tensors, packed=True
    )

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

    upw.update_weights_from_distributed(
        "g", group, 99, [engine], tensors, packed=False
    )

    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["packed"] is False
    assert kw["weight_version"] == "99"
    assert kw["names"] == ["layer.0.weight", "layer.1.weight"]


@pytest.mark.unit
def test_remote_kwargs_no_use_vllm(upw):
    """`use_vllm` is removed from the forwarded RPC kwargs (slime sglang
    never sent it; vime's legacy `use_vllm=True` is dead)."""
    group = RecordingNcclBridge()
    engine = RecordingEngine()

    upw.update_weights_from_distributed(
        "g", group, 1, [engine], _real_tensors(), packed=False
    )

    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert "use_vllm" not in kw


@pytest.mark.unit
def test_multiple_engines_each_get_call(upw):
    """RPC fanout to every engine in `rollout_engines`."""
    group = RecordingNcclBridge()
    engines = [RecordingEngine() for _ in range(3)]
    upw.update_weights_from_distributed(
        "g", group, 1, engines, _real_tensors(), packed=True
    )
    for e in engines:
        assert len(e.update_weights_from_distributed.calls) == 1


@pytest.mark.unit
def test_empty_tensor_list_still_dispatches(upw):
    """Edge case: zero tensors. Function should still RPC fanout and call
    group method (even if with empty list — caller decides whether to skip)."""
    group = RecordingNcclBridge()
    engine = RecordingEngine()

    refs = upw.update_weights_from_distributed(
        "g", group, 1, [engine], [], packed=False
    )

    assert refs == ["ref"]
    kw = engine.update_weights_from_distributed.calls[0].kwargs
    assert kw["names"] == []
    assert kw["shapes"] == []
    assert len(group.broadcast_calls) == 1
    assert group.broadcast_calls[0] == []


# ============================================================================
# Source-level invariants — confirm dead code is gone.
# ============================================================================


@pytest.mark.unit
def test_source_no_standalone_use_vllm_param(upw):
    """Grep the source for `use_vllm=` assignments — should be 0 outside of
    the unrelated `use_vllm_packed` name (which is packed-vs-bucket mode)."""
    src = inspect.getsource(upw)
    lines = [
        line.strip()
        for line in src.splitlines()
        if "use_vllm=" in line and "use_vllm_packed" not in line
    ]
    assert lines == [], f"unexpected use_vllm= leftovers: {lines}"


@pytest.mark.unit
def test_source_no_sglang_dist_broadcast_fallback(upw):
    """The old `else: dist.broadcast(...)` sglang fallback in
    `update_weights_from_distributed` is gone — confirm the function body
    has no direct `dist.broadcast` call."""
    fn_src = inspect.getsource(upw.update_weights_from_distributed)
    assert "dist.broadcast(" not in fn_src
