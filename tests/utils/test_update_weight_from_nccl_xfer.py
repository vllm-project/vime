from __future__ import annotations

from argparse import Namespace
from dataclasses import dataclass

import pytest


@dataclass
class DummyAvailability:
    available: bool
    reason: str | None = None


class DummyFallback:
    instances = []

    def __init__(self, *args, **kwargs):
        self.update_calls = 0
        self.connect_calls = []
        self.disconnect_calls = 0
        DummyFallback.instances.append(self)

    def connect_rollout_engines(self, *args, **kwargs):
        self.connect_calls.append((args, kwargs))

    def disconnect_rollout_engines(self):
        self.disconnect_calls += 1

    def update_weights(self):
        self.update_calls += 1


@pytest.fixture
def upw(monkeypatch):
    from vime.backends.megatron_utils.update_weight import update_weight_from_nccl_xfer as mod

    DummyFallback.instances.clear()
    monkeypatch.setattr(mod, "UpdateWeightFromDistributed", DummyFallback)
    return mod


@pytest.mark.unit
def test_falls_back_when_native_bridge_unavailable(upw, monkeypatch, caplog):
    monkeypatch.setattr(
        upw,
        "get_nccl_xfer_availability",
        lambda: DummyAvailability(False, "missing pybind bridge"),
    )
    updater = upw.UpdateWeightFromNcclXfer(
        Namespace(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
    )

    updater.update_weights()

    assert DummyFallback.instances[-1].update_calls == 1
    assert "missing pybind bridge" in caplog.text


@pytest.mark.unit
def test_connect_and_disconnect_delegate_to_broadcast_fallback(upw):
    updater = upw.UpdateWeightFromNcclXfer(
        Namespace(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen",
        quantization_config=None,
    )
    fallback = DummyFallback.instances[-1]

    updater.connect_rollout_engines(["engine"], "lock", engine_gpu_counts=[1], engine_gpu_offsets=[0])
    updater.disconnect_rollout_engines()

    assert fallback.connect_calls == [((["engine"], "lock"), {"engine_gpu_counts": [1], "engine_gpu_offsets": [0]})]
    assert fallback.disconnect_calls == 1
