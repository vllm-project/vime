"""Direct Ascend device tests, independent of backend registration."""

from types import SimpleNamespace

import pytest
import torch

from vime.utils.accelerator import npu as npu_backend
from vime.utils.accelerator.npu import NPUAccelerator


pytestmark = pytest.mark.unit


def test_npu_backend_describes_devices_and_process_groups(monkeypatch):
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,5")
    backend = NPUAccelerator()

    assert backend.device_type == "npu"
    assert backend.visible_devices_env == "ASCEND_RT_VISIBLE_DEVICES"
    assert backend.resolve_visible_device_id("5") == 1
    assert backend.communication_backend() == "hccl"
    assert backend.weight_update_backend() == "cpu:gloo,npu:hccl"
    assert backend.communication_backend("gloo") == "gloo"
    assert backend.distributed_device_id() is None


@pytest.mark.unit
@pytest.mark.parametrize("capturing", [False, True])
@pytest.mark.parametrize("active_pool", [None, object()])
def test_npu_empty_cache_defers_until_pool_and_capture_have_exited(monkeypatch, capturing, active_pool):
    state = {"capturing": capturing, "pool": active_pool}
    calls = []
    fake_npu = SimpleNamespace(
        is_current_stream_capturing=lambda: state["capturing"],
        memory=SimpleNamespace(MemPoolContext=SimpleNamespace(active_pool=lambda: state["pool"])),
        empty_cache=lambda: calls.append("empty_cache"),
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)
    backend = NPUAccelerator()

    backend.empty_cache()
    assert calls == (["empty_cache"] if not capturing and active_pool is None else [])

    # Leaving the export pool must not permanently suppress reclamation.
    state.update(capturing=False, pool=None)
    calls.clear()
    backend.empty_cache()
    assert calls == ["empty_cache"]


@pytest.mark.unit
def test_npu_empty_cache_propagates_allocator_errors(monkeypatch):
    def empty_cache():
        raise RuntimeError("device failure")

    fake_npu = SimpleNamespace(
        is_current_stream_capturing=lambda: False,
        memory=SimpleNamespace(MemPoolContext=SimpleNamespace(active_pool=lambda: None)),
        empty_cache=empty_cache,
    )
    monkeypatch.setattr(torch, "npu", fake_npu, raising=False)

    with pytest.raises(RuntimeError, match="device failure"):
        NPUAccelerator().empty_cache()


@pytest.mark.unit
def test_npu_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(torch, "npu", raising=False)

    assert npu_backend.is_npu_available() is False


@pytest.mark.unit
@pytest.mark.parametrize(
    ("visible_devices", "device_nodes", "expected"),
    [("-1", ["/dev/davinci0"], False), ("0", [], False), ("0", ["/dev/davinci0"], True)],
)
def test_npu_availability_requires_a_visible_device(monkeypatch, visible_devices, device_nodes, expected):
    checks = []

    def runtime_available():
        checks.append(True)
        return True

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", visible_devices)
    monkeypatch.setattr(npu_backend, "glob", lambda pattern: device_nodes)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(is_available=runtime_available), raising=False)

    assert npu_backend.is_npu_available() is expected
    assert NPUAccelerator().is_available() is expected
    assert checks == ([True, True] if expected else [])
