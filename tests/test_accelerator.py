from types import SimpleNamespace

import pytest

from vime.utils import accelerator
from vime.utils.accelerator import npu as npu_backend

NUM_GPUS = 0


class FakeAccelerator(accelerator.Accelerator):
    name = "fake"
    device_type = "fake"
    communication_backend_name = "fake"

    def is_available(self):
        return True

    def device(self, index=None):
        return accelerator.torch.device("cpu")

    def device_name(self, index=None):
        return "cpu"

    def set_device(self, index):
        return None

    def current_device(self):
        return "cpu"

    def device_count(self):
        return 0

    def synchronize(self, device=None):
        return None

    def current_stream(self, device=None):
        return None

    def empty_cache(self):
        return None

    def mem_get_info(self, device=None):
        return 0, 0

    def memory_allocated(self, device=None):
        return 0

    def memory_reserved(self, device=None):
        return 0


@pytest.fixture(autouse=True)
def reset_accelerator_selection(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    patch_imported = accelerator._MUSA_PATCH_IMPORTED
    bootstrap_checked = accelerator._MUSA_BOOTSTRAP_CHECKED
    npu_imported = accelerator._NPU_RUNTIME_IMPORTED
    npu_bootstrap_checked = accelerator._NPU_BOOTSTRAP_CHECKED
    supa_imported = accelerator._SUPA_RUNTIME_IMPORTED
    supa_bootstrap_checked = accelerator._SUPA_BOOTSTRAP_CHECKED
    for name in (
        "VIME_ACCELERATOR",
        "MUSA_VISIBLE_DEVICES",
        "MUSA_PATCH_PATH",
        "CUDA_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "ASCEND_HOME_PATH",
        "SUPA_VISIBLE_DEVICES",
        "BIREN_HOME",
    ):
        monkeypatch.delenv(name, raising=False)
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    accelerator._MUSA_PATCH_IMPORTED = False
    accelerator._MUSA_BOOTSTRAP_CHECKED = False
    accelerator._NPU_RUNTIME_IMPORTED = False
    accelerator._NPU_BOOTSTRAP_CHECKED = False
    accelerator._SUPA_RUNTIME_IMPORTED = False
    accelerator._SUPA_BOOTSTRAP_CHECKED = False
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected
    accelerator._MUSA_PATCH_IMPORTED = patch_imported
    accelerator._MUSA_BOOTSTRAP_CHECKED = bootstrap_checked
    accelerator._NPU_RUNTIME_IMPORTED = npu_imported
    accelerator._NPU_BOOTSTRAP_CHECKED = npu_bootstrap_checked
    accelerator._SUPA_RUNTIME_IMPORTED = supa_imported
    accelerator._SUPA_BOOTSTRAP_CHECKED = supa_bootstrap_checked


@pytest.mark.unit
def test_cuda_selection_does_not_bootstrap_musa(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    monkeypatch.setattr(
        accelerator,
        "_import_musa_patch",
        lambda: pytest.fail("CUDA selection must not import musa_patch"),
    )

    assert accelerator.get_accelerator().name == "cuda"
    assert accelerator.process_group_backend() == "nccl"
    assert accelerator.visible_devices_env_key() == "CUDA_VISIBLE_DEVICES"


@pytest.mark.unit
def test_selected_musa_bootstraps_patch_once(monkeypatch):
    imports = []
    fake_musa = SimpleNamespace(is_available=lambda: True)

    def import_musa_patch():
        imports.append("musa_patch")
        monkeypatch.setattr(accelerator.torch, "musa", fake_musa, raising=False)
        return True

    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_import_musa_patch", import_musa_patch)

    assert imports == []
    assert accelerator.initialize_accelerator().name == "musa"
    assert accelerator.initialize_accelerator().name == "musa"
    assert imports == ["musa_patch"]


@pytest.mark.unit
def test_cpu_only_initialization_does_not_require_an_accelerator(monkeypatch):
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_npu_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)

    assert accelerator.initialize_accelerator() is None


@pytest.mark.unit
def test_musa_backend_maps_devices_and_process_groups(monkeypatch):
    monkeypatch.setattr(accelerator.MUSAAccelerator, "is_available", lambda self: True)
    monkeypatch.setenv("MUSA_VISIBLE_DEVICES", "2,5")
    accelerator.set_accelerator(accelerator.MUSAAccelerator())

    assert accelerator.visible_devices_env_key() == "MUSA_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1
    assert accelerator.process_group_backend() == "mccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,musa:mccl"
    assert accelerator.process_group_backend("gloo") == "gloo"


@pytest.mark.unit
def test_cuda_visible_device_mapping(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "4,6")
    accelerator.set_accelerator(FakeAccelerator())

    assert accelerator.resolve_visible_device_id(4) == 0
    assert accelerator.resolve_visible_device_id(1) == 1
    with pytest.raises(RuntimeError, match="CUDA_VISIBLE_DEVICES=4,6"):
        accelerator.resolve_visible_device_id(7)

    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "GPU-aaaa,GPU-bbbb")
    assert accelerator.resolve_visible_device_id("GPU-bbbb") == 1


@pytest.mark.unit
def test_registered_backend_can_be_selected(monkeypatch):
    class RegisteredAccelerator(FakeAccelerator):
        name = "registered"

    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    accelerator.register_accelerator("registered", RegisteredAccelerator, lambda: True, priority=300)

    assert accelerator.get_accelerator().name == "registered"


@pytest.mark.unit
def test_cuda_backend_uses_torch_cuda_namespace(monkeypatch):
    monkeypatch.setattr(accelerator.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(accelerator.torch.cuda, "device_count", lambda: 2)
    monkeypatch.setattr(accelerator.torch.cuda, "current_device", lambda: 1)
    monkeypatch.setattr(accelerator.torch.cuda, "memory_allocated", lambda device=None: 123)
    backend = accelerator.CUDAAccelerator()

    assert backend.is_available()
    assert backend.device_name() == "cuda:1"
    assert backend.memory_allocated() == 123


@pytest.mark.unit
def test_routing_replay_uses_selected_backend_current_device(monkeypatch):
    from vime.utils import routing_replay

    transfers = []

    class FakeTopIndices:
        def is_pinned(self):
            return False

        def to(self, device, *, dtype, non_blocking):
            transfers.append((device, dtype, non_blocking))
            return self

    accelerator.set_accelerator(FakeAccelerator())
    monkeypatch.setattr(routing_replay.RoutingReplay, "all_routing_replays", [])
    replay = routing_replay.RoutingReplay()
    replay.top_indices_list.append(FakeTopIndices())

    replay.pop_forward()
    replay.pop_backward()
    assert transfers == [
        ("cpu", accelerator.torch.int32, False),
        ("cpu", accelerator.torch.int32, False),
    ]


@pytest.mark.unit
def test_musa_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(accelerator.torch, "musa", raising=False)

    assert accelerator.is_musa_available() is False


@pytest.mark.unit
def test_selected_npu_bootstraps_runtime_once(monkeypatch):
    imports = []
    fake_npu = SimpleNamespace(is_available=lambda: True)

    def import_torch_npu():
        imports.append("torch_npu")
        monkeypatch.setattr(accelerator.torch, "npu", fake_npu, raising=False)
        return True

    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "0")
    # Both auto-selection and runtime availability inspect device nodes.
    # Mock the hardware boundary so both checks see the same fake device.
    monkeypatch.setattr(npu_backend, "glob", lambda pattern: ["/dev/davinci0"])
    monkeypatch.setattr(accelerator, "_import_torch_npu", import_torch_npu)

    assert imports == []
    assert accelerator.initialize_accelerator().name == "npu"
    assert accelerator.initialize_accelerator().name == "npu"
    assert imports == ["torch_npu"]


@pytest.mark.unit
def test_an_ascend_toolkit_without_devices_does_not_select_npu(monkeypatch):
    """CANN exports ASCEND_HOME_PATH on hosts that have no cards at all."""
    monkeypatch.setenv("ASCEND_HOME_PATH", "/usr/local/Ascend/ascend-toolkit/latest")
    monkeypatch.setattr(accelerator, "_has_npu_device", lambda: False)
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    monkeypatch.setattr(accelerator, "_import_torch_npu", lambda: pytest.fail("the runtime must not be imported"))

    assert accelerator.initialize_accelerator() is None


@pytest.mark.unit
def test_an_explicit_request_selects_npu_without_a_device_probe(monkeypatch):
    """Explicit selection skips the auto-selection gate, then validates runtime."""
    fake_npu = SimpleNamespace(is_available=lambda: True)
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    monkeypatch.setattr(npu_backend, "glob", lambda pattern: ["/dev/davinci0"])
    monkeypatch.setattr(
        accelerator,
        "_has_npu_device",
        lambda: pytest.fail("explicit selection must skip the auto-selection probe"),
    )
    monkeypatch.setattr(
        accelerator,
        "_import_torch_npu",
        lambda: monkeypatch.setattr(accelerator.torch, "npu", fake_npu, raising=False) or True,
    )

    assert accelerator.initialize_accelerator().name == "npu"


@pytest.mark.unit
def test_npu_backend_maps_devices_and_process_groups(monkeypatch):
    monkeypatch.setattr(accelerator.NPUAccelerator, "is_available", lambda self: True)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "2,5")
    accelerator.set_accelerator(accelerator.NPUAccelerator())

    assert accelerator.device_type() == "npu"
    assert accelerator.visible_devices_env_key() == "ASCEND_RT_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1
    assert accelerator.process_group_backend() == "hccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,npu:hccl"
    assert accelerator.process_group_backend("gloo") == "gloo"
    assert accelerator.distributed_device_id() is None


@pytest.mark.unit
def test_npu_backend_is_recognized_as_accelerator_backend():
    assert accelerator.is_accelerator_backend("hccl") is True
    assert accelerator.is_accelerator_backend("cpu:gloo,npu:hccl") is True
    assert accelerator.is_accelerator_backend("gloo") is False


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
    monkeypatch.setattr(accelerator.torch, "npu", fake_npu, raising=False)
    backend = accelerator.NPUAccelerator()

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
    monkeypatch.setattr(accelerator.torch, "npu", fake_npu, raising=False)

    with pytest.raises(RuntimeError, match="device failure"):
        accelerator.NPUAccelerator().empty_cache()


@pytest.mark.unit
def test_npu_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(accelerator.torch, "npu", raising=False)

    assert accelerator.is_npu_available() is False


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
    monkeypatch.setattr(accelerator.torch, "npu", SimpleNamespace(is_available=runtime_available), raising=False)

    assert accelerator.is_npu_available() is expected
    assert accelerator.NPUAccelerator().is_available() is expected
    assert checks == ([True, True] if expected else [])


@pytest.mark.unit
def test_cuda_selection_does_not_bootstrap_supa(monkeypatch):
    monkeypatch.setenv("VIME_ACCELERATOR", "cuda")
    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: True)
    monkeypatch.setattr(accelerator.CUDAAccelerator, "is_available", lambda self: True)
    monkeypatch.setattr(
        accelerator,
        "_import_torch_supa",
        lambda: pytest.fail("CUDA selection must not import torch_supa"),
    )

    assert accelerator.get_accelerator().name == "cuda"
    assert accelerator.process_group_backend() == "nccl"
    assert accelerator.visible_devices_env_key() == "CUDA_VISIBLE_DEVICES"


@pytest.mark.unit
def test_selected_supa_bootstraps_runtime_once(monkeypatch):
    imports = []
    fake_supa = SimpleNamespace(is_available=lambda: True)

    def import_torch_supa():
        imports.append("torch_supa")
        monkeypatch.setattr(accelerator.torch, "supa", fake_supa, raising=False)
        return True

    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "0")
    monkeypatch.setattr(accelerator, "_import_torch_supa", import_torch_supa)

    assert imports == []
    assert accelerator.initialize_accelerator().name == "supa"
    assert accelerator.initialize_accelerator().name == "supa"
    assert imports == ["torch_supa"]


@pytest.mark.unit
def test_supa_backend_maps_devices_and_process_groups(monkeypatch):
    monkeypatch.setattr(accelerator.SUPAAccelerator, "is_available", lambda self: True)
    monkeypatch.setenv("SUPA_VISIBLE_DEVICES", "2,5")
    accelerator.set_accelerator(accelerator.SUPAAccelerator())

    assert accelerator.device_type() == "supa"
    assert accelerator.visible_devices_env_key() == "SUPA_VISIBLE_DEVICES"
    assert accelerator.resolve_visible_device_id("5") == 1
    assert accelerator.process_group_backend() == "bccl"
    assert accelerator.weight_update_backend() == "cpu:gloo,supa:bccl"
    assert accelerator.process_group_backend("gloo") == "gloo"
    assert accelerator.distributed_device_id() is None


@pytest.mark.unit
def test_supa_backend_is_recognized_as_accelerator_backend():
    assert accelerator.is_accelerator_backend("bccl") is True
    assert accelerator.is_accelerator_backend("cpu:gloo,supa:bccl") is True
    assert accelerator.is_accelerator_backend("gloo") is False


@pytest.mark.unit
def test_supa_availability_handles_missing_torch_namespace(monkeypatch):
    monkeypatch.delattr(accelerator.torch, "supa", raising=False)

    assert accelerator.is_supa_available() is False


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
