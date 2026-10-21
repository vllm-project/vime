"""The Ray resource seam: name, bundles, actor options and device ids."""

from types import SimpleNamespace

import pytest

from vime.ray import resources
from vime.utils import accelerator
from vime.utils.accelerator.base import Accelerator


pytestmark = pytest.mark.unit


class _CustomResourceAccelerator(Accelerator):
    """A backend whose devices Ray models outside its own GPU resource."""

    name = "fake-oot"
    device_type = "fake"
    communication_backend_name = "fccl"
    ray_device_key = "FAKE"
    ray_uses_native_gpu_resource = False

    def is_available(self) -> bool:
        return True

    # The abstract surface is irrelevant here: this seam must resolve a
    # description without touching a device.
    def device(self, index=None):
        raise AssertionError("resource description must not touch a device")

    device_name = memory_allocated = memory_reserved = device
    set_device = current_device = device_count = device
    synchronize = current_stream = default_stream = device
    stream = Stream = Event = empty_cache = ipc_collect = device
    set_allocator_expandable_segments = mem_get_info = device


@pytest.fixture(autouse=True)
def isolate_selection(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    for name in ("VIME_ACCELERATOR", "MUSA_VISIBLE_DEVICES", "MUSA_PATCH_PATH", "SUPA_VISIBLE_DEVICES", "BIREN_HOME"):
        monkeypatch.delenv(name, raising=False)
    # Say explicitly that no Ascend device is visible, rather than leaving the
    # variable unset: unset lets a CANN host's real devices be detected, and
    # then this describes that host instead of the case under test.
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "-1")
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected


@pytest.fixture
def no_local_accelerator(monkeypatch):
    """A driver host that offers no accelerator of its own."""
    # Keep this independent of the runner's hardware: no builtin backend
    # registers itself, so a test sees only the backends it registers. A CI
    # host that has real accelerators must not change these outcomes.
    monkeypatch.setattr(accelerator, "_register_builtin_backends", lambda: None)


def test_a_driver_without_any_device_keeps_rays_gpu_resource(no_local_accelerator):
    assert accelerator.initialize_accelerator() is None

    assert accelerator.ray_device_key() == "GPU"
    assert resources.actor_options(1) == {"num_gpus": 1}
    assert resources.bundle() == {"GPU": 1, "CPU": 1}


def test_a_custom_resource_backend_is_scheduled_under_its_own_name(no_local_accelerator):
    accelerator.register_accelerator("fake-oot", _CustomResourceAccelerator, lambda: True, priority=300)

    assert accelerator.ray_device_key() == "FAKE"
    assert resources.bundle(2, 4) == {"FAKE": 2, "CPU": 4}
    # num_gpus is pinned to 0 so Ray does not also reserve a GPU it has none of.
    assert resources.actor_options(0.5) == {"num_gpus": 0, "resources": {"FAKE": 0.5}}


def test_an_explicit_request_is_described_without_requiring_its_device(monkeypatch, no_local_accelerator):
    """A deviceless driver can still build requests for the job's accelerator."""
    accelerator.register_accelerator("fake-oot", _CustomResourceAccelerator, lambda: False, priority=300)
    monkeypatch.setenv("VIME_ACCELERATOR", "fake-oot")

    assert accelerator.ray_device_key() == "FAKE"
    assert resources.actor_options(1) == {"num_gpus": 0, "resources": {"FAKE": 1}}


def test_an_unknown_explicit_request_is_rejected(monkeypatch, no_local_accelerator):
    monkeypatch.setenv("VIME_ACCELERATOR", "nonexistent")

    with pytest.raises(ValueError, match="Unknown accelerator 'nonexistent'"):
        accelerator.ray_device_key()


def test_native_gpu_ids_come_from_rays_gpu_api(monkeypatch, no_local_accelerator):
    import ray

    monkeypatch.setattr(ray, "get_gpu_ids", lambda: [3, 7], raising=False)

    assert resources.accelerator_ids() == ["3", "7"]


def test_custom_resource_ids_come_from_the_accelerator_api(monkeypatch, no_local_accelerator):
    import ray

    accelerator.register_accelerator("fake-oot", _CustomResourceAccelerator, lambda: True, priority=300)
    context = SimpleNamespace(get_accelerator_ids=lambda: {"FAKE": ["1", "2"]})
    monkeypatch.setattr(ray, "get_runtime_context", lambda: context, raising=False)
    monkeypatch.setattr(ray, "get_gpu_ids", lambda: (_ for _ in ()).throw(AssertionError("wrong Ray API")))

    assert resources.accelerator_ids() == ["1", "2"]
