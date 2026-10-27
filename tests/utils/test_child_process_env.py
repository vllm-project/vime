"""What the accelerator tells a training actor, an engine actor and the
serving subprocess, and that answering it needs no local device."""

import sys
from types import ModuleType, SimpleNamespace

import pytest

from vime.utils import accelerator, device_env
from vime.utils.accelerator.cuda import CUDAAccelerator


pytestmark = pytest.mark.unit


_VENDOR_MODULE = "fake_vendor_device_env"


def _vendor_module() -> ModuleType:
    """A replacement module, resolved by name the way a real backend's is."""
    module = ModuleType(_VENDOR_MODULE)
    module.train_actor_env = lambda args, env_vars=None: {**dict(env_vars or {}), "VENDOR_TRAIN": "1"}
    module.engine_actor_env = lambda args, env_vars=None: {**dict(env_vars or {}), "VENDOR_ENGINE": "1"}

    def serving_subprocess_env(base_env, *, visible_devices, colocate):
        env = dict(base_env)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["VENDOR_VISIBLE_DEVICES"] = visible_devices
        return env

    module.serving_subprocess_env = serving_subprocess_env
    return module


class _VendorAccelerator(CUDAAccelerator):
    name = "fake-vendor"

    @classmethod
    def device_env_qualname(cls) -> str:
        return _VENDOR_MODULE


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
    # Keep this independent of the runner's hardware: no builtin backend
    # registers itself, so a test sees only the backends it registers. A CI
    # host that has real accelerators must not change these outcomes.
    monkeypatch.setattr(accelerator, "_register_builtin_backends", lambda: None)
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected


def test_without_offload_the_environment_is_untouched():
    args = SimpleNamespace(offload_train=False)

    assert device_env.current().train_actor_env(args, {"A": "1"}) == {"A": "1"}


def test_engine_and_serving_default_to_passing_the_environment_through():
    args = SimpleNamespace(offload_train=True)

    assert device_env.current().engine_actor_env(args, {"A": "1"}) == {"A": "1"}
    assert device_env.current().serving_subprocess_env({"A": "1"}, visible_devices="0,1", colocate=True) == {"A": "1"}


def test_a_deviceless_driver_still_builds_the_actor_environments():
    """Runtime envs are built before the actors exist, so on the driver."""
    assert accelerator.initialize_accelerator() is None

    args = SimpleNamespace(offload_train=False)
    assert device_env.current().train_actor_env(args, {}) == {}
    assert device_env.current().engine_actor_env(args, {}) == {}


def test_a_vendor_backend_supplies_all_three_environments(monkeypatch):
    accelerator.register_accelerator("fake-vendor", _VendorAccelerator, lambda: False, priority=300)
    monkeypatch.setitem(sys.modules, _VENDOR_MODULE, _vendor_module())
    monkeypatch.setenv("VIME_ACCELERATOR", "fake-vendor")
    args = SimpleNamespace(offload_train=True)

    assert device_env.current().train_actor_env(args, {}) == {"VENDOR_TRAIN": "1"}
    assert device_env.current().engine_actor_env(args, {}) == {"VENDOR_ENGINE": "1"}
    assert device_env.current().serving_subprocess_env(
        {"CUDA_VISIBLE_DEVICES": "0"}, visible_devices="3", colocate=False
    ) == {"VENDOR_VISIBLE_DEVICES": "3"}
