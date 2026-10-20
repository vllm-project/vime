"""What the accelerator tells a training actor, an engine actor and the
serving subprocess, and that answering it needs no local device."""

import os
from types import SimpleNamespace

import pytest

from vime.utils import accelerator
from vime.utils.accelerator.base import Accelerator
from vime.utils.accelerator.cuda import CUDAAccelerator


pytestmark = pytest.mark.unit


class _VendorAccelerator(CUDAAccelerator):
    name = "fake-vendor"

    @classmethod
    def train_actor_env(cls, args, env_vars=None):
        return {**dict(env_vars or {}), "VENDOR_TRAIN": "1"}

    @classmethod
    def engine_actor_env(cls, args, env_vars=None):
        return {**dict(env_vars or {}), "VENDOR_ENGINE": "1"}

    @classmethod
    def serving_subprocess_env(cls, base_env, *, visible_devices, colocate):
        env = dict(base_env)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env["VENDOR_VISIBLE_DEVICES"] = visible_devices
        return env


@pytest.fixture(autouse=True)
def isolate_selection(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    for name in ("VIME_ACCELERATOR", "MUSA_VISIBLE_DEVICES", "MUSA_PATCH_PATH", "SUPA_VISIBLE_DEVICES", "BIREN_HOME"):
        monkeypatch.delenv(name, raising=False)
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

    assert accelerator.train_actor_env(args, {"A": "1"}) == {"A": "1"}


def test_offload_hands_the_allocator_to_torch_memory_saver(tmp_path):
    """The same three variables the GPU path set inline, and a real library."""
    args = SimpleNamespace(offload_train=True)

    env = accelerator.train_actor_env(args, {"A": "1"})

    assert env["A"] == "1"
    assert env["TMS_INIT_ENABLE"] == "1"
    assert env["TMS_INIT_ENABLE_CPU_BACKUP"] == "1"
    assert os.path.exists(env["LD_PRELOAD"])
    assert os.path.basename(env["LD_PRELOAD"]).startswith("torch_memory_saver_hook_mode_preload")


def test_a_missing_preload_library_fails_instead_of_starting_without_offload(monkeypatch, tmp_path):
    import torch_memory_saver

    monkeypatch.setattr(torch_memory_saver, "__file__", str(tmp_path / "pkg" / "__init__.py"))

    with pytest.raises(FileNotFoundError, match="torch_memory_saver dynamic library"):
        Accelerator.offload_allocator_env({})


def test_engine_and_serving_default_to_passing_the_environment_through():
    args = SimpleNamespace(offload_train=True)

    assert accelerator.engine_actor_env(args, {"A": "1"}) == {"A": "1"}
    assert accelerator.serving_subprocess_env({"A": "1"}, visible_devices="0,1", colocate=True) == {"A": "1"}


def test_a_deviceless_driver_still_builds_the_actor_environments():
    """Runtime envs are built before the actors exist, so on the driver."""
    assert accelerator.initialize_accelerator() is None

    args = SimpleNamespace(offload_train=False)
    assert accelerator.train_actor_env(args, {}) == {}
    assert accelerator.engine_actor_env(args, {}) == {}


def test_a_vendor_backend_supplies_all_three_environments(monkeypatch):
    accelerator.register_accelerator("fake-vendor", _VendorAccelerator, lambda: False, priority=300)
    monkeypatch.setenv("VIME_ACCELERATOR", "fake-vendor")
    args = SimpleNamespace(offload_train=True)

    assert accelerator.train_actor_env(args, {}) == {"VENDOR_TRAIN": "1"}
    assert accelerator.engine_actor_env(args, {}) == {"VENDOR_ENGINE": "1"}
    assert accelerator.serving_subprocess_env(
        {"CUDA_VISIBLE_DEVICES": "0"}, visible_devices="3", colocate=False
    ) == {"VENDOR_VISIBLE_DEVICES": "3"}
