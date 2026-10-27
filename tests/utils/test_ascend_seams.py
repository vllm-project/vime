"""The Ascend backend's answers to the core seams, and what they cost.

These run anywhere: they check how the backend describes itself and that
everything it names exists, never that a device works.
"""

import importlib
import sys
from types import SimpleNamespace

import pytest

from vime.backends import megatron_lifecycle
from vime.ray import resources
from vime.utils import accelerator
from vime.utils.accelerator.npu import NPUAccelerator
from vime_plugins.ascend import device_env


pytestmark = pytest.mark.unit

ASCEND_MODULES = (
    "vime_plugins.ascend",
    "vime_plugins.ascend.device_env",
    "vime_plugins.ascend.megatron_lifecycle",
    "vime_plugins.ascend.weight_transfer",
)


@pytest.fixture
def selected_npu(monkeypatch):
    """Describe NPU as the job's backend without needing one locally."""
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    monkeypatch.setattr(accelerator, "_register_builtin_backends", lambda: None)
    accelerator.register_accelerator("npu", NPUAccelerator, lambda: False, priority=250)
    monkeypatch.setenv("VIME_ACCELERATOR", "npu")
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected


def test_npu_is_scheduled_under_rays_npu_resource(selected_npu):
    """Ray maps only its own NPU resource to ASCEND_RT_VISIBLE_DEVICES."""
    assert accelerator.ray_device_key() == "NPU"
    assert resources.bundle() == {"NPU": 1, "CPU": 1}
    assert resources.actor_options(1) == {"num_gpus": 0, "resources": {"NPU": 1}}


def test_npu_names_its_own_weight_transfer_backends(selected_npu):
    assert accelerator.weight_transfer_backend("ipc") == "npu_ipc"
    assert accelerator.weight_transfer_backend("distributed") == "hccl"


@pytest.mark.parametrize("transport", ["ipc", "distributed"])
def test_every_named_init_info_factory_exists(transport):
    """A name that does not resolve must fail now, not at transfer time."""
    qualname = NPUAccelerator.weight_transfer_init_info_qualname(transport)
    module_name, _, attr = qualname.rpartition(".")

    assert callable(getattr(importlib.import_module(module_name), attr))


def test_the_ascend_lifecycle_implements_the_whole_default_surface():
    """A missing hook would silently fall back to GPU behavior on NPU.

    Derived from the default module rather than listed, so a hook added or
    dropped there has to be answered here too.
    """
    ascend = importlib.import_module(NPUAccelerator.megatron_lifecycle_qualname())
    required = tuple(
        name
        for name, value in vars(megatron_lifecycle).items()
        if not name.startswith("_")
        and callable(value)
        and name != "current"
        and getattr(value, "__module__", None) == megatron_lifecycle.__name__
    )

    assert [name for name in required if not callable(getattr(ascend, name, None))] == []


def test_importing_the_ascend_package_loads_no_vendor_runtime():
    """Selecting or describing the backend must not drag in the stack."""
    vendor = ("vllm_ascend", "megatron_adaptor", "mindspeed")
    for name in list(sys.modules):
        if name.split(".")[0] in vendor:
            pytest.skip(f"{name} already imported by another test")

    for module in ASCEND_MODULES:
        importlib.import_module(module)

    assert [name for name in sys.modules if name.split(".")[0] in vendor] == []


def test_npu_training_adds_only_the_hook_mode_to_the_shared_contract(monkeypatch):
    """LD_PRELOAD and the TMS_INIT_* pair stay with the caller that set them.

    The patched NPU memory saver takes the allocator through the same preload
    shim as CUDA, so this only has to name the hook mode and the NPU allocator
    policy, and must preserve what the caller already put in.
    """
    monkeypatch.setattr(device_env, "_cann_site_packages", lambda: None)
    inherited = {
        "KEEP": "1",
        "LD_PRELOAD": "/set/by/the/caller.so",
        "TMS_INIT_ENABLE": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
    }

    env = device_env.train_actor_env(SimpleNamespace(offload_train=True), inherited)

    assert env["TMS_HOOK_MODE"] == "preload"
    assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
    assert env["LD_PRELOAD"] == "/set/by/the/caller.so"
    assert env["TMS_INIT_ENABLE"] == "1"
    assert env["KEEP"] == "1"
    assert inherited["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:True"
    assert device_env.train_actor_env(SimpleNamespace(offload_train=False), inherited) == inherited


def test_a_backend_without_an_override_gets_the_shared_default(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    monkeypatch.setattr(accelerator, "_register_builtin_backends", lambda: None)
    # On a CANN host an Ascend toolkit alone resolves selection to NPU, so
    # the Ascend variables have to go for this to describe no backend.
    monkeypatch.delenv("VIME_ACCELERATOR", raising=False)
    monkeypatch.setenv("ASCEND_RT_VISIBLE_DEVICES", "-1")
    try:
        assert accelerator.Accelerator.megatron_lifecycle_qualname() is None
        assert megatron_lifecycle.current() is sys.modules["vime.backends.megatron_lifecycle"]
    finally:
        accelerator._REGISTRY.clear()
        accelerator._REGISTRY.update(registry)
        accelerator._ACCELERATOR = selected
