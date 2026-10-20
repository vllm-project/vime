"""The Ascend backend's answers to the core seams, and what they cost.

These run anywhere: they check how the backend describes itself and that
everything it names exists, never that a device works.
"""

import importlib
import sys

import pytest

from vime.backends import megatron_lifecycle
from vime.ray import resources
from vime.utils import accelerator
from vime.utils.accelerator.npu import NPUAccelerator


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
    assert resources.resource_name() == "NPU"
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
    """A missing hook would silently fall back to GPU behavior on NPU."""
    ascend = importlib.import_module(NPUAccelerator.megatron_lifecycle_qualname())
    required = ("before_import", "after_args", "training_region", "initialize_optimizer_state",
                "offload_disable_context")

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


def test_npu_offload_never_suspends_the_memory_saver():
    """The NPU memory saver has no disable(); its offload is region-scoped."""
    ascend = importlib.import_module(NPUAccelerator.megatron_lifecycle_qualname())

    assert type(ascend.offload_disable_context(True)).__name__ == "nullcontext"
    assert type(ascend.training_region(False)).__name__ == "nullcontext"


def test_the_default_lifecycle_still_suspends_it_for_gpus():
    """The GPU path keeps torch_memory_saver.disable() during weight export."""
    assert type(megatron_lifecycle.offload_disable_context(False)).__name__ == "nullcontext"
    assert type(megatron_lifecycle.offload_disable_context(True)).__name__ != "nullcontext"


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
