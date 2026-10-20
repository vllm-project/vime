"""Naming the weight transport and its init-info class, without a device."""

import pytest

from vime.utils import accelerator
from vime.utils.accelerator.base import Accelerator
from vime.utils.accelerator.cuda import CUDAAccelerator


pytestmark = pytest.mark.unit


class _VendorAccelerator(CUDAAccelerator):
    name = "fake-vendor"

    @classmethod
    def weight_transfer_backend(cls, transport: str) -> str:
        return {"ipc": "vendor_ipc", "distributed": "vccl"}[transport]

    @classmethod
    def weight_transfer_init_info_qualname(cls, transport: str) -> str:
        return {"ipc": "types.SimpleNamespace", "distributed": "types.SimpleNamespace"}[transport]


@pytest.fixture(autouse=True)
def isolate_selection(monkeypatch):
    registry = accelerator._REGISTRY.copy()
    selected = accelerator._ACCELERATOR
    for name in ("VIME_ACCELERATOR", "MUSA_VISIBLE_DEVICES", "MUSA_PATCH_PATH", "SUPA_VISIBLE_DEVICES", "BIREN_HOME"):
        monkeypatch.delenv(name, raising=False)
    accelerator._REGISTRY.clear()
    accelerator.reset_accelerator()
    monkeypatch.setattr(accelerator, "is_musa_available", lambda: False)
    monkeypatch.setattr(accelerator, "is_supa_available", lambda: False)
    monkeypatch.setattr(accelerator, "_cuda_available", lambda: False)
    yield
    accelerator._REGISTRY.clear()
    accelerator._REGISTRY.update(registry)
    accelerator._ACCELERATOR = selected


def test_the_defaults_are_the_names_the_gpu_path_already_used():
    assert accelerator.weight_transfer_backend("ipc") == "ipc"
    assert accelerator.weight_transfer_backend("distributed") == "nccl"


def test_the_default_init_info_classes_are_vllms():
    assert Accelerator.weight_transfer_init_info_qualname("ipc").endswith(".IPCTrainerInitInfo")
    assert Accelerator.weight_transfer_init_info_qualname("distributed").endswith(".NCCLTrainerInitInfo")


def test_naming_a_transport_needs_no_device_and_no_runtime():
    """A host with no accelerator still resolves the names, as main's literals did."""
    assert accelerator.initialize_accelerator() is None

    assert accelerator.weight_transfer_backend("distributed") == "nccl"
    assert Accelerator.weight_transfer_backend("ipc") == "ipc"


def test_an_unknown_transport_is_rejected():
    for lookup in (accelerator.weight_transfer_backend, Accelerator.weight_transfer_init_info_qualname):
        with pytest.raises(ValueError, match="Unknown weight transport: 'bogus'"):
            lookup("bogus")


def test_a_vendor_backend_supplies_its_own_transport_names(monkeypatch):
    accelerator.register_accelerator("fake-vendor", _VendorAccelerator, lambda: False, priority=300)
    monkeypatch.setenv("VIME_ACCELERATOR", "fake-vendor")

    assert accelerator.weight_transfer_backend("ipc") == "vendor_ipc"
    assert accelerator.weight_transfer_backend("distributed") == "vccl"


def test_init_info_is_constructed_from_the_named_class(monkeypatch):
    accelerator.register_accelerator("fake-vendor", _VendorAccelerator, lambda: False, priority=300)
    monkeypatch.setenv("VIME_ACCELERATOR", "fake-vendor")

    info = accelerator.weight_transfer_init_info("ipc", rank=3, packed=True)

    assert (info.rank, info.packed) == (3, True)
