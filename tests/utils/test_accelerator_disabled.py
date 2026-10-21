"""Asking for no accelerator at all, on a host that has one."""

import pytest

from vime.backends import megatron_lifecycle
from vime.utils import accelerator


pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def disabled(monkeypatch):
    selected = accelerator._ACCELERATOR
    accelerator.reset_accelerator()
    monkeypatch.setenv("VIME_ACCELERATOR", "none")
    yield
    accelerator._ACCELERATOR = selected


def test_no_backend_is_selected_even_where_one_is_available():
    """Refused without touching any vendor visibility variable.

    Hiding devices instead would hide them from the vendor runtime too: on an
    Ascend host that breaks torch_npu's own lazy initialization.
    """
    assert accelerator.accelerator_disabled() is True
    assert accelerator.initialize_accelerator() is None


def test_descriptions_fall_back_to_the_neutral_defaults():
    assert accelerator.ray_device_key() == "GPU"
    assert accelerator.ray_uses_native_gpu_resource() is True
    assert accelerator.weight_transfer_backend("distributed") == "nccl"
    assert accelerator.megatron_lifecycle_qualname() is None
    assert megatron_lifecycle.current().__name__ == "vime.backends.megatron_lifecycle"


def test_asking_for_a_real_device_says_why_it_cannot():
    with pytest.raises(RuntimeError, match="VIME_ACCELERATOR=none disables accelerator selection"):
        accelerator.get_accelerator()


def test_an_already_selected_backend_still_wins():
    """Only selection is disabled; a backend a caller installed stands."""

    class Fake(accelerator.CUDAAccelerator):
        name = "fake"

        def is_available(self) -> bool:
            return True

    accelerator.set_accelerator(Fake())

    assert accelerator.initialize_accelerator().name == "fake"
    assert accelerator.get_accelerator().name == "fake"
