"""Direct Ascend plugin tests; no shared dispatch hooks are needed."""

import importlib
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from vime.utils.accelerator.npu import NPUAccelerator
from vime_plugins.ascend import device_env


pytestmark = pytest.mark.unit


@pytest.mark.parametrize("transport", ["ipc", "distributed"])
def test_every_named_init_info_factory_exists(transport):
    """A name that does not resolve must fail now, not at transfer time."""
    qualname = NPUAccelerator.weight_transfer_init_info_qualname(transport)
    module_name, _, attr = qualname.rpartition(".")

    assert callable(getattr(importlib.import_module(module_name), attr))


def test_importing_the_ascend_package_loads_no_vendor_runtime():
    """Importing the implementation must neither select nor register a backend."""
    program = r"""
import importlib
import importlib.abc
import sys

class RejectVendorImports(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.split(".")[0] in {"torch_npu", "vllm_ascend", "megatron_adaptor", "mindspeed"}:
            raise AssertionError(f"Unexpected vendor import: {fullname}")
        return None

sys.meta_path.insert(0, RejectVendorImports())
from vime.utils import accelerator
registry = dict(accelerator._REGISTRY)
selected = accelerator._ACCELERATOR
for name in (
    "vime.utils.accelerator.npu",
    "vime_plugins.ascend",
    "vime_plugins.ascend.device_env",
    "vime_plugins.ascend.megatron_lifecycle",
    "vime_plugins.ascend.weight_transfer",
):
    importlib.import_module(name)
assert accelerator._REGISTRY == registry
assert accelerator._ACCELERATOR is selected
"""
    env = {**os.environ, "TORCH_DEVICE_BACKEND_AUTOLOAD": "0"}
    result = subprocess.run([sys.executable, "-c", program], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr


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


@pytest.mark.parametrize("colocate", [False, True])
def test_npu_serving_pools_override_expandable_segments(monkeypatch, colocate):
    """Sleep-mode pools need this policy even when training is on other devices."""
    monkeypatch.setattr(device_env, "_cann_site_packages", lambda: None)
    inherited = {
        "KEEP": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:True",
        "CUDA_VISIBLE_DEVICES": "0,1",
        "HIP_VISIBLE_DEVICES": "0,1",
    }
    original = dict(inherited)

    actor_env = device_env.engine_actor_env(SimpleNamespace(colocate=colocate), inherited)
    subprocess_env = device_env.serving_subprocess_env(inherited, visible_devices="4,5", colocate=colocate)

    for env in (actor_env, subprocess_env):
        assert env["PYTORCH_NPU_ALLOC_CONF"] == "expandable_segments:False"
        assert env["VLLM_USE_AOT_COMPILE"] == "0"
        assert env["KEEP"] == "1"
    assert subprocess_env["ASCEND_RT_VISIBLE_DEVICES"] == "4,5"
    assert "CUDA_VISIBLE_DEVICES" not in subprocess_env
    assert "HIP_VISIBLE_DEVICES" not in subprocess_env
    assert inherited == original
