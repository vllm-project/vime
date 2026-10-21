"""Ascend NPU accelerator implementation.

Importing this module never imports ``torch_npu``. The runtime bootstrap in
``vime.utils.accelerator`` may attach ``torch.npu`` before selection.
"""

from __future__ import annotations

import os
from glob import glob
from typing import Any

import torch

from .torch_accelerator import TorchAccelerator


def npu_module() -> Any:
    return getattr(torch, "npu", None)


def has_visible_device() -> bool:
    """Whether an Ascend device is actually exposed to this process.

    Vime's own addition to the mirrored slime backend. slime treats an
    Ascend toolkit installation or a pre-set visibility variable as a request
    for this backend, but ASCEND_HOME_PATH is exported by every CANN install,
    including on hosts that only run CPU tooling or a CUDA job, and selecting
    NPU there loads the vendor stack into a process that wanted none of it.
    Checked before anything would import torch_npu.
    """
    if os.environ.get("ASCEND_RT_VISIBLE_DEVICES", "all").strip() in {"", "-1"}:
        return False
    return bool(glob("/dev/davinci[0-9]*"))


def is_npu_available() -> bool:
    # Vime checks for a device first: torch_npu is installed image-wide on a
    # CANN host, so torch.npu.is_available() alone would offer this backend on
    # a node that has the toolkit but no cards.
    if not has_visible_device():
        return False
    module = npu_module()
    checker = getattr(module, "is_available", None)
    return bool(module is not None and checker is not None and checker())


class NPUAccelerator(TorchAccelerator):
    name = "npu"
    device_type = "npu"
    communication_backend_name = "hccl"

    # Ray resolves a worker's visible-devices variable from the resource name
    # it was scheduled against, and only its own NPU resource maps to
    # ASCEND_RT_VISIBLE_DEVICES. Scheduling these devices as Ray GPUs would
    # leave every worker with no visible device.
    ray_device_key = "NPU"

    @property
    def visible_devices_env(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    # Everything below is Vime's own, not part of the slime backend this
    # module mirrors. Each one answers a seam Vime's core asks about, and
    # delegates anything that needs the Ascend stack to vime_plugins.ascend
    # so that selecting this backend still imports no vendor runtime.

    def device_uuid(self) -> str:
        # torch_npu device properties carry no uuid. vLLM Ascend derives a
        # host-IP plus physical-chip identity, and the receiving engine
        # derives it the same way, so reuse theirs rather than invent one.
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        return npu_generate_uuid()

    @classmethod
    def megatron_lifecycle_qualname(cls) -> str:
        return "vime_plugins.ascend.megatron_lifecycle"

    @classmethod
    def weight_transfer_backend(cls, transport: str) -> str:
        names = {"ipc": "npu_ipc", "distributed": "hccl"}
        if transport not in names:
            raise ValueError(f"Unknown weight transport: {transport!r}")
        return names[transport]

    @classmethod
    def weight_transfer_init_info_qualname(cls, transport: str) -> str:
        names = {
            "ipc": "vime_plugins.ascend.weight_transfer.npu_ipc_trainer_init_info",
            "distributed": "vime_plugins.ascend.weight_transfer.hccl_trainer_init_info",
        }
        if transport not in names:
            raise ValueError(f"Unknown weight transport: {transport!r}")
        return names[transport]

    @classmethod
    def device_env_qualname(cls) -> str:
        return "vime_plugins.ascend.device_env"

    def _module(self) -> Any:
        module = npu_module()
        if module is None:
            raise RuntimeError("Ascend NPU backend requires torch_npu and a runtime that exposes torch.npu")
        return module

    def is_available(self) -> bool:
        return is_npu_available()

    def weight_update_backend(self, default: str = "nccl") -> str:
        return "cpu:gloo,npu:hccl" if default == "nccl" else default

    def distributed_device_id(self, index: int | str | torch.device | None = None) -> None:
        # HCCL selects the local device through torch.npu.set_device/LOCAL_RANK.
        return None

    def empty_cache(self) -> None:
        module = self._module()
        is_capturing = getattr(module, "is_current_stream_capturing", None)
        if is_capturing is not None and is_capturing():
            return
        module.empty_cache()

    def supports(self, capability: str) -> bool:
        if capability in {"cuda_int4_extension", "nvml_affinity", "vllm_fp8_utils", "triton_kernels"}:
            return False
        if capability == "requires_cpu_initialization":
            return True
        if capability == "bf16":
            checker = getattr(self._module(), "is_bf16_supported", None)
            return bool(checker and checker())
        return super().supports(capability)
