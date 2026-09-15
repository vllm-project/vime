"""Ascend NPU implementation of the Vime platform contracts."""

from __future__ import annotations

import importlib
import logging
import os
from contextlib import nullcontext
from glob import glob
from typing import Any

from vime.utils import accelerator
from vime.utils.accelerator.torch_accelerator import TorchAccelerator

from .base import (
    CheckpointCapabilities,
    Platform,
    RayResourceSpec,
    TrainingBootstrap,
    VLLMLaunchPlatformOps,
    WeightTransferPlatformOps,
)

logger = logging.getLogger(__name__)


class NPUAccelerator(TorchAccelerator):
    name = "npu"
    device_type = "npu"
    communication_backend_name = "hccl"

    def _module(self):
        return getattr(importlib.import_module("torch"), "npu", None)

    @property
    def visible_devices_env(self) -> str:
        return "ASCEND_RT_VISIBLE_DEVICES"

    def distributed_device_id(self, index=None):
        # Preserve lazy HCCL initialization instead of CUDA's eager device binding.
        return None

    def set_allocator_expandable_segments(self) -> bool:
        # NPU allocator policy belongs to the existing TMS/runtime-env hooks.
        return False


def register_npu_accelerator() -> None:
    accelerator.register_accelerator(
        "npu",
        NPUAccelerator,
        is_available=lambda: os.environ.get("VIME_PLATFORM", "").strip().lower() != "cuda"
        and NPUAccelerator().is_available(),
        priority=300,
        communication_backends=("hccl",),
    )


def detect_npu() -> bool:
    """Return whether a usable NPU is visible, without leaking probe errors."""
    # Do not import torch/torch_npu on an unselected CUDA host merely because
    # torch_npu happens to be installed.  Its import has process-wide monkey
    # patch side effects.  An explicit VIME_PLATFORM=npu override bypasses
    # detection, while automatic selection first requires an exposed device.
    if not (os.path.exists("/dev/davinci_manager") or glob("/dev/davinci[0-9]*")):
        return False
    try:
        torch = importlib.import_module("torch")
        if getattr(torch, "npu", None) is None:
            importlib.import_module("torch_npu")
        npu = getattr(torch, "npu", None)
        return bool(npu is not None and npu.is_available())
    except Exception:  # noqa: BLE001 - detection must be safe on non-NPU hosts
        return False


def _ensure_torch_npu() -> None:
    importlib.import_module("torch_npu")


def _install_safe_empty_cache() -> None:
    """Preserve the Ascend allocator guard required by MindSpeed/TMS callers."""
    torch = importlib.import_module("torch")
    original_empty_cache = torch.npu.empty_cache
    if not getattr(original_empty_cache, "_vime_safe_empty_cache", False):

        def _safe_empty_cache(_original=original_empty_cache) -> None:
            try:
                _original()
            except RuntimeError:
                pass

        _safe_empty_cache._vime_safe_empty_cache = True
        torch.npu.empty_cache = _safe_empty_cache

    # Some shared dependencies still call the CUDA spelling after torch_npu's
    # compatibility patching. Keep that alias local to NPU-bootstrapped jobs.
    torch.cuda.empty_cache = torch.npu.empty_cache


def _cann_python_site_packages() -> str | None:
    candidates: list[str] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        base = os.environ.get(env_key)
        if not base:
            continue
        candidates.extend(
            [
                os.path.join(base, "python", "site-packages"),
                os.path.normpath(os.path.join(base, "..", "python", "site-packages")),
            ]
        )
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
    for path in candidates:
        if os.path.isdir(os.path.join(path, "acl")):
            return path
    return None


def _prepend_pythonpath(env: dict[str, str], *paths: str) -> None:
    existing = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
    existing_parts = {part for part in existing.split(os.pathsep) if part}
    prefix_parts = [path for path in paths if path and path not in existing_parts]
    if prefix_parts:
        env["PYTHONPATH"] = os.pathsep.join([*prefix_parts, existing] if existing else prefix_parts)


class NpuRayResourceSpec(RayResourceSpec):
    def __init__(self) -> None:
        super().__init__(
            resource_name="NPU",
            visible_devices_env="ASCEND_RT_VISIBLE_DEVICES",
            uses_ray_gpu_resource=False,
        )

    def train_runtime_env(self, args: Any, env_vars=None) -> dict[str, str]:
        env = dict(env_vars or {})
        if not (getattr(args, "offload_train", False) and getattr(args, "train_backend", None) == "megatron"):
            return env

        env["TMS_HOOK_MODE"] = "torch"
        env["TMS_REGION_TAG"] = "training"
        env["TMS_ENABLE_CPU_BACKUP"] = "1"
        if getattr(args, "colocate", False):
            env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        cann_python_path = _cann_python_site_packages()
        if cann_python_path is not None:
            _prepend_pythonpath(env, cann_python_path)
        return env

    def rollout_runtime_env(self, args: Any, env_vars=None) -> dict[str, str]:
        env = dict(env_vars or {})
        cann_python_path = _cann_python_site_packages()
        if cann_python_path is not None:
            _prepend_pythonpath(env, cann_python_path)
        if getattr(args, "colocate", False):
            env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        env["VLLM_USE_AOT_COMPILE"] = "0"
        return env


class NpuWeightTransferPlatformOps(WeightTransferPlatformOps):
    def current_device_uuid(self) -> str:
        # Reuse vLLM Ascend's canonical host-IP/physical-chip identifier so
        # trainer and receiver always use the exact same mapping.
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import npu_generate_uuid

        return npu_generate_uuid()

    def backend(self, backend: str) -> str:
        return {"ipc": "npu_ipc", "nccl": "hccl"}.get(backend, backend)

    def trainer_init_info(self, *, colocate: bool, **kwargs):
        _ensure_torch_npu()
        from vllm.plugins import load_general_plugins

        # Trainers, unlike vLLM workers, may not have loaded general plugins yet.
        load_general_plugins()
        if colocate:
            from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import NPUIPCTrainerInitInfo

            return NPUIPCTrainerInitInfo(**kwargs)
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerInitInfo

        return HCCLTrainerInitInfo(**kwargs)


class NpuVLLMLaunchPlatformOps(VLLMLaunchPlatformOps):
    def subprocess_env(self, base_env, *, visible_devices: str, colocate: bool) -> dict[str, str]:
        env = dict(base_env)
        env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
        env.pop("CUDA_VISIBLE_DEVICES", None)
        env.pop("HIP_VISIBLE_DEVICES", None)
        env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
        env["VLLM_USE_AOT_COMPILE"] = "0"
        cann_python_path = _cann_python_site_packages()
        if cann_python_path is not None:
            _prepend_pythonpath(env, cann_python_path)
        if colocate:
            env["PYTORCH_NPU_ALLOC_CONF"] = "expandable_segments:False"
        return env


class NpuTrainingBootstrap(TrainingBootstrap):
    def __init__(self) -> None:
        self._bootstrapping = False
        self._bootstrapped = False

    def bootstrap(self) -> None:
        if self._bootstrapped or self._bootstrapping:
            return
        self._bootstrapping = True
        try:
            _ensure_torch_npu()
            # Select NPU before MegatronAdaptor can make torch.cuda appear available.
            register_npu_accelerator()
            selected = accelerator.get_accelerator()
            if selected.name != "npu":
                raise RuntimeError(f"NPU bootstrap cannot use an already selected {selected.name!r} accelerator")
            _install_safe_empty_cache()
            # MegatronAdaptor must install its pre-patches before any Megatron module
            # is imported. Apply the NPU attention override afterwards.
            importlib.import_module("megatron_adaptor")
            importlib.import_module("vime.backends.megatron_utils.npu_attention_patch")
        except Exception:
            # A failed bootstrap may be retried after the runtime environment is
            # corrected; never leave a partially initialized success marker.
            raise
        else:
            self._bootstrapped = True
        finally:
            self._bootstrapping = False

    def repatch(self, args: Any) -> None:
        features_manager = importlib.import_module(
            "megatron_adaptor.features_manager.features_manager"
        ).FeaturesManager
        full_args = importlib.import_module("megatron_adaptor.utils.args_utils").get_full_args()
        for key, value in vars(args).items():
            setattr(full_args, key, value)
        features_manager.remove_patches()
        features_manager.apply_features_pre_patches(full_args)
        features_manager.apply_features_patches(full_args)
        # Repatch may replace attention again; importing a cached module alone
        # does not reinstall Vime's existing override.
        attention = importlib.import_module("vime.backends.megatron_utils.npu_attention_patch")
        attention.DotProductAttention.forward = attention.npu_dot_product_attention_forward

    def adjust_tp_partition_dim(self, name: str, partition_dim: int) -> int:
        if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
            return 0
        return partition_dim

    def training_context(self, offload_train: bool):
        if not offload_train:
            return nullcontext()
        from torch_memory_saver import torch_memory_saver

        return torch_memory_saver.region(tag="training", enable_cpu_backup=True)

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        """Create lazy optimizer state before leaving the training memory pool."""
        if optimizer is None:
            return
        for opt in getattr(optimizer, "chained_optimizers", [optimizer]):
            if opt.optimizer is not None and opt.init_state_fn is not None:
                opt.init_state_fn(opt.optimizer, opt.config)


class NpuCheckpointCapabilities(CheckpointCapabilities):
    def patch_default_planner(self, default_planner: Any) -> None:
        if not hasattr(default_planner, "_validate_global_plan"):
            return

        def _validate_global_plan(global_plan, metadata):
            logger.info("[NPU checkpoint] Skipping validate_access_integrity")
            return True

        default_planner._validate_global_plan = _validate_global_plan


def create_npu_platform() -> Platform:
    # Ray workers also resolve the platform without importing Megatron.
    register_npu_accelerator()
    return Platform(
        name="npu",
        ray=NpuRayResourceSpec(),
        weight_transfer=NpuWeightTransferPlatformOps(),
        vllm=NpuVLLMLaunchPlatformOps(),
        megatron=NpuTrainingBootstrap(),
        checkpoint=NpuCheckpointCapabilities(default_megatron_to_hf_mode="bridge"),
    )
