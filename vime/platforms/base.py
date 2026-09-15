"""Small contracts for behavior that genuinely differs by platform."""

from __future__ import annotations

import os
from collections.abc import Mapping
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class RayResourceSpec:
    """How one accelerator is represented and addressed by Ray."""

    resource_name: str
    visible_devices_env: str
    uses_ray_gpu_resource: bool

    def bundle_resources(self, device_count: float = 1, cpu_count: float = 1) -> dict[str, float]:
        return {self.resource_name: device_count, "CPU": cpu_count}

    def actor_options(self, fraction: float) -> dict[str, object]:
        if self.uses_ray_gpu_resource:
            return {"num_gpus": fraction}
        return {"resources": {self.resource_name: fraction}}

    def accelerator_ids(self) -> list[str]:
        import ray

        if self.uses_ray_gpu_resource:
            ids = ray.get_gpu_ids()
        else:
            ids = ray.get_runtime_context().get_accelerator_ids().get(self.resource_name, [])
        return [str(device_id) for device_id in ids]

    def local_device_id(self) -> int | str:
        device_ids = self.accelerator_ids()
        if not device_ids:
            raise RuntimeError(f"No {self.resource_name} accelerator IDs are assigned to this Ray actor")

        assigned_id = device_ids[0]
        visible_devices = os.environ.get(self.visible_devices_env)
        if visible_devices is None:
            try:
                return int(assigned_id)
            except ValueError:
                return assigned_id

        visible_ids = [value.strip() for value in visible_devices.split(",") if value.strip()]
        try:
            return visible_ids.index(assigned_id)
        except ValueError as exc:
            raise RuntimeError(
                f"Ray assigned {self.resource_name} id {assigned_id}, but it is absent from "
                f"{self.visible_devices_env}={visible_devices!r}"
            ) from exc

    def train_runtime_env(
        self,
        args: Any,
        env_vars: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        return dict(env_vars or {})

    def rollout_runtime_env(
        self,
        args: Any,
        env_vars: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        return dict(env_vars or {})


class WeightTransferPlatformOps:
    """Select vendor backends without changing the shared trainer lifecycle."""

    def backend(self, backend: str) -> str:
        return backend

    def trainer_init_info(self, *, colocate: bool, **kwargs):
        if colocate:
            from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

            return IPCTrainerInitInfo(**kwargs)
        from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

        return NCCLTrainerInitInfo(**kwargs)


class VLLMLaunchPlatformOps:
    """Platform additions to the common vLLM launch command and environment."""

    def subprocess_env(
        self,
        base_env: Mapping[str, str],
        *,
        visible_devices: str,
        colocate: bool,
    ) -> dict[str, str]:
        return dict(base_env)


class TrainingBootstrap:
    """Lazy Megatron/vendor initialization hooks."""

    def bootstrap(self) -> None:
        return None

    def repatch(self, args: Any) -> None:
        return None

    def adjust_tp_partition_dim(self, name: str, partition_dim: int) -> int:
        return partition_dim

    def training_context(self, offload_train: bool):
        return nullcontext()

    def initialize_optimizer_state(self, optimizer: Any) -> None:
        return None


@dataclass(frozen=True)
class CheckpointCapabilities:
    default_megatron_to_hf_mode: str = "raw"

    def patch_default_planner(self, default_planner: Any) -> None:
        return None


@dataclass(frozen=True)
class Platform:
    """Aggregate only the providers whose semantics differ on Ascend."""

    name: str
    ray: RayResourceSpec
    weight_transfer: WeightTransferPlatformOps
    vllm: VLLMLaunchPlatformOps
    megatron: TrainingBootstrap
    checkpoint: CheckpointCapabilities

    @property
    def is_npu(self) -> bool:
        return self.name == "npu"
