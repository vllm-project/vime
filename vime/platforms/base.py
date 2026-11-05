"""Small contracts for behavior that genuinely differs by platform."""

from __future__ import annotations

from collections.abc import Mapping
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

    def pd_port_count(self, num_devices: int) -> int:
        return 1

    def pd_config(self, worker_type: str, bootstrap_port: int | None, parallel_config: Mapping):
        if worker_type == "prefill":
            assert bootstrap_port is not None, "Prefill requires disaggregation_bootstrap_port"
        role = {"prefill": "kv_producer", "decode": "kv_consumer"}[worker_type]
        return {"kv_connector": "NixlConnector", "kv_role": role}, bootstrap_port

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


@dataclass(frozen=True)
class CheckpointCapabilities:
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
