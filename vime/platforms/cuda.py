"""Default CUDA platform assembled from the shared CUDA-compatible behavior."""

from __future__ import annotations

from .base import (
    CheckpointCapabilities,
    Platform,
    RayResourceSpec,
    TrainingBootstrap,
    VLLMLaunchPlatformOps,
    WeightTransferPlatformOps,
)


def create_cuda_platform() -> Platform:
    return Platform(
        name="cuda",
        ray=RayResourceSpec(
            resource_name="GPU", visible_devices_env="CUDA_VISIBLE_DEVICES", uses_ray_gpu_resource=True
        ),
        weight_transfer=WeightTransferPlatformOps(),
        vllm=VLLMLaunchPlatformOps(),
        megatron=TrainingBootstrap(),
        checkpoint=CheckpointCapabilities(),
    )
