"""How Ray advertises, schedules and addresses the accelerator in use.

Ray models an accelerator as a resource name plus a visible-devices variable
and resolves the second from the first, so a backend scheduled under the wrong
resource name starts with no visible device. These helpers read the name off
the accelerator rather than assuming Ray's own GPU resource; the default
remains ``GPU``, so the behavior on a GPU host is unchanged.
"""

from __future__ import annotations

from vime.utils import accelerator


def resource_name() -> str:
    return accelerator.ray_device_key()


def bundle(device_count: float = 1, cpu_count: float = 1) -> dict[str, float]:
    return {accelerator.ray_device_key(): device_count, "CPU": cpu_count}


def actor_options(fraction: float) -> dict[str, object]:
    if accelerator.ray_uses_native_gpu_resource():
        return {"num_gpus": fraction}
    # Pin num_gpus to 0: Ray would otherwise also reserve its own GPU
    # resource, which this accelerator does not provide.
    return {"num_gpus": 0, "resources": {accelerator.ray_device_key(): fraction}}


def accelerator_ids() -> list[str]:
    import ray

    if accelerator.ray_uses_native_gpu_resource():
        ids = ray.get_gpu_ids()
    else:
        ids = ray.get_runtime_context().get_accelerator_ids().get(accelerator.ray_device_key(), [])
    return [str(device_id) for device_id in ids]
