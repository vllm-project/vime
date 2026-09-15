"""Accelerator platform discovery and narrow capability providers.

``VIME_PLATFORM`` is the explicit override.  When it is not set, NPU detection
is lazy and failure-safe; CUDA is the default.
"""

from __future__ import annotations

import os
from functools import cache

from .base import (
    CheckpointCapabilities,
    Platform,
    RayResourceSpec,
    TrainingBootstrap,
    VLLMLaunchPlatformOps,
    WeightTransferPlatformOps,
)
from .cuda import create_cuda_platform
from .npu import create_npu_platform, detect_npu

_PLATFORM_FACTORIES = {
    "cuda": create_cuda_platform,
    "npu": create_npu_platform,
}


@cache
def get_platform(name: str) -> Platform:
    normalized = name.strip().lower()
    try:
        factory = _PLATFORM_FACTORIES[normalized]
    except KeyError as exc:
        available = ", ".join(_PLATFORM_FACTORIES)
        raise ValueError(f"Unknown Vime platform {name!r}; registered platforms: {available}") from exc
    return factory()


@cache
def _resolve_platform(override: str | None) -> Platform:
    if override:
        return get_platform(override)

    try:
        if detect_npu():
            return get_platform("npu")
    except Exception:  # noqa: BLE001 - a failed detector must not break imports
        pass
    return get_platform("cuda")


def current_platform() -> Platform:
    """Resolve the active platform without probing hardware at module import."""
    raw_override = os.environ.get("VIME_PLATFORM")
    override = raw_override.strip().lower() if raw_override and raw_override.strip() else None
    accelerator_override = os.environ.get("VIME_ACCELERATOR", "").strip().lower()
    if accelerator_override in {"npu", "cuda", "musa"}:
        accelerator_platform = "npu" if accelerator_override == "npu" else "cuda"
        if override in _PLATFORM_FACTORIES and override != accelerator_platform:
            raise ValueError(f"Conflicting VIME_PLATFORM={override!r} and VIME_ACCELERATOR={accelerator_override!r}")
        if override is None:
            override = accelerator_platform
    return _resolve_platform(override)


def reset_platform_cache() -> None:
    """Clear resolver/factory caches (primarily for tests and plugin registration)."""
    get_platform.cache_clear()
    _resolve_platform.cache_clear()


__all__ = [
    "CheckpointCapabilities",
    "Platform",
    "RayResourceSpec",
    "TrainingBootstrap",
    "VLLMLaunchPlatformOps",
    "WeightTransferPlatformOps",
    "current_platform",
    "get_platform",
    "reset_platform_cache",
]
