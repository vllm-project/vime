from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class NcclXferAvailability:
    available: bool
    reason: str | None = None


def get_nccl_xfer_availability() -> NcclXferAvailability:
    """Return whether a native NCCL Xfer bridge is available to Python.

    The vendored ``nccl_xfer`` tree currently exposes a C/CUDA API only. This
    shim prevents the opt-in backend from silently pretending to transfer
    payloads until a pybind/torch-extension bridge is added.
    """

    return NcclXferAvailability(
        available=False,
        reason="native NCCL Xfer Python bridge is not implemented",
    )


def reshard_with_window(*args, **kwargs):
    del args, kwargs
    raise NotImplementedError("native NCCL Xfer Python bridge is not implemented")
