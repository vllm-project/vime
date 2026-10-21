"""Points in Megatron's startup where a backend may differ.

A backend that needs vendor patches installed at an exact moment names a
replacement module through ``Accelerator.megatron_lifecycle_qualname()``. The
default below does nothing, and importing this module imports neither Megatron
nor any vendor runtime.

The contract is two functions because that is all that is left: both are about
*when* a vendor layer may touch Megatron, which nothing outside the import
order can express. Offload used to be here too; it moved into the dependencies
instead, where Megatron exposes a buffer-allocation hook and the patched NPU
memory saver takes the allocator through the same preload shim as CUDA.
"""

from __future__ import annotations

import importlib
import sys
from typing import Any


def before_import() -> None:
    """Run before Vime imports anything from Megatron."""
    return None


def after_args(args: Any) -> None:
    """Run once Megatron's arguments are fully resolved."""
    return None


def current():
    """The lifecycle module for the backend this job runs on."""
    from vime.utils import accelerator

    qualname = accelerator.megatron_lifecycle_qualname()
    if qualname is None:
        return sys.modules[__name__]
    return importlib.import_module(qualname)
