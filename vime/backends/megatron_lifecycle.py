"""Points in Megatron's startup and offload where a backend may differ.

A backend that needs vendor patches installed at an exact moment, or whose
memory saver behaves differently, names a replacement module through
``Accelerator.megatron_lifecycle_qualname()``. The default below is what the
GPU path does today, and importing this module imports neither Megatron nor
any vendor runtime.

Replacements are whole modules rather than a bag of capability flags on the
accelerator: "this backend's memory saver cannot be disabled" is not a
property of the device, it is how this function behaves for it.
"""

from __future__ import annotations

import importlib
import sys
from contextlib import nullcontext
from typing import Any


def before_import() -> None:
    """Run before Vime imports anything from Megatron."""
    return None


def after_args(args: Any) -> None:
    """Run once Megatron's arguments are fully resolved."""
    return None


def training_region(offload_train: bool):
    """Region whose allocations are offloaded and reloaded together."""
    return nullcontext()


def initialize_optimizer_state(optimizer: Any) -> None:
    """Materialize lazily created optimizer state inside the training region."""
    return None


def offload_disable_context(offload_train: bool):
    """Context in which offloaded memory is resident again, for weight export."""
    if not offload_train:
        return nullcontext()
    from torch_memory_saver import torch_memory_saver

    return torch_memory_saver.disable()


def current():
    """The lifecycle module for the backend this job runs on."""
    from vime.utils import accelerator

    qualname = accelerator.megatron_lifecycle_qualname()
    if qualname is None:
        return sys.modules[__name__]
    return importlib.import_module(qualname)
