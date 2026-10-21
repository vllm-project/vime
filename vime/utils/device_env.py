"""What a child process has to be told about the accelerator in use.

Three kinds of process need it: a training actor, a serving engine actor, and
the serving subprocess that actor launches. They are started by three
different subsystems, and the first two receive it through Ray's runtime env
while the third receives it through the process environment, but what has to
go in is one accelerator-specific question, so it is answered in one place.

A backend that needs different variables names a replacement module through
``Accelerator.device_env_qualname()``. The default below is what the GPU path
does today. Both actor environments are built before their actors exist,
which means on the driver, and a driver may own no device, so nothing here
may require one.
"""

from __future__ import annotations

import importlib
import os
import sys
from collections.abc import Mapping
from typing import Any


def train_actor_env(args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment a training actor needs, including the offload allocator."""
    env = dict(env_vars or {})
    if not getattr(args, "offload_train", False):
        return env
    return _offload_allocator_env(env)


def engine_actor_env(args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment a serving engine actor needs."""
    return dict(env_vars or {})


def serving_subprocess_env(base_env: Mapping[str, str], *, visible_devices: str, colocate: bool) -> dict[str, str]:
    """Environment the serving subprocess needs, given its device selection."""
    return dict(base_env)


def _offload_allocator_env(env_vars: Mapping[str, str]) -> dict[str, str]:
    """Hand the allocator to torch_memory_saver in the actor process.

    All three variables are required for it to take over, so a missing library
    raises rather than letting training start without offload.
    """
    import torch_memory_saver

    for name in (
        "torch_memory_saver_hook_mode_preload_cu13.abi3.so",
        "torch_memory_saver_hook_mode_preload_cu12.abi3.so",
        "torch_memory_saver_hook_mode_preload.abi3.so",
    ):
        dynlib_path = os.path.join(os.path.dirname(os.path.dirname(torch_memory_saver.__file__)), name)
        if os.path.exists(dynlib_path):
            break
    else:
        raise FileNotFoundError(
            "Cannot find torch_memory_saver dynamic library. Please make sure torch_memory_saver is properly installed."
        )

    env = dict(env_vars)
    env["LD_PRELOAD"] = dynlib_path
    env["TMS_INIT_ENABLE"] = "1"
    env["TMS_INIT_ENABLE_CPU_BACKUP"] = "1"
    return env


def current():
    """The child-environment module for the backend this job runs on."""
    from vime.utils import accelerator

    qualname = accelerator.device_env_qualname()
    if qualname is None:
        return sys.modules[__name__]
    return importlib.import_module(qualname)
