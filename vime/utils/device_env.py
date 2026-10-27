"""What a child process has to be told about the accelerator in use.

Three kinds of process need it: a training actor, a serving engine actor, and
the serving subprocess that actor launches. They are started by three
different subsystems, and the first two receive it through Ray's runtime env
while the third receives it through the process environment, but what has to
go in is one accelerator-specific question, so it is answered in one place.

A backend that needs extra variables names a replacement module through
``Accelerator.device_env_qualname()``. The default adds nothing: whatever the
GPU path already set, including the torch_memory_saver preload contract, is
set by the caller and stays where it was. Both actor environments are built
before their actors exist, which means on the driver, and a driver may own no
device, so nothing here may require one.
"""

from __future__ import annotations

import importlib
import sys
from collections.abc import Mapping
from typing import Any


def train_actor_env(args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment a training actor needs beyond what the caller already set."""
    return dict(env_vars or {})


def engine_actor_env(args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
    """Environment a serving engine actor needs."""
    return dict(env_vars or {})


def serving_subprocess_env(base_env: Mapping[str, str], *, visible_devices: str, colocate: bool) -> dict[str, str]:
    """Environment the serving subprocess needs, given its device selection."""
    return dict(base_env)


def current():
    """The child-environment module for the backend this job runs on."""
    from vime.utils import accelerator

    qualname = accelerator.device_env_qualname()
    if qualname is None:
        return sys.modules[__name__]
    return importlib.import_module(qualname)
