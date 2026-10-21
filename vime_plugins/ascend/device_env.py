"""What an Ascend child process needs in its environment.

Reached through NPUAccelerator's child-environment hooks. Nothing here
imports torch_npu or CANN: these build an environment for a process that has
not started yet.

Launch-time variables (PYTHONPATH, LD_LIBRARY_PATH, HCCL port ranges) are not
set here. They have to be exported before ``ray start`` so the raylet and
every actor it spawns inherit them, which is before any of this runs, so the
image and the run scripts own them.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from typing import Any

_ALLOC_CONF = "expandable_segments:False"


def _cann_site_packages() -> str | None:
    """Locate the CANN python bindings, which a child imports as ``acl``."""
    candidates: list[str] = []
    for env_key in ("ASCEND_TOOLKIT_HOME", "ASCEND_HOME_PATH"):
        base = os.environ.get(env_key)
        if not base:
            continue
        candidates.append(os.path.join(base, "python", "site-packages"))
        candidates.append(os.path.normpath(os.path.join(base, "..", "python", "site-packages")))
    candidates.append("/usr/local/Ascend/ascend-toolkit/latest/python/site-packages")
    for path in candidates:
        if os.path.isdir(os.path.join(path, "acl")):
            return path
    return None


def _with_cann_and_allocator(env: dict[str, str]) -> dict[str, str]:
    cann_python_path = _cann_site_packages()
    if cann_python_path is not None:
        existing = env.get("PYTHONPATH", os.environ.get("PYTHONPATH", ""))
        if cann_python_path not in {part for part in existing.split(os.pathsep) if part}:
            env["PYTHONPATH"] = os.pathsep.join([cann_python_path, existing] if existing else [cann_python_path])
    # NPU memory pools require this even outside colocate: vLLM sleep mode
    # and training offload both use pools incompatible with expandable segments.
    env["PYTORCH_NPU_ALLOC_CONF"] = _ALLOC_CONF
    return env


def train_actor_env(args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(env_vars or {})
    if not getattr(args, "offload_train", False):
        return env
    # The NPU build of torch_memory_saver hooks the allocator from Python
    # instead of through an LD_PRELOAD shim, so none of the CUDA preload
    # contract applies here.
    env["TMS_HOOK_MODE"] = "torch"
    return _with_cann_and_allocator(env)


def engine_actor_env(args: Any, env_vars: Mapping[str, str] | None = None) -> dict[str, str]:
    env = dict(env_vars or {})
    env["VLLM_USE_AOT_COMPILE"] = "0"
    return _with_cann_and_allocator(env)


def serving_subprocess_env(base_env: Mapping[str, str], *, visible_devices: str, colocate: bool) -> dict[str, str]:
    env = dict(base_env)
    # The caller set the CUDA selection before asking; replace it rather than
    # leave both, so the subprocess cannot be pointed at two device sets.
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.pop("CUDA_VISIBLE_DEVICES", None)
    env.pop("HIP_VISIBLE_DEVICES", None)
    env["ASCEND_RT_VISIBLE_DEVICES"] = visible_devices
    env["VLLM_USE_AOT_COMPILE"] = "0"
    return _with_cann_and_allocator(env)
