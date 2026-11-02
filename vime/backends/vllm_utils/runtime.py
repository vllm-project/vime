"""Start vLLM with its own Python using an ordinary subprocess."""

import dataclasses
import json
import os
import re
import subprocess
import sys
import time

import psutil
from pydantic import TypeAdapter
from pydantic_core import to_jsonable_python


def prepare_server_environment(environ):
    """Remove unsupported allocator settings and avoid Mooncake/HTTP port collisions."""
    for name in ("PYTORCH_CUDA_ALLOC_CONF", "PYTORCH_ALLOC_CONF"):
        value = environ.get(name)
        if value is None:
            continue
        value = re.sub(r"(^|,)\s*expandable_segments\s*:\s*(?:True|False|true|false|1|0)(?=,|$)", "", value).strip(
            ", "
        )
        if value:
            environ[name] = value
        else:
            environ.pop(name, None)

    # Preserve explicit ranges, including Mooncake's legacy PRC spelling.
    if not any(f"MC_{bound}_{spelling}_PORT" in environ for bound in ("MIN", "MAX") for spelling in ("RPC", "PRC")):
        environ["MC_MIN_PRC_PORT"] = "27000"
        environ["MC_MAX_PRC_PORT"] = "28000"


def _json_default(value):
    if dataclasses.is_dataclass(value):
        return TypeAdapter(type(value)).dump_python(
            value,
            mode="json",
            include={field.name for field in dataclasses.fields(value) if field.init},
            exclude_defaults=True,
        )
    return to_jsonable_python(value)


def launch_server_subprocess(executable, server_args, *, environ=None):
    # JSON keeps the child independent of the parent's imported Torch classes.
    payload = json.dumps(
        {
            "server_args": server_args,
            "parent_pid": os.getpid(),
            "config_keys": [key for key, value in server_args.items() if dataclasses.is_dataclass(value)],
        },
        default=_json_default,
    )
    environ = os.environ.copy() if environ is None else environ.copy()
    prepare_server_environment(environ)
    process = subprocess.Popen(
        [executable, "-m", "vime.backends.vllm_utils.runtime"],
        stdin=subprocess.PIPE,
        env=environ,
        text=True,
    )
    try:
        process.stdin.write(payload)
        process.stdin.close()
    except BaseException:
        process.kill()
        process.wait()
        raise
    return process


def terminate_server_process(process, timeout=30):
    """Wait for this server's process tree to exit before reusing its GPUs."""
    if isinstance(process, subprocess.Popen):
        if process.poll() is not None:
            return
    elif not process.is_alive():
        process.join(timeout=0)
        return
    try:
        parent = psutil.Process(process.pid)
        owned = parent.children(recursive=True) + [parent]
    except psutil.NoSuchProcess:
        return
    for child in owned:
        try:
            child.kill()
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + timeout
    while owned:
        alive = []
        for child in owned:
            try:
                # Orphaned zombies can await init's reaper; their CUDA resources
                # are already gone. psutil.wait_procs alone can wait on these.
                if child.is_running() and child.status() != psutil.STATUS_ZOMBIE:
                    alive.append(child)
            except psutil.NoSuchProcess:
                pass
        owned = alive
        if not owned:
            break
        if time.monotonic() >= deadline:
            raise TimeoutError(f"vLLM processes still exiting: {[child.pid for child in owned]}")
        time.sleep(0.05)
    if isinstance(process, subprocess.Popen):
        process.wait(timeout=max(0, deadline - time.monotonic()))
    else:
        process.join(timeout=max(0, deadline - time.monotonic()))


def main():
    payload = json.load(sys.stdin)
    prepare_server_environment(os.environ)
    from threading import Thread

    from vllm.utils.system_utils import kill_process_tree

    from vime.backends.vllm_utils.vllm_engine import _run_vllm_server

    if os.getppid() != payload["parent_pid"]:
        return
    parent = psutil.Process(payload["parent_pid"])

    def stop_when_parent_exits():
        parent.wait()
        kill_process_tree(os.getpid())

    Thread(target=stop_when_parent_exits, daemon=True).start()
    _run_vllm_server(payload["server_args"], os.environ.copy(), config_keys=payload["config_keys"])


if __name__ == "__main__":
    main()
