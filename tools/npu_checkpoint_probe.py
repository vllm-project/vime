"""TEST ONLY: observe NPU checkpoint stalls; remove after CI diagnosis.

Enabled only around the converter's save call. Delegates to the original functions
without changing tensors, synchronization or checkpoint contents.
"""

import inspect
import json
import os
import socket
import subprocess
import threading
import time
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path


def _emit(event, **fields):
    record = {
        "event": event,
        "rank": os.environ.get("RANK"),
        "pid": os.getpid(),
        "wall_time": time.time(),
        "monotonic": time.monotonic(),
        **fields,
    }
    try:
        os.write(1, ("[NPU_CHECKPOINT_PROBE] " + json.dumps(record, sort_keys=True) + "\n").encode())
    except OSError:
        pass


def _read(path):
    try:
        return Path(path).read_text().strip()
    except OSError as exc:
        return f"unavailable: {exc}"


def _selected(path, keys):
    result = {}
    for line in _read(path).splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].rstrip(":") in keys:
            result[parts[0].rstrip(":")] = " ".join(parts[1:])
    return result


def _sample_resources():
    _emit(
        "resources",
        meminfo=_selected(
            "/proc/meminfo",
            {"MemFree", "MemAvailable", "Cached", "Dirty", "Writeback", "AnonPages", "SwapFree", "SReclaimable"},
        ),
        vmstat=_selected(
            "/proc/vmstat",
            {
                "pgfault",
                "pgmajfault",
                "pgscan_direct",
                "pgscan_kswapd",
                "pgsteal_direct",
                "pgsteal_kswapd",
                "allocstall_normal",
                "allocstall_movable",
                "compact_stall",
                "nr_dirty",
                "nr_writeback",
            },
        ),
        pressure={name: _read(f"/proc/pressure/{name}") for name in ("memory", "io", "cpu")},
        cgroup={
            name: _read(f"/sys/fs/cgroup/{name}")
            for name in ("memory.current", "memory.max", "memory.events", "cpu.max", "cpu.stat", "io.stat")
        },
    )


def _timed(name, fn):
    @wraps(fn)
    def wrapper(*args, **kwargs):
        started = time.monotonic()
        _emit("enter", phase=name)
        try:
            result = fn(*args, **kwargs)
        except BaseException as exc:
            _emit("error", phase=name, seconds=time.monotonic() - started, error_type=type(exc).__name__)
            raise
        _emit(
            "return",
            phase=name,
            seconds=time.monotonic() - started,
            process_memory=_selected("/proc/self/status", {"VmRSS", "VmHWM", "VmLck"}),
            process_io=_read("/proc/self/io"),
        )
        return result

    return wrapper


def _wrap(stack, owner, name, phase, static=False):
    original = inspect.getattr_static(owner, name)
    wrapped = _timed(phase, getattr(owner, name))
    setattr(owner, name, staticmethod(wrapped) if static else wrapped)
    stack.callback(setattr, owner, name, original)


@contextmanager
def observe_checkpoint_save(checkpoint_dir):
    import torch.distributed as dist
    from megatron.core.dist_checkpointing.strategies.filesystem_async import FileSystemWriterAsync

    rank = dist.get_rank()
    _emit(
        "save_enter",
        checkpoint_dir=str(checkpoint_dir),
        hostname=socket.gethostname(),
        hccl_connect_timeout=os.environ.get("HCCL_CONNECT_TIMEOUT"),
    )
    if rank == 0:
        try:
            mount = subprocess.run(
                ["findmnt", "-T", str(checkpoint_dir), "-o", "TARGET,SOURCE,FSTYPE,OPTIONS"],
                capture_output=True,
                text=True,
                timeout=5,
                check=False,
            )
            _emit("mount", output=mount.stdout.strip(), error=mount.stderr.strip(), exit_code=mount.returncode)
        except (OSError, subprocess.TimeoutExpired) as exc:
            _emit("mount", error=str(exc))
        _emit("cgroup_membership", value=_read("/proc/self/cgroup"))
        _sample_resources()

    stop = threading.Event()

    def sample():
        while not stop.wait(10):
            _sample_resources()

    sampler = threading.Thread(target=sample, name="checkpoint-probe", daemon=True) if rank == 0 else None
    with ExitStack() as stack:
        _wrap(stack, FileSystemWriterAsync, "preload_tensors", "cpu_staging", static=True)
        # The current Megatron uses threads; keep the probe usable with the preceding baseline.
        for name in ("write_preloaded_data_multithread", "write_preloaded_data_multiproc"):
            if hasattr(FileSystemWriterAsync, name):
                _wrap(stack, FileSystemWriterAsync, name, "write", static=True)
        _wrap(stack, os, "fsync", "fsync")
        _wrap(stack, dist, "barrier", "barrier")
        if sampler is not None:
            sampler.start()
        completed = False
        try:
            yield
            completed = True
        finally:
            stop.set()
            if sampler is not None:
                sampler.join(timeout=1)
                _sample_resources()
            _emit("save_exit", completed=completed)
