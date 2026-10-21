"""Capture owned server output and fence its processes before replacement."""

import os
import select
import signal
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path

import psutil


def snapshot_process_tree(pid: int) -> dict:
    parent = psutil.Process(pid)
    identities = [{"pid": pid, "created": parent.create_time()}]
    for child in parent.children(recursive=True):
        try:
            identities.append({"pid": child.pid, "created": child.create_time()})
        except psutil.NoSuchProcess:
            pass
    return {
        "pid": pid,
        "process_group": pid if hasattr(os, "getpgid") and os.getpgid(pid) == pid else None,
        "processes": identities,
    }


def _matching_process(identity):
    try:
        process = psutil.Process(identity["pid"])
        if process.create_time() == identity["created"]:
            return process
    except psutil.NoSuchProcess:
        pass
    return None


def _in_group(process, group):
    try:
        return os.getpgid(process.pid) == group
    except ProcessLookupError:
        return False


def terminate_process_tree(snapshot: dict, timeout: float = 10) -> None:
    """Wait for the recorded server to release its devices; never kill reused PIDs."""
    live = [p for item in snapshot["processes"] if (p := _matching_process(item)) is not None]
    if not live:
        return
    for process in list(live):
        try:
            live.extend(process.children(recursive=True))
        except psutil.NoSuchProcess:
            pass
    if any(process.pid == os.getpid() for process in live):
        raise RuntimeError("Refusing to terminate the cleanup process")
    group = snapshot.get("process_group")
    # The server starts a dedicated session. A still-matching member anchors
    # the group identity even when its original parent was already killed.
    if group and group != os.getpgrp() and any(_in_group(p, group) for p in live):
        try:
            os.killpg(group, signal.SIGKILL)
        except ProcessLookupError:
            pass
        for process in psutil.process_iter(["pid", "create_time"]):
            try:
                if os.getpgid(process.pid) == group:
                    live.append(process)
            except ProcessLookupError:
                pass
    # Also fence known descendants that started a separate process session.
    for process in reversed(live):
        try:
            process.kill()
        except psutil.NoSuchProcess:
            pass
    deadline = time.monotonic() + timeout
    while live:
        remaining = []
        for process in live:
            try:
                # An unreaped zombie has released its device handles already.
                if process.is_running() and process.status() != psutil.STATUS_ZOMBIE:
                    remaining.append(process)
            except psutil.NoSuchProcess:
                pass
        if not remaining:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(f"Server processes still alive before replacement: {[p.pid for p in remaining]}")
        live = remaining
        time.sleep(0.05)


@contextmanager
def capture_process_output(path: str | None, max_bytes: int = 8 * 1024 * 1024):
    """Tee Python and native stdout/stderr to a per-server diagnostic file."""
    if path is None:
        yield
        return
    file = Path(path)
    file.parent.mkdir(parents=True, exist_ok=True)
    output = file.open("w+b", buffering=0)
    sys.stdout.flush()
    sys.stderr.flush()
    saved = (os.dup(1), os.dup(2))
    read_fd, write_fd = os.pipe()
    stop = threading.Event()

    def copy():
        try:
            with output:
                while True:
                    ready, _, _ = select.select([read_fd], [], [], 0.1)
                    if not ready:
                        if stop.is_set():
                            return
                        continue
                    data = os.read(read_fd, 65536)
                    if not data:
                        return
                    output.write(data)
                    if output.tell() > max_bytes:
                        output.seek(-max(1, max_bytes // 2), os.SEEK_END)
                        tail = output.read()
                        output.seek(0)
                        output.write(tail)
                        output.truncate()
                    try:
                        os.write(saved[0], data)
                    except BrokenPipeError:
                        pass
        finally:
            os.close(read_fd)
            os.close(saved[0])
            os.close(saved[1])

    thread = threading.Thread(target=copy, name="server-output", daemon=True)
    thread.start()
    os.dup2(write_fd, 1)
    os.dup2(write_fd, 2)
    os.close(write_fd)
    try:
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved[0], 1)
        os.dup2(saved[1], 2)
        stop.set()
        thread.join(timeout=2)
