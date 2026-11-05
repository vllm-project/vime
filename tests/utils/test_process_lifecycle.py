"""Use actual CPU child processes to verify replacement fences and log capture."""

import subprocess
import sys
import time

import psutil
import pytest

from vime.utils.process_lifecycle import snapshot_process_tree, terminate_process_tree

NUM_GPUS = 0


@pytest.mark.parametrize("parent_dies", [False, True])
@pytest.mark.parametrize("detached_child", [False, True])
def test_fence_kills_server_and_independent_descendants(tmp_path, parent_dies, detached_child):
    marker = tmp_path / "child"
    code = """
import os, subprocess, sys, time
from pathlib import Path
child = subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(300)'], start_new_session=sys.argv[2] == '1')
Path(sys.argv[1]).write_text(str(child.pid))
time.sleep(300)
"""
    parent = subprocess.Popen(
        [sys.executable, "-c", code, str(marker), str(int(detached_child))], start_new_session=True
    )
    snapshot = snapshot_process_tree(parent.pid)
    try:
        deadline = time.monotonic() + 5
        while not marker.exists():
            assert time.monotonic() < deadline
            time.sleep(0.01)
        child_pid = int(marker.read_text())
        snapshot = snapshot_process_tree(parent.pid)
        if parent_dies:
            parent.kill()
            parent.wait(timeout=5)
        terminate_process_tree(snapshot, timeout=2)
        parent.wait(timeout=5)
        for pid in (parent.pid, child_pid):
            assert not psutil.pid_exists(pid) or psutil.Process(pid).status() == psutil.STATUS_ZOMBIE
    finally:
        terminate_process_tree(snapshot)
        parent.wait(timeout=5)


def test_reused_pid_identity_never_kills_an_unrelated_process():
    child = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(300)"], start_new_session=True)
    try:
        snapshot = snapshot_process_tree(child.pid)
        snapshot["processes"][0]["created"] -= 1
        terminate_process_tree(snapshot)
        assert child.poll() is None
    finally:
        child.kill()
        child.wait(timeout=5)


def test_native_output_and_fatal_traceback_are_bounded_and_preserved(tmp_path):
    log = tmp_path / "server.log"
    code = """
import os, sys, traceback
from vime.utils.process_lifecycle import capture_process_output
with capture_process_output(sys.argv[1], max_bytes=1024):
    os.write(1, b'x' * 4096)
    os.write(2, b'native-driver-error\\n')
    try:
        raise RuntimeError('startup-fatal')
    except RuntimeError:
        traceback.print_exc()
print('output-restored', flush=True)
"""
    result = subprocess.run([sys.executable, "-c", code, str(log)], capture_output=True, check=True)
    assert b"native-driver-error" in result.stdout and b"startup-fatal" in result.stdout
    assert result.stdout.endswith(b"output-restored\n")
    assert log.stat().st_size <= 1024
    assert "native-driver-error" in log.read_text() and "startup-fatal" in log.read_text()
