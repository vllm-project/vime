"""Empty the NPU test cache's tmp directory once before launching new tests.

Run after older builds finish: those builds do not participate in this cleanup's
lock. Model/dataset/hub caches outside tmp remain untouched. The filename stays
unchanged for the existing CI entry point.
"""

import fcntl
import json
import os
import shutil
import stat
import tempfile
from pathlib import Path

MARKER = ".vime-ci-tmp-cleanup-72026cf1-v2.json"


def cleanup_once(test_root):
    root = Path(test_root).expanduser().resolve()
    scratch = root / "tmp"
    if root == Path(root.anchor) or scratch in (Path("/tmp"), Path("/var/tmp")):
        raise ValueError(f"Refusing a system temporary directory: {scratch}")
    if scratch.is_symlink():
        raise ValueError(f"Refusing a symlink scratch directory: {scratch}")
    scratch.mkdir(parents=True, exist_ok=True)
    # Every new migration job enters here before creating scratch. Coordination
    # lives outside tmp so clearing its entire contents cannot delete the lock
    # inode or completion marker and let a later job delete new test outputs.
    flags = os.O_CREAT | os.O_RDWR | os.O_NOFOLLOW
    with os.fdopen(os.open(root / (MARKER + ".lock"), flags, 0o600), "r+") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        marker = root / MARKER
        if marker.is_symlink():
            raise ValueError(f"Refusing a symlink cleanup marker: {marker}")
        if marker.exists():
            print(f"NPU tmp cleanup already completed: {marker}", flush=True)
            return json.loads(marker.read_text())
        report = {"scratch": str(scratch), "removed": []}
        for path in sorted(scratch.iterdir()):
            print(f"Removing NPU tmp entry: {path}", flush=True)
            if stat.S_ISDIR(path.lstat().st_mode):
                shutil.rmtree(path)
            else:
                # Remove symlinks themselves, including dangling links, without
                # following them into model/data caches or external directories.
                path.unlink()
            report["removed"].append(path.name)
        # Write only after every deletion succeeds. On failure the next job retries.
        with tempfile.NamedTemporaryFile(mode="w", dir=root, prefix=MARKER, delete=False) as file:
            json.dump(report, file, indent=2)
            file.write("\n")
        Path(file.name).replace(marker)
        print(json.dumps({"npu_tmp_cleanup": report}), flush=True)
        return report


if __name__ == "__main__":
    cleanup_once(os.environ.get("VIME_TEST_ROOT") or os.environ.get("HF_HOME") or "/root")
