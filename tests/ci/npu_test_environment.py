"""Print bounded, non-secret resource evidence before an NPU migration test.

All values describe this job. Host memory and requested resource labels are
reported separately from container limits; neither implies a queue capacity.
This does not start Ray or allocate model tensors.
"""

import importlib.metadata
import json
import os
import shutil
import subprocess
from pathlib import Path


def collect_environment():
    report = {
        "env": {
            key: os.environ.get(key)
            for key in (
                "BUILDKITE_COMMIT",
                "BUILDKITE_JOB_ID",
                "BUILDKITE_LABEL",
                "VIME_IMAGE_TAG",
                "VIME_TEST_RESOURCE_CLASS",
                "VIME_PLATFORM",
                "ASCEND_VISIBLE_DEVICES",
                "ASCEND_RT_VISIBLE_DEVICES",
                "HF_HOME",
                "VIME_TEST_ROOT",
            )
        },
        "versions": {},
        "cgroup": {},
        "filesystems": {},
    }
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "ray", "fla-core", "flash-linear-attention"):
        try:
            report["versions"][package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            report["versions"][package] = "not installed"
    report["cpu_affinity"] = sorted(os.sched_getaffinity(0))
    for filename in (
        "/sys/fs/cgroup/memory.max",
        "/sys/fs/cgroup/memory.current",
        "/sys/fs/cgroup/cpu.max",
        "/sys/fs/cgroup/cpuset.cpus.effective",
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/cpu/cpu.cfs_quota_us",
        "/sys/fs/cgroup/cpu/cpu.cfs_period_us",
        "/proc/self/cgroup",
    ):
        try:
            report["cgroup"][filename] = Path(filename).read_text().strip()
        except OSError as exc:
            report["cgroup"][filename] = f"unavailable: {exc}"
    root = os.environ.get("VIME_TEST_ROOT") or os.environ.get("HF_HOME") or "/root"
    for directory in dict.fromkeys(("/dev/shm", "/tmp", root)):
        try:
            usage = shutil.disk_usage(directory)
            report["filesystems"][directory] = {"total_bytes": usage.total, "free_bytes": usage.free}
        except OSError as exc:
            report["filesystems"][directory] = {"error": str(exc)}
    try:
        import torch
        import torch_npu  # noqa: F401

        report["npu_device_count"] = torch.npu.device_count()
    except Exception as exc:
        report["npu_device_error"] = f"{type(exc).__name__}: {exc}"
    try:
        result = subprocess.run(["npu-smi", "info"], capture_output=True, text=True, timeout=30)
        report["npu_smi"] = {
            "returncode": result.returncode,
            "output": (result.stdout + result.stderr)[-16000:],
        }
    except (OSError, subprocess.TimeoutExpired) as exc:
        report["npu_smi"] = {"error": str(exc)}
    return report


if __name__ == "__main__":
    print("--- NPU migration environment (observed resources; not queue limits)", flush=True)
    print(json.dumps(collect_environment(), indent=2), flush=True)
