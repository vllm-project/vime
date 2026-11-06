"""TEST ONLY: capture intermittent NPU CI stalls; remove after diagnosis.

Stdlib-only supervisor; trainer hooks import the already bootstrapped runtime.
No added collectives, device synchronizations, tensor reads or timeout changes.
Native HCCL diagnostic settings are left unchanged.
"""

from __future__ import annotations

import atexit
import faulthandler
import functools
import importlib
import importlib.metadata
import itertools
import json
import os
import signal
import socket
import subprocess
import sys
import tarfile
import tempfile
import time
from pathlib import Path

_worker_pid = None
_event_fd = None
_stack_file = None
_calls = itertools.count()


def _read(path):
    try:
        return Path(path).read_text()
    except OSError as exc:
        return f"unavailable: {exc}"


def _environment():
    keys = {
        "RANK",
        "LOCAL_RANK",
        "WORLD_SIZE",
        "MASTER_ADDR",
        "MASTER_PORT",
        "HOSTNAME",
        "NODE_NAME",
        "KUBERNETES_NODE_NAME",
        "ASCEND_VISIBLE_DEVICES",
        "ASCEND_RT_VISIBLE_DEVICES",
        "OMP_NUM_THREADS",
        "PYTORCH_NPU_ALLOC_CONF",
        "LD_PRELOAD",
        "RAY_DEDUP_LOGS",
        "VIME_PLATFORM",
        "VIME_TEST_NPU_DIAGNOSTICS",
        "VIME_NPU_DIAGNOSTICS_DIR",
        "HF_HOME",
        "TMPDIR",
        "ASCEND_PROCESS_LOG_PATH",
        "BUILDKITE_BUILD_NUMBER",
        "BUILDKITE_JOB_ID",
        "BUILDKITE_COMMIT",
        "BUILDKITE_RETRY_COUNT",
    }
    return {k: v for k, v in os.environ.items() if k in keys or k.startswith(("HCCL_", "TORCH_HCCL_"))}


def _event(stage, phase, **fields):
    if _event_fd is None:
        return
    record = {
        "wall_ns": time.time_ns(),
        "mono_ns": time.monotonic_ns(),
        "pid": os.getpid(),
        "rank": os.environ.get("RANK"),
        "local_rank": os.environ.get("LOCAL_RANK"),
        "stage": stage,
        "phase": phase,
        **fields,
    }
    try:
        os.write(_event_fd, (json.dumps(record, default=str) + "\n").encode())
    except OSError:
        pass  # Diagnostic I/O must not replace a training result or exception.


def start_worker_diagnostics():
    global _worker_pid, _event_fd, _stack_file
    if os.environ.get("VIME_TEST_NPU_DIAGNOSTICS") != "1" or _worker_pid == os.getpid():
        return
    destination = os.environ.get("VIME_NPU_DIAGNOSTICS_DIR")
    if not destination:
        return
    try:
        root = Path(destination) / "workers" / f"{socket.gethostname()}-{os.getpid()}"
        root.mkdir(parents=True, exist_ok=True)
        _event_fd = os.open(root / "events.jsonl", os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
        _stack_file = (root / "stacks.log").open("a")
        _stack_file.write(f"started_wall_ns={time.time_ns()} dump_interval_seconds=120 pid={os.getpid()}\n")
        _stack_file.flush()
        # C-level watchdog can dump stacks even when another thread holds the GIL.
        faulthandler.dump_traceback_later(120, repeat=True, file=_stack_file)
        atexit.register(faulthandler.cancel_dump_traceback_later)
        _worker_pid = os.getpid()
        _event("worker", "start", env=_environment(), affinity=sorted(os.sched_getaffinity(0)))
        print(f"[NPU CI diagnostic] pid={os.getpid()} evidence={root}", flush=True)
    except Exception as exc:
        print(f"[NPU CI diagnostic] setup failed: {exc}", file=sys.stderr, flush=True)


def _group_info(group):
    import torch.distributed as dist

    # Reloadable groups carry their rank list without accessing a communicator.
    info = getattr(group, "group_info", None)
    if isinstance(info, dict):
        return {"ranks": info.get("ranks"), "backend": info.get("backend"), "reloadable": True}
    group = dist.group.WORLD if group is None else group
    return {"name": group.group_name, "ranks": dist.get_process_group_ranks(group), "backend": dist.get_backend(group)}


def _wrap(owner, name, stage, describe=None, on_return=None):
    original = getattr(owner, name, None)
    if original is None:
        _event(stage, "hook_missing", name=name)
        return
    if getattr(original, "_vime_ci_diagnostic", False):
        return

    @functools.wraps(original)
    def wrapped(*args, **kwargs):
        call = next(_calls)
        started = time.monotonic_ns()
        details = {}
        if describe:
            try:
                details = describe(args, kwargs)
            except Exception as exc:
                details = {"description_error": str(exc)}
        _event(stage, "enter", call=call, **details)
        try:
            result = original(*args, **kwargs)
        except BaseException as exc:
            _event(stage, "error", call=call, elapsed_ns=time.monotonic_ns() - started, error=type(exc).__name__)
            raise
        _event(stage, "return", call=call, elapsed_ns=time.monotonic_ns() - started)
        if on_return:
            try:
                on_return(result)
            except Exception as exc:
                _event(stage, "hook_error", error=str(exc))
        return result

    wrapped._vime_ci_diagnostic = True
    setattr(owner, name, wrapped)
    _event(
        stage,
        "hook_installed",
        implementation=f"{original.__module__}.{original.__qualname__}",
        source=getattr(getattr(original, "__code__", None), "co_filename", None),
    )


def _optimizer_hooks(optimizer):
    # Observe the actual CPU optimizer selected by Adaptor, not a guessed class.
    for item in getattr(optimizer, "chained_optimizers", [optimizer]):
        _wrap(item, "step", "optimizer.step")
        inner = getattr(item, "optimizer", None)
        if inner is None:
            continue
        _wrap(inner, "step", "inner_optimizer.step")
        if hasattr(inner, "_set_sub_optimizer_grads"):
            _wrap(inner, "_set_sub_optimizer_grads", "hybrid._set_sub_optimizer_grads")
        for cpu in getattr(inner, "cpu_optimizers", []):
            _wrap(cpu, "step", "cpu_optimizer.step")


def install_training_diagnostics(args):
    start_worker_diagnostics()
    if _event_fd is None:
        return
    try:
        model = importlib.import_module("vime.backends.megatron_utils.model")
        checkpoint = importlib.import_module("megatron.training.checkpointing")
        hybrid = importlib.import_module("megatron.core.optimizer.cpu_offloading.hybrid_optimizer")
        buffers = importlib.import_module("megatron.core.distributed.param_and_grad_buffer")
        _event(
            "trainer",
            "configured",
            env=_environment(),
            world=_group_info(None),
            modules={
                name: {"file": getattr(module, "__file__", None), "version": getattr(module, "__version__", None)}
                for name in ("torch", "torch_npu", "megatron.core", "megatron_adaptor", "transformer_engine")
                if (module := sys.modules.get(name)) is not None
            },
            args={
                k: getattr(args, k, None)
                for k in (
                    "distributed_timeout_minutes",
                    "optimizer_cpu_offload",
                    "optimizer_offload_fraction",
                    "pin_cpu_grads",
                    "pin_cpu_params",
                    "overlap_param_gather",
                    "tensor_model_parallel_size",
                    "pipeline_model_parallel_size",
                    "expert_model_parallel_size",
                )
            },
        )
        for name in ("get_model", "setup_model_and_optimizer", "clear_memory", "train_one_step"):
            _wrap(model, name, name)
        _wrap(model, "get_megatron_optimizer", "optimizer.init", on_return=_optimizer_hooks)
        _wrap(
            checkpoint,
            "read_metadata",
            "checkpoint.read_metadata",
            describe=lambda a, k: {
                "tracker": str(a[0] if a else k.get("tracker_filename")),
                "group": _group_info(None),
            },
        )
        _wrap(hybrid.HybridDeviceOptimizer, "_get_sub_optimizer_param_groups", "hybrid.cpu_parameter_copies")
        _wrap(buffers._ParamAndGradBucketGroup, "start_param_sync", "params.start_sync")
        _wrap(
            buffers,
            "dist_all_gather_func",
            "params.all_gather",
            describe=lambda a, k: {
                "output_numel": a[0].numel(),
                "input_numel": a[1].numel(),
                "dtype": str(a[0].dtype),
                "device": str(a[0].device),
                "group": _group_info(k.get("group")),
                "async_op": k.get("async_op", False),
            },
        )
    except Exception as exc:
        _event("trainer", "hook_error", error=str(exc))
        print(f"[NPU CI diagnostic] incomplete hooks: {exc}", file=sys.stderr, flush=True)


def _cgroup_paths():
    """Resolve namespaced v1/v2 mounts, including ancestor limits."""
    memberships = {}
    for line in _read("/proc/self/cgroup").splitlines():
        parts = line.split(":", 2)
        if len(parts) == 3:
            for controller in parts[1].split(","):
                memberships[controller] = parts[2]
    paths = set()
    for line in _read("/proc/self/mountinfo").splitlines():
        if " - " not in line:
            continue
        left, right = line.split(" - ", 1)
        fields, fs = left.split(), right.split()
        if fs[0] not in ("cgroup", "cgroup2"):
            continue
        controllers = [""] if fs[0] == "cgroup2" else fs[2].split(",")
        for controller in controllers:
            if controller not in memberships:
                continue
            member, mount_root = Path(memberships[controller]), Path(fields[3])
            relative = (
                member.relative_to(mount_root) if member.is_relative_to(mount_root) else Path(str(member).lstrip("/"))
            )
            mount = Path(fields[4])
            path = (mount / relative).resolve()
            # Container namespaces can expose the hierarchy root instead of the
            # host membership path. Never traverse above the visible cgroup mount.
            if not path.is_relative_to(mount) or not path.is_dir():
                path = mount
            while path.is_relative_to(mount):
                paths.add(path)
                if path == mount:
                    break
                path = path.parent
    names = (
        "memory.current",
        "memory.max",
        "memory.events",
        "memory.stat",
        "memory.pressure",
        "memory.usage_in_bytes",
        "memory.limit_in_bytes",
        "memory.failcnt",
        "memory.oom_control",
        "cpu.stat",
        "cpu.max",
        "cpu.cfs_quota_us",
        "cpu.cfs_period_us",
        "cpuacct.usage",
        "cpu.pressure",
        "cpuset.cpus.effective",
        "cpuset.mems.effective",
        "cpuset.cpus",
        "cpuset.mems",
        "io.stat",
        "io.pressure",
        "blkio.throttle.io_service_bytes",
    )
    return sorted({path / name for path in paths for name in names if (path / name).is_file()})


def _capture(root, name, command, timeout=10):
    try:
        with (root / name).open("w") as output:
            result = subprocess.run(command, stdout=output, stderr=subprocess.STDOUT, timeout=timeout, check=False)
            output.write(f"\nexit_status={result.returncode}\n")
    except (OSError, subprocess.TimeoutExpired) as exc:
        with (root / name).open("a") as output:
            output.write(f"\ncapture unavailable: {exc}\n")


def _snapshot(root, cgroups, child_pid):
    paths = [
        Path(p)
        for p in (
            "/proc/meminfo",
            "/proc/vmstat",
            "/proc/loadavg",
            "/proc/stat",
            "/proc/diskstats",
            "/proc/pressure/cpu",
            "/proc/pressure/memory",
            "/proc/pressure/io",
            "/proc/net/rpc/nfs",
            "/proc/buddyinfo",
        )
    ]
    paths.extend(Path("/sys/devices/system/node").glob("node*/meminfo"))
    pids = {child_pid}
    for worker in (root / "workers").glob("*"):
        pids.add(int(worker.name.rsplit("-", 1)[1]))
    processes = {
        str(pid): {
            name: _read(f"/proc/{pid}/{name}")
            for name in (
                "status",
                "stat",
                "schedstat",
                "io",
                "wchan",
            )
        }
        for pid in sorted(pids)
        if Path(f"/proc/{pid}").exists()
    }
    record = {
        "wall_ns": time.time_ns(),
        "mono_ns": time.monotonic_ns(),
        "system": {str(p): _read(p) for p in paths + cgroups},
        "processes": processes,
    }
    nfs = []
    include = False
    mountstats = _read("/proc/self/mountstats")
    for line in mountstats.splitlines():
        if line.startswith("device "):
            include = "with fstype nfs" in line
        if include:
            nfs.append(line)
    record["nfs_mountstats"] = "\n".join(nfs)
    if mountstats.startswith("unavailable:"):
        record["nfs_mountstats_error"] = mountstats
    with (root / "resources.jsonl").open("a") as output:
        output.write(json.dumps(record) + "\n")


def _collect_logs(root, started):
    # Bounded tails, including successful attempts, for matching phase/node comparisons.
    manifest = []
    budget = 192 * 1024**2
    candidates = []
    for directory in (Path("/tmp/ray/session_latest/logs"), Path("/root/ascend/log/debug/plog")):
        if directory.exists():
            candidates.extend(p for p in directory.rglob("*") if p.is_file() and not p.is_symlink())
    for path in candidates:
        try:
            stat = path.stat()
            if stat.st_mtime < started:
                continue
            size = min(stat.st_size, 4 * 1024**2, budget)
            if size <= 0:
                continue
            dest = root / "runtime-logs" / path.relative_to("/")
            dest.parent.mkdir(parents=True, exist_ok=True)
            with path.open("rb") as source:
                source.seek(max(0, stat.st_size - size))
                dest.write_bytes(source.read(size))
            budget -= size
            manifest.append({"source": str(path), "size": stat.st_size, "saved_tail": size})
        except OSError as exc:
            manifest.append({"source": str(path), "error": str(exc)})
    (root / "log-manifest.json").write_text(json.dumps(manifest, indent=2))


def _archive(root, destination):
    # Prioritize our small observations and flight records over verbose runtime logs.
    budget = 384 * 1024**2
    skipped = []
    with tarfile.open(destination, "w:gz", compresslevel=1) as archive:
        for path in sorted(root.rglob("*"), key=lambda p: ("runtime-logs" in p.parts, "ascend" in p.parts, str(p))):
            try:
                if path.is_file() and not path.is_symlink():
                    size = path.stat().st_size
                    if size > budget:
                        skipped.append(str(path.relative_to(root)))
                        continue
                    archive.add(path, arcname=str(path.relative_to(root)), recursive=False)
                    budget -= size
            except OSError as exc:
                skipped.append(f"{path}: {exc}")
    return skipped


def _worker_summary(root):
    workers = []
    for path in (root / "workers").glob("*/events.jsonl"):
        pending = {}
        worker = {"worker": path.parent.name, "configured": False, "hook_errors": []}
        with path.open() as source:
            for line in source:
                try:
                    record = json.loads(line)
                except ValueError:
                    continue  # A still-running worker may be midway through its final write.
                if record["rank"] is not None:
                    worker["rank"] = record["rank"]
                if record["stage"] == "trainer" and record["phase"] == "configured":
                    worker["configured"] = True
                    worker["world_ranks"] = record.get("world", {}).get("ranks", [])
                if record["phase"] in ("hook_missing", "hook_error"):
                    worker["hook_errors"].append(record)
                if record["phase"] == "enter":
                    pending[record["call"]] = record
                elif record["phase"] in ("return", "error"):
                    pending.pop(record["call"], None)
        worker["pending"] = list(pending.values())
        workers.append(worker)
    (root / "worker-summary.json").write_text(json.dumps(workers, indent=2))
    return workers


def main():
    command = sys.argv[1:]
    if command[:1] == ["--"]:
        command = command[1:]
    if not command:
        raise SystemExit("usage: python -m vime.utils.npu_ci_diagnostics -- COMMAND [ARGS...]")
    if os.environ.get("VIME_TEST_NPU_DIAGNOSTICS") != "1":
        status = subprocess.call(command)
        return status if status >= 0 else 128 - status
    root = Path(tempfile.mkdtemp(prefix="vime-npu-ci-", dir="/tmp"))
    artifact = Path.cwd() / ".npu-ci-diagnostics" / root.name
    artifact.mkdir(parents=True)
    os.environ["VIME_NPU_DIAGNOSTICS_DIR"] = str(root)
    os.environ["RAY_DEDUP_LOGS"] = "0"
    (root / "ascend").mkdir()
    os.environ.setdefault("ASCEND_PROCESS_LOG_PATH", str(root / "ascend"))
    started = time.time()
    versions = {}
    for package in ("torch", "torch-npu", "vllm", "vllm-ascend", "megatron-core", "torch-memory-saver"):
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = "not registered"
    metadata = {
        "env": _environment(),
        "versions": versions,
        "python": sys.version,
        "host": socket.gethostname(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "kernel_boot_id": _read("/proc/sys/kernel/random/boot_id"),
        "dmi_product_uuid": _read("/sys/class/dmi/id/product_uuid"),
    }
    (root / "metadata.json").write_text(json.dumps(metadata, indent=2))
    for name in ("mountinfo", "cgroup", "limits"):
        (root / f"proc-{name}.txt").write_text(_read(f"/proc/self/{name}"))
    for repo in (
        Path.cwd(),
        Path("/root/vime"),
        Path("/root/Megatron-LM"),
        Path("/root/MegatronAdaptor"),
        Path("/root/TransformerEngineNPU"),
        Path("/root/vllm"),
        Path("/root/vllm-ascend"),
    ):
        if repo.is_dir():
            _capture(root, f"source-{str(repo).replace('/', '_')}.txt", ["git", "-C", str(repo), "rev-parse", "HEAD"])
    _capture(root, "npu-before.log", ["npu-smi", "info"])
    _capture(
        root,
        "checkpoint-filesystem.txt",
        [
            "findmnt",
            "-T",
            os.environ.get("HF_HOME", "/root/.cache/huggingface"),
            "-o",
            "TARGET,SOURCE,FSTYPE,OPTIONS",
        ],
    )
    cgroups = _cgroup_paths()
    (root / "cgroup-paths.json").write_text(json.dumps([str(p) for p in cgroups], indent=2))
    print(f"[NPU CI diagnostic] evidence={root}; artifacts={artifact}", flush=True)
    child = subprocess.Popen(command, start_new_session=True)
    terminating = None

    def forward_signal(signum, _frame):
        nonlocal terminating
        terminating = time.monotonic()
        try:
            os.killpg(child.pid, signum)
        except ProcessLookupError:
            pass

    for signum in (signal.SIGTERM, signal.SIGINT):
        signal.signal(signum, forward_signal)
    next_sample = 0.0
    while child.poll() is None:
        if terminating is not None and time.monotonic() - terminating > 10:
            try:
                os.killpg(child.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
        if time.monotonic() >= next_sample:
            try:
                _snapshot(root, cgroups, child.pid)
            except Exception as exc:
                print(f"[NPU CI diagnostic] sample failed: {exc}", file=sys.stderr, flush=True)
            next_sample = time.monotonic() + 15
        time.sleep(1)
    status = child.wait()
    summary = {"exit_status": status, "elapsed_s": time.time() - started, "evidence": str(root)}
    try:
        _snapshot(root, cgroups, child.pid)
        if terminating is None:
            _capture(root, "npu-after.log", ["npu-smi", "info"])
        _collect_logs(root, started)
    except Exception as exc:
        summary["collection_error"] = str(exc)
    try:
        workers = _worker_summary(root)
        summary["trainer_ranks"] = sorted({w["rank"] for w in workers if w["configured"] and "rank" in w})
        expected = {str(rank) for w in workers for rank in w.get("world_ranks", [])}
        summary["missing_trainer_ranks"] = sorted(expected - set(summary["trainer_ranks"]))
        summary["hook_errors"] = sum(len(w["hook_errors"]) for w in workers)
        if not summary["trainer_ranks"]:
            summary["coverage_warning"] = (
                "No trainer hooks observed; inspect environment propagation before drawing conclusions."
            )
    except Exception as exc:
        summary["coverage_error"] = str(exc)
    try:
        summary["archive_skipped"] = _archive(root, artifact / "diagnostics.tar.gz")
    except Exception as exc:
        summary["archive_error"] = str(exc)
    (artifact / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"[NPU CI diagnostic] {json.dumps(summary)}", flush=True)
    return status if status >= 0 else 128 - status


if __name__ == "__main__":
    raise SystemExit(main())
