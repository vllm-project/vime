"""Manual Qwen3-4B GPU/NVMe A/B and live-rollout soak; not an automatic CI job.

Capture real vLLM responses once, then replay the identical files in both arms.
Use a dedicated Vime container: the common launcher manages its Ray/vLLM processes.
Synthetic within-group rewards exercise gradients, not task quality.
"""

import argparse
import json
import math
import os
import shlex
import subprocess
import time
from pathlib import Path

import test_qwen3_4B_ckpt as e2e


def set_option(tokens, name, value=None):
    """Replace a known zero/one-value option without shell/string substitution."""
    while name in tokens:
        index = tokens.index(name)
        del tokens[index]
        if index < len(tokens) and not tokens[index].startswith("--"):
            del tokens[index]
    if value is not None:
        tokens.append(name)
        if value is not True:
            tokens.append(str(value))


def benchmark_args(phase, root, num_rollouts, *, checkpoint_interval=20, resume_step=None, deterministic=False):
    root = Path(root)
    run = root / phase
    tokens = shlex.split(e2e.build_train_args("", "nvme", str(run / "checkpoint")))
    for name in ("--ci-test", "--ci-save-grad-norm"):
        set_option(tokens, name)
    set_option(tokens, "--num-rollout", num_rollouts)
    if deterministic:
        set_option(tokens, "--deterministic-mode", True)
    if phase in ("capture", "gpu", "gpu-repeat"):
        for name in (
            "--stream-optimizer-state-to-disk",
            "--offload-train-disk-dir",
            "--offload-train-disk-chunk-mb",
            "--stream-optimizer-state-moment-dtype",
        ):
            set_option(tokens, name)
    if phase == "capture":
        set_option(tokens, "--debug-rollout-only", True)
        set_option(tokens, "--save-debug-rollout-data", root / "capture" / "rollout_{rollout_id}.pt")
    else:
        set_option(
            tokens, "--custom-megatron-before-train-step-hook-path", "test_qwen3_4B_nvme_benchmark.measure_step"
        )
        if phase in ("gpu", "gpu-repeat", "nvme"):
            set_option(tokens, "--load-debug-rollout-data", root / "replay" / "rollout_{rollout_id}.pt")
        else:
            set_option(tokens, "--save", run / "checkpoint")
            set_option(tokens, "--save-interval", checkpoint_interval)
            if phase == "resume":
                if resume_step is None:
                    raise ValueError("resume requires a checkpoint step")
                set_option(tokens, "--load", root / "soak" / "checkpoint")
                set_option(tokens, "--ckpt-step", resume_step)
                # Extending the rollout budget changes the derived schedule
                # length. Restore the saved scheduler and samples-seen counter
                # rather than resetting them or rejecting the extended run.
                set_option(tokens, "--use-checkpoint-opt-param-scheduler", True)
    return shlex.join(tokens)


def measure_step(args, rollout_id, step_id, model, optimizer, scheduler):
    """Record synchronized per-rank train/optimizer time without production edits."""
    import torch
    import torch.distributed as dist

    context = {
        "rollout": rollout_id,
        "step_in_rollout": step_id,
        "step": rollout_id * 2 + step_id,
        "rank": dist.get_rank(),
    }
    if not hasattr(optimizer, "_nvme_benchmark_original_step"):
        optimizer._nvme_benchmark_original_step = optimizer.step

        def measured_step():
            current = optimizer._nvme_benchmark_context
            torch.cuda.synchronize()
            begin = time.perf_counter()
            result = optimizer._nvme_benchmark_original_step()
            torch.cuda.synchronize()
            end = time.perf_counter()
            norm = float(result[1])
            assert result[0] and math.isfinite(norm) and norm > 0, result
            record = {
                **current,
                "time": time.time(),
                "optimizer_seconds": end - begin,
                "forward_backward_optimizer_seconds": end - current["begin"],
                "peak_allocated_bytes": torch.cuda.max_memory_allocated(),
                "peak_reserved_bytes": torch.cuda.max_memory_reserved(),
                "grad_norm": norm,
            }
            if current["step"] == 0:
                import sys

                record["source_files"] = {
                    name: getattr(sys.modules.get(name), "__file__", None)
                    for name in ("vime.backends.megatron_utils.model", "vime_plugins.optimizers.nvme_stream")
                }
            del record["begin"]
            # Bounded samples across every model tensor: explicitly NOT a full
            # tensor equality claim. Telemetry is collected after the timed region.
            if current["step"] in (0, 1, 43):
                record["model_samples"] = [
                    param.detach().flatten()[:: max(1, param.numel() // 16)][:16].float().cpu().tolist()
                    for chunk in model
                    for param in chunk.parameters()
                ]
            if current["step"] == 43:
                import hashlib

                digest = hashlib.sha256()
                for chunk in model:
                    for name, param in chunk.named_parameters():
                        value = param.detach().cpu().contiguous()
                        digest.update(name.encode())
                        digest.update(str(tuple(value.shape)).encode())
                        digest.update(memoryview(value.view(torch.uint8).numpy()).cast("B"))
                        del value
                record["full_model_sha256"] = digest.hexdigest()
            directory = Path(os.environ["VIME_NVME_BENCH_DIR"])
            with (directory / f"steps-rank{current['rank']:02d}.jsonl").open("a") as handle:
                handle.write(json.dumps(record) + "\n")
            return result

        optimizer.step = measured_step
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    context["begin"] = time.perf_counter()
    optimizer._nvme_benchmark_context = context


def prepare_replay(root, num_rollouts):
    """Cycle captured files unchanged and record their hashes for the A/B audit."""
    import hashlib

    root = Path(root)
    sources = sorted((root / "capture").glob("rollout_*.pt"))
    if not sources:
        raise ValueError("Capture real rollout data before preparing replay")
    replay = root / "replay"
    replay.mkdir(exist_ok=False)
    manifest = []
    for index in range(num_rollouts):
        source = sources[index % len(sources)].resolve()
        (replay / f"rollout_{index}.pt").symlink_to(source)
        manifest.append(
            {"rollout": index, "source": str(source), "sha256": hashlib.sha256(source.read_bytes()).hexdigest()}
        )
    (replay / "manifest.json").write_text(json.dumps(manifest, indent=2))


def validate_phase_output(phase, run, num_rollouts, resume_step=None):
    """Do not treat an interrupted Ray job's zero CLI exit code as completed work."""
    run = Path(run)
    if phase == "capture":
        for rollout in range(num_rollouts):
            assert (run / f"rollout_{rollout}.pt").is_file()
        return
    start = resume_step + 1 if phase == "resume" else 0
    expected = list(range(start * 2, num_rollouts * 2))
    for rank in range(8):
        path = run / f"steps-rank{rank:02d}.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()]
        assert [row["step"] for row in records] == expected, (phase, rank, "incomplete or repeated steps")
        assert all(math.isfinite(row["grad_norm"]) and row["grad_norm"] > 0 for row in records)


def require_idle_gpus(wait_seconds=30, poll_seconds=2):
    """Allow prior workers to exit, but never launch on a busy/shared GPU.

    This bounded wait is not a resource reservation.
    """
    deadline = time.monotonic() + wait_seconds
    while True:
        status = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=memory.used,utilization.gpu", "--format=csv,noheader,nounits"],
            text=True,
            timeout=15,
        )
        devices = [tuple(map(int, line.split(","))) for line in status.splitlines()]
        if len(devices) == 8 and all(memory <= 1024 and utilization == 0 for memory, utilization in devices):
            return
        if time.monotonic() >= deadline:
            raise RuntimeError(f"Validation requires eight idle GPUs; observed (MiB, utilization%): {devices}")
        time.sleep(poll_seconds)


def compare_replay_outputs(root, num_rollouts=22, warmup_steps=4):
    """Require completed GPU/repeated-GPU/NVMe runs before publishing an A/B result."""
    import statistics

    root = Path(root)
    count = 2 * num_rollouts
    if not 0 <= warmup_steps < count:
        raise ValueError("warmup steps must leave at least one measured step")
    data = {}
    for phase in ("gpu", "gpu-repeat", "nvme"):
        validate_phase_output(phase, root / phase, num_rollouts)
        data[phase] = [
            [json.loads(line) for line in (root / phase / f"steps-rank{rank:02d}.jsonl").read_text().splitlines()]
            for rank in range(8)
        ]
    max_norm_error = 0.0
    for rank in range(8):
        reference = data["gpu"][rank]
        for phase in ("gpu-repeat", "nvme"):
            candidate = data[phase][rank]
            assert reference[-1]["full_model_sha256"] == candidate[-1]["full_model_sha256"], (phase, rank)
            for lhs, rhs in zip(reference, candidate, strict=True):
                error = abs(lhs["grad_norm"] - rhs["grad_norm"])
                max_norm_error = max(max_norm_error, error)
                assert math.isclose(lhs["grad_norm"], rhs["grad_norm"], rel_tol=1e-4, abs_tol=1e-5), (
                    phase,
                    rank,
                    lhs["step"],
                    error,
                )
    summary = {"full_model_hashes_equal_all_ranks": True, "max_grad_norm_abs_difference": max_norm_error}
    for phase in ("gpu", "nvme"):
        values = [
            max(data[phase][rank][step]["forward_backward_optimizer_seconds"] for rank in range(8))
            for step in range(warmup_steps, count)
        ]
        summary[phase] = {
            "warmup_steps": warmup_steps,
            "measured_steps": len(values),
            "mean_forward_backward_optimizer_seconds": statistics.mean(values),
            "peak_training_allocated_bytes": max(
                row["peak_allocated_bytes"] for rows in data[phase] for row in rows[warmup_steps:]
            ),
        }
    (root / "checked-ab-comparison.json").write_text(json.dumps(summary, indent=2) + "\n")
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "phase", choices=["capture", "prepare-replay", "gpu", "gpu-repeat", "nvme", "compare", "soak", "resume"]
    )
    parser.add_argument("--root", required=True)
    parser.add_argument("--num-rollouts", type=int, default=22, help="Two optimizer steps per rollout")
    parser.add_argument("--checkpoint-interval", type=int, default=20)
    parser.add_argument("--resume-step", type=int)
    parser.add_argument(
        "--deterministic", action="store_true", help="Use deterministic Megatron/TE/NCCL controls for replay"
    )
    parser.add_argument("--warmup-steps", type=int, default=4, help="Steps excluded by the compare phase")
    options = parser.parse_args()
    if options.num_rollouts <= 0:
        parser.error("--num-rollouts must be positive")
    if options.phase == "prepare-replay":
        prepare_replay(options.root, options.num_rollouts)
    elif options.phase == "compare":
        print(json.dumps(compare_replay_outputs(options.root, options.num_rollouts, options.warmup_steps), indent=2))
    else:
        require_idle_gpus()
        run = Path(options.root).resolve() / options.phase
        run.mkdir(parents=True, exist_ok=False)
        train_args = benchmark_args(
            options.phase,
            options.root,
            options.num_rollouts,
            checkpoint_interval=options.checkpoint_interval,
            resume_step=options.resume_step,
            deterministic=options.deterministic,
        )
        (run / "command.txt").write_text(train_args + "\n")
        e2e.U.execute_train(
            train_args=train_args,
            num_gpus_per_node=8,
            megatron_model_type=e2e.MODEL_TYPE,
            extra_env_vars={
                "PYTHONPATH": f"{Path(__file__).resolve().parents[1]}:{Path(__file__).resolve().parent}:/root/Megatron-LM/",
                "VIME_NVME_BENCH_DIR": str(run),
                **(
                    {
                        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
                        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
                        "NCCL_ALGO": "Ring",
                    }
                    if options.deterministic
                    else {}
                ),
            },
        )
        validate_phase_output(options.phase, run, options.num_rollouts, options.resume_step)
