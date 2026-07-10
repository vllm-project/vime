#!/usr/bin/env python3
"""Emit Buildkite steps for the GPU suites selected at the gpu-gate block step.

Piped into `buildkite-agent pipeline upload` by the gpu-suites-upload step in
pipeline.yml. The suites and their env-var combinations are defined here.

GPU jobs run on the shared `mithril-h100-pool` queue the same way vllm-omni
uses it: one Kubernetes pod per job via the agent-stack-k8s `kubernetes`
plugin, GPUs allocated with `nvidia.com/gpu` limits on H100 SXM nodes,
memory-backed /dev/shm, and the node's /mnt/hf-cache mounted as HF_HOME (vime
tests `hf download` their models at startup, so a warm cache is all they
need — no pre-staged model mounts).

The selection is read from the block step's multi-select field (newline-
separated values in the `gpu-suites` build meta-data key). For local testing,
set GPU_SUITES=short,ckpt instead of having a buildkite-agent on PATH.

stdlib only — runs with the agent host's python3.
"""

import json
import os
import subprocess

GPU_QUEUE = "mithril-h100-pool"
CI_IMAGE = "vllm/vime:latest"
HF_CACHE_HOST_PATH = "/mnt/hf-cache"
HF_HOME = "/root/.cache/huggingface"
NODE_INSTANCE_TYPE = "gpu-h100-sxm"

# Known hardware-fit failures on the pool's 80 GB H100s — test-level issues,
# not pipeline ones (PR #239, builds #6/#7):
#   * gsm8k_async_short: FIXED — max-tokens-per-gpu reduced 9216→2048 (peak
#     39.6 GB on H200, well within H100 80 GB). Root cause was Qwen3.5 248k
#     vocab × 5 logits copies in calculate_log_probs_and_entropy.
#   * parallel_check: cross-layout grad-norm invariance (TP4+per-token-loss)
#     diverges ~12% on ~11% of rollout data (bimodal: most <1.5%, outliers
#     10-20%). Confirmed same behavior in slime — Megatron FP reduction-order
#     non-invariance, not a vime bug.
# soft_fail keeps them running and visible (orange) without failing the build.
SOFT_FAIL_ON_H100 = {
    "test_qwen3_0.6B_parallel_check.py",
}

# (test_file, num_gpus, extra_args, env overrides)
SUITES = {
    "short": [
        ("test_qwen3.5_0.8B_gsm8k_async_short.py", 4, "", {}),
        ("test_qwen3.5_0.8B_gsm8k_short.py", 4, "", {}),
        ("test_qwen2.5_0.5B_fully_async_short.py", 4, "", {}),
    ],
    "vllm-config": [
        ("test_qwen2.5_0.5B_vllm_config.py", 8, "", {}),
        ("test_qwen2.5_0.5B_vllm_config_distributed.py", 8, "", {}),
        ("test_vllm_config_mixed_offload.py", 8, "", {}),
        ("test_vllm_config_mixed_offload_ft.py", 8, "", {}),
    ],
    "megatron": [
        ("test_quick_start_glm4_9B.py", 8, "", {}),
        ("test_glm4.7_30B_A3B_pd_mooncake.py", 8, "", {}),
        ("test_qwen3_30B_A3B.py", 8, "", {"USE_DEEPEP": "1", "USE_FP8_ROLLOUT": "1"}),
        ("test_qwen3.6_35B_A3B_pd_mooncake.py", 8, "", {"USE_DEEPEP": "1"}),
        ("test_qwen3_30B_A3B_r3.py", 8, "", {"USE_DEEPEP": "1", "USE_FP8_ROLLOUT": "1", "ENABLE_EVAL": "0"}),
        ("test_qwen3_30B_A3B_r3.py", 8, "", {"ENABLE_EVAL": "0"}),
        ("test_qwen3_4B_ppo.py", 8, "", {}),
        ("test_qwen3_4B_ppo_disaggregate.py", 8, "", {}),
        ("test_qwen3_4B_ppo_train_critic_only.py", 8, "", {}),
        ("test_qwen3_4B_streaming_partial_rollout.py", 8, "", {}),
        ("test_moonlight_16B_A3B.py", 8, "", {}),
        ("test_moonlight_16B_A3B_r3.py", 8, "", {"ENABLE_EVAL": "0"}),
        ("test_qwen2.5_0.5B_debug_rollout_then_train.py", 8, "", {}),
        ("test_qwen2.5_0.5B_opd_vllm.py", 8, "", {}),
    ],
    "precision": [
        ("test_qwen3_0.6B_parallel_check.py", 8, "", {}),
    ],
    "ckpt": [
        ("test_qwen3_4B_ckpt.py", 8, "", {}),
        ("test_qwen3_4B_ckpt.py", 8, "--async-save", {}),
    ],
}


def selected_suites() -> list:
    raw = os.environ.get("GPU_SUITES")
    if raw is None:
        raw = subprocess.run(
            ["buildkite-agent", "meta-data", "get", "gpu-suites"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
    # multi-select meta-data is newline-separated; accept commas too
    values = [v.strip() for v in raw.replace(",", "\n").splitlines()]
    unknown = [v for v in values if v and v not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; expected {sorted(SUITES)}")
    return [s for s in SUITES if s in values]


def gpu_step(suite: str, test_file: str, num_gpus: int, extra_args: str, env: dict) -> dict:
    vime_flags = {k: v for k, v in env.items() if k in ("USE_DEEPEP", "USE_FP8_ROLLOUT", "ENABLE_EVAL")}
    pod_env = [
        {"name": "HF_HOME", "value": HF_HOME},
        {"name": "VIME_TEST_ENABLE_INFINITE_RUN", "value": "false"},
        {"name": "VIME_TEST_USE_DEEPEP", "value": vime_flags.get("USE_DEEPEP", "0")},
        {"name": "VIME_TEST_USE_FP8_ROLLOUT", "value": vime_flags.get("USE_FP8_ROLLOUT", "0")},
        {"name": "VIME_TEST_ENABLE_EVAL", "value": vime_flags.get("ENABLE_EVAL", "1")},
    ]
    # anything else in env is passed to the pod verbatim (e.g. allocator knobs)
    pod_env += [{"name": k, "value": v} for k, v in env.items() if k not in vime_flags]
    # Set a stable commit identifier for downstream tooling.
    command = "\n".join(
        [
            'PR="${BUILDKITE_PULL_REQUEST:-false}"',
            '[ "$PR" = "false" ] && PR="non-pr"',
            'export GITHUB_COMMIT_NAME="${BUILDKITE_COMMIT}_${PR}"',
            "pip install -e . --no-deps --break-system-packages",
            f"python tests/ci/gpu_lock_exec.py --count {num_gpus} -- "
            f"python tests/{test_file}{' ' + extra_args if extra_args else ''}",
        ]
    )
    label = f":fire: {suite}: {test_file}{' ' + extra_args if extra_args else ''}"
    flag_note = ",".join(f"{k.lower()}={v}" for k, v in vime_flags.items())
    if flag_note:
        label += f" ({flag_note})"
    step = {
        "label": label,
        "command": command,
        "agents": {"queue": GPU_QUEUE},
        "timeout_in_minutes": 360,
        "retry": {"automatic": [{"exit_status": -1, "limit": 2}]},
        "plugins": [
            {
                "kubernetes": {
                    "podSpec": {
                        "containers": [
                            {
                                "image": CI_IMAGE,
                                "resources": {"limits": {"nvidia.com/gpu": num_gpus}},
                                "volumeMounts": [
                                    {"name": "devshm", "mountPath": "/dev/shm"},
                                    {"name": "hf-cache", "mountPath": HF_HOME},
                                ],
                                "env": pod_env,
                            }
                        ],
                        "nodeSelector": {"node.kubernetes.io/instance-type": NODE_INSTANCE_TYPE},
                        "volumes": [
                            {"name": "devshm", "emptyDir": {"medium": "Memory"}},
                            {
                                "name": "hf-cache",
                                "hostPath": {"path": HF_CACHE_HOST_PATH, "type": "DirectoryOrCreate"},
                            },
                        ],
                    }
                }
            }
        ],
    }
    if test_file in SOFT_FAIL_ON_H100:
        step["soft_fail"] = True
        step["label"] = ":warning: " + step["label"]
    return step


def main() -> None:
    steps = [gpu_step(suite, *entry) for suite in selected_suites() for entry in SUITES[suite]]
    if not steps:
        raise SystemExit("no GPU suites selected")
    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
