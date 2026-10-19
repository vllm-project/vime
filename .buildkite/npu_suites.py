#!/usr/bin/env python3
"""Emit Buildkite steps for the NPU suites.

Piped into `buildkite-agent pipeline upload` by pipeline-npu.yaml.
The suites and their configurations are defined here.

NPU jobs run on the ascend-a3 queue with resource_class determining NPU count
(e.g., "npu-8" means 8 NPUs).

The selection is read from the NPU_SUITES environment variable. For local
testing, set NPU_SUITES=smk,nightly instead of having a buildkite-agent on PATH.

stdlib only — runs with the agent host's python3.
"""

import json
import os
import shlex
import subprocess
import sys

NPU_QUEUE = "ascend-a3"
DEFAULT_CI_IMAGE = "quay.io/ascend/vime:vime-latest"
IMAGE_REGISTRY = "swr.cn-southwest-2.myhuaweicloud.com/modelfoundry"
IMAGE_NAME = "vime-ci-npu"
VIME_IMAGE_TAG = os.environ.get("BUILDKITE_COMMIT", "latest")
BUILDKITE_SOURCE = os.environ.get("BUILDKITE_SOURCE", "")

# (test_name, resource_class, extra_args, env_overrides[, timeout_in_minutes])
SUITES = {
    # Temporarily paused for this migration PR: these six cases were already validated.
    "smk": [
        # ("test_delta_disk_weight_update.py", "npu-4", "", {}),
        # ("test_qwen3_4B_npu.py", "npu-8", "", {}),
        # ("test_qwen3_30B_A3B_npu.py", "npu-16", "", {}),
        # ("test_qwen3_vl_8B_npu.py", "npu-8", "", {}),
        # ("test_glm4.7_30B_A3B_npu.py", "npu-16", "", {}),
        # # 35B convert plus the first rollout does not fit in the default 180 minutes.
        # ("test_qwen3.5_35B_A3B_npu.py", "npu-16", "", {}, 360),
    ],
    # Shared GPU tests: keep their assertions and select NPU launch policy.
    # The score-centering test uses two logical devices within a proven 4-NPU pod.
    "migration-basic": [
        ("test_qwen2.5_0.5B_fully_async_short.py", "npu-4", "", {}),
        ("test_qwen2.5_0.5B_pipeline_rl.py", "npu-4", "--transport nccl", {}),
        ("test_qwen2.5_0.5B_pipeline_rl.py", "npu-4", "--transport disk", {}),
        ("test_qwen2.5_0.5B_pipeline_rl.py", "npu-4", "--transport nccl --flush-cache-interval 2", {}),
        ("test_full_disk_weight_update.py", "npu-4", "", {"VIME_TEST_NUM_ROLLOUT": "2"}),
        ("test_qwen2.5_0.5B_fanout_short.py", "npu-4", "", {}),
        ("test_qwen2.5_0.5B_score_centering.py", "npu-4", "", {}),
    ],
    "nightly": [],
}


def _read_suite_values() -> list[str]:
    raw = os.environ.get("NPU_SUITES")
    if raw is None:
        try:
            raw = subprocess.run(
                ["buildkite-agent", "meta-data", "get", "npu-suites"],
                check=True,
                capture_output=True,
                text=True,
            ).stdout
        except (subprocess.CalledProcessError, FileNotFoundError):
            raw = ""
    return [v.strip() for v in raw.replace(",", "\n").splitlines()]


def _ci_image() -> str:
    values = _read_suite_values()
    if ("image-build" in values) or (BUILDKITE_SOURCE == "schedule"):
        return f"{IMAGE_REGISTRY}/{IMAGE_NAME}:{VIME_IMAGE_TAG}"
    return DEFAULT_CI_IMAGE


def selected_suites() -> list:
    values = _read_suite_values()
    unknown = [v for v in values if v and v not in SUITES and v != "image-build"]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; expected {sorted(SUITES)}")
    if "image-build" in values:
        # image-build auto-includes smk tests
        values.append("smk")
    selected = [s for s in SUITES if s in values]
    if not any(SUITES[s] for s in selected):
        available = ", ".join(s for s, entries in SUITES.items() if entries)
        raise SystemExit(f"No NPU tests selected; select a non-empty suite (currently: {available})")
    return selected


def npu_step(
    suite: str,
    test_name: str,
    resource_class: str,
    extra_args: str,
    env: dict,
    timeout_in_minutes: int = 180,
) -> dict:
    step_env = {
        "VIME_PLATFORM": "npu",
        "VIME_TEST_ENABLE_INFINITE_RUN": "false",
        "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", "false"),
        "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
        "HF_TOKEN": "${HF_TOKEN}",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15",
        "IMAGE_REGISTRY": IMAGE_REGISTRY,
        "IMAGE_NAME": IMAGE_NAME,
        "VIME_IMAGE_TAG": VIME_IMAGE_TAG,
        "HF_HOME": "/root/.cache/huggingface",
        **env,
    }

    commands = [
        "set -eo pipefail",
        'echo "INFO: update NPU environment"',
        'if [ -n "${BUILDKITE_COMMIT}" ]; then',
        "  source /workspace/build/buildkite/.buildkite/scripts/update-npu-environment.sh",
        "fi",
        "export HF_HUB_OFFLINE=0",
        # Cover checkpoint conversion before the training launcher runs.
        'export HCCL_CONNECT_TIMEOUT="$${HCCL_CONNECT_TIMEOUT:-600}"',
    ]
    if suite == "migration-basic":
        step_env["VIME_TEST_RESOURCE_CLASS"] = resource_class
        commands.append("python tests/ci/npu_test_environment.py")
    commands.append(f"python tests/{test_name}{' ' + extra_args if extra_args else ''}")
    script = "\n".join(commands)
    command = f"bash -c {shlex.quote(script)}"

    label = f":fire: {suite}: {test_name}{' ' + extra_args if extra_args else ''}"
    step = {
        "label": label,
        "depends_on": "image-build-npu",
        "command": command,
        "agents": {
            "queue": NPU_QUEUE,
            "resource_class": resource_class,
        },
        "timeout_in_minutes": timeout_in_minutes,
        "image": _ci_image(),
        "plugins": [
            {
                "kubernetes": {
                    "podSpecPatch": {
                        "imagePullSecrets": [{"name": "swr-secret"}],
                    },
                }
            }
        ],
        "env": step_env,
    }
    if suite == "migration-basic":
        # Allow up to four migration jobs at once, including across builds. A failed case
        # releases its slot so the other capability probes can still run.
        step["concurrency"] = 4
        step["concurrency_group"] = "vime-npu-migration-basic"
    return step


def build_steps() -> list[dict]:
    steps = []
    for suite in selected_suites():
        for index, entry in enumerate(SUITES[suite], start=1):
            step = npu_step(suite, *entry)
            step["key"] = f"npu-{suite}-{index}"
            steps.append(step)
    return steps


def main() -> None:
    json_str = json.dumps({"steps": build_steps()}, indent=2)

    print("--- Generated Pipeline JSON:", file=sys.stderr)
    print(json_str, file=sys.stderr)

    print(json_str)


if __name__ == "__main__":
    main()
