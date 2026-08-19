#!/usr/bin/env python3
"""Emit Buildkite steps for the NPU suites.

Piped into `buildkite-agent pipeline upload` by the npu step in pipeline.yml.
The suites and their configurations are defined here.

NPU jobs run on the ascend-a3 queue with resource_class determining NPU count
(e.g., "npu-8" means 8 NPUs).

The selection is read from the NPU_SUITES environment variable. For local
testing, set NPU_SUITES=smk,nightly instead of having a buildkite-agent on PATH.

stdlib only — runs with the agent host's python3.
"""

import json
import os
import subprocess
import sys

NPU_QUEUE = "ascend-a3"
DEFAULT_CI_IMAGE = "quay.io/ascend/vime:vime-latest"
IMAGE_REGISTRY = "swr.cn-southwest-2.myhuaweicloud.com/modelfoundry"
IMAGE_NAME = "vime-ci-npu"
VIME_IMAGE_TAG = os.environ.get("BUILDKITE_COMMIT", "latest")
BUILDKITE_SOURCE = os.environ.get("BUILDKITE_SOURCE", "")

# (test_name, resource_class, extra_args, env_overrides)
SUITES = {
    "smk": [
        ("test_qwen3_4B_npu.py", "npu-8", "", {}),
        ("test_qwen3_30B_A3B_npu.py", "npu-16", "", {}),
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
        except subprocess.CalledProcessError:
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
    return [s for s in SUITES if s in values]


def npu_step(suite: str, test_name: str, resource_class: str, extra_args: str, env: dict) -> dict:
    step_env = {
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

    commands = "\n".join(
        [
            'echo "INFO: update NPU environment"',
            'if [ -n "${BUILDKITE_COMMIT}" ]; then',
            "  source /workspace/build/buildkite/.buildkite/scripts/update-npu-environment.sh",
            "fi",
            f"python tests/{test_name}{' ' + extra_args if extra_args else ''}",
        ]
    )
    command = f"bash -c '{commands}'"

    label = f":fire: {suite}: {test_name}{' ' + extra_args if extra_args else ''}"
    step = {
        "label": label,
        "depends_on": "image-build-npu",
        "command": command,
        "agents": {
            "queue": NPU_QUEUE,
            "resource_class": resource_class,
        },
        "timeout_in_minutes": 180,
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
    return step


def main() -> None:
    steps = [npu_step(suite, *entry) for suite in selected_suites() for entry in SUITES[suite]]
    json_str = json.dumps({"steps": steps}, indent=2)

    print("--- Generated Pipeline JSON:", file=sys.stderr)
    print(json_str, file=sys.stderr)

    print(json_str)


if __name__ == "__main__":
    main()
