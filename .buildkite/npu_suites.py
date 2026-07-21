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

NPU_QUEUE = "ascend-a3"
CI_IMAGE = os.environ.get("IMAGE_BUILD", "quay.io/ascend/vime:0.3.0-a3-vllm0.22.1rc1")
IMAGE_REGISTRY = "swr.cn-southwest-2.myhuaweicloud.com/modelfoundry"
IMAGE_NAME = "vime-ci-npu"
VIME_IMAGE_TAG = os.environ.get("BUILDKITE_COMMIT", "latest")

# (test_name, resource_class, extra_args, env_overrides)
SUITES = {
    "smk": [
        ("test_qwen3_4B_npu.py", "npu-8", "", {}),
        ("test_qwen3_30B_A3B_npu.py", "npu-16", "", {}),
    ],
    "nightly": [],
}


def selected_suites() -> list:
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
    values = [v.strip() for v in raw.replace(",", "\n").splitlines()]
    unknown = [v for v in values if v and v not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; expected {sorted(SUITES)}")
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
        "command": command,
        "agents": {
            "queue": NPU_QUEUE,
            "resource_class": resource_class,
        },
        "timeout_in_minutes": 180,
        "image": CI_IMAGE,
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
    print(json.dumps({"steps": steps}, indent=2))


if __name__ == "__main__":
    main()
