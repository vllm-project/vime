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
from pathlib import Path

NPU_QUEUE = "ascend-a3"
CI_IMAGE = "swr.cn-southwest-2.myhuaweicloud.com/modelfoundry/" "vime-ci-npu:f6e700ec7f19bcc567e70a910ad9868ff7360dbd"
REPO_ROOT = Path(__file__).resolve().parents[1]

# (test_name, resource_class, extra_args, env_overrides[, timeout_in_minutes])
SUITES = {
    "smk": [
        ("test_npu_ci_smoke.py", "npu-4", "", {}),
    ],
    "nightly": [("test_npu_ci_smoke.py", "npu-4", "", {})],
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
    values = [v.strip() for v in raw.replace(",", "\n").splitlines() if v.strip()]
    return values or ["smk"]


def selected_suites() -> list:
    values = _read_suite_values()
    unknown = [v for v in values if v not in SUITES]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; expected {sorted(SUITES)}")
    return [s for s in SUITES if s in values]


def _validate_tests(selected: list[str]) -> None:
    missing = []
    for suite in selected:
        for test_name, *_ in SUITES[suite]:
            path = REPO_ROOT / "tests" / test_name
            if not path.is_file():
                missing.append(str(path))
    if missing:
        raise SystemExit(f"NPU suite references missing test file(s): {missing}")


def npu_step(
    suite: str,
    test_name: str,
    resource_class: str,
    extra_args: str,
    env: dict,
    timeout_in_minutes: int = 180,
) -> dict:
    step_env = {
        "VIME_TEST_ENABLE_INFINITE_RUN": "false",
        "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", "false"),
        "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
        "HF_TOKEN": "${HF_TOKEN}",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "ASCEND_RT_VISIBLE_DEVICES": "0",
        "HF_HOME": "/root/.cache/huggingface",
        **env,
    }

    commands = f"PYTHONPATH=\"$PWD:$PYTHONPATH\" python tests/{test_name}{' ' + extra_args if extra_args else ''}"
    command = f"bash -c '{commands}'"

    label = f":fire: {suite}: {test_name}{' ' + extra_args if extra_args else ''}"
    step = {
        "label": label,
        "command": command,
        "agents": {
            "queue": NPU_QUEUE,
            "resource_class": resource_class,
        },
        "timeout_in_minutes": timeout_in_minutes,
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
    selected = selected_suites()
    if not selected:
        raise SystemExit("NPU suite selection is empty")
    _validate_tests(selected)
    steps = [npu_step(suite, *entry) for suite in selected for entry in SUITES[suite]]
    json_str = json.dumps({"steps": steps}, indent=2)

    print("--- Generated Pipeline JSON:", file=sys.stderr)
    print(json_str, file=sys.stderr)

    print(json_str)


if __name__ == "__main__":
    main()
