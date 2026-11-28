#!/usr/bin/env python3
"""Emit Buildkite steps for the NPU suites.

With --pipeline, include the image builds needed for a single Buildkite upload.
The suites and their configurations are defined here.

Each test's hardware preset selects its Ascend queue and resource_class
(e.g., "npu-8" means 8 NPUs).

The selection is read from the NPU_SUITES environment variable. For local
testing, set NPU_SUITES=smk,nightly instead of having a buildkite-agent on PATH.

The uploader uses PyYAML to load the shared image-build step.
"""

import argparse
import http.client
import json
import os
import re
import subprocess
import sys
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import NamedTuple

import yaml

IMAGE_REGISTRY = "swr.cn-southwest-2.myhuaweicloud.com/modelfoundry"
IMAGE_NAME = "vime-ci-npu"
VIME_IMAGE_TAG = os.environ.get("BUILDKITE_COMMIT", "latest")
BUILDKITE_SOURCE = os.environ.get("BUILDKITE_SOURCE", "")


@dataclass(frozen=True)
class Hardware:
    queue: str
    default_image: str
    build_suffix: str
    soc_version: str = ""  # Empty uses the Dockerfile default.
    test_env: dict = field(default_factory=dict)
    toolkit_env: str = ""

    @property
    def build_key(self) -> str:
        return f"image-build-npu{self.build_suffix}"

    @property
    def build_tag(self) -> str:
        return f"{VIME_IMAGE_TAG}{self.build_suffix}"


# Keys match the hardware suffix on the vLLM-Ascend base image.
HARDWARE = {
    "a3": Hardware(
        queue="ascend-a3",
        default_image="quay.io/ascend/vime:vime-latest",
        build_suffix="",
        test_env={"ASCEND_RT_VISIBLE_DEVICES": "0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15"},
    ),
    "a5": Hardware(
        queue="ascend-a5",
        default_image="quay.io/ascend/vime:vime-latest-a5",
        build_suffix="-a5",
        soc_version="ascend950dt_9582",
        toolkit_env="/usr/local/Ascend/ascend-toolkit/set_env.sh",
    ),
}


class Preset(NamedTuple):
    variant: str
    resource_class: str


A3_4 = Preset("a3", "npu-4")
A3_8 = Preset("a3", "npu-8")
A3_16 = Preset("a3", "npu-16")
A5_8 = Preset("a5", "npu-8")


@dataclass(frozen=True)
class Case:
    script: str
    preset: Preset
    extra_args: str = ""
    env: dict = field(default_factory=dict)
    timeout_in_minutes: int = 180


SUITES = {
    "smk": [
        # Temporarily pause A3 cases while validating the A5 runtime environment.
        # Case("test_delta_disk_weight_update.py", A3_4),
        # Case("test_qwen3_4B_npu.py", A3_8),
        Case("test_qwen3_4B_npu.py", A5_8),
        # Case("test_qwen3_4B_eagle3_npu.py", A3_8),
        # Case("test_qwen3_4B_dspark_npu.py", A3_8),
        # Case("test_qwen3_30B_A3B_npu.py", A3_16),
        # Case("test_qwen3_vl_8B_npu.py", A3_8),
        # Case("test_glm4.7_30B_A3B_npu.py", A3_16),
        # 35B convert plus the first rollout does not fit in the default 180 minutes.
        # Case("test_qwen3.6_35B_A3B_npu.py", A3_16, timeout_in_minutes=360),
    ],
    "nightly": [],
}


def _read_suite_values() -> list[str]:
    raw = os.environ.get("NPU_SUITES")
    if raw is None:
        raw = subprocess.run(
            ["buildkite-agent", "meta-data", "get", "npu-suites", "--default", ""],
            check=True,
            stdout=subprocess.PIPE,
            text=True,
        ).stdout
    return [v.strip() for v in raw.replace(",", "\n").splitlines()]


def resolve_digest(default_image: str) -> str:
    image, _, tag = default_image.rpartition(":")
    if not image.startswith("quay.io/") or not tag:
        raise SystemExit(f"Expected a tagged quay.io image for digest resolution, got {default_image}")
    repository = image.removeprefix("quay.io/")
    url = f"https://quay.io/api/v1/repository/{repository}/tag/?specificTag={tag}&onlyActiveTags=true"
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data = json.load(response)
    except (OSError, http.client.HTTPException, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SystemExit(f"Failed to resolve Quay digest for {default_image}: {exc}") from exc
    tags = data.get("tags") if isinstance(data, dict) else None
    if not isinstance(tags, list):
        raise SystemExit(f"Invalid Quay response for {default_image}: expected a list of tags")
    for entry in tags:
        if not isinstance(entry, dict) or entry.get("name") != tag:
            continue
        digest = entry.get("manifest_digest")
        if isinstance(digest, str) and re.fullmatch(r"sha256:[0-9a-f]{64}", digest):
            return f"{image}@{digest}"
    raise SystemExit(f"Quay returned no valid manifest digest for {default_image}")


def selected_suites(values: list[str]) -> list:
    unknown = [v for v in values if v and v not in SUITES and v != "image-build"]
    if unknown:
        raise SystemExit(f"unknown suite(s) {unknown}; expected {sorted(SUITES)}")
    if "image-build" in values:
        # image-build auto-includes smk tests
        values = [*values, "smk"]
    return [s for s in SUITES if s in values]


def npu_step(suite: str, case: Case, image: str) -> dict:
    hardware = HARDWARE[case.preset.variant]
    step_env = {
        "VIME_TEST_ENABLE_INFINITE_RUN": "false",
        "BUILDKITE_PULL_REQUEST": os.environ.get("BUILDKITE_PULL_REQUEST", "false"),
        "BUILDKITE_COMMIT": os.environ.get("BUILDKITE_COMMIT", ""),
        "HF_TOKEN": "${HF_TOKEN}",
        "HF_ENDPOINT": "https://hf-mirror.com",
        "IMAGE_REGISTRY": IMAGE_REGISTRY,
        "IMAGE_NAME": IMAGE_NAME,
        "VIME_IMAGE_TAG": VIME_IMAGE_TAG,
        "HF_HOME": "/root/.cache/huggingface",
        **hardware.test_env,
        **case.env,
    }

    commands = [
        'echo "INFO: update NPU environment"',
        'if [ -n "${BUILDKITE_COMMIT}" ]; then',
        "  source /workspace/build/buildkite/.buildkite/scripts/update-npu-environment.sh",
        "fi",
        "export HF_HUB_OFFLINE=0",
        # Cover checkpoint conversion before the training launcher runs.
        'export HCCL_CONNECT_TIMEOUT="$${HCCL_CONNECT_TIMEOUT:-600}"',
    ]
    if hardware.toolkit_env:
        commands.append(f"source {hardware.toolkit_env} || exit 1")
    commands.append(f"python tests/{case.script} {case.extra_args}".rstrip())
    command = "bash -c '" + "\n".join(commands) + "'"

    label = f":fire: {suite} [{case.preset.variant.upper()} {case.preset.resource_class}]: {case.script} {case.extra_args}".rstrip()
    step = {
        "label": label,
        "command": command,
        "agents": {
            "queue": hardware.queue,
            "resource_class": case.preset.resource_class,
        },
        "timeout_in_minutes": case.timeout_in_minutes,
        "image": image,
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


def build_image_step(variant: str, hardware: Hardware) -> dict:
    template = Path(__file__).with_name("pipeline-npu-image.yaml")
    (step,) = yaml.safe_load(template.read_text(encoding="utf-8"))["steps"]
    step["key"] = hardware.build_key
    step["label"] = f":docker: Build and Push NPU Test Image - {variant.upper()}"
    step["env"].update(
        NPU_IMAGE_VARIANT=variant,
        NPU_SOC_VERSION=hardware.soc_version,
        IMAGE_REGISTRY=IMAGE_REGISTRY,
        IMAGE_NAME=IMAGE_NAME,
        VIME_IMAGE_TAG=hardware.build_tag,
    )
    return step


def generate_steps(values: list[str], *, include_builds: bool) -> list[dict]:
    suites = selected_suites(values)
    build_images = "image-build" in values or BUILDKITE_SOURCE == "schedule"
    cases = [(suite, case) for suite in suites for case in SUITES[suite]]
    variants = {case.preset.variant for _, case in cases}
    # Temporarily pause the scheduled A3 image refresh for the A5-only run.
    # if BUILDKITE_SOURCE == "schedule":
    #     variants.add("a3")

    images = {}
    steps = []
    for variant in sorted(variants):
        hardware = HARDWARE[variant]
        if build_images:
            images[variant] = f"{IMAGE_REGISTRY}/{IMAGE_NAME}:{hardware.build_tag}"
            if include_builds:
                steps.append(build_image_step(variant, hardware))
        else:
            images[variant] = resolve_digest(hardware.default_image)

    for suite, case in cases:
        variant = case.preset.variant
        step = npu_step(suite, case, images[variant])
        if build_images:
            step["depends_on"] = HARDWARE[variant].build_key
        steps.append(step)
    return steps


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pipeline", action="store_true", help="include image builds alongside the test steps")
    args = parser.parse_args()
    steps = generate_steps(_read_suite_values(), include_builds=args.pipeline)
    json_str = json.dumps({"steps": steps}, indent=2)

    print("--- Generated Pipeline JSON:", file=sys.stderr)
    print(json_str, file=sys.stderr)

    print(json_str)


if __name__ == "__main__":
    main()
