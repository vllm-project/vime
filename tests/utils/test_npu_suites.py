"""Buildkite generation contracts; no agent API or hardware jobs are invoked."""

import importlib.util
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

import pytest
import yaml

NUM_GPUS = 0
REPO = Path(__file__).resolve().parents[2]


@pytest.fixture
def suites(monkeypatch):
    monkeypatch.setenv("BUILDKITE_COMMIT", "test-pr-commit")
    monkeypatch.setenv("BUILDKITE_SOURCE", "webhook")
    spec = importlib.util.spec_from_file_location("npu_suites_under_test", REPO / ".buildkite/npu_suites.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_image_build_runs_selected_migration_with_smoke_paused(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "image-build,migration-basic,smk,migration-basic")
    steps = suites.build_steps()
    assert len(steps) == 7
    assert len({step["key"] for step in steps}) == 7
    for step in steps:
        assert step["image"].endswith("/vime-ci-npu:test-pr-commit")
        assert step["depends_on"] == "image-build-npu"
        assert step["env"]["VIME_PLATFORM"] == "npu"
        assert step["env"]["BUILDKITE_COMMIT"] == "test-pr-commit"
        command = shlex.split(step["command"])
        assert command[:2] == ["bash", "-c"] and len(command) == 3
        assert command[2].startswith("set -eo pipefail\n")
    migration = [step for step in steps if step["key"].startswith("npu-migration-basic-")]
    assert len(migration) == 7
    for step in migration:
        assert step["agents"] == {"queue": "ascend-a3", "resource_class": "npu-4"}
        assert step["concurrency"] == 4
        assert step["concurrency_group"] == "vime-npu-migration-basic"
        assert step["timeout_in_minutes"] == 180
        assert "python tests/ci/npu_test_environment.py" in step["command"]
        assert step["depends_on"] == "image-build-npu"  # failures must not skip subsequent probes
    full_disk = next(step for step in migration if "test_full_disk_weight_update.py" in step["command"])
    assert full_disk["env"]["VIME_TEST_NUM_ROLLOUT"] == "2"
    pipeline = [step for step in migration if "test_qwen2.5_0.5B_pipeline_rl.py" in step["command"]]
    assert len(pipeline) == 3
    assert any("--transport disk" in step["command"] for step in pipeline)
    assert any("--flush-cache-interval 2" in step["command"] for step in pipeline)
    assert not any(step["key"].startswith("npu-smk-") for step in steps)


@pytest.mark.parametrize("selection", ["", "smk", "image-build", "nightly", "unknown", "image-build,unknown"])
def test_invalid_or_empty_selection_fails_before_emitting_json(monkeypatch, selection):
    env = {**os.environ, "NPU_SUITES": selection}
    result = subprocess.run(
        [sys.executable, str(REPO / ".buildkite/npu_suites.py")],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    assert result.returncode != 0
    assert not result.stdout
    if selection in ("", "smk", "image-build", "nightly"):
        assert "select a non-empty suite (currently: migration-basic)" in result.stderr


def test_image_build_auto_includes_smoke_when_restored(suites, monkeypatch):
    # Verify the existing auto-inclusion policy without re-enabling real CI jobs.
    monkeypatch.setitem(suites.SUITES, "smk", [("test_qwen3.5_35B_A3B_npu.py", "npu-16", "", {}, 360)])
    monkeypatch.setenv("NPU_SUITES", "image-build")
    steps = suites.build_steps()
    assert len(steps) == 1
    assert steps[0]["key"] == "npu-smk-1"
    assert steps[0]["agents"]["resource_class"] == "npu-16"
    assert steps[0]["timeout_in_minutes"] == 360
    assert "concurrency" not in steps[0]
    assert steps[0]["image"].endswith(":test-pr-commit")


def test_migration_does_not_require_image_build(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "migration-basic")
    assert all(step["image"] == suites.DEFAULT_CI_IMAGE for step in suites.build_steps())
    monkeypatch.setattr(suites, "BUILDKITE_SOURCE", "schedule")
    assert all(step["image"].endswith(":test-pr-commit") for step in suites.build_steps())


def test_gate_exposes_every_suite(suites):
    pipeline = yaml.safe_load((REPO / ".buildkite/pipeline-npu.yaml").read_text())
    gate = next(step for step in pipeline["steps"][0]["steps"] if step.get("key") == "npu-gate")
    assert {option["value"] for option in gate["fields"][0]["options"]} == set(suites.SUITES) | {"image-build"}


@pytest.mark.parametrize("from_env", [True, False])
@pytest.mark.parametrize("selection", ["image-build,migration-basic", "image-build", "image-build,unknown", "nightly"])
def test_upload_uses_same_selection_for_image_and_tests(tmp_path, from_env, selection):
    pipeline = yaml.safe_load((REPO / ".buildkite/pipeline-npu.yaml").read_text())
    upload = next(step for step in pipeline["steps"][0]["steps"] if step.get("key") == "upload-npu-suites")
    # Emulate Buildkite's dollar escaping. The agent executable only records
    # uploads locally; it cannot reach Buildkite or create a real job.
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    agent = bin_dir / "buildkite-agent"
    agent.write_text(
        f"#!{sys.executable}\n"
        "import os, pathlib, sys\n"
        "if sys.argv[1:3] == ['meta-data', 'get']:\n"
        " print(os.environ['TEST_SUITE_METADATA'])\n"
        "elif sys.argv[1:3] == ['pipeline', 'upload']:\n"
        " data = pathlib.Path(sys.argv[3]).read_text() if len(sys.argv) > 3 else sys.stdin.read()\n"
        " with open(os.environ['TEST_UPLOAD_LOG'], 'a') as f: f.write(data + '\\n---UPLOAD---\\n')\n"
        "else: sys.exit(99)\n"
    )
    agent.chmod(0o755)
    log = tmp_path / "uploads"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "BUILDKITE_SOURCE": "webhook",
        "BUILDKITE_COMMIT": "test-pr-commit",
        "TEST_SUITE_METADATA": selection,
        "TEST_UPLOAD_LOG": str(log),
    }
    env.pop("NPU_SUITES", None)
    if from_env:
        env["NPU_SUITES"] = selection
        env["TEST_SUITE_METADATA"] = "unknown"  # metadata must not override the explicit environment
    result = subprocess.run(
        ["bash", "-c", upload["command"].replace("$$", "$")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if selection != "image-build,migration-basic":
        assert result.returncode != 0
        assert not log.exists(), "Invalid selection must fail before uploading the image build"
        return
    assert result.returncode == 0, result.stderr
    image_yaml, test_json, _ = log.read_text().split("\n---UPLOAD---\n")
    assert yaml.safe_load(image_yaml)["steps"][0]["skip"] is False
    generated = json.loads(test_json)["steps"]
    assert len(generated) == 7
    assert all(step["key"].startswith("npu-migration-basic-") for step in generated)
    assert all(step["image"].endswith(":test-pr-commit") for step in generated)
