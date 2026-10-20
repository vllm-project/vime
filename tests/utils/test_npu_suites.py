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
    monkeypatch.delenv("NPU_CI_IMAGE", raising=False)
    monkeypatch.delenv("NPU_KEEP_ARTIFACTS", raising=False)
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
        assert (
            "select a non-empty suite (currently: migration-basic, migration-parallel, migration-lifecycle, migration-recovery, migration-checkpoint, migration-moe, migration-moe-recovery, migration-models, migration-foundation)"
            in result.stderr
        )


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


def test_batch_two_preserves_parallel_layouts(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "migration-parallel")
    steps = suites.build_steps()
    assert len(steps) == 9
    assert len({step["key"] for step in steps}) == 9
    assert sum(step["agents"]["resource_class"] == "npu-8" for step in steps) == 7
    for step in steps:
        assert step["concurrency"] == 4
        assert step["concurrency_group"] == "vime-npu-migration-basic"
        assert "python tests/ci/npu_test_environment.py" in step["command"]
        assert step["depends_on"] == "image-build-npu"
        assert step["timeout_in_minutes"] == (360 if "parallel_check" in step["command"] else 180)
    ppo = [step for step in steps if "ppo" in step["command"]]
    assert len(ppo) == 3
    assert all(step["env"]["VIME_TEST_ENABLE_EVAL"] == "0" for step in ppo)
    disaggregate = next(step for step in ppo if "disaggregate" in step["command"])
    assert "--transport object-store" in disaggregate["command"]
    monkeypatch.setenv("NPU_SUITES", "migration-basic,migration-parallel")
    steps = suites.build_steps()
    assert len(steps) == 16
    assert sum("test_full_disk_weight_update.py" in step["command"] for step in steps) == 1
    monkeypatch.setenv("NPU_SUITES", "migration-full-disk")
    with pytest.raises(SystemExit, match="unknown suite"):
        suites.build_steps()


def test_batch_three_keeps_all_gpu_variants_and_shared_resource_limit(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "migration-lifecycle,migration-recovery,migration-checkpoint")
    steps = suites.build_steps()
    assert len(steps) == 17 and len({step["key"] for step in steps}) == 17
    assert [
        len(suites.SUITES[name]) for name in ("migration-lifecycle", "migration-recovery", "migration-checkpoint")
    ] == [5, 7, 5]
    for step in steps:
        assert step["concurrency"] == 4 and step["concurrency_group"] == "vime-npu-migration-basic"
        assert step["agents"]["resource_class"] in ("npu-4", "npu-8")
        assert "python tests/ci/npu_test_environment.py" in step["command"]
        assert step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == "0"
        script = shlex.split(step["command"])[2]
        assert "cleanup_npu_batch3.py" not in script
    spec = importlib.util.spec_from_file_location("gpu_suites_reference", REPO / ".buildkite/gpu_suites.py")
    gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu)
    originals = {(name, extra) for entries in gpu.SUITES.values() for name, _, extra, _ in entries}
    for group in ("migration-lifecycle", "migration-recovery", "migration-checkpoint"):
        for name, _, extra, *_ in suites.SUITES[group]:
            assert (name, extra) in originals
    checkpoint = [step for step in steps if "test_qwen3_4B_ckpt.py" in step["command"]]
    assert len(checkpoint) == 5
    assert all(
        step["agents"]["resource_class"] == "npu-8" and step["timeout_in_minutes"] == 360 for step in checkpoint
    )
    monkeypatch.setenv(
        "NPU_SUITES", "migration-basic,migration-parallel,migration-lifecycle,migration-recovery,migration-checkpoint"
    )
    assert len(suites.build_steps()) == 33


def test_existing_image_is_explicit_and_schedule_still_builds(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "migration-parallel")
    monkeypatch.setenv("NPU_CI_IMAGE", "registry.example/vime:previous-commit")
    steps = suites.build_steps()
    assert all(step["image"] == "registry.example/vime:previous-commit" for step in steps)
    assert all(step["env"]["VIME_TEST_IMAGE"] == step["image"] for step in steps)
    assert all(step["env"]["BUILDKITE_COMMIT"] == "test-pr-commit" for step in steps)
    monkeypatch.setenv("NPU_SUITES", "image-build,migration-parallel")
    with pytest.raises(SystemExit, match="not both"):
        suites.build_steps()
    monkeypatch.setattr(suites, "BUILDKITE_SOURCE", "schedule")
    assert all(step["image"].endswith(":test-pr-commit") for step in suites.build_steps())


def test_gate_exposes_every_suite(suites):
    pipeline = yaml.safe_load((REPO / ".buildkite/pipeline-npu.yaml").read_text())
    gate = next(step for step in pipeline["steps"][0]["steps"] if step.get("key") == "npu-gate")
    assert {option["value"] for option in gate["fields"][0]["options"]} == set(suites.SUITES) | {"image-build"}
    keep = next(field for field in gate["fields"] if field["key"] == "npu-keep-artifacts")
    assert keep["default"] == "0" and {item["value"] for item in keep["options"]} == {"0", "1"}


@pytest.mark.parametrize("from_env", [True, False])
@pytest.mark.parametrize(
    "selection",
    [
        "image-build,migration-basic",
        "migration-parallel",
        "migration-lifecycle,migration-recovery,migration-checkpoint",
        "migration-moe,migration-moe-recovery,migration-models",
        "migration-foundation",
        "image-build",
        "image-build,unknown",
        "nightly",
    ],
)
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
        " print(os.environ[{'npu-suites': 'TEST_SUITE_METADATA', 'npu-ci-image': 'TEST_IMAGE_METADATA', 'npu-keep-artifacts': 'TEST_KEEP_METADATA'}[sys.argv[3]]])\n"
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
        "TEST_IMAGE_METADATA": ("registry.example/vime:previous-commit" if selection.startswith("migration-") else ""),
        "TEST_UPLOAD_LOG": str(log),
        "TEST_KEEP_METADATA": "1",
    }
    env.pop("NPU_SUITES", None)
    env.pop("NPU_CI_IMAGE", None)
    env.pop("NPU_KEEP_ARTIFACTS", None)
    if from_env:
        env["NPU_SUITES"] = selection
        env["TEST_SUITE_METADATA"] = "unknown"  # metadata must not override the explicit environment
        env["NPU_CI_IMAGE"] = env["TEST_IMAGE_METADATA"]
        env["TEST_IMAGE_METADATA"] = ""  # explicit image must also take precedence
        env["NPU_KEEP_ARTIFACTS"] = "0"
    result = subprocess.run(
        ["bash", "-c", upload["command"].replace("$$", "$")],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=15,
    )
    if selection not in (
        "image-build,migration-basic",
        "migration-parallel",
        "migration-lifecycle,migration-recovery,migration-checkpoint",
        "migration-moe,migration-moe-recovery,migration-models",
        "migration-foundation",
    ):
        assert result.returncode != 0
        assert not log.exists(), "Invalid selection must fail before uploading the image build"
        return
    assert result.returncode == 0, result.stderr
    image_yaml, test_json, _ = log.read_text().split("\n---UPLOAD---\n")
    building = selection.startswith("image-build")
    assert yaml.safe_load(image_yaml)["steps"][0]["skip"] is not building
    generated = json.loads(test_json)["steps"]
    assert len(generated) == (
        7
        if building
        else (
            9
            if selection == "migration-parallel"
            else 8 if "migration-models" in selection else 4 if selection == "migration-foundation" else 17
        )
    )
    expected_image = ":test-pr-commit" if building else "registry.example/vime:previous-commit"
    assert all(step["image"].endswith(expected_image) for step in generated)
    if any(group in selection for group in ("migration-checkpoint", "migration-models", "migration-foundation")):
        assert all(step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == ("0" if from_env else "1") for step in generated)


@pytest.mark.parametrize("keep", ["0", "1", "invalid"])
def test_batch_three_retention_selection(suites, monkeypatch, keep):
    monkeypatch.setenv("NPU_SUITES", "migration-lifecycle,migration-recovery,migration-checkpoint")
    monkeypatch.setenv("NPU_KEEP_ARTIFACTS", keep)
    if keep == "invalid":
        with pytest.raises(SystemExit, match="NPU_KEEP_ARTIFACTS"):
            suites.build_steps()
    else:
        assert all(step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == keep for step in suites.build_steps())


def test_batch_four_preserves_eight_gpu_scenarios_and_cleanup_policy(suites, monkeypatch):
    groups = ("migration-moe", "migration-moe-recovery", "migration-models")
    monkeypatch.setenv("NPU_SUITES", ",".join(groups))
    steps = suites.build_steps()
    assert [len(suites.SUITES[name]) for name in groups] == [4, 1, 3]
    assert len(steps) == 8 and len({step["key"] for step in steps}) == 8
    spec = importlib.util.spec_from_file_location("gpu_reference", REPO / ".buildkite/gpu_suites.py")
    gpu = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gpu)
    original_files = {entry[0] for entries in gpu.SUITES.values() for entry in entries}
    for group in groups:
        assert all(entry[0] in original_files for entry in suites.SUITES[group])
    for step in steps:
        assert step["agents"] == {"queue": "ascend-a3", "resource_class": "npu-8"}
        assert step["concurrency"] == 4 and step["concurrency_group"] == "vime-npu-migration-basic"
        assert step["timeout_in_minutes"] == 360 and step["depends_on"] == "image-build-npu"
        assert step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == "0"
        script = shlex.split(step["command"])[2]
        assert "cleanup_npu_batch3.py" not in script
    r3 = next(step for step in steps if "test_qwen3_30B_A3B_r3.py" in step["command"])
    assert r3["env"]["VIME_TEST_USE_DEEPEP"] == r3["env"]["VIME_TEST_USE_FP8_ROLLOUT"] == "0"
    assert r3["env"]["VIME_TEST_ENABLE_EVAL"] == "0"
    gpu_r3 = next(
        entry
        for entries in gpu.SUITES.values()
        for entry in entries
        if entry[0] == "test_qwen3_30B_A3B_r3.py" and entry[3] == {"ENABLE_EVAL": "0"}
    )
    assert {name: gpu_r3[3].get(name, "0") for name in ("USE_DEEPEP", "USE_FP8_ROLLOUT", "ENABLE_EVAL")} == {
        "USE_DEEPEP": "0",
        "USE_FP8_ROLLOUT": "0",
        "ENABLE_EVAL": "0",
    }
    assert not any("--pd" in step["command"] for step in steps)
    monkeypatch.setenv("NPU_KEEP_ARTIFACTS", "1")
    assert all(step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == "1" for step in suites.build_steps())
    monkeypatch.setenv("NPU_SUITES", ",".join(name for name, entries in suites.SUITES.items() if entries))
    assert len(suites.build_steps()) == 45


def test_all_migration_jobs_dispose_failed_sessions_without_global_sweeps(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", ",".join(name for name in suites.SUITES if name.startswith("migration-")))
    steps = suites.build_steps()
    assert len(steps) == 45
    for step in steps:
        script = shlex.split(step["command"])[2]
        assert "cleanup_npu_batch3.py" not in script
        assert step["env"]["VIME_TEST_DISPOSE_ON_FAILURE"] == "1"
        assert script.index("python tests/ci/npu_test_environment.py") < script.index("python tests/test_")


def test_batch_five_keeps_variants_explicit_and_shares_concurrency(suites, monkeypatch):
    monkeypatch.setenv("NPU_SUITES", "migration-foundation")
    steps = suites.build_steps()
    assert len(steps) == 4
    assert [step["agents"]["resource_class"] for step in steps] == ["npu-4", "npu-4", "npu-8", "npu-4"]
    for step in steps:
        assert step["concurrency"] == 4 and step["concurrency_group"] == "vime-npu-migration-basic"
        assert step["timeout_in_minutes"] == 180
        assert step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == "0"
    assert "--transport object-store" in steps[0]["command"]
    assert "test_ppo_logprob_entropy_npu.py" in steps[1]["command"]
    monkeypatch.setenv("NPU_KEEP_ARTIFACTS", "1")
    assert all(step["env"]["VIME_TEST_KEEP_ARTIFACTS"] == "1" for step in suites.build_steps())
