"""Deletion boundaries and concurrent one-time CI cleanup, using disposable fixtures."""

import importlib.util
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vime.utils.external_utils import command_utils as U

NUM_GPUS = 0
REPO = Path(__file__).resolve().parents[2]
SCRIPT = REPO / "tests/ci/cleanup_npu_batch3.py"


@pytest.fixture
def cleanup():
    spec = importlib.util.spec_from_file_location("batch3_cleanup", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def fixture_directory(parent, name):
    path = parent / name
    path.mkdir(parents=True)
    (path / "checkpoint").write_text("fixture")
    return path


def test_cleanup_empties_all_tmp_entries_but_preserves_sibling_caches_and_link_targets(cleanup, tmp_path):
    scratch = tmp_path / "tmp"
    directories = [
        fixture_directory(scratch, "vime_training_recovery_12345678"),
        fixture_directory(scratch, "vime_straw_new_output"),
        fixture_directory(scratch, "Qwen3-30B-A3B_torch_dist"),
        fixture_directory(scratch, ".hidden_directory"),
    ]
    caches = [fixture_directory(tmp_path, name) for name in ("models", "datasets", "hub", "outside")]
    (tmp_path / "sibling.txt").write_text("preserve")
    files = [scratch / name for name in ("output.pt", ".hidden_file", ".vime-batch3-cleanup-72026cf1-v1.json")]
    for path in files:
        path.write_text("fixture")
    # Neither a future timestamp nor an old completion marker limits the sweep.
    os.utime(files[0], (4000000000, 4000000000))
    (scratch / "model_link").symlink_to(caches[0], target_is_directory=True)
    (scratch / "file_link").symlink_to(tmp_path / "sibling.txt")
    (scratch / "dangling_link").symlink_to(tmp_path / "missing")
    (directories[0] / "nested_link").symlink_to(caches[1], target_is_directory=True)
    expected = {path.name for path in scratch.iterdir()}
    report = cleanup.cleanup_once(tmp_path)
    assert set(report["removed"]) == expected
    assert scratch.is_dir() and not list(scratch.iterdir())
    assert all((path / "checkpoint").read_text() == "fixture" for path in caches)
    assert (tmp_path / "sibling.txt").read_text() == "preserve"
    assert (tmp_path / cleanup.MARKER).is_file()
    assert (tmp_path / (cleanup.MARKER + ".lock")).is_file()
    # Kept output created after this one-time sweep survives every later call.
    later = fixture_directory(scratch, "kept_after_sweep")
    assert cleanup.cleanup_once(tmp_path) == report and later.exists()


def test_failed_deletion_does_not_mark_cleanup_complete(cleanup, monkeypatch, tmp_path):
    old = fixture_directory(tmp_path / "tmp", "any_old_directory")
    remove = cleanup.shutil.rmtree

    def fail(path):
        raise PermissionError("fixture")

    monkeypatch.setattr(cleanup.shutil, "rmtree", fail)
    with pytest.raises(PermissionError):
        cleanup.cleanup_once(tmp_path)
    assert old.exists() and not (tmp_path / cleanup.MARKER).exists()
    monkeypatch.setattr(cleanup.shutil, "rmtree", remove)
    cleanup.cleanup_once(tmp_path)
    assert not old.exists() and (tmp_path / cleanup.MARKER).is_file()


def test_symlink_scratch_is_rejected(cleanup, tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    outside = fixture_directory(tmp_path, "outside")
    (root / "tmp").symlink_to(outside, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink scratch"):
        cleanup.cleanup_once(root)
    assert (outside / "checkpoint").read_text() == "fixture"


@pytest.mark.parametrize("test_root", ["/", "/var"])
def test_system_temporary_directories_are_rejected_before_access(cleanup, test_root):
    with pytest.raises(ValueError, match="system temporary"):
        cleanup.cleanup_once(test_root)


@pytest.mark.parametrize("metadata", ["marker", "lock"])
def test_cleanup_coordination_cannot_follow_symlinks(cleanup, tmp_path, metadata):
    target = tmp_path / "outside.json"
    target.write_text("preserve")
    name = cleanup.MARKER + (".lock" if metadata == "lock" else "")
    (tmp_path / name).symlink_to(target)
    with pytest.raises((ValueError, OSError)):
        cleanup.cleanup_once(tmp_path)
    assert target.read_text() == "preserve"


def test_four_concurrent_ci_jobs_sweep_once_before_creating_new_outputs(cleanup, tmp_path):
    fixture_directory(tmp_path / "tmp", "legacy_output")
    env = {**os.environ, "VIME_TEST_ROOT": str(tmp_path)}

    def run(index):
        result = subprocess.run([sys.executable, str(SCRIPT)], env=env, capture_output=True, text=True, timeout=15)
        if result.returncode == 0:
            (tmp_path / "tmp" / f"new_job_{index}").write_text("live output")
        return result

    with ThreadPoolExecutor(max_workers=4) as executor:
        results = list(executor.map(run, range(4)))
    assert all(result.returncode == 0 for result in results), results
    assert sum("Removing NPU tmp entry:" in result.stdout for result in results) == 1
    assert sum("already completed" in result.stdout for result in results) == 3
    assert {path.name for path in (tmp_path / "tmp").iterdir()} == {f"new_job_{index}" for index in range(4)}
    assert all((tmp_path / "tmp" / f"new_job_{index}").read_text() == "live output" for index in range(4))


@pytest.mark.parametrize("failure", [False, True])
@pytest.mark.parametrize("keep", [False, True])
def test_temporary_directory_policy_cleans_failures_and_supports_keep(monkeypatch, tmp_path, keep, failure):
    monkeypatch.setattr(U, "get_test_tmp_dir", lambda: str(tmp_path))
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "1" if keep else "0")
    path = None
    try:
        with U.test_work_directory("fixture_") as path:
            (path / "output").write_text("fixture")
            if failure:
                raise RuntimeError("fixture")
    except RuntimeError:
        assert failure
    assert path.exists() is keep


def test_explicit_directory_is_preserved_and_cli_keep_overrides_environment(monkeypatch, tmp_path):
    monkeypatch.setattr(U, "get_test_tmp_dir", lambda: str(tmp_path))
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "0")
    explicit = tmp_path / "explicit"
    with pytest.raises(RuntimeError), U.test_work_directory("fixture_", directory=explicit) as path:
        (path / "sentinel").write_text("preserve")
        raise RuntimeError("fixture")
    assert (explicit / "sentinel").read_text() == "preserve"
    with U.test_work_directory("fixture_", keep=True) as kept:
        pass
    assert kept.exists()
    monkeypatch.setenv("VIME_TEST_KEEP_ARTIFACTS", "1")
    with U.test_work_directory("fixture_", keep=False) as deleted:
        pass
    assert not deleted.exists()
