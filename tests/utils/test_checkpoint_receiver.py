"""CPU tests for the transactional full-disk checkpoint receiver."""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from vime.backends.vllm_utils import checkpoint_receiver as receiver


def _publish(root: Path, version: int, payload: bytes = b"weights-v1") -> Path:
    checkpoint = root / f"weight_v{version:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "config.json").write_text("{}", encoding="utf-8")
    (checkpoint / "model-00001.safetensors").write_bytes(payload)
    (checkpoint / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": {"total_size": len(payload)},
                "weight_map": {"model.weight": "model-00001.safetensors"},
            }
        ),
        encoding="utf-8",
    )
    return checkpoint


def test_materialize_is_atomic_and_repeated_pull_is_idempotent(tmp_path: Path):
    source = tmp_path / "published"
    _publish(source, 1)
    local = tmp_path / "local"

    first = receiver.materialize_checkpoint(
        source_dir=str(source), local_checkpoint_dir=str(local), target_version=1
    )
    assert first["status"] == "materialized"
    assert (local / "model-00001.safetensors").read_bytes() == b"weights-v1"

    repeated = receiver.materialize_checkpoint(
        source_dir=str(source), local_checkpoint_dir=str(local), target_version=1
    )
    assert repeated["status"] == "already_applied"
    assert repeated["manifest_sha256"] == first["manifest_sha256"]


def test_invalid_source_and_malformed_checkpoint_are_rejected(tmp_path: Path):
    with pytest.raises(receiver.CheckpointReceiveError, match="does not exist"):
        receiver.materialize_checkpoint(
            source_dir=str(tmp_path / "missing"),
            local_checkpoint_dir=str(tmp_path / "local"),
            target_version=1,
        )

    source = tmp_path / "published"
    checkpoint = source / "weight_v000001"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors.index.json").write_text("not-json", encoding="utf-8")
    with pytest.raises(receiver.CheckpointReceiveError, match="malformed checkpoint index"):
        receiver.materialize_checkpoint(
            source_dir=str(source),
            local_checkpoint_dir=str(tmp_path / "local"),
            target_version=1,
        )


def test_stale_and_conflicting_versions_are_rejected(tmp_path: Path):
    source = tmp_path / "published"
    _publish(source, 1, b"one")
    _publish(source, 2, b"two")
    local = tmp_path / "local"
    receiver.materialize_checkpoint(source_dir=str(source), local_checkpoint_dir=str(local), target_version=2)

    with pytest.raises(receiver.StaleCheckpointError, match="stale checkpoint"):
        receiver.materialize_checkpoint(source_dir=str(source), local_checkpoint_dir=str(local), target_version=1)

    # Same version with different bytes is a conflict, never an overwrite.
    (source / "weight_v000002" / "model-00001.safetensors").write_bytes(b"changed")
    with pytest.raises(receiver.CheckpointConflictError, match="different manifest"):
        receiver.materialize_checkpoint(source_dir=str(source), local_checkpoint_dir=str(local), target_version=2)
    assert (local / "model-00001.safetensors").read_bytes() == b"two"


def test_failed_copy_keeps_old_checkpoint_available(tmp_path: Path, monkeypatch):
    source = tmp_path / "published"
    _publish(source, 1, b"old")
    _publish(source, 2, b"new")
    local = tmp_path / "local"
    receiver.materialize_checkpoint(source_dir=str(source), local_checkpoint_dir=str(local), target_version=1)

    original_copy2 = receiver.shutil.copy2
    calls = 0

    def fail_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls >= 2:
            raise OSError("simulated partial copy")
        return original_copy2(*args, **kwargs)

    monkeypatch.setattr(receiver.shutil, "copy2", fail_after_first)
    with pytest.raises(OSError, match="simulated partial copy"):
        receiver.materialize_checkpoint(source_dir=str(source), local_checkpoint_dir=str(local), target_version=2)
    assert json.loads((local / receiver._CHECKPOINT_MARKER).read_text())["version"] == 1
    assert (local / "model-00001.safetensors").read_bytes() == b"old"


def test_concurrent_same_version_pulls_are_serialized(tmp_path: Path):
    source = tmp_path / "published"
    _publish(source, 1)
    local = tmp_path / "local"

    def pull():
        return receiver.materialize_checkpoint(
            source_dir=str(source), local_checkpoint_dir=str(local), target_version=1
        )["status"]

    with ThreadPoolExecutor(max_workers=2) as pool:
        statuses = sorted(pool.map(lambda _: pull(), range(2)))
    assert statuses == ["already_applied", "materialized"]
    assert (local / "model.safetensors.index.json").is_file()


def test_materialize_rejects_paths_outside_configured_roots(tmp_path: Path):
    configured_source = tmp_path / "configured-source"
    other_source = tmp_path / "other-source"
    _publish(configured_source, 1)
    _publish(other_source, 1)
    configured_local = tmp_path / "configured-local"

    with pytest.raises(receiver.CheckpointReceiveError, match="configured checkpoint source"):
        receiver.materialize_checkpoint(
            source_dir=str(other_source),
            local_checkpoint_dir=str(configured_local),
            target_version=1,
            expected_source_dir=str(configured_source),
            expected_local_checkpoint_dir=str(configured_local),
        )

    with pytest.raises(receiver.CheckpointReceiveError, match="configured local checkpoint destination"):
        receiver.materialize_checkpoint(
            source_dir=str(configured_source),
            local_checkpoint_dir=str(tmp_path / "other-local"),
            target_version=1,
            expected_source_dir=str(configured_source),
            expected_local_checkpoint_dir=str(configured_local),
        )

    result = receiver.materialize_checkpoint(
        source_dir=str(configured_source),
        local_checkpoint_dir=str(configured_local),
        target_version=1,
        expected_source_dir=str(configured_source),
        expected_local_checkpoint_dir=str(configured_local),
    )
    assert result["status"] == "materialized"


def test_materialize_rejects_source_symlink(tmp_path: Path):
    source = tmp_path / "source"
    _publish(source, 1)
    source_link = tmp_path / "source-link"
    try:
        source_link.symlink_to(source, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"directory symlinks are unavailable: {exc}")

    with pytest.raises(receiver.CheckpointReceiveError, match="must not be a symlink"):
        receiver.materialize_checkpoint(
            source_dir=str(source_link),
            local_checkpoint_dir=str(tmp_path / "local"),
            target_version=1,
        )


def test_publish_checkpoint_directory_is_immutable_and_retryable(tmp_path: Path):
    destination = tmp_path / "weight_v000001"
    staging = tmp_path / ".weight_v000001.staging"
    _publish(tmp_path / "first", 1).rename(staging)

    assert receiver.publish_checkpoint_directory(staging, destination) == "published"
    assert (destination / "model-00001.safetensors").read_bytes() == b"weights-v1"

    retry_staging = tmp_path / ".weight_v000001.retry"
    _publish(tmp_path / "retry", 1).rename(retry_staging)
    assert receiver.publish_checkpoint_directory(retry_staging, destination) == "already_published"
    assert retry_staging.exists()

    conflict_staging = tmp_path / ".weight_v000001.conflict"
    _publish(tmp_path / "conflict", 1, b"different").rename(conflict_staging)
    with pytest.raises(receiver.CheckpointConflictError, match="different contents"):
        receiver.publish_checkpoint_directory(conflict_staging, destination)
    assert (destination / "model-00001.safetensors").read_bytes() == b"weights-v1"
