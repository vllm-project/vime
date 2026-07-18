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
