from pathlib import Path

import pytest

from vime.backends.vllm_utils import checkpoint_receiver


def _checkpoint(source_dir: Path, version: int, content: bytes) -> Path:
    checkpoint = source_dir / f"weight_v{version:06d}"
    checkpoint.mkdir(parents=True)
    (checkpoint / "model.safetensors").write_bytes(content)
    return checkpoint


@pytest.mark.unit
def test_materialize_checkpoint_replaces_local_copy(tmp_path: Path) -> None:
    source_dir = tmp_path / "published"
    _checkpoint(source_dir, 1, b"new")
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "model.safetensors").write_bytes(b"old")

    result = checkpoint_receiver.materialize_checkpoint(
        source_dir=str(source_dir),
        local_checkpoint_dir=str(local_dir),
        target_version=1,
    )

    assert result == {
        "success": True,
        "version": 1,
        "local_checkpoint_dir": str(local_dir),
    }
    assert (local_dir / "model.safetensors").read_bytes() == b"new"


@pytest.mark.unit
@pytest.mark.parametrize("version", [0, -1, True, "1"])
def test_materialize_checkpoint_rejects_invalid_version(tmp_path: Path, version: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        checkpoint_receiver.materialize_checkpoint(
            source_dir=str(tmp_path / "published"),
            local_checkpoint_dir=str(tmp_path / "local"),
            target_version=version,  # type: ignore[arg-type]
        )


@pytest.mark.unit
def test_failed_copy_preserves_local_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    source_dir = tmp_path / "published"
    _checkpoint(source_dir, 1, b"new")
    local_dir = tmp_path / "local"
    local_dir.mkdir()
    (local_dir / "model.safetensors").write_bytes(b"old")

    def fail_copy(*args: object, **kwargs: object) -> None:
        raise OSError("copy failed")

    monkeypatch.setattr(checkpoint_receiver.shutil, "copytree", fail_copy)

    with pytest.raises(OSError, match="copy failed"):
        checkpoint_receiver.materialize_checkpoint(
            source_dir=str(source_dir),
            local_checkpoint_dir=str(local_dir),
            target_version=1,
        )

    assert (local_dir / "model.safetensors").read_bytes() == b"old"
