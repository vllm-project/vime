"""Transactional receiver for full HuggingFace checkpoints published on disk.

The vLLM HTTP endpoint is intentionally kept thin.  This module owns the
filesystem transaction so it can be tested without importing vLLM or starting
an HTTP server.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import threading
import uuid
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

_CHECKPOINT_MARKER = ".vime_checkpoint_manifest.json"
_INDEX_NAMES = ("model.safetensors.index.json", "pytorch_model.bin.index.json")
_SINGLE_FILE_WEIGHT_NAMES = ("model.safetensors", "pytorch_model.bin")
_MATERIALIZE_LOCK = threading.RLock()


class CheckpointReceiveError(ValueError):
    """A malformed request or checkpoint that should be returned as HTTP 400."""

    status_code = 400


class StaleCheckpointError(CheckpointReceiveError):
    """A checkpoint older than the active local checkpoint."""

    status_code = 409


class CheckpointConflictError(CheckpointReceiveError):
    """A version was already applied with different checkpoint contents."""

    status_code = 409


def materialize_checkpoint(
    *,
    local_checkpoint_dir: str,
    source_dir: str,
    target_version: int,
    expected_local_checkpoint_dir: str | None = None,
    expected_source_dir: str | None = None,
) -> dict[str, Any]:
    """Materialize one published checkpoint into a host-local directory.

    The source is validated and checksummed before any destination is touched.
    Files are copied into a sibling temporary directory, then that directory is
    swapped into place.  A failed copy or swap therefore leaves the previous
    local checkpoint intact.
    """

    version = _parse_version(target_version)
    if expected_source_dir is not None:
        expected_source_root = _resolve_existing_directory(expected_source_dir, "configured source_dir")
        source_root = _resolve_existing_directory(source_dir, "source_dir")
        if source_root != expected_source_root:
            raise CheckpointReceiveError("source_dir does not match the configured checkpoint source")
    else:
        source_root = _resolve_existing_directory(source_dir, "source_dir")
    version_dir = source_root / f"weight_v{version:06d}"
    version_dir = _resolve_existing_directory(version_dir, "checkpoint version")

    if expected_local_checkpoint_dir is not None:
        expected_local_dir = _prepare_local_directory(expected_local_checkpoint_dir)
        local_dir = _resolve_local_directory(local_checkpoint_dir)
        if local_dir != expected_local_dir:
            raise CheckpointReceiveError(
                "local_checkpoint_dir does not match the configured local checkpoint destination"
            )
    else:
        local_dir = _prepare_local_directory(local_checkpoint_dir)
    if local_dir.exists() and local_dir.resolve() == version_dir:
        raise CheckpointReceiveError("local_checkpoint_dir must differ from source checkpoint")

    with _MATERIALIZE_LOCK:
        source_manifest = _build_checkpoint_manifest(version_dir)
        source_manifest_hash = _manifest_hash(source_manifest["files"])
        current = _read_marker(local_dir)

        if current is not None:
            current_version = current["version"]
            if current_version > version:
                raise StaleCheckpointError(
                    f"stale checkpoint version {version}; active version is {current_version}"
                )
            if current_version == version:
                if current["manifest_sha256"] != source_manifest_hash:
                    raise CheckpointConflictError(
                        f"checkpoint version {version} is already active with a different manifest"
                    )
                if _local_manifest_is_valid(local_dir, current):
                    return _result("already_applied", version, current)

        staging_dir = Path(tempfile.mkdtemp(prefix=f".{local_dir.name}.", dir=local_dir.parent))
        try:
            _copy_manifest_files(version_dir, staging_dir, source_manifest["files"])
            marker = {
                "version": version,
                "manifest_sha256": source_manifest_hash,
                "files": source_manifest["files"],
            }
            _write_json(staging_dir / _CHECKPOINT_MARKER, marker)
            _atomic_replace_directory(staging_dir, local_dir)
            staging_dir = None  # ownership transferred to local_dir
        except Exception:
            if staging_dir is not None:
                shutil.rmtree(staging_dir, ignore_errors=True)
            raise

        return _result("materialized", version, marker)


def _parse_version(value: int) -> int:
    if isinstance(value, bool):
        raise CheckpointReceiveError("target_version must be a positive integer")
    try:
        version = int(value)
    except (TypeError, ValueError) as exc:
        raise CheckpointReceiveError("target_version must be a positive integer") from exc
    if version <= 0 or str(value).strip() != str(version):
        raise CheckpointReceiveError("target_version must be a positive integer")
    return version


def _resolve_existing_directory(value: str | Path, name: str) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CheckpointReceiveError(f"{name} must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_symlink():
        raise CheckpointReceiveError(f"{name} must not be a symlink: {path}")
    try:
        resolved = path.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReceiveError(f"{name} does not exist: {path}") from exc
    if not resolved.is_dir():
        raise CheckpointReceiveError(f"{name} must be a directory: {path}")
    return resolved


def _prepare_local_directory(value: str | Path) -> Path:
    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CheckpointReceiveError("local_checkpoint_dir must be a non-empty path")
    path = Path(value).expanduser()
    if path.exists() and (path.is_symlink() or not path.is_dir()):
        raise CheckpointReceiveError(f"local_checkpoint_dir must be a directory: {path}")
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReceiveError(f"cannot prepare local checkpoint parent: {path.parent}") from exc
    return parent / path.name


def _resolve_local_directory(value: str | Path) -> Path:
    """Resolve an untrusted destination without creating caller-selected parents."""

    if not isinstance(value, (str, Path)) or not str(value).strip():
        raise CheckpointReceiveError("local_checkpoint_dir must be a non-empty path")
    path = Path(value).expanduser()
    if path.is_symlink() or (path.exists() and not path.is_dir()):
        raise CheckpointReceiveError(f"local_checkpoint_dir must be a directory: {path}")
    try:
        parent = path.parent.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReceiveError(f"local checkpoint parent does not exist: {path.parent}") from exc
    return parent / path.name


def publish_checkpoint_directory(staging_dir: str | Path, destination_dir: str | Path) -> str:
    """Publish an immutable checkpoint directory, allowing an identical retry."""

    if not isinstance(staging_dir, (str, Path)) or not str(staging_dir).strip():
        raise CheckpointReceiveError("checkpoint staging directory must be a non-empty path")
    staging_path = Path(staging_dir).expanduser()
    if staging_path.is_symlink():
        raise CheckpointReceiveError(f"checkpoint staging directory must not be a symlink: {staging_path}")
    try:
        staging_parent = staging_path.parent.resolve(strict=True)
    except OSError as exc:
        raise CheckpointReceiveError(
            f"checkpoint staging directory parent does not exist: {staging_path.parent}"
        ) from exc
    destination = _prepare_local_directory(destination_dir)
    if staging_parent != destination.parent:
        raise CheckpointReceiveError("checkpoint staging and destination directories must be siblings")
    staging_path = staging_parent / staging_path.name

    try:
        staging = _resolve_existing_directory(staging_path, "checkpoint staging directory")
    except CheckpointReceiveError:
        if not staging_path.exists() and _published_checkpoint_is_valid(destination):
            return "published_by_peer"
        raise

    if destination.exists():
        destination_manifest = _build_checkpoint_manifest(destination)
        try:
            staging_manifest = _build_checkpoint_manifest(staging)
        except (CheckpointReceiveError, OSError):
            if not staging.exists() and _published_checkpoint_is_valid(destination):
                return "published_by_peer"
            raise
        destination_hash = _manifest_hash(destination_manifest["files"])
        staging_hash = _manifest_hash(staging_manifest["files"])
        if destination_hash != staging_hash:
            raise CheckpointConflictError(
                f"published checkpoint already exists with different contents: {destination}"
            )
        return "already_published"

    try:
        os.replace(staging, destination)
    except FileNotFoundError:
        # On a shared filesystem another writer rank may have renamed the same
        # staging directory after the distributed write barrier.
        if _published_checkpoint_is_valid(destination):
            return "published_by_peer"
        raise
    _fsync_directory(destination.parent)
    return "published"


def _published_checkpoint_is_valid(destination: Path) -> bool:
    if not destination.is_dir() or destination.is_symlink():
        return False
    try:
        _build_checkpoint_manifest(destination)
    except (CheckpointReceiveError, OSError):
        return False
    return True


def _build_checkpoint_manifest(root: Path) -> dict[str, Any]:
    index_candidates = [root / name for name in _INDEX_NAMES if (root / name).exists()]
    if any(path.is_symlink() for path in index_candidates):
        raise CheckpointReceiveError("checkpoint index must not be a symlink")
    indexes = [path for path in index_candidates if path.is_file()]
    if len(indexes) > 1:
        raise CheckpointReceiveError(
            f"checkpoint must contain exactly one supported weight index: {', '.join(_INDEX_NAMES)}"
        )

    if not indexes:
        direct_weights = [root / name for name in _SINGLE_FILE_WEIGHT_NAMES if (root / name).is_file()]
        if not direct_weights or any(path.stat().st_size <= 0 for path in direct_weights):
            raise CheckpointReceiveError(
                "checkpoint must contain a supported weight index or a non-empty single-file weight"
            )
        return _manifest_files(root)

    index_path = indexes[0]
    try:
        with index_path.open("r", encoding="utf-8") as file:
            index = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointReceiveError(f"malformed checkpoint index: {index_path.name}") from exc

    weight_map = index.get("weight_map") if isinstance(index, dict) else None
    if not isinstance(weight_map, dict) or not weight_map:
        raise CheckpointReceiveError("checkpoint index must contain a non-empty weight_map")
    for tensor_name, filename in weight_map.items():
        if not isinstance(tensor_name, str) or not isinstance(filename, str):
            raise CheckpointReceiveError("checkpoint index contains a non-string weight mapping")
        _resolve_checkpoint_file(root, filename)

    return _manifest_files(root)


def _manifest_files(root: Path) -> dict[str, Any]:
    files: dict[str, dict[str, Any]] = {}
    for path in sorted(root.rglob("*")):
        if path.is_symlink():
            raise CheckpointReceiveError(f"checkpoint contains a symlink: {path.relative_to(root)}")
        if not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        try:
            size = path.stat().st_size
        except OSError as exc:
            raise CheckpointReceiveError(f"cannot stat checkpoint file: {relative}") from exc
        files[relative] = {"size": size, "sha256": _sha256(path)}
    if not files:
        raise CheckpointReceiveError("checkpoint contains no files")
    return {"files": files}


def _resolve_checkpoint_file(root: Path, filename: str) -> Path:
    candidate = root / filename
    try:
        resolved = candidate.resolve(strict=True)
        resolved.relative_to(root)
    except (OSError, ValueError) as exc:
        raise CheckpointReceiveError(f"checkpoint index references missing or unsafe file: {filename}") from exc
    if candidate.is_symlink() or not resolved.is_file() or resolved.stat().st_size <= 0:
        raise CheckpointReceiveError(f"checkpoint index references invalid file: {filename}")
    return resolved


def _copy_manifest_files(source: Path, destination: Path, files: dict[str, dict[str, Any]]) -> None:
    for relative, metadata in files.items():
        source_file = source / Path(relative)
        destination_file = destination / Path(relative)
        destination_file.parent.mkdir(parents=True, exist_ok=True)
        try:
            shutil.copy2(source_file, destination_file)
            copied_size = destination_file.stat().st_size
            copied_hash = _sha256(destination_file)
        except OSError as exc:
            raise OSError(f"failed to copy checkpoint file {relative}: {exc}") from exc
        if copied_size != metadata["size"] or copied_hash != metadata["sha256"]:
            raise OSError(f"checkpoint file changed while copying: {relative}")


def _read_marker(local_dir: Path) -> dict[str, Any] | None:
    marker_path = local_dir / _CHECKPOINT_MARKER
    if not marker_path.exists():
        return None
    try:
        with marker_path.open("r", encoding="utf-8") as file:
            marker = json.load(file)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise CheckpointReceiveError("local checkpoint version marker is malformed") from exc
    if (
        not isinstance(marker, dict)
        or isinstance(marker.get("version"), bool)
        or not isinstance(marker.get("version"), int)
        or marker["version"] <= 0
        or not isinstance(marker.get("manifest_sha256"), str)
        or not isinstance(marker.get("files"), dict)
    ):
        raise CheckpointReceiveError("local checkpoint version marker is malformed")
    return marker


def _local_manifest_is_valid(local_dir: Path, marker: dict[str, Any]) -> bool:
    if _manifest_hash(marker["files"]) != marker["manifest_sha256"]:
        return False
    for relative, metadata in marker["files"].items():
        if not isinstance(relative, str) or not isinstance(metadata, dict):
            return False
        path = local_dir / Path(relative)
        try:
            path.resolve(strict=True).relative_to(local_dir.resolve(strict=True))
            if path.is_symlink() or not path.is_file() or path.stat().st_size != metadata.get("size"):
                return False
            if _sha256(path) != metadata.get("sha256"):
                return False
        except (OSError, ValueError):
            return False
    return True


def _manifest_hash(files: dict[str, Any]) -> str:
    encoded = json.dumps(files, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_json(path: Path, value: dict[str, Any]) -> None:
    with path.open("w", encoding="utf-8") as file:
        json.dump(value, file, sort_keys=True, separators=(",", ":"))
        file.flush()
        os.fsync(file.fileno())


def _fsync_directory(path: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    try:
        descriptor = os.open(path, os.O_RDONLY | os.O_DIRECTORY)
    except OSError:
        logger.warning("Could not open checkpoint directory for fsync: %s", path, exc_info=True)
        return
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _atomic_replace_directory(staging: Path, destination: Path) -> None:
    backup: Path | None = None
    if destination.exists() or destination.is_symlink():
        if destination.is_symlink() or not destination.is_dir():
            raise CheckpointReceiveError(f"local checkpoint destination is not a directory: {destination}")
        backup = destination.parent / f".{destination.name}.backup-{uuid.uuid4().hex}"
        os.replace(destination, backup)
    try:
        os.replace(staging, destination)
    except Exception:
        if backup is not None and backup.exists() and not destination.exists():
            os.replace(backup, destination)
        raise
    _fsync_directory(destination.parent)
    if backup is not None:
        try:
            shutil.rmtree(backup)
        except OSError:
            logger.warning("Could not remove old checkpoint backup %s", backup, exc_info=True)


def _result(status: str, version: int, marker: dict[str, Any]) -> dict[str, Any]:
    return {
        "success": True,
        "status": status,
        "version": version,
        "manifest_sha256": marker["manifest_sha256"],
        "files": len(marker["files"]),
    }
