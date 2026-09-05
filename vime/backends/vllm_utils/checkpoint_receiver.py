"""Copy a published checkpoint to rollout-host-local storage."""

from __future__ import annotations

import shutil
import uuid
from pathlib import Path


def materialize_checkpoint(
    *,
    local_checkpoint_dir: str,
    source_dir: str,
    target_version: int,
) -> dict[str, object]:
    """Copy one complete checkpoint and atomically make it active."""

    if isinstance(target_version, bool) or not isinstance(target_version, int) or target_version <= 0:
        raise ValueError("target_version must be a positive integer")

    source = Path(source_dir).expanduser() / f"weight_v{target_version:06d}"
    if not source.is_dir():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")

    destination = Path(local_checkpoint_dir).expanduser()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and destination.resolve() == source.resolve():
        raise ValueError("local checkpoint directory must differ from the source")

    suffix = uuid.uuid4().hex
    staging = destination.parent / f".{destination.name}.staging-{suffix}"
    backup = destination.parent / f".{destination.name}.backup-{suffix}"

    try:
        shutil.copytree(source, staging)
        if destination.exists():
            destination.replace(backup)
        staging.replace(destination)
    except Exception:
        if backup.exists() and not destination.exists():
            backup.replace(destination)
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    shutil.rmtree(backup, ignore_errors=True)
    return {
        "success": True,
        "version": target_version,
        "local_checkpoint_dir": str(destination),
    }
