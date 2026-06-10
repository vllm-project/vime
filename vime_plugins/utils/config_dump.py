"""Resolved-config dump, the vime port of ai21-verl's ``config_dump.py``.

Writes the fully-resolved run arguments (every attribute on ``args``, after CLI parsing,
``--custom-config-path`` YAML overrides, and derived defaults) to a JSON file at run start,
and optionally mirrors it to GCS — a durable record of exactly what a run was launched with.

vime has no core "run started" hook, so this stays additive: the AI21 plugin entrypoints
(snoozing filter, length-reward post-process, ai21 evaluators rm) call
:func:`maybe_dump_resolved_config` on their first invocation; it is a no-op unless enabled.
You can also call :func:`dump_args` directly from any custom function that receives ``args``.

Config is read from ``args`` if present (settable via ``--custom-config-path`` YAML),
else the matching env var:

    config_dump_path        AI21_CONFIG_DUMP_PATH       enables dumping; the output JSON path.
                                                        The literal value "auto" resolves to
                                                        ``<args.save or cwd>/resolved_config.json``.
    config_dump_gcs_dest    AI21_CONFIG_DUMP_GCS_DEST   optional ``gs://`` destination; the JSON is
                                                        copied there via gcs_sync (needs gsutil).
"""

import json
import logging
import os
import threading
from pathlib import Path

logger = logging.getLogger(__name__)

__all__ = ["dump_args", "maybe_dump_resolved_config"]

_dump_lock = threading.Lock()
_dumped = False


def _cfg(args, attr, env, default=None):
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


def _json_safe(value):
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        if isinstance(value, dict):
            return {str(k): _json_safe(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, set)):
            return [_json_safe(v) for v in value]
        return repr(value)


def dump_args(args, path: str | os.PathLike, gcs_dest: str | None = None) -> Path:
    """Write all attributes of ``args`` as JSON to ``path``; optionally copy to ``gcs_dest``."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    resolved = {k: _json_safe(v) for k, v in sorted(vars(args).items()) if not k.startswith("_")}
    with open(path, "w") as f:
        json.dump(resolved, f, indent=2)
    logger.info(f"Dumped resolved config ({len(resolved)} keys) to {path}")

    if gcs_dest:
        try:
            from vime_plugins.checkpoint.gcs_sync import sync_file_gs

            sync_file_gs(str(path), gcs_dest.rstrip("/") + "/" + path.name)
        except Exception:
            logger.exception(f"Failed to sync resolved config to {gcs_dest}")
    return path


def maybe_dump_resolved_config(args) -> Path | None:
    """Dump the resolved config once per process, if enabled (else no-op).

    Safe to call from any hot path — after the first call it returns immediately.
    """
    global _dumped
    if _dumped:
        return None
    with _dump_lock:
        if _dumped:
            return None
        _dumped = True

    path = _cfg(args, "config_dump_path", "AI21_CONFIG_DUMP_PATH")
    if not path:
        return None
    if str(path) == "auto":
        path = Path(getattr(args, "save", None) or ".") / "resolved_config.json"
    gcs_dest = _cfg(args, "config_dump_gcs_dest", "AI21_CONFIG_DUMP_GCS_DEST")
    try:
        return dump_args(args, path, gcs_dest=gcs_dest)
    except Exception:
        logger.exception("Failed to dump resolved config")
        return None


def reset_dump_state() -> None:
    """Clear the once-per-process guard (used by tests)."""
    global _dumped
    _dumped = False
