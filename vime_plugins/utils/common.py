"""Small helpers shared across the AI21 vime plugins.

These were previously duplicated verbatim in several plugin modules
(``_cfg`` in 7 files, ``_flatten``/``_flatten_group`` in 3). Keeping a single
copy here avoids the drift that comes with copy-paste while staying fully
additive to the upstream tree.
"""

from __future__ import annotations

import os
from typing import Any

__all__ = ["cfg", "flatten_samples"]


def cfg(args, attr: str, env: str, default: Any = None) -> Any:
    """Resolve a config value from ``args``, then an env var, then ``default``.

    ``args`` attributes win when set (not ``None``); otherwise fall back to the
    environment variable ``env`` and finally to ``default``.
    """
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


def flatten_samples(groups) -> list:
    """Flatten arbitrarily nested ``list``-of-``list`` groups into a flat list.

    The standard rollout path passes ``list[Sample]``; compact/fanout passes
    ``list[list[Sample]]``. Recurses to any depth while preserving order, so it
    is safe for callers that map results back onto the original samples.
    """
    out: list = []
    for item in groups:
        if isinstance(item, list):
            out.extend(flatten_samples(item))
        else:
            out.append(item)
    return out
