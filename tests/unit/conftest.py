"""Unit-test collection hooks (e.g. stub optional heavy deps on dev machines)."""

from __future__ import annotations

import importlib.util
import sys
from unittest.mock import MagicMock


def _ensure_ray_stub() -> None:
    # Only stub ray when it is genuinely absent (bare-deps dev machine). When the real ray is
    # installed (as in the CI image), a bare MagicMock stub shadows it and breaks unrelated
    # submodule imports like ``from ray.util.placement_group import placement_group`` in
    # sibling test packages — this conftest's module-level mutation leaks session-wide.
    if "ray" in sys.modules:
        return
    try:
        if importlib.util.find_spec("ray") is not None:
            return
    except (ImportError, ValueError):
        pass
    ray = MagicMock()
    sys.modules["ray"] = ray
    sys.modules["ray._private"] = MagicMock()
    sys.modules["ray._private.services"] = MagicMock()
    sys.modules["ray.actor"] = MagicMock()


_ensure_ray_stub()
