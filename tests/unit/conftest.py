"""Unit-test collection hooks (e.g. stub optional heavy deps on dev machines)."""

from __future__ import annotations

import sys
from unittest.mock import MagicMock


def _ensure_ray_stub() -> None:
    if "ray" in sys.modules:
        return
    ray = MagicMock()
    sys.modules["ray"] = ray
    sys.modules["ray._private"] = MagicMock()
    sys.modules["ray._private.services"] = MagicMock()
    sys.modules["ray.actor"] = MagicMock()


_ensure_ray_stub()
