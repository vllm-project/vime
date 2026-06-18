"""CPU-only import stubs for tests under ``tests/utils``."""

from __future__ import annotations

import sys
from pathlib import Path

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

_unit_stubs.install_rollout_optional_stubs()
