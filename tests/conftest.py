"""Make the CPU test suite independent of the runner's accelerator.

``vime/backends/megatron_utils/__init__.py`` installs the training backend's
vendor layer at module import time, so on a host with Ascend devices visible,
importing anything under that package pulls in ``megatron_adaptor`` and the
real Megatron modules -- through whatever stubs a test installed, and whether
or not the test has anything to do with NPU. The same host makes the
weight-transfer seam resolve to the Ascend factories, which import
``vllm_ascend``.

Hide the devices rather than pin a backend: naming one explicitly would make
selection fail fast on a runner that does not have it, which is most of them.
With none visible the suite falls back to the neutral defaults on
``Accelerator``, which are the GPU answers these tests were written against.

Tests that exercise selection itself clear or set these variables in their
own fixtures and are unaffected.
"""

from __future__ import annotations

import os

import pytest


def pytest_configure(config: pytest.Config) -> None:
    # Set before any test module is imported: the installation this guards
    # against runs at import time, not inside a fixture.
    os.environ.setdefault("ASCEND_RT_VISIBLE_DEVICES", "-1")
