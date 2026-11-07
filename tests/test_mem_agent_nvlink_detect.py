"""Regression tests for ``mem_agent_detect_nvlink`` in examples/mem_agent/_common.sh.

The launcher sources that file under ``set -euo pipefail``, so the detector has
to tell three cases apart that previously collapsed into one:

* the topology query succeeds and matches nothing -- a machine with no NVLink.
  This is legal: it must set ``NCCL_NVLS_ENABLE=0`` and return 0, not abort the
  launcher through ``pipefail``;
* the topology query succeeds and matches -- set ``NCCL_NVLS_ENABLE=1``;
* the topology query fails, or produces nothing at all -- must be loud on stderr
  and return non-zero rather than be reported as "no NVLink".

Every case runs in its own bash subprocess with only ``nvidia-smi`` mocked, so no
GPU, Ray or vLLM process is touched and ``mem_agent_cleanup`` is never called.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
COMMON_SH = REPO_ROOT / "examples" / "mem_agent" / "_common.sh"

# Only the connectivity matrix is printed by the real tool; the legend below it
# mentions "NV#" (no digits), which must not be counted as a match.
FAKE_NVIDIA_SMI = """#!/bin/sh
set -eu
if [ "$#" -ne 2 ] || [ "$1" != "topo" ] || [ "$2" != "-m" ]; then
    echo "unexpected nvidia-smi arguments" >&2
    exit 64
fi
case "$FAKE_TOPO_CASE" in
  pci_only)
    printf '\\tGPU0\\tGPU1\\nGPU0\\tX\\tPHB\\nGPU1\\tPHB\\tX\\n'
    ;;
  pci_only_with_legend)
    printf '\\tGPU0\\tGPU1\\nGPU0\\tX\\tPHB\\nGPU1\\tPHB\\tX\\n\\nLegend:\\n\\n'
    printf '  NV#  = Connection traversing a bonded set of # NVLinks\\n'
    ;;
  nv_link)
    printf '\\tGPU0\\tGPU1\\nGPU0\\tX\\tNV2\\nGPU1\\tNV2\\tX\\n'
    ;;
  empty)
    exit 0
    ;;
  query_error)
    echo "mock topology query failure" >&2
    exit 17
    ;;
  *)
    exit 65
    ;;
esac
"""

# `set -euo pipefail` mirrors the real launcher. The function is called directly
# rather than inside `if ...; then`, because an `if` condition suppresses errexit
# and would hide the defect this file exists to catch.
CHILD_SCRIPT = """
set -euo pipefail
source "$1"
mem_agent_detect_nvlink
printf '\\n__NVLS__=%s\\n' "${NCCL_NVLS_ENABLE-UNSET}"
"""

UNSET = "UNSET"


def _tool_path(tmp_path: Path, *, with_nvidia_smi: bool) -> str:
    """Build a PATH that always exposes grep/wc but only optionally nvidia-smi."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir(exist_ok=True)
    if with_nvidia_smi:
        smi = bin_dir / "nvidia-smi"
        smi.write_text(FAKE_NVIDIA_SMI, encoding="utf-8")
        smi.chmod(0o755)
        return str(bin_dir) + os.pathsep + os.environ["PATH"]
    # No nvidia-smi anywhere: expose only what the detector itself shells out to.
    for tool in ("grep", "wc"):
        real = shutil.which(tool)
        assert real is not None, f"{tool} is required to run these tests"
        (bin_dir / tool).symlink_to(real)
    return str(bin_dir)


def _run_detector(tmp_path: Path, *, case: str | None, with_nvidia_smi: bool, preset_nvls: str | None):
    env = {
        "PATH": _tool_path(tmp_path, with_nvidia_smi=with_nvidia_smi),
        "HOME": str(tmp_path),
    }
    if case is not None:
        env["FAKE_TOPO_CASE"] = case
    if preset_nvls is not None:
        env["NCCL_NVLS_ENABLE"] = preset_nvls
    bash = shutil.which("bash")
    assert bash is not None, "bash is required to run these tests"
    return subprocess.run(
        [bash, "-c", CHILD_SCRIPT, "vime-mem-agent-nvlink", str(COMMON_SH)],
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
        check=False,
    )


@pytest.fixture
def fake_smi(tmp_path: Path) -> Path:
    return tmp_path


def _nvls_of(proc: subprocess.CompletedProcess[str]) -> str:
    match = re.search(r"^__NVLS__=(.*)$", proc.stdout, re.MULTILINE)
    return match.group(1) if match else UNSET


def test_common_sh_exists() -> None:
    assert COMMON_SH.is_file(), f"missing {COMMON_SH}"


@pytest.mark.parametrize(
    ("case", "expected_nvls"),
    [
        ("pci_only", "0"),
        ("pci_only_with_legend", "0"),
        ("nv_link", "1"),
    ],
)
def test_zero_match_is_a_legal_no_nvlink_result(fake_smi: Path, case: str, expected_nvls: str) -> None:
    """A successful query with no NVLink must not abort the launcher."""
    proc = _run_detector(fake_smi, case=case, with_nvidia_smi=True, preset_nvls=None)
    assert proc.returncode == 0, f"detector aborted: stderr={proc.stderr!r}"
    assert _nvls_of(proc) == expected_nvls


@pytest.mark.parametrize("case", ["query_error", "empty"])
def test_failed_or_empty_query_is_not_reported_as_no_nvlink(fake_smi: Path, case: str) -> None:
    """Neither a failing query nor an empty matrix may be silently read as zero NVLink."""
    proc = _run_detector(fake_smi, case=case, with_nvidia_smi=True, preset_nvls=None)
    assert proc.returncode != 0, f"{case} was accepted as a valid topology"
    assert _nvls_of(proc) == UNSET, "NCCL_NVLS_ENABLE was exported despite an unusable query"
    assert proc.stderr.strip(), f"{case} failed without a diagnostic on stderr"


def test_missing_nvidia_smi_is_not_reported_as_no_nvlink(fake_smi: Path) -> None:
    proc = _run_detector(fake_smi, case=None, with_nvidia_smi=False, preset_nvls=None)
    assert proc.returncode != 0
    assert _nvls_of(proc) == UNSET
    assert proc.stderr.strip()


def test_query_error_keeps_the_underlying_message(fake_smi: Path) -> None:
    """The tool's own stderr must not be discarded, or the failure is undiagnosable."""
    proc = _run_detector(fake_smi, case="query_error", with_nvidia_smi=True, preset_nvls=None)
    assert "mock topology query failure" in proc.stderr


def test_detector_overwrites_a_preconfigured_value(fake_smi: Path) -> None:
    """A stale inherited NCCL_NVLS_ENABLE must be replaced by the detected topology."""
    proc = _run_detector(fake_smi, case="pci_only", with_nvidia_smi=True, preset_nvls="1")
    assert proc.returncode == 0
    assert _nvls_of(proc) == "0"
