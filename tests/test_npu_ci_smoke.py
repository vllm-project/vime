"""Minimal hardware smoke test for the NPU Buildkite lane."""

from __future__ import annotations

import os
import tempfile


def main() -> None:
    import torch

    __import__("torch_npu")
    npu = getattr(torch, "npu", None)
    if npu is None or not npu.is_available():
        raise RuntimeError("NPU CI image does not expose a usable torch.npu device")

    values = torch.arange(1024, dtype=torch.float32, device="npu")
    result = values.mul(2).add(1)
    npu.synchronize()
    expected = torch.arange(1024, dtype=torch.float32).mul(2).add(1)
    torch.testing.assert_close(result.cpu(), expected)

    from vime_plugins.optimizers.nvme_stream import _align, _Stager

    fd, path = tempfile.mkstemp(prefix="vime_npu_nvme_smoke_")
    try:
        stager = _Stager(1 << 16)
        stager.transfer(fd, 0, values, torch.float32, to_disk=True)
        values.zero_()
        stager.transfer(fd, 0, values, torch.float32, to_disk=False)
        npu.synchronize()
        torch.testing.assert_close(values.cpu(), torch.arange(1024, dtype=torch.float32))
    finally:
        os.close(fd)
        os.unlink(path)

    assert _align(1) == 4096
    print("NPU CI smoke passed: device arithmetic and NVMe staging")


if __name__ == "__main__":
    main()
