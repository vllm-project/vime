"""Temporary CI probe for Buildkite #752. Remove together with its launcher hook and A5 case flag."""

import ctypes
import os
import re
import socket
import subprocess
from datetime import timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

WORLD_SIZE = 4


def read_card_ids() -> dict[int, str]:
    """Map CANN physical chip IDs to the NPU card IDs reported by npu-smi."""
    try:
        output = subprocess.check_output(["npu-smi", "info", "-m"], text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        print(f"[A5 HCCL] npu-smi mapping unavailable: {exc}", flush=True)
        return {}

    print(f"[A5 HCCL] npu-smi info -m\n{output}", flush=True)
    cards = {}
    columns = []
    for line in output.splitlines():
        if "Chip Phy-ID" in line and "NPU ID" in line:
            columns = re.split(r"\s{2,}|\t", line.strip())
        elif columns:
            row = dict(zip(columns, line.split(), strict=False))
            if row.get("Chip Phy-ID", "").isdigit() and row.get("NPU ID", "").isdigit():
                cards[int(row["Chip Phy-ID"])] = row["NPU ID"]
    return cards


def physical_chip_id(device: int) -> int:
    # The runtime applies both ASCEND_RT_VISIBLE_DEVICES and container device remapping.
    lookup = ctypes.CDLL("libascendcl.so").aclrtGetPhyDevIdByUserDevId
    lookup.argtypes = [ctypes.c_int32, ctypes.POINTER(ctypes.c_int32)]
    lookup.restype = ctypes.c_int
    physical = ctypes.c_int32()
    status = lookup(device, ctypes.byref(physical))
    if status:
        raise RuntimeError(f"aclrtGetPhyDevIdByUserDevId({device}) returned {status}")
    return physical.value


def probe_rank(rank: int, rendezvous: str, cards: dict[int, str]) -> None:
    import torch
    import torch.distributed as dist
    import torch_npu

    prefix = f"[A5 HCCL rank={rank} pid={os.getpid()}]"
    torch_npu.npu.set_device(rank)
    device = torch_npu.npu.current_device()
    try:
        chip = physical_chip_id(device)
        mapping = f"physical_chip={chip} npu_card={cards.get(chip, 'unknown')}"
    except (AttributeError, OSError, RuntimeError) as exc:
        mapping = f"physical mapping unavailable: {exc}"
    print(f"{prefix} local_device={device} {mapping}; initializing HCCL", flush=True)
    dist.init_process_group(
        "hccl", init_method=rendezvous, world_size=WORLD_SIZE, rank=rank, timeout=timedelta(seconds=90)
    )
    value = torch.tensor([rank + 1], dtype=torch.float32, device=f"npu:{device}")
    print(f"{prefix} entering all_reduce", flush=True)
    dist.all_reduce(value)
    result = value.item()
    print(f"{prefix} all_reduce sum={result} expected=10", flush=True)
    if result != 10:
        raise RuntimeError(f"{prefix} incorrect all_reduce result: {result}")
    dist.destroy_process_group()


if __name__ == "__main__":
    mask = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
    devices = mask.split(",")
    print(f"[A5 HCCL] host={socket.gethostname()} rollout ASCEND_RT_VISIBLE_DEVICES={mask}", flush=True)
    if len(devices) != WORLD_SIZE or len(set(devices)) != WORLD_SIZE or not all(d.isdigit() for d in devices):
        raise ValueError(f"Expected four distinct rollout device IDs, got {mask!r}")
    os.environ.update(
        ASCEND_SLOG_PRINT_TO_STDOUT="1", ASCEND_MODULE_LOG_LEVEL="HCCL=1:HCCP=1", HCCL_ENTRY_LOG_ENABLE="1"
    )

    import torch.multiprocessing as mp

    with TemporaryDirectory(prefix="vime-a5-hccl-") as tmp:
        mp.spawn(probe_rank, args=((Path(tmp) / "store").as_uri(), read_card_ids()), nprocs=WORLD_SIZE)
    print("[A5 HCCL] All four ranks passed and exited. Starting vLLM.", flush=True)
