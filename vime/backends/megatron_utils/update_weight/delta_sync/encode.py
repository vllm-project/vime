"""On-wire schema for values-only seeds and sparse delta flushes.

The manifest matches verl's delta checkpoint engine. Sparse positions are
absolute int32 offsets in the flattened final Hugging Face tensor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import torch

DeltaEncodingName = Literal["indices", "dense"]


@dataclass
class DeltaParam:
    name: str
    dtype: str
    shape: list[int]
    pos_start: int
    pos_end: int
    pos_width: int
    val_start: int
    val_end: int


def checksum(positions: torch.Tensor, values: torch.Tensor) -> int:
    """Return verl's XOR checksum for one homogeneous-dtype payload."""
    p = int(torch.hash_tensor(positions).item()) if positions.numel() else 0
    v = int(torch.hash_tensor(values).item()) if values.numel() else 0
    # Keep the wire value JSON/msgpack-safe. ``hash_tensor`` yields 64-bit
    # values, but the left shift can otherwise create a Python integer wider
    # than uint64 and the vLLM HTTP control plane rejects it.
    return (p ^ (v << 1)) & ((1 << 64) - 1)


@dataclass
class DeltaFlush:
    encoding: DeltaEncodingName
    params: list[DeltaParam]
    positions: torch.Tensor
    values: torch.Tensor
    checksum: int

    @property
    def nnz(self) -> int:
        return self.values.numel()

    @property
    def wire_bytes(self) -> int:
        return self.positions.numel() + self.values.numel() * self.values.element_size()
