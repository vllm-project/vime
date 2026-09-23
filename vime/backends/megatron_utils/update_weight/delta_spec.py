"""Megatron shard export contract used by VIME's delta updater.

This keeps the explicit-placement half of verl's ``ShardSpec``. VIME's
Megatron tensors are not DTensors; their TP/EP/PP geometry is described by
Megatron parameter attributes and process groups instead.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from torch.distributed import ProcessGroup


@dataclass
class ShardSpec:
    full_shape: tuple[int, ...]
    place: int = 0
    gather_group: ProcessGroup | None = None
    contributes: bool = True

