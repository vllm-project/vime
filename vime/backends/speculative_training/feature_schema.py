from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import torch


FEATURE_SCHEMA_VERSION = 1
_ALGORITHMS = {"eagle3", "dspark"}


def _cpu_contiguous(tensor: torch.Tensor, *, dtype: torch.dtype | None = None) -> torch.Tensor:
    result = tensor.detach()
    if dtype is not None:
        result = result.to(dtype=dtype)
    return result.to(device="cpu", non_blocking=False).contiguous()


@dataclass(slots=True)
class DraftFeatureSample:
    input_ids: torch.Tensor
    loss_mask: torch.Tensor
    position_ids: torch.Tensor
    aux_hidden_states: torch.Tensor
    final_hidden_states: torch.Tensor
    rollout_id: int
    target_weight_version: str
    original_sample_id: str
    prompt_length: int
    response_length: int
    window_start: int
    window_end: int
    aux_layer_ids: tuple[int, ...]
    algorithm: str = "eagle3"
    schema_version: int = FEATURE_SCHEMA_VERSION

    @classmethod
    def from_payload(cls, payload: dict[str, Any], *, strict: bool = True) -> DraftFeatureSample:
        algorithm = str(payload.get("algorithm", "eagle3")).lower()
        sample = cls(
            schema_version=int(payload.get("schema_version", FEATURE_SCHEMA_VERSION)),
            algorithm=algorithm,
            input_ids=payload["input_ids"],
            loss_mask=payload["loss_mask"],
            position_ids=payload["position_ids"],
            aux_hidden_states=payload["aux_hidden_states"],
            final_hidden_states=payload["final_hidden_states"],
            rollout_id=int(payload["rollout_id"]),
            target_weight_version=str(payload["target_weight_version"]),
            original_sample_id=str(payload["original_sample_id"]),
            prompt_length=int(payload["prompt_length"]),
            response_length=int(payload["response_length"]),
            window_start=int(payload["window_start"]),
            window_end=int(payload["window_end"]),
            aux_layer_ids=tuple(int(item) for item in payload["aux_layer_ids"]),
        )
        sample.validate(strict=strict)
        return sample

    def validate(self, *, strict: bool = True) -> None:
        if self.schema_version != FEATURE_SCHEMA_VERSION and strict:
            raise ValueError(f"Unsupported Draft feature schema version {self.schema_version}")
        algorithm = self.algorithm.lower()
        if algorithm not in _ALGORITHMS:
            raise ValueError(f"Unsupported Draft feature algorithm {self.algorithm!r}")
        tensor_fields = (
            "input_ids",
            "loss_mask",
            "position_ids",
            "aux_hidden_states",
            "final_hidden_states",
        )
        for name in tensor_fields:
            if not torch.is_tensor(getattr(self, name)):
                raise TypeError(f"DraftFeatureSample.{name} must be a torch.Tensor")
        for name in ("input_ids", "loss_mask", "position_ids"):
            value = getattr(self, name)
            if value.dim() != 1:
                raise ValueError(f"DraftFeatureSample.{name} must be one-dimensional, got {tuple(value.shape)}")
        if self.aux_hidden_states.dim() != 2 or self.final_hidden_states.dim() != 2:
            raise ValueError("Draft hidden state tensors must have shape [rows, hidden]")
        rows = int(self.input_ids.numel())
        actual_rows = {
            "loss_mask": int(self.loss_mask.numel()),
            "position_ids": int(self.position_ids.numel()),
            "aux_hidden_states": int(self.aux_hidden_states.size(0)),
            "final_hidden_states": int(self.final_hidden_states.size(0)),
        }
        mismatches = {name: value for name, value in actual_rows.items() if value != rows}
        if mismatches:
            raise ValueError(f"Draft feature row mismatch: input_ids={rows}, others={mismatches}")
        if rows < 3:
            raise ValueError("Draft feature windows require at least three token rows")
        if self.window_start < 0 or self.window_end <= self.window_start:
            raise ValueError(f"Invalid Draft feature window [{self.window_start}, {self.window_end})")
        if self.window_end - self.window_start != rows:
            raise ValueError("Draft feature window length does not match tensor rows")
        if strict and rows > 1 and not torch.all(self.position_ids[1:] == self.position_ids[:-1] + 1):
            raise ValueError("Draft position ids must be contiguous")
        if self.prompt_length < 0 or self.response_length < 0:
            raise ValueError("Prompt and response lengths must be non-negative")
        if not self.target_weight_version:
            raise ValueError("Draft feature target_weight_version must be non-empty")
        if not self.aux_layer_ids:
            raise ValueError("Draft feature aux_layer_ids must be non-empty")

    def to_payload(self) -> dict[str, Any]:
        self.validate(strict=True)
        return {
            "schema_version": self.schema_version,
            "algorithm": self.algorithm,
            "input_ids": _cpu_contiguous(self.input_ids, dtype=torch.long),
            "loss_mask": _cpu_contiguous(self.loss_mask, dtype=torch.float32),
            "position_ids": _cpu_contiguous(self.position_ids, dtype=torch.long),
            "aux_hidden_states": _cpu_contiguous(self.aux_hidden_states, dtype=torch.bfloat16),
            "final_hidden_states": _cpu_contiguous(self.final_hidden_states, dtype=torch.bfloat16),
            "rollout_id": int(self.rollout_id),
            "target_weight_version": str(self.target_weight_version),
            "original_sample_id": self.original_sample_id,
            "prompt_length": int(self.prompt_length),
            "response_length": int(self.response_length),
            "window_start": int(self.window_start),
            "window_end": int(self.window_end),
            "aux_layer_ids": list(self.aux_layer_ids),
        }


class VersionedFeatureQueue:
    """A bounded FIFO queue that never mixes Target versions implicitly."""

    def __init__(self, max_samples: int = 2048) -> None:
        if max_samples <= 0:
            raise ValueError("VersionedFeatureQueue max_samples must be positive")
        self.max_samples = int(max_samples)
        self._items: dict[str, deque[DraftFeatureSample]] = defaultdict(deque)
        self._order: deque[tuple[str, str]] = deque()
        self._repeat_offsets: dict[str, int] = {}
        self.rejected_version_mismatch = 0

    def __len__(self) -> int:
        return sum(len(items) for items in self._items.values())

    def versions(self) -> tuple[str, ...]:
        return tuple(version for version, items in self._items.items() if items)

    def count(self, version: str) -> int:
        """Return the number of queued samples for one exact Target version."""
        return len(self._items.get(str(version), ()))

    def add(
        self,
        samples: Iterable[DraftFeatureSample | dict[str, Any]],
        *,
        expected_version: str | None = None,
    ) -> int:
        accepted = 0
        for value in samples:
            sample = value if isinstance(value, DraftFeatureSample) else DraftFeatureSample.from_payload(value)
            if expected_version is not None and sample.target_weight_version != str(expected_version):
                self.rejected_version_mismatch += 1
                continue
            version = sample.target_weight_version
            self._items[version].append(sample)
            self._order.append((version, sample.original_sample_id))
            accepted += 1
            self._evict_if_needed()
        return accepted

    def _evict_if_needed(self) -> None:
        while len(self) > self.max_samples and self._order:
            version, sample_id = self._order.popleft()
            items = self._items.get(version)
            # Records consumed by ``take`` are removed from ``_order``.  Keep
            # this guard as corruption protection for duplicate/stale records.
            if items and items[0].original_sample_id == sample_id:
                items.popleft()
                self._repeat_offsets[version] = 0
                if not items:
                    del self._items[version]
                    self._repeat_offsets.pop(version, None)

    def _remove_order_record(self, version: str, sample_id: str) -> None:
        try:
            self._order.remove((version, sample_id))
        except ValueError:
            # ``_order`` is only an eviction index; the version deque remains
            # the source of truth if a caller legitimately reused sample ids.
            pass

    def take(self, version: str, count: int, *, repeat: bool = False) -> list[DraftFeatureSample]:
        if count <= 0:
            return []
        items = self._items.get(str(version))
        if not items:
            return []
        result: list[DraftFeatureSample] = []
        if repeat:
            source = list(items)
            offset = self._repeat_offsets.get(str(version), 0) % len(source)
            for index in range(count):
                result.append(source[(offset + index) % len(source)])
            self._repeat_offsets[str(version)] = (offset + count) % len(source)
            return result
        for _ in range(min(count, len(items))):
            sample = items.popleft()
            result.append(sample)
            self._remove_order_record(str(version), sample.original_sample_id)
        if not items:
            self._items.pop(str(version), None)
            self._repeat_offsets.pop(str(version), None)
        else:
            self._repeat_offsets[str(version)] = 0
        return result

    def clear_except(self, version: str) -> int:
        version = str(version)
        removed = sum(len(items) for key, items in self._items.items() if key != version)
        self._items = defaultdict(deque, {version: self._items.get(version, deque())})
        self._order = deque((key, sample_id) for key, sample_id in self._order if key == version)
        self._repeat_offsets = {version: self._repeat_offsets.get(version, 0)}
        return removed
