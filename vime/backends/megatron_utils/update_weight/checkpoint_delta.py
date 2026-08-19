from __future__ import annotations

from collections.abc import Iterable, Iterator
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Literal, Protocol

import torch

DeltaEncoding = Literal["dense", "indices"]


@dataclass(frozen=True)
class CheckpointPatchSpec:
    """Metadata for one checkpoint-coordinate patch inside a wire chunk."""

    name: str
    shape: tuple[int, ...]
    dtype_name: str
    value_start: int
    value_end: int
    position_start: int
    position_end: int

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "shape": list(self.shape),
            "dtype_name": self.dtype_name,
            "value_start": self.value_start,
            "value_end": self.value_end,
            "position_start": self.position_start,
            "position_end": self.position_end,
        }


@dataclass
class CheckpointDeltaChunk:
    """One dense-seed or sparse checkpoint-coordinate transfer chunk."""

    schema_version: int
    base_version: int
    target_version: int
    sequence_no: int
    is_final: bool
    encoding: DeltaEncoding
    patches: list[CheckpointPatchSpec]
    positions: torch.Tensor
    values: torch.Tensor

    @property
    def wire_bytes(self) -> int:
        return (
            self.positions.numel() * self.positions.element_size() + self.values.numel() * self.values.element_size()
        )

    def update_info(self) -> dict:
        return {
            "schema_version": self.schema_version,
            "base_version": self.base_version,
            "target_version": self.target_version,
            "sequence_no": self.sequence_no,
            "is_final": self.is_final,
            "encoding": self.encoding,
            "patches": [patch.to_dict() for patch in self.patches],
            "position_count": self.positions.numel(),
            "value_count": self.values.numel(),
            "value_dtype_name": str(self.values.dtype).removeprefix("torch."),
        }

    def wire_tensors(self) -> Iterator[tuple[str, torch.Tensor]]:
        if self.positions.numel():
            yield "__positions__", self.positions
        if self.values.numel():
            yield "__values__", self.values


class DeltaWeightSource(Protocol):
    """Source-side transaction boundary shared by future delta transports."""

    def begin_update(self, *, base_version: int, target_version: int) -> None: ...

    def encode_chunk(self, named_tensors: Iterable[tuple[str, torch.Tensor]]) -> list[CheckpointDeltaChunk]: ...

    def finish_update(self) -> CheckpointDeltaChunk: ...

    def commit(self) -> None: ...

    def abort(self) -> None: ...


@dataclass(frozen=True)
class _CheckpointTensorLayout:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    start: int
    end: int


@dataclass
class _BucketSnapshot:
    layout: tuple[_CheckpointTensorLayout, ...]
    values: torch.Tensor


@dataclass
class _PendingBucketUpdate:
    bucket_index: int
    indices: torch.Tensor
    values: torch.Tensor


class CheckpointDeltaSource:
    """Create checkpoint-coordinate patches from deterministic HF buckets.

    The first update sends dense weights and saves one flat CPU snapshot for
    each exporter bucket. Later updates compare one complete bucket at a time
    on the export device. This keeps the existing HF names and native vLLM
    loader while avoiding one GPU synchronization and CPU snapshot update per
    checkpoint tensor. Snapshots advance only after every rollout worker has
    applied the update.
    """

    def __init__(self, wire_dtype: torch.dtype = torch.bfloat16) -> None:
        if wire_dtype != torch.bfloat16:
            raise NotImplementedError("The direct DWU MVP supports BF16 only")
        self.wire_dtype = wire_dtype
        self._snapshot: list[_BucketSnapshot] = []
        self._pending_seed: list[_BucketSnapshot] = []
        self._pending_updates: list[_PendingBucketUpdate] = []
        self._seen_names: set[str] = set()
        self._active = False
        self._finished = False
        self._seed_update = True
        self._committed_version = 0
        self._base_version = 0
        self._target_version = 0
        self._next_sequence_no = 0
        self._next_bucket_index = 0
        self.total_elements = 0
        self.changed_elements = 0
        self.wire_bytes = 0

    @property
    def is_seed_update(self) -> bool:
        return self._seed_update

    def begin_update(self, *, base_version: int, target_version: int) -> None:
        if self._active:
            raise RuntimeError("A checkpoint delta update is already active")
        if target_version != base_version + 1:
            raise ValueError("Checkpoint delta target_version must be base_version + 1")
        if not self._snapshot and base_version != 0:
            raise ValueError("The first direct DWU transaction must start at version 0")
        if self._snapshot and base_version != self._committed_version:
            raise RuntimeError(
                f"Checkpoint delta source version mismatch: snapshot={self._committed_version}, update={base_version}"
            )
        self._active = True
        self._finished = False
        self._seed_update = not self._snapshot
        self._base_version = base_version
        self._target_version = target_version
        self._next_sequence_no = 0
        self._next_bucket_index = 0
        self._pending_seed = []
        self._pending_updates = []
        self._seen_names = set()
        self.total_elements = 0
        self.changed_elements = 0
        self.wire_bytes = 0

    @torch.no_grad()
    def encode_chunk(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
    ) -> list[CheckpointDeltaChunk]:
        self._require_active()
        if self._finished:
            raise RuntimeError("Cannot encode checkpoint deltas after finish_update()")
        layout, current = self._flatten_bucket(named_tensors)
        if not layout:
            return []

        bucket_index = self._next_bucket_index
        self._next_bucket_index += 1
        if self._seed_update:
            self._pending_seed.append(
                _BucketSnapshot(
                    layout=layout,
                    values=current.to(device="cpu", copy=True),
                )
            )
            chunk = self._build_dense_chunk(layout, current)
        else:
            if bucket_index >= len(self._snapshot):
                raise RuntimeError(f"HF exporter added bucket {bucket_index}")
            snapshot = self._snapshot[bucket_index]
            self._validate_bucket_layout(bucket_index, expected=snapshot.layout, actual=layout)
            chunk = self._build_sparse_chunk(bucket_index, snapshot, current)
            if chunk is None:
                return []

        self.wire_bytes += chunk.wire_bytes
        return [chunk]

    def finish_update(self) -> CheckpointDeltaChunk:
        self._require_active()
        if self._finished:
            raise RuntimeError("finish_update() was already called")
        if not self._seed_update and self._next_bucket_index != len(self._snapshot):
            raise RuntimeError(
                f"HF exporter bucket count changed: expected {len(self._snapshot)}, got {self._next_bucket_index}"
            )
        self._finished = True
        chunk = CheckpointDeltaChunk(
            schema_version=1,
            base_version=self._base_version,
            target_version=self._target_version,
            sequence_no=self._next_sequence_no,
            is_final=True,
            encoding="dense" if self._seed_update else "indices",
            patches=[],
            positions=torch.empty(0, dtype=torch.int32),
            values=torch.empty(0, dtype=self.wire_dtype),
        )
        self._next_sequence_no += 1
        return chunk

    def commit(self) -> None:
        self._require_active()
        if not self._finished:
            raise RuntimeError("finish_update() must be called before commit()")
        if self._seed_update:
            self._snapshot = self._pending_seed
        else:
            worker_count = min(16, len(self._pending_updates))
            if worker_count == 1:
                self._commit_bucket(self._pending_updates[0])
            elif worker_count > 1:
                # Each update writes to an independent bucket snapshot. Parallel
                # CPU writes shorten a commit that otherwise extends the rollout pause.
                with ThreadPoolExecutor(max_workers=worker_count) as executor:
                    list(executor.map(self._commit_bucket, self._pending_updates))
        self._committed_version = self._target_version
        self._clear_transaction()

    def abort(self) -> None:
        if self._active:
            self._clear_transaction()

    def _flatten_bucket(
        self,
        named_tensors: Iterable[tuple[str, torch.Tensor]],
    ) -> tuple[tuple[_CheckpointTensorLayout, ...], torch.Tensor]:
        tensors: list[torch.Tensor] = []
        layout: list[_CheckpointTensorLayout] = []
        offset = 0
        device: torch.device | None = None
        for name, tensor in named_tensors:
            if name in self._seen_names:
                raise ValueError(f"Duplicate HF tensor in one update: {name!r}")
            self._seen_names.add(name)

            current = tensor.detach()
            if not current.is_floating_point():
                raise NotImplementedError(f"{name}: direct DWU supports floating-point weights only")
            if current.dtype != self.wire_dtype:
                current = current.to(self.wire_dtype)
            current = current.contiguous()
            if current.numel() >= 1 << 31:
                raise NotImplementedError(f"{name}: direct DWU int32 indices require fewer than 2^31 elements")
            if device is None:
                device = current.device
            elif current.device != device:
                raise ValueError("All tensors in one HF exporter bucket must use the same device")

            end = offset + current.numel()
            layout.append(
                _CheckpointTensorLayout(
                    name=name,
                    shape=tuple(current.shape),
                    dtype=current.dtype,
                    start=offset,
                    end=end,
                )
            )
            tensors.append(current.reshape(-1))
            offset = end
            self.total_elements += current.numel()

        if not tensors:
            return (), torch.empty(0, dtype=self.wire_dtype)
        return tuple(layout), torch.cat(tensors)

    def _build_dense_chunk(
        self,
        layout: tuple[_CheckpointTensorLayout, ...],
        values: torch.Tensor,
    ) -> CheckpointDeltaChunk:
        patches = [
            CheckpointPatchSpec(
                name=tensor.name,
                shape=tensor.shape,
                dtype_name=str(tensor.dtype).removeprefix("torch."),
                value_start=tensor.start,
                value_end=tensor.end,
                position_start=0,
                position_end=0,
            )
            for tensor in layout
        ]
        self.changed_elements += values.numel()
        return self._new_chunk(
            encoding="dense",
            patches=patches,
            positions=torch.empty(0, dtype=torch.int32, device=values.device),
            values=values,
        )

    def _build_sparse_chunk(
        self,
        bucket_index: int,
        snapshot: _BucketSnapshot,
        current: torch.Tensor,
    ) -> CheckpointDeltaChunk | None:
        previous = snapshot.values.to(device=current.device)
        changed = torch.nonzero(
            current.view(torch.int16) != previous.view(torch.int16),
            as_tuple=False,
        ).reshape(-1)
        if not changed.numel():
            return None

        values = current.index_select(0, changed)
        ends = torch.tensor(
            [tensor.end for tensor in snapshot.layout],
            dtype=torch.int64,
            device=current.device,
        )
        # The checkpoint patch API reserves NaN as its unchanged-value sentinel
        # and rejects NaN wire values on every receiver; fail here on the source
        # rank instead, where the offending tensor can still be named.
        nan_mask = torch.isnan(values)
        if bool(nan_mask.any()):
            first_flat = changed[nan_mask.nonzero(as_tuple=False).reshape(-1)[0]]
            tensor_index = int(torch.searchsorted(ends, first_flat, right=True).item())
            raise ValueError(
                f"{snapshot.layout[tensor_index].name}: training produced NaN weight "
                "values; refusing to ship a sparse delta"
            )
        cumulative_counts = torch.searchsorted(changed, ends)
        counts = cumulative_counts.clone()
        counts[1:] -= cumulative_counts[:-1]
        starts = torch.tensor(
            [tensor.start for tensor in snapshot.layout],
            dtype=torch.int64,
            device=current.device,
        )
        positions = (
            changed
            - torch.repeat_interleave(
                starts,
                counts,
                output_size=changed.numel(),
            )
        ).to(torch.int32)

        counts_cpu = counts.to(device="cpu").tolist()
        patches: list[CheckpointPatchSpec] = []
        offset = 0
        for tensor, count in zip(snapshot.layout, counts_cpu, strict=True):
            if not count:
                continue
            end = offset + count
            patches.append(
                CheckpointPatchSpec(
                    name=tensor.name,
                    shape=tensor.shape,
                    dtype_name=str(tensor.dtype).removeprefix("torch."),
                    value_start=offset,
                    value_end=end,
                    position_start=offset,
                    position_end=end,
                )
            )
            offset = end

        self._pending_updates.append(
            _PendingBucketUpdate(
                bucket_index=bucket_index,
                indices=changed.to(device="cpu", copy=True),
                values=values.to(device="cpu", copy=True),
            )
        )
        self.changed_elements += values.numel()
        return self._new_chunk(
            encoding="indices",
            patches=patches,
            positions=positions,
            values=values,
        )

    def _new_chunk(
        self,
        *,
        encoding: DeltaEncoding,
        patches: list[CheckpointPatchSpec],
        positions: torch.Tensor,
        values: torch.Tensor,
    ) -> CheckpointDeltaChunk:
        chunk = CheckpointDeltaChunk(
            schema_version=1,
            base_version=self._base_version,
            target_version=self._target_version,
            sequence_no=self._next_sequence_no,
            is_final=False,
            encoding=encoding,
            patches=patches,
            positions=positions,
            values=values,
        )
        self._next_sequence_no += 1
        return chunk

    @staticmethod
    def _validate_bucket_layout(
        bucket_index: int,
        *,
        expected: tuple[_CheckpointTensorLayout, ...],
        actual: tuple[_CheckpointTensorLayout, ...],
    ) -> None:
        if expected == actual:
            return
        if len(expected) != len(actual):
            raise RuntimeError(
                f"HF exporter bucket {bucket_index} tensor count changed: expected {len(expected)}, got {len(actual)}"
            )
        for tensor_index, (old, new) in enumerate(zip(expected, actual, strict=True)):
            if old != new:
                raise RuntimeError(
                    f"HF exporter bucket {bucket_index} tensor {tensor_index} changed: expected {old}, got {new}"
                )
        raise AssertionError("Mismatched HF bucket layouts did not contain a mismatched tensor")

    def _require_active(self) -> None:
        if not self._active:
            raise RuntimeError("begin_update() must be called first")

    def _commit_bucket(self, update: _PendingBucketUpdate) -> None:
        self._snapshot[update.bucket_index].values.index_copy_(
            0,
            update.indices,
            update.values,
        )

    def _clear_transaction(self) -> None:
        self._pending_seed = []
        self._pending_updates = []
        self._seen_names = set()
        self._next_bucket_index = 0
        self._active = False
        self._finished = False
