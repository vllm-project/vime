from __future__ import annotations

import hashlib
import logging
import os
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle
from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerSendWeightsArgs
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import SparseHCCLWeightTransferEngine
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch

from vime.utils import megatron_bridge_utils
from vime.utils.distributed_utils import get_gloo_group

from ..misc_utils import strip_param_name_prefix
from .hf_weight_iterator_base import HfWeightIteratorBase
from .megatron_sparse_export import (
    build_sparse_export_index,
    clone_cpu_snapshot,
    local_bit_exact_diff,
    sparse_hf_entry,
)
from .sparse_gather import gather_slot_entries_to_rank0
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
)

logger = logging.getLogger(__name__)


class _GatherQueue:
    """Count-triggered queues keep collective ordering identical on all ranks."""

    def __init__(self, batch_size: int, max_round_bytes: int, is_source: bool, consume):
        self.batch_size = max(int(batch_size), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_source = is_source
        self.consume = consume
        self.queues: dict[int, tuple] = {}

    def put(self, group, slots, counts, indices, values) -> None:
        _group, entries = self.queues.setdefault(id(group), (group, []))
        entries.append((slots, counts, indices, values))
        if len(entries) >= self.batch_size:
            self._flush(group, entries)

    def flush_all(self) -> None:
        for group, entries in self.queues.values():
            self._flush(group, entries)

    def _flush(self, group, entries) -> None:
        if not entries:
            return
        batch = list(entries)
        entries.clear()
        if group is None:
            if self.is_source:
                for slots, counts, indices, values in batch:
                    offset = 0
                    for (name, shape), count in zip(slots, counts.tolist(), strict=True):
                        self.consume(name, shape, indices[offset : offset + count], values[offset : offset + count])
                        offset += count
            return

        device = batch[0][2].device
        counts = torch.cat([entry[1] for entry in batch]).to(device)
        indices = torch.cat([entry[2] for entry in batch])
        values = torch.cat([entry[3] for entry in batch])
        gathered = gather_slot_entries_to_rank0(
            indices, values, counts, group=group, max_round_bytes=self.max_round_bytes
        )
        if self.is_source and gathered is not None:
            slot_index = 0
            for slots, _counts, _indices, _values in batch:
                for name, shape in slots:
                    merged_indices, merged_values = gathered[slot_index]
                    slot_index += 1
                    self.consume(name, shape, merged_indices, merged_values)


class UpdateWeightFromSparseDistributed:
    """Diff Megatron shards locally and gather only final-HF sparse entries."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        if quantization_config:
            raise NotImplementedError("Sparse HCCL weight sync currently supports unquantized rollout weights only")
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self._snapshot: dict[str, torch.Tensor] = {}
        self._slot_cache: dict[str, list[tuple[str, tuple[int, ...]]]] = {}
        self._export_index = None
        self._baseline_captured = False
        self._model_update_groups = None
        self._iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )
        self._is_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0
            and mpu.get_tensor_model_parallel_rank() == 0
            and mpu.get_pipeline_model_parallel_rank() == 0
        )
        self._verify_full_diff = os.getenv("VIME_SPARSE_HCCL_VERIFY_FULL_DIFF", "0").lower() in {
            "1", "true", "yes"
        }
        self._legacy_snapshot: dict[str, torch.Tensor] = {}
        self._distributed_signatures: dict[str, tuple] = {}
        if self._is_src_rank:
            self._group_name = "vime-sparse-hccl"

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        del engine_gpu_offsets
        self.rollout_engines = list(rollout_engines)
        self.rollout_engine_lock = rollout_engine_lock
        if self._is_src_rank:
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args, self._group_name, self.rollout_engines, engine_gpu_counts=engine_gpu_counts
            )

    def disconnect_rollout_engines(self) -> None:
        if self._is_src_rank and self._model_update_groups is not None:
            disconnect_rollout_engines_from_distributed(
                self.args, self._group_name, self._model_update_groups, self.rollout_engines
            )
            self._model_update_groups = None

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    def _local_weights(self) -> dict[str, torch.Tensor]:
        return {strip_param_name_prefix(name): tensor for name, tensor in self.weights_getter().items()}

    def _iter_hf_tensors(self):
        for chunk in self._iterator.get_hf_weight_chunks(
            self.weights_getter(), progress_desc="Sparse HCCL full-diff verification"
        ):
            if self._is_src_rank:
                yield from chunk

    def _capture_baseline(self) -> None:
        local_weights = self._local_weights()
        with megatron_bridge_utils.patch_megatron_model(self.model):
            self._export_index = build_sparse_export_index(
                self._iterator._bridge, self.model, local_weights, self._slot_cache
            )
        for record in self._export_index:
            self._snapshot[record.weight_key] = clone_cpu_snapshot(local_weights[record.weight_key])

        if self._verify_full_diff:
            for name, tensor in self._iter_hf_tensors():
                self._legacy_snapshot[name] = tensor.detach().cpu().contiguous().clone()
        self._baseline_captured = True
        dist.barrier(group=get_gloo_group())
        local_bytes = sum(t.numel() * t.element_size() for t in self._snapshot.values())
        if self._is_src_rank:
            logger.info(
                "[sparse HCCL] captured rank-local baseline: %d shards, %.2f MB%s",
                len(self._snapshot), local_bytes / 1e6,
                " (full-diff verification enabled)" if self._verify_full_diff else "",
            )

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            self._capture_baseline()
            return

        next_weight_version = self.weight_version + 1
        started = time.perf_counter()
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            ray.get([
                engine.start_weight_update.remote(is_checkpoint_format=False)
                for engine in self.rollout_engines
            ])
        dist.barrier(group=get_gloo_group())

        statistics = {"changed": 0, "total": 0, "wire": 0}
        next_snapshot: dict[str, torch.Tensor] = {}
        seen_names: set[str] = set()
        self._distributed_signatures.clear()

        def consume(name, shape, indices, values) -> None:
            if name in seen_names:
                raise RuntimeError(f"Sparse Bridge emitted duplicate HF tensor {name!r}")
            seen_names.add(name)
            total = 1
            for dimension in shape:
                total *= dimension
            statistics["total"] += total
            statistics["changed"] += indices.numel()
            if self._verify_full_diff:
                self._distributed_signatures[name] = self._patch_signature(shape, indices, values)
            if indices.numel() == 0:
                return
            statistics["wire"] += (
                indices.numel() * indices.element_size() + values.numel() * values.element_size()
            )
            patch = SparseWeightPatch(
                name=name, indices=indices.to(torch.int32).contiguous(), values=values.contiguous()
            )
            self._send_patch(patch, list(shape), next_weight_version)

        queue = _GatherQueue(
            batch_size=32,
            max_round_bytes=self.args.update_weight_buffer_size,
            is_source=self._is_src_rank,
            consume=consume,
        )
        try:
            local_weights = self._local_weights()
            with megatron_bridge_utils.patch_megatron_model(self.model):
                for record in self._export_index:
                    current = local_weights[record.weight_key].detach().cpu().contiguous()
                    local_indices, local_values = local_bit_exact_diff(current, self._snapshot[record.weight_key])
                    # Commit snapshots only after every collective and rollout
                    # update has succeeded.  A failed update can then be
                    # retried without silently dropping its local changes.
                    # ``current`` may already be the mutable CPU tensor owned by
                    # TensorBackuper.  Keeping it directly would make the next
                    # backup mutate both the live weight and our baseline, so
                    # every update after v1 would compare the tensor with itself.
                    next_snapshot[record.weight_key] = clone_cpu_snapshot(current)
                    if not record.contributes:
                        local_indices = local_indices[:0]
                        local_values = local_values[:0]
                    device = record.param.device
                    slots, counts, hf_indices, hf_values = sparse_hf_entry(
                        record,
                        local_indices.to(device=device, non_blocking=False),
                        local_values.to(device=device, non_blocking=False),
                        self._slot_cache,
                    )
                    queue.put(record.gather_group, slots, counts, hf_indices, hf_values)
                queue.flush_all()
            if self._verify_full_diff:
                next_legacy_snapshot = self._verify_against_full_diff()
            else:
                next_legacy_snapshot = None
            if self._is_src_rank:
                torch.npu.synchronize()
        finally:
            dist.barrier(group=get_gloo_group())
            if dist.get_rank() == 0:
                try:
                    ray.get([engine.finish_weight_update.remote() for engine in self.rollout_engines])
                finally:
                    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            dist.barrier(group=get_gloo_group())

        self._snapshot.update(next_snapshot)
        if next_legacy_snapshot is not None:
            self._legacy_snapshot = next_legacy_snapshot
        self.weight_version = next_weight_version

        if self._is_src_rank:
            elapsed = time.perf_counter() - started
            changed, total, wire = statistics["changed"], statistics["total"], statistics["wire"]
            self.update_weight_metrics.update({
                "perf/update_weights_density": changed / max(total, 1),
                "perf/update_weights_wire_bytes": wire,
                "perf/update_weights_sparse_hccl_time": elapsed,
            })
            logger.info(
                "[sparse HCCL v=%d] density=%.4f%% wire=%.2f MB elapsed=%.3fs",
                self.weight_version, 100 * changed / max(total, 1), wire / 1e6, elapsed,
            )

    @staticmethod
    def _patch_signature(shape, indices, values) -> tuple:
        indices = indices.detach().cpu().to(torch.int64)
        values = values.detach().cpu().contiguous()
        if indices.numel():
            order = torch.argsort(indices)
            indices = indices[order]
            values = values[order]
        digest = hashlib.sha256()
        digest.update(indices.numpy().tobytes())
        digest.update(values.view(torch.uint8).numpy().tobytes())
        return tuple(shape), indices.numel(), digest.hexdigest()

    def _verify_against_full_diff(self) -> dict[str, torch.Tensor]:
        expected: dict[str, tuple] = {}
        next_snapshot: dict[str, torch.Tensor] = {}
        for name, tensor in self._iter_hf_tensors():
            current = tensor.detach().cpu().contiguous()
            snapshot = self._legacy_snapshot.get(name)
            if snapshot is None:
                raise RuntimeError(f"Full-diff baseline is missing {name!r}")
            indices, values = local_bit_exact_diff(current, snapshot)
            expected[name] = self._patch_signature(current.shape, indices, values)
            next_snapshot[name] = current
        if self._is_src_rank and expected != self._distributed_signatures:
            missing = sorted(expected.keys() - self._distributed_signatures.keys())
            extra = sorted(self._distributed_signatures.keys() - expected.keys())
            mismatched = sorted(
                name for name in expected.keys() & self._distributed_signatures.keys()
                if expected[name] != self._distributed_signatures[name]
            )
            raise AssertionError(
                "Sparse shard export differs from full HF bit-exact diff: "
                f"missing={missing[:8]}, extra={extra[:8]}, mismatched={mismatched[:8]}"
            )
        if self._is_src_rank:
            logger.info(
                "[sparse HCCL] TP shard export matched full HF bit-exact diff for %d tensors", len(expected)
            )
        return next_snapshot

    def _send_patch(
        self, patch: SparseWeightPatch, shape: list[int], weight_version: int
    ) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            refs = [
                engine.update_sparse_weights_from_distributed.remote(
                    names=[patch.name], dtypes=[patch.values.dtype], shapes=[shape],
                    num_updates_list=[patch.indices.numel()], group_name=self._group_name,
                    weight_version=str(weight_version),
                )
                for engine in self.rollout_engines
            ]
            SparseHCCLWeightTransferEngine.trainer_send_weights(
                iter([patch]), HCCLTrainerSendWeightsArgs(group=self._model_update_groups, packed=False)
            )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())
