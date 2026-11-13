"""Megatron weight synchronization through sparse HCCL patches."""

from __future__ import annotations

import logging
import socket
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from math import prod

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from vllm.distributed.weight_transfer.sparse_nccl_engine import SparseWeightPatch
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerInitInfo,
    SparseHCCLTrainerWeightTransferEngine,
)

from vime.utils.distributed_utils import get_gloo_group

from ..delta_sync.hf_weight_iterator_bridge import HfWeightIteratorBridge
from ..delta_sync.megatron_delta_export import (
    build_export_index,
    iter_delta_entries,
    prime_delta_snapshots,
    validate_export_weights,
)
from ..delta_sync.sparse_gather import _GatherQueue
from .common import VimeRayWeightSyncClient

logger = logging.getLogger(__name__)


def prepare_sparse_patches(patches: Sequence[SparseWeightPatch], device) -> list[SparseWeightPatch]:
    """Place payloads on the communication device and reject unsafe values.

    The sender checks patch structure. This adds numeric checks before its RPC.
    Ordered indices need only a linear scan; unordered indices retain a full
    duplicate check without changing the association between indices and values.
    """
    prepared = []
    checks = []
    order_checks = []
    unordered_candidates = []
    for patch in patches:
        if not isinstance(patch.name, str) or not patch.name:
            raise ValueError("Sparse checkpoint name must be non-empty")
        if patch.full_shape is None or any(type(dim) is not int or dim < 0 for dim in patch.full_shape):
            raise ValueError(f"Invalid sparse checkpoint shape: {patch.name}")
        numel = prod(patch.full_shape)
        if numel >= 2**31:
            raise ValueError(f"Sparse loader requires checkpoint tensors smaller than 2**31 elements: {patch.name}")
        if patch.values.dtype not in (torch.float16, torch.bfloat16, torch.float32, torch.float64):
            raise ValueError(f"Sparse values must have floating dtype: {patch.name}")
        indices = patch.indices.to(device=device).contiguous()
        values = patch.values.to(device=device).contiguous()
        if indices.numel():
            flat = indices.view(-1)
            # The supported shape bound fits int32, so no int64 copy is needed.
            checks.extend([((flat < 0) | (flat >= numel)).any(), torch.isnan(values).any()])
            order_checks.append((flat[1:] <= flat[:-1]).any())
            unordered_candidates.append(flat)
        prepared.append(SparseWeightPatch(patch.name, indices, values, tuple(patch.full_shape)))
    if not checks:
        return prepared
    flags = torch.stack([torch.stack(checks).any(), *order_checks]).cpu().tolist()
    if flags[0]:
        raise ValueError("Sparse payload contains out-of-range or duplicate indices, or NaN values")
    duplicate_checks = []
    for indices, unordered in zip(unordered_candidates, flags[1:], strict=True):
        if unordered:
            ordered = torch.sort(indices).values
            duplicate_checks.append((ordered[1:] == ordered[:-1]).any())
    if duplicate_checks and torch.stack(duplicate_checks).any().item():
        raise ValueError("Sparse payload contains duplicate indices")
    return prepared


class _SparseFlushBucket:
    """One homogeneous-dtype sparse flush with bounded wire size."""

    def __init__(self, capacity: int, publish):
        self.capacity = int(capacity)
        self.publish = publish
        self.patches: list[SparseWeightPatch] = []
        self.nbytes = 0

    def add(self, name, shape, indices, values) -> None:
        if not indices.numel():
            return
        max_elements = max(self.capacity // (4 + values.element_size()), 1)
        for start in range(0, indices.numel(), max_elements):
            end = min(start + max_elements, indices.numel())
            piece_bytes = (end - start) * (4 + values.element_size())
            if self.patches and self.nbytes + piece_bytes > self.capacity:
                self.flush()
            self.patches.append(
                SparseWeightPatch(
                    name=name,
                    indices=indices[start:end].contiguous(),
                    values=values[start:end].contiguous(),
                    full_shape=tuple(shape),
                )
            )
            self.nbytes += piece_bytes
            if self.nbytes >= self.capacity:
                self.flush()

    def flush(self) -> None:
        if not self.patches:
            return
        self.publish(self.patches)
        self.patches, self.nbytes = [], 0


class UpdateWeightFromSparseHCCL:
    """Full-coordinate seed and sparse deltas using the same public engine API."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.weight_version = 0
        self._iterator = HfWeightIteratorBridge(args, model)
        self._index = None
        self._snapshots: dict[str, torch.Tensor] = {}
        self._seeded = False
        self._payload_started = False
        self._failed = False
        self._trainer = None
        self._needs_reconnect = False
        self._engine_gpu_counts = None
        self._gather_workspaces = {}
        self._client = None
        self.rollout_engines: list[ActorHandle] = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        del rollout_engine_lock, engine_gpu_offsets
        for config in engine_parallel_configs or [
            {
                "pipeline_parallel_size": getattr(self.args, "vllm_pipeline_parallel_size", 1) or 1,
                "prefill_context_parallel_size": getattr(self.args, "vllm_prefill_context_parallel_size", 1) or 1,
            }
        ]:
            if int(config.get("pipeline_parallel_size", config.get("pp_size", 1)) or 1) != 1:
                raise ValueError("Sparse HCCL requires inference pipeline_parallel_size=1")
            if int(config.get("prefill_context_parallel_size", config.get("pcp_size", 1)) or 1) != 1:
                raise ValueError("Sparse HCCL requires inference prefill_context_parallel_size=1")
        self.disconnect_rollout_engines()
        self._seeded = False
        self._failed = False
        self._snapshots.clear()
        self._index = None
        self.rollout_engines = list(rollout_engines)
        gpu_counts = list(engine_gpu_counts or [self.args.rollout_num_gpus_per_engine] * len(self.rollout_engines))
        self._engine_gpu_counts = gpu_counts
        self._needs_reconnect = False
        self._client = VimeRayWeightSyncClient(
            self.rollout_engines,
            lambda: self.weight_version,
            gpu_counts,
        )
        rendezvous = [None]
        if dist.get_rank() == 0:
            with socket.socket() as sock:
                sock.bind(("", 0))
                rendezvous[0] = (
                    ray._private.services.get_node_ip_address(),
                    sock.getsockname()[1],
                )
        dist.broadcast_object_list(rendezvous, src=0, group=get_gloo_group())
        master_address, master_port = rendezvous[0]
        if dist.get_rank() == 0:
            init_info = SparseHCCLTrainerInitInfo(
                master_address=master_address,
                master_port=master_port,
                world_size=sum(gpu_counts) + 1,
                rank=0,
            )
            self._trainer = SparseHCCLTrainerWeightTransferEngine.trainer_init(
                init_info,
                client=self._client,
            )
        dist.barrier(group=get_gloo_group())

    def disconnect_rollout_engines(self) -> None:
        if self._trainer is not None:
            self._trainer.shutdown()
            self._trainer = None
        self._client = None

    def pop_metrics(self) -> dict[str, float]:
        return {}

    def _ensure_export_index(self) -> None:
        if self._index is not None and not self.args.offload_train:
            return
        with self._iterator.model_context():
            self._index = build_export_index(self._iterator.bridge, self.model)

    def _begin_update(self) -> None:
        if dist.get_rank() != 0:
            return
        assert self._client is not None
        ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        self._client.start_weight_update()

    def _finish_update(self) -> None:
        if dist.get_rank() != 0:
            return
        assert self._client is not None
        self._client.finish_weight_update(str(self.weight_version))
        torch.accelerator.synchronize()
        ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])

    def _sync_update_state(self, error: BaseException | None) -> tuple[bool, bool]:
        """Replace the existing phase barrier with synchronized failure state."""
        state = torch.tensor(
            [int(error is not None), int(self._payload_started)],
            dtype=torch.int32,
        )
        dist.all_reduce(state, op=dist.ReduceOp.MAX, group=get_gloo_group())
        return bool(state[0].item()), bool(state[1].item())

    def _recover_before_payload(self) -> None:
        """Restore a no-payload session; reconnect before a new dense baseline."""
        cleanup_error = None
        if dist.get_rank() == 0:
            assert self._client is not None
            try:
                self._client.finish_weight_update(str(self.weight_version - 1))
            except Exception as error:
                cleanup_error = error
                logger.warning(
                    "Sparse HCCL failure cleanup could not finish the worker update session",
                    exc_info=True,
                )
            try:
                if cleanup_error is None:
                    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            except Exception as error:
                cleanup_error = error
                logger.warning(
                    "Sparse HCCL failure cleanup could not resume every rollout engine",
                    exc_info=True,
                )
        cleanup_failed, _ = self._sync_update_state(cleanup_error)
        self._failed = cleanup_failed
        self._needs_reconnect = not cleanup_failed
        # Diff preparation may already have advanced snapshots. Force the next
        # successful attempt through a dense seed instead of reusing them.
        self.weight_version -= 1
        self._seeded = False
        self._snapshots.clear()

    @staticmethod
    def _raise_synchronized_error(error: BaseException | None, stage: str) -> None:
        if error is not None:
            raise error
        raise RuntimeError(f"Sparse HCCL weight update failed on another trainer rank during {stage}")

    def _publish_flush(self, patches: list[SparseWeightPatch]) -> None:
        if dist.get_rank() != 0 or not patches:
            return
        assert self._trainer is not None
        patches = prepare_sparse_patches(patches, self._trainer.model_update_group.device)
        self._payload_started = True
        self._trainer.send_weight_chunk(patches)

    def _sync_publish_error(self, error: BaseException | None) -> None:
        failed, _ = self._sync_update_state(error)
        if failed:
            self._raise_synchronized_error(error, "gather/publish")

    def _send_seed(self) -> None:
        """Seed new or reconnected workers from the complete training weights."""
        bucket = _SparseFlushBucket(self.args.update_weight_buffer_size, self._publish_flush)
        for chunk in self._iterator.get_hf_weight_chunks():
            if dist.get_rank() != 0 or not chunk:
                continue
            for name, tensor in chunk:
                flat = tensor.reshape(-1)
                if flat.numel() >= 2**31:
                    raise ValueError(f"Sparse loader requires checkpoint tensors smaller than 2**31 elements: {name}")
                max_elements = max(self.args.update_weight_buffer_size // (4 + flat.element_size()), 1)
                for start in range(0, flat.numel(), max_elements):
                    end = min(start + max_elements, flat.numel())
                    indices = torch.arange(start, end, dtype=torch.int32, device=flat.device)
                    bucket.add(name, tensor.shape, indices, flat[start:end])
            # Bridge may reuse export buffers when the iterator advances.
            bucket.flush()

    def _send_sparse_delta(self) -> None:
        assert self._index is not None
        buckets: dict[str, _SparseFlushBucket] = {}

        def consume(name, dtype_name, shape, indices, values):
            if dtype_name not in buckets:
                buckets[dtype_name] = _SparseFlushBucket(self.args.update_weight_buffer_size, self._publish_flush)
            buckets[dtype_name].add(name, shape, indices, values)

        queue = _GatherQueue(
            self.args.update_weight_delta_batch_gather,
            self.args.update_weight_buffer_size,
            dist.get_rank() == 0,
            consume,
            workspaces=self._gather_workspaces,
            sync_error=self._sync_publish_error,
        )
        for slots, dtype_name, counts, indices, values, group in iter_delta_entries(
            self._index,
            self._snapshots,
            batch_size=self.args.update_weight_delta_batch_diff,
        ):
            queue.put(group, slots, dtype_name, counts, indices, values)
        queue.flush_all()
        if dist.get_rank() == 0:
            for bucket in buckets.values():
                bucket.flush()

    @torch.no_grad()
    def update_weights(self) -> None:
        if self._client is None:
            raise RuntimeError("Sparse HCCL updater is not connected")
        if self._failed:
            raise RuntimeError(
                "Sparse HCCL updater cannot be reused after a potentially partial update; reconnect it to force a full seed"
            )
        if self._needs_reconnect:
            self.connect_rollout_engines(self.rollout_engines, None, engine_gpu_counts=self._engine_gpu_counts)
        self.weight_version += 1
        self._payload_started = False
        begin_error: BaseException | None = None
        try:
            self._begin_update()
        except BaseException as error:
            begin_error = error
        failed, _ = self._sync_update_state(begin_error)
        if failed:
            self._recover_before_payload()
            dist.barrier(group=get_gloo_group())
            self._raise_synchronized_error(begin_error, "begin")

        phase_error: BaseException | None = None
        try:
            self._ensure_export_index()
            validate_export_weights(self._index)
            if not self._seeded:
                self._send_seed()
                prime_delta_snapshots(self._index, self._snapshots, pin=True)
                torch.accelerator.synchronize()
                self._seeded = True
            else:
                self._send_sparse_delta()
        except BaseException as error:
            phase_error = error
        failed, payload_started = self._sync_update_state(phase_error)
        if failed:
            if payload_started:
                self._failed = True
                if dist.get_rank() == 0:
                    logger.error(
                        "Sparse HCCL update failed after payload transmission began; generation remains paused"
                    )
            else:
                self._recover_before_payload()
            dist.barrier(group=get_gloo_group())
            self._raise_synchronized_error(phase_error, "prepare/publish")

        finish_error: BaseException | None = None
        try:
            self._finish_update()
        except BaseException as error:
            finish_error = error
        failed, _ = self._sync_update_state(finish_error)
        if failed:
            self._failed = True
            if dist.get_rank() == 0:
                logger.error("Sparse HCCL finish failed; reconnect the updater before reusing it")
            self._raise_synchronized_error(finish_error, "finish")
