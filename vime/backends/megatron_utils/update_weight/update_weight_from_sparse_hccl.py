"""verl-style Megatron delta synchronization over Sparse HCCL."""

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

from .common import VimeRayWeightSyncClient
from .delta_sync import gather_slot_entries_to_rank0
from .delta_sync.sparse_gather import GatherWorkspace
from .hf_weight_iterator_sparse_bridge import HfWeightIteratorSparseBridge
from .megatron_delta_export import (
    build_export_index,
    iter_delta_entries,
    prime_delta_snapshots,
    validate_export_weights,
)

logger = logging.getLogger(__name__)


class _GatherQueue:
    """Count-triggered queues preserving collective order on every rank."""

    def __init__(self, batch_size: int, max_round_bytes: int, is_wire_master: bool, consume, workspaces=None):
        self.batch_size = max(int(batch_size), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_wire_master = is_wire_master
        self.consume = consume
        self.workspaces = workspaces if workspaces is not None else {}
        self.queues: dict[tuple[int, str], tuple] = {}

    def put(self, group, slots, dtype_name, counts, indices, values) -> None:
        key = (id(group), dtype_name)
        _group, _dtype, entries = self.queues.setdefault(key, (group, dtype_name, []))
        entries.append((slots, counts, indices, values))
        if len(entries) >= self.batch_size:
            self._flush(group, dtype_name, entries)

    def flush_all(self) -> None:
        for group, dtype_name, entries in self.queues.values():
            self._flush(group, dtype_name, entries)

    def _flush(self, group, dtype_name, entries) -> None:
        if not entries:
            return
        batch = list(entries)
        entries.clear()
        if group is None:
            if self.is_wire_master:
                for slots, counts, indices, values in batch:
                    offset = 0
                    for (name, shape), count in zip(slots, counts.tolist(), strict=True):
                        self.consume(
                            name,
                            dtype_name,
                            tuple(shape),
                            indices[offset : offset + count],
                            values[offset : offset + count],
                        )
                        offset += count
            return

        counts = torch.cat([entry[1] for entry in batch])
        indices = torch.cat([entry[2] for entry in batch])
        values = torch.cat([entry[3] for entry in batch])
        workspace_key = (id(group), dtype_name)
        workspace = self.workspaces.get(workspace_key)
        if workspace is None:
            workspace = self.workspaces[workspace_key] = GatherWorkspace()
        gathered = gather_slot_entries_to_rank0(
            indices,
            values,
            counts,
            group=group,
            max_round_bytes=self.max_round_bytes,
            workspace=workspace,
        )
        if self.is_wire_master and gathered is not None:
            gathered_index = 0
            for slots, _counts, _indices, _values in batch:
                for name, shape in slots:
                    merged_indices, merged_values = gathered[gathered_index]
                    gathered_index += 1
                    self.consume(name, dtype_name, tuple(shape), merged_indices, merged_values)


class _SparseFlushBucket:
    """One homogeneous-dtype sparse flush with bounded wire size."""

    def __init__(self, capacity: int, publish):
        self.capacity = int(capacity)
        self.publish = publish
        self.patches: list[SparseWeightPatch] = []
        self.shapes: list[list[int]] = []
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
            self.shapes.append(list(shape))
            self.nbytes += piece_bytes
            if self.nbytes >= self.capacity:
                self.flush()

    def flush(self) -> None:
        if not self.patches:
            return
        self.publish(self.patches)
        self.patches, self.shapes, self.nbytes = [], [], 0


class UpdateWeightFromSparseHCCL:
    """Dense seed followed by shard-local sparse delta synchronization."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        if getattr(args, "update_weight_delta_verify_every", 0):
            raise ValueError("Sparse HCCL does not expose rank-local verify; set --update-weight-delta-verify-every 0. Correctness is checked by the dedicated E2E oracle.")
        if quantization_config:
            raise NotImplementedError("Sparse HCCL delta sync does not support quantized rollout weights")
        if args.colocate:
            raise ValueError("Sparse HCCL delta sync requires non-colocated rollout engines")
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self._iterator = HfWeightIteratorSparseBridge(args, model)
        self._index = None
        self._snapshots: dict[str, torch.Tensor] = {}
        self._seeded = False
        self._steady_updates = 0
        self._payload_started = False
        self._failed = False
        self._trainer = None
        self._needs_reconnect = False
        self._engine_gpu_counts = None
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
        del rollout_engine_lock, engine_gpu_offsets, engine_parallel_configs
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
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    def _ensure_export_index(self) -> None:
        if self._index is not None and not self.args.offload_train:
            return
        with self._iterator.model_context():
            self._index = build_export_index(self._iterator.bridge, self.model, {})

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
        self._steady_updates = 0
        self._snapshots.clear()

    @staticmethod
    def _raise_synchronized_error(error: BaseException | None, stage: str) -> None:
        if error is not None:
            raise error
        raise RuntimeError(f"Sparse HCCL weight update failed on another trainer rank during {stage}")

    def _publish_flush(self, patches: list[SparseWeightPatch]) -> None:
        if dist.get_rank() != 0 or not patches:
            return
        assert self._client is not None and self._trainer is not None
        # Native send_weight_chunk owns metadata and matching worker RPCs.
        # The surrounding VIME update owns start/finish and generation pause.
        self._payload_started = True
        self._trainer.send_weight_chunk(patches)

    def _send_dense(self) -> tuple[int, int]:
        """Seed all values using the native sparse protocol, without a wire extension.

        Complete checkpoint indices cost more wire bytes than dense HCCL. Bound
        allocations by the configured buffer and report seed separately.
        """
        statistics = {"flushes": 0, "wire_bytes": 0}

        def publish(patches):
            self._publish_flush(patches)
            statistics["flushes"] += 1
            statistics["wire_bytes"] += sum(patch.indices.numel() * (4 + patch.values.element_size()) for patch in patches)

        bucket = _SparseFlushBucket(self.args.update_weight_buffer_size, publish)
        for chunk in self._iterator.get_hf_weight_chunks(self.weights_getter(), progress_desc="Native sparse HCCL full-index seed"):
            if dist.get_rank() != 0:
                continue
            for name, tensor in chunk:
                flat = tensor.detach().contiguous().view(-1)
                if flat.numel() > torch.iinfo(torch.int32).max + 1:
                    raise ValueError(f"Native int32 checkpoint coordinates cannot represent all seed indices: {name}")
                count = max(self.args.update_weight_buffer_size // (4 + flat.element_size()), 1)
                for start in range(0, flat.numel(), count):
                    end = min(start + count, flat.numel())
                    indices = torch.arange(start, end, dtype=torch.int32, device=flat.device)
                    bucket.add(name, tuple(tensor.shape), indices, flat[start:end])
            # Batch small tensors through the native list API while releasing
            # all exported tensor references before the next Bridge chunk.
            bucket.flush()
        return statistics["flushes"], statistics["wire_bytes"]

    def _send_sparse_delta(self) -> tuple[int, int, int]:
        assert self._index is not None
        statistics = {"flushes": 0, "wire_bytes": 0, "updates": 0}

        def publish(patches):
            self._publish_flush(patches)
            statistics["flushes"] += 1
            statistics["updates"] += sum(patch.indices.numel() for patch in patches)
            statistics["wire_bytes"] += sum(patch.indices.numel() * (4 + patch.values.element_size()) for patch in patches)

        buckets: dict[str, _SparseFlushBucket] = {}

        def consume(name, dtype_name, shape, indices, values):
            bucket = buckets.get(dtype_name)
            if bucket is None:
                bucket = buckets[dtype_name] = _SparseFlushBucket(
                    self.args.update_weight_buffer_size,
                    publish,
                )
            bucket.add(name, shape, indices, values)

        if not hasattr(self, "_gather_workspaces"):
            self._gather_workspaces = {}
        queue = _GatherQueue(
            getattr(self.args, "update_weight_delta_batch_gather", 32),
            self.args.update_weight_buffer_size,
            dist.get_rank() == 0,
            consume,
            workspaces=self._gather_workspaces,
        )
        for slots, dtype_name, counts, indices, values, group in iter_delta_entries(
            self._index,
            self._snapshots,
            batch_size=getattr(self.args, "update_weight_delta_batch_diff", 32),
        ):
            queue.put(group, slots, dtype_name, counts, indices, values)
        queue.flush_all()
        if dist.get_rank() == 0:
            for bucket in buckets.values():
                bucket.flush()
        return statistics["flushes"], statistics["wire_bytes"], statistics["updates"]

    @torch.no_grad()
    def update_weights(self) -> None:
        if self._client is None:
            raise RuntimeError("Sparse HCCL updater is not connected")
        if self._failed:
            raise RuntimeError("Sparse HCCL updater cannot be reused after a potentially partial update; restart or reconnect it to force a dense seed")
        if getattr(self, "_needs_reconnect", False):
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
                flushes, wire_bytes = self._send_dense()
                assert self._index is not None
                prime_delta_snapshots(
                    self._index,
                    self._snapshots,
                    pin=True,
                )
                torch.accelerator.synchronize()
                updates = sum(prod(shape) for record in self._index for _name, shape in record.slots or [])
                self._seeded = True
            else:
                flushes, wire_bytes, updates = self._send_sparse_delta()
                self._steady_updates += 1
        except BaseException as error:
            phase_error = error

        failed, payload_started = self._sync_update_state(phase_error)
        if failed:
            if payload_started:
                self._failed = True
                if dist.get_rank() == 0:
                    logger.error("Sparse HCCL update failed after payload transmission began; generation remains paused because partial sparse apply cannot be rolled back")
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
                logger.error("Sparse HCCL finish failed after weights were applied; generation may remain paused and the updater must not be reused")
            self._raise_synchronized_error(finish_error, "finish")
        if dist.get_rank() == 0:
            self.update_weight_metrics = {
                "perf/update_weights_sparse_hccl_wire_mbytes": wire_bytes / (1024**2),
                "perf/update_weights_sparse_hccl_flushes": float(flushes),
                "perf/update_weights_sparse_hccl_updates": float(updates),
            }
            logger.info(
                "Sparse HCCL weight sync v=%d seed=%s flushes=%d wire=%.2f MiB updates=%d",
                self.weight_version,
                self.weight_version == 1,
                flushes,
                wire_bytes / (1024**2),
                updates,
            )
