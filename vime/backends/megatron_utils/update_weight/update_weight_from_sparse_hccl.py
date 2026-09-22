"""verl-style Megatron delta synchronization over Sparse HCCL."""

from __future__ import annotations

import logging
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict
from math import prod

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from vllm_ascend.distributed.weight_transfer.sparse_hccl_engine import (
    SparseHCCLTrainerSendWeightsArgs,
    SparseHCCLWeightTransferEngine,
    SparseHCCLWeightTransferUpdateInfo,
)
from vllm_ascend.distributed.weight_transfer.sparse_weight_patch import SparseWeightPatch

from vime.utils.distributed_utils import get_gloo_group

from .common import VimeRayWeightSyncClient
from .delta_sync import (
    DeltaFlush,
    DeltaParam,
    checksum,
    gather_slot_entries_to_rank0,
)
from .hf_weight_iterator_direct import HfWeightIteratorDirect
from .megatron_delta_export import (
    build_export_index,
    iter_delta_entries,
    prime_delta_snapshots,
)

logger = logging.getLogger(__name__)


class _GatherQueue:
    """Count-triggered queues preserving collective order on every rank."""

    def __init__(self, batch_size: int, max_round_bytes: int, is_wire_master: bool, consume):
        self.batch_size = max(int(batch_size), 1)
        self.max_round_bytes = int(max_round_bytes)
        self.is_wire_master = is_wire_master
        self.consume = consume
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

        device = batch[0][2].device
        counts = torch.cat([entry[1] for entry in batch]).to(device)
        indices = torch.cat([entry[2] for entry in batch])
        values = torch.cat([entry[3] for entry in batch])
        gathered = gather_slot_entries_to_rank0(
            indices,
            values,
            counts,
            group=group,
            max_round_bytes=self.max_round_bytes,
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
                )
            )
            self.shapes.append(list(shape))
            self.nbytes += piece_bytes
            if self.nbytes >= self.capacity:
                self.flush()

    def flush(self) -> None:
        if not self.patches:
            return
        position_offset = value_offset = 0
        params = []
        for patch, shape in zip(self.patches, self.shapes, strict=True):
            position_bytes = patch.indices.numel() * patch.indices.element_size()
            params.append(
                DeltaParam(
                    name=patch.name,
                    dtype=str(patch.values.dtype).replace("torch.", ""),
                    shape=shape,
                    pos_start=position_offset,
                    pos_end=position_offset + position_bytes,
                    pos_width=4,
                    val_start=value_offset,
                    val_end=value_offset + patch.values.numel(),
                )
            )
            position_offset += position_bytes
            value_offset += patch.values.numel()
        positions = torch.cat([patch.indices for patch in self.patches]).contiguous().view(torch.uint8)
        values = torch.cat([patch.values for patch in self.patches]).contiguous()
        self.publish(
            DeltaFlush(
                encoding="indices",
                params=params,
                positions=positions,
                values=values,
                checksum=checksum(positions, values),
            )
        )
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
        self._iterator = HfWeightIteratorDirect(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._param_infos = [
            info
            for bucket in self._iterator.megatron_local_param_info_buckets
            for info in bucket
        ]
        self._index = None
        self._snapshots: dict[str, torch.Tensor] = {}
        self._seeded = False
        self._steady_updates = 0
        self._group = None
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
        self.rollout_engines = list(rollout_engines)
        gpu_counts = list(
            engine_gpu_counts
            or [self.args.rollout_num_gpus_per_engine] * len(self.rollout_engines)
        )
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
            init_info = {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": 1,
                "world_size": sum(gpu_counts) + 1,
            }
            executor = ThreadPoolExecutor(max_workers=1)
            try:
                future = executor.submit(self._client.init_weight_transfer_engine, init_info)
                self._group = SparseHCCLWeightTransferEngine.trainer_init(init_info)
                future.result()
            finally:
                executor.shutdown(wait=False)
        dist.barrier(group=get_gloo_group())

    def disconnect_rollout_engines(self) -> None:
        self._group = None
        self._client = None

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    def _ensure_export_index(self) -> None:
        if self._index is not None:
            return
        local_weights = dict(self.weights_getter())
        self._index = build_export_index(
            self.args,
            self.model_name,
            self.quantization_config,
            self._param_infos,
            local_weights,
        )

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

    def _publish(
        self,
        patches: list[SparseWeightPatch],
        shapes: list[list[int]],
        encoding: str,
        *,
        verify: bool = False,
        wire_checksum: int | None = None,
    ) -> None:
        if dist.get_rank() != 0 or not patches:
            return
        assert self._client is not None and self._group is not None
        dtype_names = [str(patch.values.dtype).replace("torch.", "") for patch in patches]
        if len(set(dtype_names)) != 1:
            raise ValueError("Each delta flush must contain exactly one values dtype")
        num_updates = [patch.values.numel() for patch in patches]
        positions = (
            torch.cat([patch.indices for patch in patches]).contiguous().view(torch.uint8)
            if encoding == "indices"
            else torch.empty(0, dtype=torch.uint8, device=patches[0].values.device)
        )
        values = torch.cat([patch.values for patch in patches]).contiguous()
        update_info = SparseHCCLWeightTransferUpdateInfo(
            names=[patch.name for patch in patches],
            dtype_names=dtype_names,
            shapes=shapes,
            num_updates_list=num_updates,
            encoding=encoding,
            checksum=(checksum(positions, values) if wire_checksum is None else wire_checksum),
            verify=verify,
        )
        executor = ThreadPoolExecutor(max_workers=1)
        try:
            future = executor.submit(self._client.update_weights, asdict(update_info))
            # verl-equivalent handshake-free publish: HCCL broadcast is
            # rendezvous-safe (an early sender simply waits for the receiver
            # to join the collective), so no fixed sleep is needed before
            # entering it.  A rejected/dead receiver hangs until the job
            # supervisor kills us -- the same failure semantics verl's
            # ZMQ PUB manifest + NCCL broadcast accepts; receiver-side
            # errors (checksum/decode/apply) still surface via the
            # future.result() below.
            SparseHCCLWeightTransferEngine.trainer_send_weights(
                iter(patches),
                SparseHCCLTrainerSendWeightsArgs(group=self._group),
            )
            future.result()
        finally:
            executor.shutdown(wait=False)

    def _publish_flush(self, flush: DeltaFlush, *, verify: bool = False) -> None:
        patches = []
        shapes = []
        for param in flush.params:
            positions = flush.positions[param.pos_start : param.pos_end]
            indices = (
                positions.view(torch.int32)
                if flush.encoding == "indices"
                else torch.empty(0, dtype=torch.int32, device=flush.values.device)
            )
            values = flush.values[param.val_start : param.val_end]
            patches.append(SparseWeightPatch(param.name, indices, values))
            shapes.append(param.shape)
        self._publish(
            patches,
            shapes,
            flush.encoding,
            verify=verify,
            wire_checksum=flush.checksum,
        )

    def _send_dense(self, *, verify: bool = False) -> tuple[int, int]:
        flushes = wire_bytes = 0
        for chunk in self._iterator.get_hf_weight_chunks(
            self.weights_getter(),
            progress_desc=(
                "Sparse HCCL state verification"
                if verify
                else "Sparse HCCL dense seed"
            ),
        ):
            if dist.get_rank() != 0:
                continue
            by_dtype: dict[torch.dtype, list[tuple[str, torch.Tensor]]] = {}
            for name, tensor in chunk:
                by_dtype.setdefault(tensor.dtype, []).append((name, tensor))
            for entries in by_dtype.values():
                params = []
                flat_values = []
                value_offset = 0
                for name, tensor in entries:
                    flat = tensor.detach().contiguous().view(-1)
                    params.append(
                        DeltaParam(
                            name=name,
                            dtype=str(tensor.dtype).replace("torch.", ""),
                            shape=list(tensor.shape),
                            pos_start=0,
                            pos_end=0,
                            pos_width=4,
                            val_start=value_offset,
                            val_end=value_offset + flat.numel(),
                        )
                    )
                    flat_values.append(flat)
                    value_offset += flat.numel()
                values = torch.cat(flat_values)
                positions = torch.empty(0, dtype=torch.uint8, device=values.device)
                flush = DeltaFlush(
                    encoding="dense",
                    params=params,
                    positions=positions,
                    values=values,
                    checksum=checksum(positions, values),
                )
                self._publish_flush(flush, verify=verify)
                flushes += 1
                wire_bytes += flush.wire_bytes
        return flushes, wire_bytes

    def _verify_due(self) -> bool:
        interval = int(getattr(self.args, "update_weight_delta_verify_every", 0))
        return interval > 0 and self._steady_updates % interval == 0

    def _send_sparse_delta(self) -> tuple[int, int, int]:
        assert self._index is not None
        statistics = {"flushes": 0, "wire_bytes": 0, "updates": 0}

        def publish(flush):
            self._publish_flush(flush)
            statistics["flushes"] += 1
            statistics["updates"] += flush.nnz
            statistics["wire_bytes"] += flush.wire_bytes

        buckets: dict[str, _SparseFlushBucket] = {}

        def consume(name, dtype_name, shape, indices, values):
            bucket = buckets.setdefault(
                dtype_name,
                _SparseFlushBucket(self.args.update_weight_buffer_size, publish),
            )
            bucket.add(name, shape, indices, values)

        queue = _GatherQueue(
            getattr(self.args, "update_weight_delta_batch_gather", 32),
            self.args.update_weight_buffer_size,
            dist.get_rank() == 0,
            consume,
        )
        for slots, dtype_name, counts, indices, values, group in iter_delta_entries(
            self.args,
            self.model_name,
            self.quantization_config,
            self._index,
            self._snapshots,
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
        started = time.perf_counter()
        self.weight_version += 1
        self._begin_update()
        dist.barrier(group=get_gloo_group())
        if not self._seeded:
            flushes, wire_bytes = self._send_dense()
            self._ensure_export_index()
            assert self._index is not None
            prime_delta_snapshots(self._index, self._snapshots, pin=False)
            torch.accelerator.synchronize()
            updates = sum(prod(shape) for record in self._index for _name, shape in record.slots or [])
            self._seeded = True
        else:
            flushes, wire_bytes, updates = self._send_sparse_delta()
            self._steady_updates += 1
            if self._verify_due():
                verify_flushes, verify_bytes = self._send_dense(verify=True)
                flushes += verify_flushes
                wire_bytes += verify_bytes
        dist.barrier(group=get_gloo_group())
        self._finish_update()
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            self.update_weight_metrics = {
                "perf/update_weights_sparse_hccl_seconds": time.perf_counter() - started,
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
