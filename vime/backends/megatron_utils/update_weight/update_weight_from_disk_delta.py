from __future__ import annotations

import json
import logging
import os
import queue
import shutil
from argparse import Namespace
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import ray
import safetensors.numpy
import torch
import torch.distributed as dist
import zstandard
from ray.actor import ActorHandle

from vime.utils.disk_delta import NUM_WORKERS, checksum, make_tensor_reader, overwrite_encode
from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase

logger = logging.getLogger(__name__)


class UpdateWeightFromDiskDelta:
    """Publish byte-level HF weight deltas and reload them through local checkpoints.

    This is intentionally independent of ``UpdateWeightFromDistributed``: all
    training ranks still participate in the Ascend PP/TP/EP conversion iterator,
    while only global rank zero publishes the canonical result.  No HCCL group is
    created for this transport.
    """

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
        self.weights_getter = weights_getter
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        self.rollout_engines: list[ActorHandle] = []
        self.delta_dir = args.update_weight_disk_dir
        self.delta_encoding = args.update_weight_delta_encoding
        self.checksum_algorithm = args.update_weight_delta_checksum
        self._iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._snapshot: dict[str, np.ndarray] = {}
        self._baseline_captured = False
        self._post_write_hook: Callable | None = None
        if args.custom_update_weight_post_write_path:
            from vime.utils.misc import load_function

            self._post_write_hook = load_function(args.custom_update_weight_post_write_path)

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        del rollout_engine_lock, engine_gpu_counts, engine_gpu_offsets
        self.rollout_engines = list(rollout_engines)

    def disconnect_rollout_engines(self) -> None:
        return

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        if not self._baseline_captured:
            self._capture_baseline()
            self._baseline_captured = True
            return

        self.weight_version += 1
        self._publish()
        self._reload_engines()
        self._record_metrics()

    def _capture_baseline(self) -> None:
        """Use the serving checkpoint as the byte-exact version-zero baseline."""
        pulls = []
        if dist.get_rank() == 0:
            shutil.rmtree(self.delta_dir, ignore_errors=True)
            os.makedirs(self.delta_dir, exist_ok=True)
            if self._post_write_hook is not None:
                self._post_write_hook(self.args, self.delta_dir, self.rollout_engines)
            pulls = [engine.pull_weights.remote(target_version=0) for engine in self.rollout_engines]
        dist.barrier(group=get_gloo_group())

        read_hf = make_tensor_reader(self.args.hf_checkpoint)
        for name, tensor in self._iter_hf_tensors(progress_desc="Capture disk delta baseline"):
            try:
                self._snapshot[name] = read_hf(name)
            except KeyError:
                self._snapshot[name] = _tensor_bytes(tensor)
                logger.warning("delta baseline: %s absent from hf_checkpoint; using current converted weight", name)

        if dist.get_rank() == 0:
            ray.get(pulls)
            logger.info("[disk delta] captured baseline for %d tensors", len(self._snapshot))

    def _publish(self) -> None:
        self._encode_delta()
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            self._write_delta_files()
        dist.barrier(group=get_gloo_group())

    def _iter_hf_tensors(self, *, progress_desc: str):
        """All ranks execute conversion collectives; only rank zero publishes tensors."""
        for chunk in self._iterator.get_hf_weight_chunks(self.weights_getter(), progress_desc=progress_desc):
            if dist.get_rank() == 0:
                yield from chunk

    def _encode_delta(self) -> None:
        self._version_dir = os.path.join(self.delta_dir, f"weight_v{self.weight_version:06d}")
        self._delta: dict[str, np.ndarray] = {}
        self._checksums: dict[str, str] = {}
        self.changed_bytes = 0
        self.total_bytes = 0
        self.wire_bytes = 0

        if dist.get_rank() != 0:
            # Do not return: every rank must execute the conversion iterator's collectives.
            for _name, _tensor in self._iter_hf_tensors(progress_desc="Encode disk delta"):
                pass
            return

        os.makedirs(self._version_dir, exist_ok=True)
        max_bytes = max((value.nbytes for value in self._snapshot.values()), default=0)
        free_buffers: queue.Queue[torch.Tensor] = queue.Queue()
        use_pinned = max_bytes > 0
        if use_pinned:
            try:
                pool_size = max(2, min(2 * NUM_WORKERS, (8 << 30) // max(max_bytes, 1)))
                for _ in range(pool_size):
                    free_buffers.put(torch.empty(max_bytes, dtype=torch.uint8, pin_memory=True))
            except RuntimeError as exc:
                logger.warning("Pinned host buffers unavailable (%s); using pageable copies", exc)
                use_pinned = False

        def diff_and_compress(name: str, new: np.ndarray) -> tuple[str, np.ndarray, np.ndarray | None, str | None, int]:
            old = self._snapshot[name]
            if new.nbytes != old.nbytes:
                raise ValueError(f"Delta tensor size changed for {name}: {old.nbytes} != {new.nbytes}")
            if self.delta_encoding == "xor":
                diff = new ^ old
                changed = int(np.count_nonzero(diff))
            else:
                mask = new != old
                changed = int(np.count_nonzero(mask))
                diff = overwrite_encode(new, mask)
            if not changed:
                return name, new, None, None, 0
            compressed = np.frombuffer(zstandard.ZstdCompressor(level=1).compress(diff), dtype=np.uint8)
            return name, new, compressed, checksum(self.checksum_algorithm, new), changed

        pool = ThreadPoolExecutor(max_workers=NUM_WORKERS)
        inflight: deque = deque()
        try:
            for name, tensor in self._iter_hf_tensors(progress_desc="Encode disk delta"):
                new = _tensor_bytes(tensor, free_buffers=free_buffers if use_pinned else None)
                self.total_bytes += new.nbytes
                inflight.append(pool.submit(diff_and_compress, name, new))
                if len(inflight) >= 2 * NUM_WORKERS:
                    self._collect_encoded(inflight.popleft())
            while inflight:
                self._collect_encoded(inflight.popleft())
        finally:
            pool.shutdown()

    def _collect_encoded(self, future) -> None:
        name, new, compressed, digest, changed = future.result()
        self._snapshot[name] = new
        if changed:
            self.changed_bytes += changed
            assert compressed is not None and digest is not None
            self._delta[name] = compressed
            self._checksums[name] = digest

    def _write_delta_files(self) -> None:
        if self._delta:
            filename = "model-00000-of-00001.safetensors"
            blob = safetensors.numpy.save(self._delta, metadata=self._checksums)
            self.wire_bytes = len(blob)
            _atomic_write(os.path.join(self._version_dir, filename), blob)
        else:
            filename = None
        index = {
            "metadata": {
                "version": f"{self.weight_version:06d}",
                "base_version": f"{self.weight_version - 1:06d}",
                "delta_encoding": self.delta_encoding,
                "compression_format": "zstd",
                "checksum_format": self.checksum_algorithm,
            },
            "weight_map": {name: filename for name in self._delta},
        }
        _atomic_write(
            os.path.join(self._version_dir, "model.safetensors.index.json"),
            json.dumps(index).encode(),
        )

    def _reload_engines(self) -> None:
        if self._post_write_hook is not None:
            self._post_write_hook(self.args, self._version_dir, self.rollout_engines)
        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            ray.get([engine.pull_weights.remote(self.weight_version) for engine in self.rollout_engines])
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            try:
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
                ray.get(
                    [
                        engine.update_weights_from_disk.remote(
                            self.args.update_weight_local_checkpoint_dir,
                            weight_version=str(self.weight_version),
                        )
                        for engine in self.rollout_engines
                    ]
                )
            finally:
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _record_metrics(self) -> None:
        device = _metric_device()
        counts = torch.tensor(
            [self.changed_bytes, self.total_bytes, self.wire_bytes], dtype=torch.int64, device=device
        )
        dist.all_reduce(counts)
        changed, total, wire = counts.tolist()
        self.update_weight_metrics["perf/update_weights_density"] = changed / max(total, 1)
        self.update_weight_metrics["perf/update_weights_wire_bytes"] = wire
        if dist.get_rank() == 0:
            logger.info("[disk delta v=%s] density=%.2f%% wire=%.2f GB", self.weight_version, 100 * changed / max(total, 1), wire / 1e9)


def _tensor_bytes(tensor: torch.Tensor, *, free_buffers: queue.Queue[torch.Tensor] | None = None) -> np.ndarray:
    flat = tensor.detach().contiguous().view(torch.uint8).reshape(-1)
    if flat.device.type == "cpu":
        return flat.numpy().copy()
    if free_buffers is None:
        return flat.cpu().numpy().copy()

    buffer = free_buffers.get()
    try:
        buffer[: flat.numel()].copy_(flat, non_blocking=True)
        if flat.device.type == "npu":
            torch.npu.current_stream().synchronize()
        elif flat.device.type == "cuda":
            torch.cuda.current_stream().synchronize()
        return buffer[: flat.numel()].numpy().copy()
    finally:
        free_buffers.put(buffer)


def _metric_device() -> torch.device:
    if hasattr(torch, "npu") and torch.npu.is_available():
        return torch.device("npu", torch.npu.current_device())
    if torch.cuda.is_available():
        return torch.device("cuda", torch.cuda.current_device())
    return torch.device("cpu")


def _atomic_write(path: str, data: bytes) -> None:
    temporary = path + ".tmp"
    with open(temporary, "wb") as output:
        output.write(data)
        output.flush()
        os.fsync(output.fileno())
    os.replace(temporary, path)
