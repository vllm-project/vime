from __future__ import annotations

import json
import logging
import os
import shutil
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle
from safetensors.torch import save_file

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase

logger = logging.getLogger(__name__)

_HF_WEIGHT_FILE_NAMES = {
    "model.safetensors.index.json",
    "pytorch_model.bin.index.json",
    "tf_model.h5",
    "flax_model.msgpack",
}
_HF_WEIGHT_FILE_SUFFIXES = (".safetensors", ".bin", ".pt", ".pth", ".ckpt", ".msgpack")


class UpdateWeightFromDisk:
    """Publish a full HF checkpoint and reload non-colocated rollout engines from disk."""

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
        self._iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
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
        self.weight_version += 1
        version_dir = Path(self.args.update_weight_disk_dir) / f"weight_v{self.weight_version:06d}"
        started = time.perf_counter()

        self._prepare_version_dir(version_dir)
        write_started = time.perf_counter()
        self._write_full_checkpoint(version_dir)
        dist.barrier(group=get_gloo_group())
        write_finished = time.perf_counter()

        if self._post_write_hook is not None:
            self._post_write_hook(self.args, str(version_dir), self.rollout_engines)
        dist.barrier(group=get_gloo_group())

        if dist.get_rank() == 0:
            reload_started = time.perf_counter()
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            paused = time.perf_counter()
            try:
                ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
                flushed = time.perf_counter()
                ray.get(
                    [
                        engine.update_weights_from_disk.remote(
                            model_path=str(version_dir),
                            weight_version=str(self.weight_version),
                        )
                        for engine in self.rollout_engines
                    ]
                )
                reloaded = time.perf_counter()
            finally:
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            resumed = time.perf_counter()
            self.update_weight_metrics.update(
                {
                    "perf/update_weights_full_disk_write_time": write_finished - write_started,
                    "perf/update_weights_full_disk_pause_time": paused - reload_started,
                    "perf/update_weights_full_disk_flush_time": flushed - paused,
                    "perf/update_weights_full_disk_reload_time": reloaded - flushed,
                    "perf/update_weights_full_disk_resume_time": resumed - reloaded,
                    "perf/update_weights_full_disk_total_time": resumed - started,
                }
            )
            logger.info("[disk full timings v=%s] %s", self.weight_version, self.update_weight_metrics)
        dist.barrier(group=get_gloo_group())

    def _prepare_version_dir(self, version_dir: Path) -> None:
        if dist.get_rank() == 0:
            shutil.rmtree(version_dir, ignore_errors=True)
            version_dir.mkdir(parents=True, exist_ok=True)
            source_dir = Path(self.args.hf_checkpoint)
            for source in source_dir.iterdir():
                if source.is_file() and not _is_hf_weight_file(source):
                    shutil.copy2(source, version_dir / source.name)
        dist.barrier(group=get_gloo_group())

    def _write_full_checkpoint(self, version_dir: Path) -> None:
        is_writer = dist.get_rank() == 0
        weight_map: dict[str, str] = {}
        total_size = 0
        shard_files: list[str] = []

        for chunk_index, chunk in enumerate(
            self._iterator.get_hf_weight_chunks(
                self.weights_getter(), progress_desc="Save full disk checkpoint"
            ),
            start=1,
        ):
            if not is_writer:
                continue
            state_dict: dict[str, torch.Tensor] = {}
            for name, tensor in chunk:
                if name in weight_map or name in state_dict:
                    raise ValueError(f"Duplicate HF tensor while saving full disk checkpoint: {name}")
                tensor = tensor.detach()
                if not tensor.is_contiguous():
                    tensor = tensor.contiguous()
                if tensor.device.type != "cpu":
                    tensor = tensor.cpu()
                state_dict[name] = tensor
                total_size += tensor.numel() * tensor.element_size()
            if not state_dict:
                continue
            filename = f"model-{chunk_index:05d}.safetensors"
            save_file(state_dict, version_dir / filename, metadata={"format": "pt"})
            shard_files.append(filename)
            weight_map.update({name: filename for name in state_dict})

        if is_writer:
            if not shard_files:
                raise ValueError("No HF tensors were produced for full disk checkpoint")
            rename_map: dict[str, str] = {}
            total_files = len(shard_files)
            for index, old_name in enumerate(shard_files, start=1):
                new_name = f"model-{index:05d}-of-{total_files:05d}.safetensors"
                os.replace(version_dir / old_name, version_dir / new_name)
                rename_map[old_name] = new_name
            index_data = {
                "metadata": {"total_size": total_size},
                "weight_map": {name: rename_map[filename] for name, filename in weight_map.items()},
            }
            index_path = version_dir / "model.safetensors.index.json"
            temporary = index_path.with_suffix(index_path.suffix + ".tmp")
            with temporary.open("w", encoding="utf-8") as output:
                json.dump(index_data, output)
                output.flush()
                os.fsync(output.fileno())
            os.replace(temporary, index_path)


def _is_hf_weight_file(path: Path) -> bool:
    return path.name in _HF_WEIGHT_FILE_NAMES or path.name.endswith(_HF_WEIGHT_FILE_SUFFIXES)
