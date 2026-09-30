import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from functools import partial

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from ..dspark.export import export_dspark_model_weights
from .common import HfWeightSource, VimeRayWeightSyncClient, create_nccl_trainer
from .hf_weight_iterator_direct import HfWeightIteratorDirect


class UpdateWeightFromDistributed:
    """Update distributed vLLM engines through its stateful NCCL trainer API."""

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
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}
        iterator = HfWeightIteratorDirect(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        draft_weights_getter = (
            partial(
                export_dspark_model_weights,
                model,
                use_policy_embedding=not self.args.dspark_pretrained_model,
            )
            if self.args.dspark_enabled
            else None
        )
        self._source = HfWeightSource(iterator, weights_getter, draft_weights_getter)
        self._trainer = None

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
        engine_gpu_counts = list(
            engine_gpu_counts or [self.args.rollout_num_gpus_per_engine] * len(self.rollout_engines)
        )
        client = VimeRayWeightSyncClient(
            self.rollout_engines,
            lambda: self.weight_version,
            engine_gpu_counts,
        )
        self._trainer = create_nccl_trainer(
            client,
            self._source,
            engine_gpu_counts,
        )

    def disconnect_rollout_engines(self) -> None:
        if self._trainer is not None:
            self._trainer.shutdown()
            self._trainer = None

    def pop_metrics(self) -> dict[str, float]:
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    @torch.no_grad()
    def update_weights(self) -> None:
        assert self._trainer is not None
        self.update_weight_metrics = {}
        if hasattr(self._source, "reset_metrics"):
            self._source.reset_metrics()
        client = self._trainer.client
        client.prepare_seconds = 0.0
        client.finish_seconds = 0.0
        total_started = time.perf_counter()
        self.weight_version += 1

        pause_started = time.perf_counter()
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())
        pause_flush_seconds = time.perf_counter() - pause_started

        transfer_started = time.perf_counter()
        client.draft = False
        self._trainer.send_weights()
        update_draft = self.args.dspark_enabled or (
            self.args.enable_mtp_training and (self.args.vllm_speculative_config or {}).get("method") == "mtp"
        )
        if update_draft:
            self._source.draft = self.args.dspark_enabled
            client.draft = True
            self._trainer.send_weights()
            self._source.draft = False
            client.draft = False
        transfer_phase_seconds = time.perf_counter() - transfer_started

        resume_started = time.perf_counter()
        if dist.get_rank() == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
        resume_seconds = time.perf_counter() - resume_started

        source_metrics = self._source.stop_metrics() if hasattr(self._source, "stop_metrics") else {}
        transferred_bytes = source_metrics.get("weight_update_bytes", 0.0)
        transfer_load_seconds = max(
            0.0,
            transfer_phase_seconds
            - source_metrics.get("weight_update_export_seconds", 0.0)
            - client.prepare_seconds
            - client.finish_seconds,
        )
        self.update_weight_metrics = {
            "weight_update_total_seconds": time.perf_counter() - total_started,
            "weight_update_pause_flush_seconds": pause_flush_seconds,
            "weight_update_prepare_seconds": client.prepare_seconds,
            "weight_update_transfer_phase_seconds": transfer_phase_seconds,
            "weight_update_export_seconds": source_metrics.get("weight_update_export_seconds", 0.0),
            "weight_update_transfer_load_seconds": transfer_load_seconds,
            "weight_update_finish_seconds": client.finish_seconds,
            "weight_update_resume_seconds": resume_seconds,
            "weight_update_bytes": transferred_bytes,
            "weight_update_chunks": source_metrics.get("weight_update_chunks", 0.0),
            "weight_update_effective_gib_per_second": (
                transferred_bytes / (1024**3) / transfer_phase_seconds if transfer_phase_seconds > 0 else 0.0
            ),
        }


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
) -> None:
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
