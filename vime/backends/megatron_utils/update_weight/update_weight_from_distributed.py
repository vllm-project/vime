from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .common import HfWeightSource, VimeRayWeightSyncClient, create_nccl_trainer
from .hf_weight_iterator_base import HfWeightIteratorBase


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
        iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._source = HfWeightSource(iterator, weights_getter)
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
        self.weight_version += 1

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

        client = self._trainer.client
        client.draft = False
        self._trainer.send_weights()
        if self.args.enable_mtp_training and (self.args.vllm_speculative_config or {}).get("method") == "mtp":
            client.draft = True
            self._trainer.send_weights()
            client.draft = False

        if dist.get_rank() == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())


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
