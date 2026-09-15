from __future__ import annotations

from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from time import perf_counter
from typing import Any

import ray
import torch
import torch.distributed as dist
from modelexpress_rl import (
    ModelExpressControlClient,
    ModelExpressTrainerClient,
    ModelExpressTrainerConfig,
    ObjectStorageConfig,
    ObjectStorageSource,
    ObjectStorageType,
    TrainerStagingMode,
    WeightPayloadFormat,
    WeightVersionRef,
    WeightVersionState,
)
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_direct import HfWeightIteratorDirect


class UpdateWeightFromModelExpress:
    """Publish Megatron weights through ModelExpress and install them in vLLM.

    The current path publishes canonical S3 XOR deltas with optional periodic full
    checkpoints. Future integrations may add P2P NIXL/RDMA transfer formats behind
    the same updater lifecycle.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self._config = dict(args.modelexpress_config)
        self._weights_getter = weights_getter
        self._weight_iterator = HfWeightIteratorDirect(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self.weight_version = 0
        full_checkpoint_interval = self._config.get("full_hf_checkpoint_interval")
        if full_checkpoint_interval is not None and (
            isinstance(full_checkpoint_interval, bool)
            or not isinstance(full_checkpoint_interval, int)
            or full_checkpoint_interval <= 0
        ):
            raise ValueError("full_hf_checkpoint_interval must be a positive integer")
        self._full_hf_checkpoint_interval = full_checkpoint_interval
        self.rollout_engines: Sequence[ActorHandle] | None = None
        self._baseline_captured = False
        self._update_engine_weights_time = 0.0
        self.update_weight_metrics: dict[str, int | float] = {}
        self._initialize()

    def _initialize(self) -> None:
        """Initialize object storage and rank-local ModelExpress clients."""
        self._rpc_timeout_seconds = self._config.get("rpc_timeout_seconds", 30.0)
        self._object_storage_config = ObjectStorageConfig(
            storage_type=ObjectStorageType.S3,
            uri_prefix=self._config.get("s3_uri_prefix") or "",
            initial_base_version_id=self._config.get("initial_base_version_id") or "",
            seed_checkpoint_path=self._config.get("seed_checkpoint_path") or "",
            endpoint_url=self._config.get("s3_endpoint_url"),
            region_name=self._config.get("s3_region_name"),
        )
        self._current_version_id = self._object_storage_config.initial_base_version_id

        registration_ttl = self._config.get("registration_ttl_seconds")
        self._trainer = ModelExpressTrainerClient.initialize(
            ModelExpressTrainerConfig(
                model_name=self._config.get("model_name"),
                staging_mode=TrainerStagingMode.WRITE_TO_STORAGE,
                payload_format=WeightPayloadFormat.XOR_DELTA,
                server_url=self._config.get("server_url"),
                registration_ttl_seconds=registration_ttl,
                rpc_timeout_seconds=self._rpc_timeout_seconds,
                process_group=get_gloo_group(),
                object_storage=self._object_storage_config,
            )
        )

        if dist.get_rank() == 0:
            self._control = ModelExpressControlClient.connect(
                server_url=self._trainer.server_url,
                rpc_timeout_seconds=self._rpc_timeout_seconds,
            )
        else:
            self._control = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        """Initialize ModelExpress on a newly connected rollout-engine cohort."""
        del rollout_engine_lock, engine_gpu_counts, engine_gpu_offsets, engine_parallel_configs
        connected = tuple(rollout_engines)
        if self.rollout_engines == connected:
            return

        if not self._baseline_captured:
            base_uri = f"{self._object_storage_config.uri_prefix.rstrip('/')}" "/v0/model.safetensors.index.json"
            self._rank_zero_call(
                lambda: self._control.create_weight_version(
                    uid=self._current_version_id,
                    model_name=self._trainer.model_name,
                    idempotency_key=f"vime:{base_uri}",
                    payload_format=WeightPayloadFormat.FULL_TENSOR,
                    object_storage=ObjectStorageSource(
                        storage_type=self._object_storage_config.storage_type,
                        uri=base_uri,
                    ),
                    state=WeightVersionState.READY,
                ),
                f"ModelExpress baseline version {self._current_version_id} registration failed",
            )

        registration_ttl = self._config.get("registration_ttl_seconds")
        lease_ttl = self._config.get("lease_ttl_seconds")

        init_info = {
            "model_name": self._trainer.model_name,
            "server_url": self._trainer.server_url,
            "initial_base_version_id": self._current_version_id,
            "seed_checkpoint_path": self._object_storage_config.seed_checkpoint_path,
            "refit_checkpoint_dir": self._config.get("refit_checkpoint_dir"),
            "object_storage_type": self._object_storage_config.storage_type.value,
            "object_storage_endpoint_url": self._config.get("s3_endpoint_url"),
            "object_storage_region_name": self._config.get("s3_region_name"),
            "registration_ttl_seconds": registration_ttl,
            "lease_ttl_seconds": lease_ttl,
            "max_transfer_attempts": self._config.get("max_transfer_attempts", 3),
            "rpc_timeout_seconds": self._rpc_timeout_seconds,
        }

        self._rank_zero_call(
            lambda: ray.get(
                [engine.init_weight_transfer_engine.remote({"init_info": init_info}) for engine in connected]
            ),
            "vLLM ModelExpress initialization failed",
        )
        self.rollout_engines = connected

    def disconnect_rollout_engines(self) -> None:
        """Forget the current rollout-engine cohort."""
        self.rollout_engines = None

    def pop_metrics(self) -> dict[str, int | float]:
        """Return and clear metrics from the latest weight update."""
        metrics, self.update_weight_metrics = self.update_weight_metrics, {}
        return metrics

    def _rank_zero_call(self, action: Callable[[], Any], description: str) -> Any:
        """Run an action on rank zero and broadcast its result or failure."""
        result = [None, None]
        if dist.get_rank() == 0:
            try:
                result[0] = action()
            except Exception as error:
                result[1] = str(error)
        dist.broadcast_object_list(result, src=0, group=get_gloo_group())
        if result[1] is not None:
            raise RuntimeError(f"{description}: {result[1]}")
        return result[0]

    @torch.no_grad()
    def update_weights(self) -> None:
        """Capture the initial base or publish and install one policy update."""
        if not self._baseline_captured:
            self._trainer.prepare_delta_base(hf_tensor_iter=self._iter_hf_buckets())
            self._baseline_captured = True
            return

        self._update_engine_weights_time = 0.0
        update_number = self.weight_version + 1

        version = self._create_weight_version(update_number)
        self._stage_and_publish(version)

        self._rank_zero_call(
            lambda: self._update_engine_weights(version.version_id),
            f"ModelExpress version {version.version_id} install failed",
        )

        self._current_version_id = version.version_id
        self.weight_version += 1
        self.update_weight_metrics = self._gather_metrics(
            group=get_gloo_group(),
        )

    def _create_weight_version(self, update_number: int) -> WeightVersionRef:
        """Create one version and return its opaque ModelExpress identity."""
        publish_full_checkpoint = (
            self._full_hf_checkpoint_interval is not None and update_number % self._full_hf_checkpoint_interval == 0
        )
        object_storage_uri = (
            f"{self._object_storage_config.uri_prefix.rstrip('/')}" f"/v{update_number}/model.safetensors.index.json"
        )
        create_kwargs = {
            "model_name": self._trainer.model_name,
            "idempotency_key": f"vime:{object_storage_uri}",
            "payload_format": (
                WeightPayloadFormat.FULL_HF_CHECKPOINT if publish_full_checkpoint else WeightPayloadFormat.XOR_DELTA
            ),
            "object_storage": ObjectStorageSource(
                storage_type=self._object_storage_config.storage_type,
                uri=object_storage_uri,
            ),
            "state": WeightVersionState.STAGING,
        }
        if not publish_full_checkpoint:
            create_kwargs["base_version_id"] = self._current_version_id
        target_version_id = str(
            self._rank_zero_call(
                lambda: self._control.create_weight_version(**create_kwargs).version_id,
                f"ModelExpress update {update_number} creation failed",
            )
        )
        return WeightVersionRef(target_version_id)

    def _stage_and_publish(self, version: WeightVersionRef) -> None:
        """Stage a version, publish its artifacts, and mark it READY."""
        staged = self._trainer.stage_shard(
            version=version,
            hf_tensor_iter=self._iter_hf_buckets(),
        )

        staged.publish()

        dist.barrier(group=get_gloo_group())
        self._rank_zero_call(
            lambda: self._control.update_weight_version_state(
                version.version_id,
                WeightVersionState.READY,
            ),
            f"ModelExpress version {version.version_id} activation failed",
        )

    def _iter_hf_buckets(self):
        """Yield canonical non-expert and expert Hugging Face weight buckets."""
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        yield from self._weight_iterator.get_hf_weight_chunks(
            self._weights_getter(),
            progress_desc="Stage ModelExpress weights",
            should_convert_chunk=lambda chunk_idx: chunk_idx % world_size == rank,
        )

    def _gather_metrics(
        self,
        *,
        group: Any,
    ) -> dict[str, int | float]:
        """Aggregate ModelExpress publication and serving-cutover metrics."""
        local_metrics = self._trainer.pop_metrics()
        counts = torch.tensor(
            [
                local_metrics.get("changed_bytes", 0),
                local_metrics.get("total_bytes", 0),
                local_metrics.get("wire_bytes", 0),
            ],
            dtype=torch.int64,
        )
        dist.all_reduce(counts, op=dist.ReduceOp.SUM, group=group)

        timings = torch.tensor(
            [
                local_metrics.get("stage_delta_time", 0.0),
                local_metrics.get("publish_object_storage_time", 0.0),
                self._update_engine_weights_time,
            ],
            dtype=torch.float64,
        )
        dist.all_reduce(timings, op=dist.ReduceOp.MAX, group=group)

        changed_bytes, total_bytes, wire_bytes = counts.tolist()
        stage_delta_time, publish_object_storage_time, update_engine_weights_time = timings.tolist()
        return {
            "perf/update_weights_density": changed_bytes / max(total_bytes, 1),
            "perf/update_weights_wire_bytes": wire_bytes,
            "perf/mx_stage_delta_time": stage_delta_time,
            "perf/mx_publish_object_storage_time": publish_object_storage_time,
            "perf/mx_update_engine_weights_time": update_engine_weights_time,
        }

    def _update_engine_weights(self, target_version_id: str) -> None:
        """Pause rollout engines, install one version, and resume generation."""
        engines = tuple(self.rollout_engines or ())
        if not engines:
            raise RuntimeError("ModelExpress requires rollout engines")
        phase_started = perf_counter()
        ray.get([engine.pause_generation.remote() for engine in engines])
        ray.get([engine.flush_cache.remote() for engine in engines])
        ray.get([engine.start_weight_update.remote() for engine in engines])
        ray.get([engine.update_weights.remote({"version_id": target_version_id}) for engine in engines])
        ray.get([engine.finish_weight_update.remote() for engine in engines])
        ray.get([engine.continue_generation.remote() for engine in engines])
        self._update_engine_weights_time = perf_counter() - phase_started
