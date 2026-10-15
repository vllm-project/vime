"""Push FSDP2 weights to the vLLM rollout engines.

FSDP2 keeps HF parameter names, so the weights need no name or layout
conversion: every rank materializes each parameter with ``full_tensor()`` (a
collective, so all ranks iterate in the same order) and the vLLM weight
transfer trainer sends them, exactly as the Megatron backend does with its
converted weights. Colocated engines receive CUDA IPC handles; engines on
other GPUs receive an NCCL broadcast from rank 0.
"""

from __future__ import annotations

import logging
from collections.abc import Iterator, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate

from vime.utils.distributed_utils import get_gloo_group

logger = logging.getLogger(__name__)


class FSDPWeightSource:
    """vLLM weight source over the sharded model's parameters, in HF names.

    Parameters are materialized bucket by bucket: every parameter of a bucket
    (at most ``bucket_bytes`` in the rollout dtype) starts its all-gather
    before the first one is yielded, and a bucket is released before the next
    one is gathered. Shards offloaded to the host are moved to ``device``
    first, so the source also works while the trainer sleeps.
    """

    def __init__(self, model: torch.nn.Module, *, dtype: torch.dtype, device: torch.device, bucket_bytes: int):
        self.model = model
        self.dtype = dtype
        self.device = torch.device(device)
        self.bucket_bytes = bucket_bytes
        self._metadata = None

    def max_tensor_bytes(self) -> int:
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        return max((param.numel() * itemsize for param in self.model.parameters()), default=0)

    def metadata(self):
        if self._metadata is None:
            from vllm.distributed.weight_transfer.base import ParamMeta

            # DTensor.shape is the global shape, so this never communicates.
            self._metadata = [
                ParamMeta(name, self.dtype, tuple(param.shape)) for name, param in self.model.named_parameters()
            ]
        return self._metadata

    def __iter__(self) -> Iterator[tuple[str, torch.Tensor]]:
        itemsize = torch.empty((), dtype=self.dtype).element_size()
        bucket, size = [], 0
        for name, param in self.model.named_parameters():
            nbytes = param.numel() * itemsize
            if bucket and size + nbytes > self.bucket_bytes:
                yield from self._materialize(bucket)
                bucket, size = [], 0
            bucket.append((name, param))
            size += nbytes
        if bucket:
            yield from self._materialize(bucket)

    @torch.no_grad()
    def _materialize(self, bucket) -> Iterator[tuple[str, torch.Tensor]]:
        pending = []
        for name, param in bucket:
            tensor = param.detach().to(device=self.device, dtype=self.dtype)
            if isinstance(tensor, DTensor):
                tensor = tensor.redistribute(placements=[Replicate()] * tensor.device_mesh.ndim, async_op=True)
                tensor = tensor.to_local()
            pending.append((name, tensor))
        for name, tensor in pending:
            if hasattr(tensor, "wait"):
                tensor = tensor.wait()
            yield name, tensor


class FSDPWeightUpdater:
    """Connects to the updatable engines and publishes a new weight version per call."""

    def __init__(self, args, model: torch.nn.Module, device: torch.device) -> None:
        self.args = args
        self.source = FSDPWeightSource(
            model,
            dtype=torch.float16 if args.fp16 else torch.bfloat16,
            device=device,
            bucket_bytes=args.update_weight_buffer_size,
        )
        self.weight_version = args.update_weight_start_version
        self.rollout_engines = []
        self._trainers = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence,
        rollout_engine_lock,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, object]] | None = None,
    ) -> None:
        from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

        from vime.backends.megatron_utils.update_weight.common import VimeRayWeightSyncClient, create_nccl_trainer

        del rollout_engine_lock, engine_parallel_configs
        self.disconnect_rollout_engines()
        self.rollout_engines = list(rollout_engines)
        engine_gpu_counts = list(engine_gpu_counts or [self.args.rollout_num_gpus_per_engine] * len(rollout_engines))
        if engine_gpu_offsets is None:
            engine_gpu_offsets, offset = [], 0
            for count in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += count

        # Under --colocate the engines whose GPUs lie inside the trainer's GPU
        # range get CUDA IPC handles; any engines beyond it get NCCL.
        num_colocated = 0
        if self.args.colocate:
            total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
            for offset, count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
                if offset + count > total_actor_gpus:
                    break
                num_colocated += 1
        colocated = self.rollout_engines[:num_colocated]
        distributed = self.rollout_engines[num_colocated:]

        if colocated:
            client = VimeRayWeightSyncClient(colocated, lambda: self.weight_version)
            self._trainers.append(
                WeightTransferTrainerFactory.trainer_init(
                    IPCTrainerInitInfo(
                        rank=dist.get_rank(),
                        packed=True,
                        packed_buffer_size_bytes=max(
                            self.args.update_weight_buffer_size, self.source.max_tensor_bytes()
                        ),
                    ),
                    client=client,
                    source=self.source,
                )
            )
        if distributed:
            client = VimeRayWeightSyncClient(
                distributed, lambda: self.weight_version, engine_gpu_counts[num_colocated:]
            )
            self._trainers.append(create_nccl_trainer(client, self.source, engine_gpu_counts[num_colocated:]))

    def disconnect_rollout_engines(self) -> None:
        for trainer in self._trainers:
            trainer.shutdown()
        self._trainers = []

    @torch.no_grad()
    def update_weights(self) -> None:
        assert self._trainers, "connect_rollout_engines must run before update_weights"
        self.weight_version += 1
        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        for trainer in self._trainers:
            trainer.send_weights()

        if dist.get_rank() == 0:
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())
