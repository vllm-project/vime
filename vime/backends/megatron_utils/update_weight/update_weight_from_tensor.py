from __future__ import annotations

from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm

from vime.utils.distributed_utils import get_gloo_group
from vime.utils.types import ParamInfo

from ..lora_utils import (
    build_lora_weight_update_request,
    export_lora_named_tensors,
    is_lora_enabled,
    save_lora_adapter_for_vllm,
)
from ..megatron_to_hf import convert_to_hf
from .common import HfWeightSource, VimeRayWeightSyncClient, create_nccl_trainer
from .expert_routing import configure_expert_routing
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import post_process_weights


def _native_ipc_buffer_size(args: Namespace, param_info_buckets: Sequence[Sequence[ParamInfo]] | None) -> int:
    buffer_size = args.update_weight_buffer_size
    if not param_info_buckets:
        return buffer_size

    tensor_parallel_size = mpu.get_tensor_model_parallel_world_size()
    expert_tensor_parallel_size = mpu.get_expert_tensor_parallel_world_size()
    for bucket in param_info_buckets:
        for info in bucket:
            parallel_size = expert_tensor_parallel_size if ".experts." in info.name else tensor_parallel_size
            buffer_size = max(buffer_size, info.size * parallel_size)
    return buffer_size


def _build_packed_ipc_update_info(
    named_tensors: Sequence[tuple[str, torch.Tensor]],
) -> tuple[dict[str, Any], torch.Tensor | None]:
    if not named_tensors:
        return (
            {
                "names": [],
                "dtype_names": [],
                "shapes": [],
                "tensor_sizes": [],
                "ipc_handles": {},
            },
            None,
        )

    from torch.multiprocessing.reductions import reduce_tensor
    from vllm.distributed.weight_transfer.packed_tensor import pack_tensors

    chunk = pack_tensors(
        iter(named_tensors),
        post_iter_func=lambda item: item[1],
        buffer_size_bytes=sum(tensor.numel() * tensor.element_size() for _, tensor in named_tensors),
    )
    assert chunk is not None
    _, ipc_args = reduce_tensor(chunk.packed_tensor)
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return (
        {
            "names": chunk.names,
            "dtype_names": [str(dtype).split(".")[-1] for dtype in chunk.dtypes],
            "shapes": chunk.shapes,
            "tensor_sizes": chunk.tensor_sizes,
            "ipc_handles": {gpu_uuid: ipc_args},
        },
        chunk.packed_tensor,
    )


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → gather_object(Gloo CPU, over the engine
    slot ranks) → Ray IPC to engine.  Distributed: GPU NCCL broadcast to remote engines.
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
        """
        Compute param buckets.  IPC Gloo groups are created later in
        ``connect_rollout_engines`` once ``engine_gpu_counts`` is known.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.rank = dist.get_rank()
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._lora_enabled = is_lora_enabled(args)
        if self._lora_enabled:
            self._hf_weight_iterator = None
            self._full_param_info_buckets = None
            self._source = None
        else:
            self._hf_weight_iterator = HfWeightIteratorBase.create(
                args=args, model=model, model_name=model_name, quantization_config=quantization_config
            )
            param_info_buckets = getattr(self._hf_weight_iterator, "megatron_local_param_info_buckets", None)
            self._full_param_info_buckets = (
                tuple(tuple(bucket) for bucket in param_info_buckets) if param_info_buckets is not None else None
            )
            self._source = HfWeightSource(self._hf_weight_iterator, self.weights_getter)
        self._non_expert_param_info_buckets: list[list[ParamInfo]] | None = None

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        # The vLLM LoRA tensor-update path can only update an adapter that is
        # already registered, so the first sync always loads from disk to register it;
        # later syncs stream the adapter over IPC.
        self._lora_adapter_registered = False
        self._expert_transfer_plan = []
        self._native_trainers = []

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
        engine_parallel_configs: Sequence[Mapping[str, Any]] | None = None,
    ) -> None:
        del rollout_engine_lock
        for trainer in self._native_trainers:
            trainer.shutdown()
        self._all_rollout_engines = list(rollout_engines)
        self.rollout_engines = []
        self._ipc_engine = None
        self._native_trainers = []

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            # Fallback: assume engines are densely packed (no placeholder gaps).
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        # Compute colocated engine count: engines whose GPUs fall within actor GPU range.
        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self.rollout_engines = list(rollout_engines[:colocate_engine_nums])
        distributed_rollout_engines = list(rollout_engines[colocate_engine_nums:])
        use_distribute = bool(distributed_rollout_engines)
        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]
        colocate_parallel_configs = (
            engine_parallel_configs[:colocate_engine_nums] if engine_parallel_configs is not None else None
        )

        if self._lora_enabled:
            if distributed_rollout_engines:
                raise RuntimeError("LoRA weight updates require all rollout engines to be colocated.")
            if getattr(self.args, "lora_sync_from_tensor", False):
                if self._ipc_gather_group is None:
                    for index in range(colocate_engine_nums):
                        group_ranks = list(
                            range(
                                colocate_gpu_offsets[index],
                                colocate_gpu_offsets[index] + colocate_gpu_counts[index],
                            )
                        )
                        new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                        if self.rank in group_ranks:
                            self._ipc_gather_group = new_group
                            self._ipc_gather_src = colocate_gpu_offsets[index]

                for index, engine in enumerate(self.rollout_engines):
                    start = colocate_gpu_offsets[index]
                    if start <= self.rank < start + colocate_gpu_counts[index]:
                        self._ipc_engine = engine

                if self.rank == 0:
                    ray.get(
                        [
                            engine.init_weight_transfer_engine.remote({"init_info": {"packed": True}})
                            for engine in self.rollout_engines
                        ]
                    )
            return

        self._non_expert_param_info_buckets, self._expert_transfer_plan = configure_expert_routing(
            args=self.args,
            full_param_info_buckets=self._full_param_info_buckets,
            get_local_weight_names=self.weights_getter,
            engine_gpu_counts=colocate_gpu_counts,
            engine_gpu_offsets=colocate_gpu_offsets,
            engine_parallel_configs=colocate_parallel_configs,
            use_distribute=use_distribute,
        )

        if not self._expert_transfer_plan:
            if self.rollout_engines:
                from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
                from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerInitInfo

                client = VimeRayWeightSyncClient(self.rollout_engines, lambda: self.weight_version)
                trainer = WeightTransferTrainerFactory.trainer_init(
                    IPCTrainerInitInfo(
                        rank=dist.get_rank(),
                        packed=True,
                        packed_buffer_size_bytes=_native_ipc_buffer_size(
                            self.args,
                            self._full_param_info_buckets,
                        ),
                    ),
                    client=client,
                    source=self._source,
                )
                self._native_trainers.append(trainer)
            if distributed_rollout_engines:
                distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
                client = VimeRayWeightSyncClient(
                    distributed_rollout_engines,
                    lambda: self.weight_version,
                    distributed_gpu_counts,
                )
                trainer = create_nccl_trainer(
                    client,
                    self._source,
                    distributed_gpu_counts,
                )
                self._native_trainers.append(trainer)
            return

        # Rank-local expert routing is the one case the generic IPC API cannot
        # express: each rollout EP rank receives a different expert subset.
        if self._ipc_gather_group is None:
            for index in range(colocate_engine_nums):
                group_ranks = list(
                    range(
                        colocate_gpu_offsets[index],
                        colocate_gpu_offsets[index] + colocate_gpu_counts[index],
                    )
                )
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[index]

        for index, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[index]
            if start <= dist.get_rank() < start + colocate_gpu_counts[index]:
                self._ipc_engine = engine

        if dist.get_rank() == 0:
            ray.get(
                [
                    engine.init_weight_transfer_engine.remote({"init_info": {"packed": True}})
                    for engine in self.rollout_engines
                ]
            )

    def pop_metrics(self) -> dict[str, float]:
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    def _prepare_expert_weight_batch(
        self,
        transfers: Sequence[Any],
        megatron_local_weights: Mapping[str, torch.Tensor],
        staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]],
    ) -> list[tuple[str, torch.Tensor]]:
        local_params = []
        p2p_ops = []
        buffer_offsets: dict[tuple[torch.dtype, tuple[int, ...]], int] = defaultdict(int)
        for transfer in transfers:
            for expert_param in transfer.params:
                info = expert_param.info
                if self.rank != transfer.source_rank and self.rank not in transfer.target_ranks:
                    continue
                key = (info.dtype, tuple(info.shape))
                pool = staging_buffers.setdefault(key, [])
                offset = buffer_offsets[key]
                buffer_offsets[key] = offset + 1
                if offset == len(pool):
                    pool.append(torch.empty(info.shape, dtype=info.dtype, device="cuda"))
                tensor = pool[offset]
                if self.rank == transfer.source_rank:
                    source = megatron_local_weights[info.name]
                    if source.shape != info.shape or source.dtype != info.dtype:
                        raise ValueError(f"expert metadata changed for {info.name}")
                    tensor.copy_(source, non_blocking=True)
                    p2p_ops.extend(
                        dist.P2POp(dist.isend, tensor, target_rank)
                        for target_rank in transfer.target_ranks
                        if target_rank != self.rank
                    )
                    if self.rank in expert_param.target_ranks:
                        local_params.append((expert_param, tensor))
                else:
                    p2p_ops.append(dist.P2POp(dist.irecv, tensor, transfer.source_rank))
                    local_params.append((expert_param, tensor))

        for request in dist.batch_isend_irecv(p2p_ops) if p2p_ops else ():
            request.wait()

        hf_named_tensors = []
        for expert_param, tensor in local_params:
            hf_named_tensors.extend(
                convert_to_hf(
                    self.args,
                    self.model_name,
                    expert_param.info.name,
                    tensor,
                    self.quantization_config,
                )
            )
        return hf_named_tensors

    def _update_expert_weights(
        self,
        megatron_local_weights: Mapping[str, torch.Tensor],
    ) -> None:
        dist.barrier(group=get_gloo_group())
        # Initialize WORLD on all ranks before subset batched P2P.
        dist.barrier()
        # Reuse staging across layers instead of fragmenting the CUDA allocator.
        staging_buffers: dict[tuple[torch.dtype, tuple[int, ...]], list[torch.Tensor]] = {}
        for transfer_group in tqdm(
            self._expert_transfer_plan,
            disable=self.rank != 0,
            desc="Update expert weights",
        ):
            for transfer_batch in transfer_group:
                hf_named_tensors = self._prepare_expert_weight_batch(
                    transfer_batch,
                    megatron_local_weights,
                    staging_buffers,
                )
                refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
                ray.get(refs)
                dist.barrier(group=get_gloo_group())
                torch.cuda.synchronize()
                del refs, long_lived_tensors, hf_named_tensors
                torch.cuda.ipc_collect()
                torch.cuda.empty_cache()
        del staging_buffers
        torch.cuda.empty_cache()

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1
        if self._lora_enabled:
            self._update_lora_adapter()
            return

        if self.rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self._all_rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self._all_rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self._all_rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        if self._native_trainers:
            for trainer in self._native_trainers:
                trainer.client.draft = False
                trainer.send_weights()
            if self.args.enable_mtp_training and (self.args.vllm_speculative_config or {}).get("method") == "mtp":
                for trainer in self._native_trainers:
                    trainer.client.draft = True
                    trainer.send_weights()
                    trainer.client.draft = False
        else:
            megatron_local_weights = self.weights_getter()
            self._update_rollout_weights(megatron_local_weights, draft=False)

            if self.args.enable_mtp_training and (self.args.vllm_speculative_config or {}).get("method") == "mtp":
                self._update_rollout_weights(megatron_local_weights, draft=True)

        # int4/fp4 post_process
        if self.rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self._all_rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self._all_rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _update_rollout_weights(self, megatron_local_weights, *, draft: bool) -> None:
        if self._ipc_engine is not None and self.rank == self._ipc_gather_src:
            method = self._ipc_engine.start_draft_weight_update if draft else self._ipc_engine.start_weight_update
            ray.get(method.remote())
        dist.barrier(group=get_gloo_group())

        self._send_weight_chunks(megatron_local_weights)
        dist.barrier(group=get_gloo_group())
        torch.cuda.ipc_collect()
        torch.cuda.empty_cache()

        if self._ipc_engine is not None and self.rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote(weight_version=str(self.weight_version)))
        dist.barrier(group=get_gloo_group())

    def _send_weight_chunks(self, megatron_local_weights) -> None:
        param_info_buckets = (
            self._non_expert_param_info_buckets if self._expert_transfer_plan else self._full_param_info_buckets
        )
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(
            megatron_local_weights,
            param_info_buckets=param_info_buckets,
        ):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            del refs, long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()
            torch.cuda.empty_cache()
        if self._expert_transfer_plan:
            self._update_expert_weights(megatron_local_weights)

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        return _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
        )

    @torch.no_grad()
    def _update_lora_adapter(self) -> None:
        """Push the current adapter to the colocated vLLM engines.

        Disk path (default): export the adapter to a PEFT dir and have vLLM reload it.
        Tensor path (--lora-sync-from-tensor): stream the adapter over IPC into vLLM's
        in-memory update target (#48409), avoiding the multi-GB disk write+read. The
        first sync always uses the disk path so the adapter is registered before any
        in-memory update can target it.
        """
        use_tensor = getattr(self.args, "lora_sync_from_tensor", False) and self._lora_adapter_registered

        # The barriers below must always be reached on rank 0, even when a Ray call
        # raises, otherwise the other ranks would block on the barrier indefinitely.
        if self.rank == 0:
            try:
                ray.get([engine.pause_generation.remote() for engine in self._all_rollout_engines])
                ray.get([engine.flush_cache.remote() for engine in self._all_rollout_engines])
            finally:
                dist.barrier(group=get_gloo_group())
        else:
            dist.barrier(group=get_gloo_group())

        if use_tensor:
            self._send_lora_adapter_via_ipc()
        else:
            adapter_path = save_lora_adapter_for_vllm(self.model, self.args, self.weight_version)
            if self.rank == 0:
                refs = [
                    engine.load_lora_adapter.remote(
                        self.args.lora_adapter_name,
                        adapter_path,
                        weight_version=str(self.weight_version),
                    )
                    for engine in self._all_rollout_engines
                ]
                ray.get(refs)
            self._lora_adapter_registered = True

        if self.rank == 0:
            try:
                ray.get([engine.continue_generation.remote() for engine in self._all_rollout_engines])
            finally:
                dist.barrier(group=get_gloo_group())
        else:
            dist.barrier(group=get_gloo_group())

    @torch.no_grad()
    def _send_lora_adapter_via_ipc(self) -> None:
        """Stream the adapter tensors into vLLM's LoRA update target over IPC (#48409).

        Reuses the colocate IPC channel the full-parameter path uses: open a LoRA update
        transaction on each engine, send the adapter tensors with the same
        ``update_weights_from_tensor`` call (which now routes into the adapter), then
        commit. Each colocate group's gather-source rank drives its own engine's HTTP.
        """
        # vime registers exactly one adapter (`--lora-adapter-name`), so vLLM assigns it
        # int id 1 on first load.
        lora_int_id = 1

        named = export_lora_named_tensors(self.model, self.args)  # collective across TP
        tensor_names = [name for name, _ in named]

        if self._ipc_engine is not None and self.rank == self._ipc_gather_src:
            request = build_lora_weight_update_request(self.args, lora_int_id, tensor_names)
            ray.get(self._ipc_engine.start_lora_weight_update.remote(request, weight_version=str(self.weight_version)))
        dist.barrier(group=get_gloo_group())

        refs, _long_lived = _send_to_colocated_engine(
            named,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
        )
        if refs:
            ray.get(refs)

        if self._ipc_engine is not None and self.rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote(weight_version=str(self.weight_version)))
        dist.barrier(group=get_gloo_group())


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    local_info, weight_ref = _build_packed_ipc_update_info(hf_named_tensors)

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        if not local_info["names"]:
            return [], weight_ref
        ref = ipc_engine.update_weights.remote(local_info)
        return [ref], weight_ref

    gathered_infos = [None] * slot_size if dist.get_rank() == ipc_gather_src else None
    dist.gather_object(local_info, object_gather_list=gathered_infos, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(info is None for info in gathered_infos):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_infos!r}")
        rank_local_infos = [info if info["names"] else None for info in gathered_infos]
        if any(info is not None for info in rank_local_infos):
            refs.append(ipc_engine.update_weights.remote(rank_local_infos))

    return refs, weight_ref
