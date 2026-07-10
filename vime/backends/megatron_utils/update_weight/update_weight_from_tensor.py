"""
Colocated vLLM weight sync (trainer side)
=========================================

``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC handles
→ ``POST /update_weights`` to vLLM's native ``IPCWeightTransferEngine``.

vLLM handles UUID routing + device_index remapping + layerwise reload
internally; no worker extension or monkey-patch is needed.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from vime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)


def _current_gpu_uuid() -> str:
    device_index = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device_index)
    return str(props.uuid)


def _build_ipc_update_info_from_named_tensors(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, list], list[torch.Tensor]]:
    """Build vLLM IPC ``update_info`` payload from tensors on this rank's GPU.

    Each handle is keyed by the physical GPU UUID of the producing rank rather
    than by a local device index. The coordinator gathers all ranks' dicts and
    merges them; the receiver looks up its own UUID to pick the matching handle,
    then vLLM unconditionally overwrites ``args[6]`` (device_index) with its own
    local index before ``rebuild_cuda_tensor``. This UUID-keyed routing makes
    the path correct under any ``CUDA_VISIBLE_DEVICES`` ordering without
    relying on a torch reductions monkey-patch.

    Return the contiguous tensor refs alongside the payload. ``reduce_tensor``
    only exports CUDA IPC metadata, so the producer storage must stay alive
    until the receiver opens the handle.
    """
    from torch.multiprocessing.reductions import reduce_tensor

    names: list[str] = []
    dtype_names: list[str] = []
    shapes: list[list[int]] = []
    ipc_handles: list[dict[str, tuple]] = []
    weight_refs: list[torch.Tensor] = []
    gpu_uuid = _current_gpu_uuid()

    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        weight = tensor.detach().contiguous()
        weight_refs.append(weight)
        _, ipc_args = reduce_tensor(weight)
        ipc_handles.append({gpu_uuid: ipc_args})

    return (
        {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "ipc_handles": ipc_handles,
        },
        weight_refs,
    )


def _serialize_ipc_update_info(info: dict[str, list]) -> str:
    """Pickle IPC handles for cross-rank gather (Gloo ``all_gather_object`` cannot carry them)."""
    import base64

    import cloudpickle

    return base64.b64encode(cloudpickle.dumps(info)).decode("ascii")


def _deserialize_ipc_update_info(payload: str) -> dict[str, list]:
    import base64

    import cloudpickle

    return cloudpickle.loads(base64.b64decode(payload.encode("ascii")))


def _merge_ipc_update_infos(infos: Sequence[dict[str, list]]) -> dict[str, list]:
    """Merge per-rank IPC payloads so each weight has handles for every GPU UUID in the slot."""
    if not infos:
        raise ValueError("no IPC update_info payloads to merge")
    base = infos[0]
    merged_handles: list[dict[str, tuple]] = []
    num_params = len(base["names"])
    for i in range(num_params):
        combined: dict[str, tuple] = {}
        for info in infos:
            combined.update(info["ipc_handles"][i])
        merged_handles.append(combined)
    return {
        "names": base["names"],
        "dtype_names": base["dtype_names"],
        "shapes": base["shapes"],
        "ipc_handles": merged_handles,
    }


class UpdateWeightFromTensor:
    """
    Update rollout engines from tensor dict:
    gather TP(GPU NCCL) → convert HF(GPU) → send.
    Colocated: build CUDA IPC handles → all_gather_object(Gloo CPU, over the engine
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
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self.update_weight_metrics: dict[str, float] = {}

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args, model=model, model_name=model_name, quantization_config=quantization_config
        )

        self._ipc_gather_group = None
        self._ipc_gather_src = None
        self._ipc_engine = None
        self._model_update_groups = None
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # vLLM IPC handle payloads may use cloudpickle on the Ray/HTTP bridge.
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    # ------------------------------------------------------------------
    # connect / disconnect
    # ------------------------------------------------------------------

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Split colocated/distributed engines. Global source rank (DP=TP=PP=0) creates NCCL
        for distributed. Map ranks to colocated IPC engines.
        """
        self.rollout_engines = rollout_engines

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

        self.use_distribute = len(rollout_engines) > colocate_engine_nums

        if self.use_distribute:
            self.rollout_engines = rollout_engines[:colocate_engine_nums]
            self.distributed_rollout_engines = rollout_engines[colocate_engine_nums:]
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            self._group_name = "vime"
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args, self._group_name, self._model_update_groups, self.distributed_rollout_engines
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self.distributed_rollout_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]

        # Create IPC Gloo gather groups (only on first call; partitioning is
        # fixed across reconnects).
        if self._ipc_gather_group is None:
            for i in range(colocate_engine_nums):
                group_ranks = list(range(colocate_gpu_offsets[i], colocate_gpu_offsets[i] + colocate_gpu_counts[i]))
                new_group = dist.new_group(ranks=group_ranks, backend="gloo")
                if dist.get_rank() in group_ranks:
                    self._ipc_gather_group = new_group
                    self._ipc_gather_src = colocate_gpu_offsets[i]

        # Map training ranks to colocated engine actors.
        for i, engine in enumerate(self.rollout_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

        # vLLM #39212: one-time IPC transfer-engine init on each colocated engine.
        if dist.get_rank() == 0 and self.rollout_engines and not self._ipc_initialized:
            ray.get([engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self.rollout_engines])
            self._ipc_initialized = True

    def pop_metrics(self) -> dict[str, float]:
        """
        Return and clear ``update_weight_metrics``. Empty under colocate today;
        kept symmetric with UpdateWeightFromDistributed so the actor can drain unconditionally.
        """
        out, self.update_weight_metrics = self.update_weight_metrics, {}
        return out

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        version++, flush caches, process buckets. Progress on rank 0.
        """
        self.weight_version += 1

        rank = dist.get_rank()
        if rank == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        # vLLM #39212: enter weight-update mode on each slot leader.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, long_lived_tensors = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free GPU tensors so the caching allocator can reuse the blocks,
            # then release CUDA IPC cache entries whose consumers (vLLM engines)
            # have already closed their IPC handles.
            del long_lived_tensors, hf_named_tensors
            torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        torch.cuda.ipc_collect()

        # vLLM #39212: exit weight-update mode.
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            ray.get(self._ipc_engine.finish_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        # int4/fp4 post_process
        if rank == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_hf_params(self, hf_named_tensors) -> tuple[list[ObjectRef], Any]:
        all_refs = []

        refs_colocated, long_lived_tensors = _send_to_colocated_engine(
            hf_named_tensors,
            ipc_engine=self._ipc_engine,
            ipc_gather_src=self._ipc_gather_src,
            ipc_gather_group=self._ipc_gather_group,
            weight_version=self.weight_version,
        )
        all_refs.extend(refs_colocated)

        if self.use_distribute and self._is_distributed_src_rank:
            refs_distributed = update_weights_from_distributed(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
                packed=False,
            )
            if refs_distributed:
                all_refs.extend(refs_distributed)

        return all_refs, long_lived_tensors


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    ipc_engine,
    ipc_gather_src,
    ipc_gather_group,
    weight_version,
) -> tuple[list[ObjectRef], Any]:
    # Placeholder ranks (GPU slots reserved but no engine) have no gather group.
    # all_gather_object is only collective among group members, so we skip entirely.
    if ipc_gather_group is None:
        return [], None

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
        ref = ipc_engine.update_weights_from_tensor.remote(**local_info, weight_version=str(weight_version))
        return [ref], weight_refs

    local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
    payload = _serialize_ipc_update_info(local_info)

    gathered_payloads = [None] * slot_size if dist.get_rank() == ipc_gather_src else None
    dist.gather_object(payload, object_gather_list=gathered_payloads, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        merged = _merge_ipc_update_infos(slot_infos)
        refs.append(ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version)))

    return refs, weight_refs
