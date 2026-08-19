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

import logging
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

from .coordinator import WeightUpdateCoordinator
from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    update_weights_from_distributed,
)

_MAX_COLOCATED_UPDATES_INFLIGHT = 4
logger = logging.getLogger(__name__)


def _build_packed_ipc_update_info(
    named_tensors: Iterable[tuple[str, torch.Tensor]],
) -> tuple[dict[str, Any], torch.Tensor]:
    from torch.multiprocessing.reductions import reduce_tensor

    names, dtype_names, shapes, tensor_sizes, byte_tensors = [], [], [], [], []
    for name, tensor in named_tensors:
        names.append(name)
        dtype_names.append(str(tensor.dtype).split(".")[-1])
        shapes.append(list(tensor.shape))
        byte_tensor = tensor.detach().contiguous().view(torch.uint8).flatten()
        tensor_sizes.append(byte_tensor.numel())
        byte_tensors.append(byte_tensor)
    if not byte_tensors:
        raise ValueError("cannot build an empty packed IPC update")

    packed_tensor = torch.cat(byte_tensors)
    _, ipc_args = reduce_tensor(packed_tensor)
    gpu_uuid = str(torch.cuda.get_device_properties(torch.cuda.current_device()).uuid)
    return (
        {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": shapes,
            "tensor_sizes": tensor_sizes,
            "ipc_handles": {gpu_uuid: ipc_args},
        },
        packed_tensor,
    )


def _serialize_ipc_update_info(info: dict[str, Any]) -> str:
    """Pickle IPC handles for cross-rank gather (Gloo ``all_gather_object`` cannot carry them)."""
    import base64

    import cloudpickle

    return base64.b64encode(cloudpickle.dumps(info)).decode("ascii")


def _deserialize_ipc_update_info(payload: str) -> dict[str, Any]:
    import base64

    import cloudpickle

    return cloudpickle.loads(base64.b64decode(payload.encode("ascii")))


def _merge_ipc_update_infos(infos: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Merge the per-rank handles for one packed IPC update."""
    if not infos:
        raise ValueError("no IPC update_info payloads to merge")

    metadata_keys = ("names", "dtype_names", "shapes", "tensor_sizes")
    base = infos[0]
    if "tensor_sizes" not in base or any(
        "tensor_sizes" not in info or any(info[key] != base[key] for key in metadata_keys) for info in infos[1:]
    ):
        raise ValueError("packed IPC metadata must match across all ranks in a slot")
    handles = {}
    for info in infos:
        handles.update(info["ipc_handles"])
    return {**base, "ipc_handles": handles}


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
        self._all_rollout_engines = list(rollout_engines)
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
        else:
            self.distributed_rollout_engines = []

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
        """Run one coordinated full-weight update and commit on success."""
        committed_version = self.weight_version
        rollout_engines = getattr(self, "_all_rollout_engines", self.rollout_engines)
        coordinator = WeightUpdateCoordinator(rollout_engines, self.quantization_config)
        megatron_local_weights = None

        def transfer_target(candidate_version: int) -> None:
            nonlocal megatron_local_weights
            self.weight_version = candidate_version
            megatron_local_weights = self._run_weight_update_session()

        draft_transfer: Callable[[int], None] | None = None
        if (
            not self.use_distribute
            and self.args.enable_mtp_training
            and (self.args.vllm_speculative_config or {}).get("method") == "mtp"
        ):

            def transfer_draft(candidate_version: int) -> None:
                assert candidate_version == self.weight_version
                assert megatron_local_weights is not None
                self._run_weight_update_session(megatron_local_weights, draft=True)

            draft_transfer = transfer_draft

        try:
            self.weight_version = coordinator.run(
                current_version=committed_version,
                transfer_target=transfer_target,
                transfer_draft=draft_transfer,
            )
        except BaseException:
            self.weight_version = committed_version
            raise

    def _run_weight_update_session(
        self,
        megatron_local_weights: Mapping[str, torch.Tensor] | None = None,
        *,
        draft: bool = False,
    ) -> Mapping[str, torch.Tensor]:
        self._start_weight_update_session(draft=draft)
        try:
            # Preserve the colocated memory order: vLLM first enters layerwise
            # reload mode, then Megatron materializes the source weights.
            if megatron_local_weights is None:
                megatron_local_weights = self.weights_getter()
            self._send_weight_chunks(megatron_local_weights)
            dist.barrier(group=get_gloo_group())
            # Every engine has returned, so the consumers have released the
            # final chunk's IPC handles.
            torch.cuda.ipc_collect()
        except BaseException:
            try:
                self._finish_weight_update_session()
            except BaseException:
                logger.warning("Failed to finalize vLLM session after weight transfer error", exc_info=True)
            raise
        else:
            self._finish_weight_update_session()
        return megatron_local_weights

    def _start_weight_update_session(self, *, draft: bool = False) -> None:
        rank = dist.get_rank()
        refs = []
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            if draft:
                refs.append(self._ipc_engine.start_draft_weight_update.remote())
            else:
                refs.append(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
        if self.use_distribute and self._is_distributed_src_rank:
            if draft:
                refs.extend(engine.start_draft_weight_update.remote() for engine in self.distributed_rollout_engines)
            else:
                refs.extend(
                    engine.start_weight_update.remote(is_checkpoint_format=True)
                    for engine in self.distributed_rollout_engines
                )
        if refs:
            ray.get(refs)
        dist.barrier(group=get_gloo_group())

    def _finish_weight_update_session(self) -> None:
        rank = dist.get_rank()
        refs = []
        if self._ipc_engine is not None and rank == self._ipc_gather_src:
            refs.append(self._ipc_engine.finish_weight_update.remote())
        if self.use_distribute and self._is_distributed_src_rank:
            refs.extend(engine.finish_weight_update.remote() for engine in self.distributed_rollout_engines)
        try:
            if refs:
                ray.get(refs)
        finally:
            dist.barrier(group=get_gloo_group())

    def _send_weight_chunks(self, megatron_local_weights) -> None:
        max_inflight = 1 if self.use_distribute else _MAX_COLOCATED_UPDATES_INFLIGHT
        pending = []
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs, weight_refs = self._send_hf_params(hf_named_tensors)
            pending.append((refs, weight_refs))
            if len(pending) >= max_inflight:
                self._drain_ipc_updates(pending)
        self._drain_ipc_updates(pending)

    def _drain_ipc_updates(self, pending) -> None:
        if not pending:
            return
        ray.get([ref for refs, _ in pending for ref in refs])
        if self._ipc_gather_group is not None:
            dist.barrier(group=self._ipc_gather_group)
        pending.clear()
        torch.cuda.ipc_collect()

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
                self._model_update_groups,
                self.weight_version,
                self.distributed_rollout_engines,
                hf_named_tensors,
                record_weight_version=False,
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

    local_info, weight_ref = _build_packed_ipc_update_info(hf_named_tensors)

    slot_size = dist.get_world_size(ipc_gather_group)
    if slot_size <= 1:
        ref = ipc_engine.update_weights_from_tensor.remote(
            **local_info,
            weight_version=str(weight_version),
            record_weight_version=False,
        )
        return [ref], weight_ref

    payload = _serialize_ipc_update_info(local_info)

    gathered_payloads = [None] * slot_size if dist.get_rank() == ipc_gather_src else None
    dist.gather_object(payload, object_gather_list=gathered_payloads, dst=ipc_gather_src, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        merged = _merge_ipc_update_infos(slot_infos)
        refs.append(
            ipc_engine.update_weights_from_tensor.remote(
                **merged,
                weight_version=str(weight_version),
                record_weight_version=False,
            )
        )

    return refs, weight_ref
