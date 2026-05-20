"""
UpdateVLLMWeightFromTensor
==========================

Update vLLM rollout engines using CUDA IPC (Ray mode) when colocated on the
same GPU(s) as the trainer, following the vLLM RLHF IPC approach:
https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/

The flow for colocated engines:
1. Megatron params → HF conversion (via HfWeightIteratorBase)
2. All trainer ranks call ``IPCWeightTransferEngine.trainer_send_weights()``
   with ``send_mode="ray"`` pointing at the list of colocated vLLM engine
   actors.  Each rank creates a CUDA IPC handle for its GPU; the engine
   collects all handles via ``_all_gather_and_merge_handles`` so every vLLM
   worker can pick the handle belonging to its physical GPU UUID.

For non-colocated overflow engines the existing NCCL distributed broadcast
(``update_weights_from_distributed``) is used unchanged.

API is intentionally compatible with ``MegatronTrainRayActor``: same
``__init__``, ``connect_rollout_engines``, and ``update_weights`` signatures
as the sglang-IPC ``UpdateWeightFromTensor``.
"""

from __future__ import annotations

import logging
import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray.actor import ActorHandle

from slime.utils.distributed_utils import get_gloo_group

from .hf_weight_iterator_base import HfWeightIteratorBase
from .update_weight_from_distributed import (
    connect_rollout_engines_from_distributed,
    disconnect_rollout_engines_from_distributed,
    post_process_weights,
    update_weights_from_distributed,
)

logger = logging.getLogger(__name__)


class UpdateVLLMWeightFromTensor:
    """
    Update colocated vLLM engines from tensors via CUDA IPC (Ray send mode).

    Colocated path:
        Megatron weights → HF conversion → CUDA IPC to vLLM engine actors via
        ``IPCWeightTransferEngine.trainer_send_weights(send_mode="ray")``.
        All trainer ranks participate in the IPC handle all-gather; only rank 0
        actually delivers the merged payload to the vLLM actors.

    Distributed overflow path (optional):
        Falls back to NCCL distributed broadcast via
        ``update_weights_from_distributed`` for engines whose GPUs lie outside
        the actor GPU range.

    Engine lifecycle per ``update_weights`` call::

        colocated:   sleep(level=0)             (rank 0)
        distributed: pause_generation / flush_cache (rank 0)
        init_weight_transfer_engine              (rank 0, colocated, first call only)
        start_weight_update                      (rank 0, colocated)
        [for each HF chunk]
          trainer_send_weights                   (all ranks, colocated)
          update_weights_from_distributed        (src rank, distributed)
        finish_weight_update                     (rank 0, colocated)
        colocated:   wake_up(tags=["scheduling"]) (rank 0)
        distributed: continue_generation          (rank 0)
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
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0

        self._hf_weight_iterator = HfWeightIteratorBase.create(
            args=args,
            model=model,
            model_name=model_name,
            quantization_config=quantization_config,
        )

        # Populated by connect_rollout_engines
        self._colocated_engines: list[ActorHandle] = []
        self._distributed_engines: list[ActorHandle] = []
        self._model_update_groups = None
        self._is_distributed_src_rank: bool = False
        self._group_name = "slime"
        # IPC weight transfer engine is initialized once per set of colocated
        # engines (not per update call).
        self._ipc_initialized: bool = False

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
        Split engines into colocated (IPC) vs distributed (NCCL) buckets.

        Colocated engines are those whose GPU range fits entirely within the
        trainer actor GPU range.  The remainder are treated as distributed and
        receive weights via NCCL broadcast.

        The NCCL bridge for distributed engines is (re-)created whenever the
        engine set changes, matching the behaviour of
        ``UpdateWeightFromTensor.connect_rollout_engines``.
        """
        self.rollout_engine_lock = rollout_engine_lock

        if engine_gpu_counts is None:
            engine_gpu_counts = [self.args.rollout_num_gpus_per_engine] * len(rollout_engines)
        if engine_gpu_offsets is None:
            engine_gpu_offsets = []
            offset = 0
            for c in engine_gpu_counts:
                engine_gpu_offsets.append(offset)
                offset += c

        total_actor_gpus = self.args.actor_num_nodes * self.args.actor_num_gpus_per_node
        colocate_engine_nums = 0
        for gpu_offset, gpu_count in zip(engine_gpu_offsets, engine_gpu_counts, strict=True):
            if gpu_offset + gpu_count > total_actor_gpus:
                break
            colocate_engine_nums += 1

        self._colocated_engines = list(rollout_engines[:colocate_engine_nums])
        self._distributed_engines = list(rollout_engines[colocate_engine_nums:])

        # Set up NCCL bridge for any overflow (non-colocated) engines.
        if self._distributed_engines:
            distributed_gpu_counts = engine_gpu_counts[colocate_engine_nums:]
            self._is_distributed_src_rank = (
                mpu.get_data_parallel_rank(with_context_parallel=True) == 0
                and mpu.get_tensor_model_parallel_rank() == 0
                and mpu.get_pipeline_model_parallel_rank() == 0
            )
            if self._is_distributed_src_rank:
                if self._model_update_groups is not None:
                    disconnect_rollout_engines_from_distributed(
                        self.args,
                        self._group_name,
                        self._model_update_groups,
                        self._distributed_engines,
                    )
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    self._distributed_engines,
                    engine_gpu_counts=distributed_gpu_counts,
                )

    # ------------------------------------------------------------------
    # weight update
    # ------------------------------------------------------------------

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Transfer updated Megatron weights to all rollout engines.

        Colocated engines receive weights via CUDA IPC (all trainer ranks
        participate).  Distributed overflow engines receive weights via NCCL
        broadcast (source rank only).
        """
        self.weight_version += 1
        rank = dist.get_rank()
        all_engines = self._colocated_engines + self._distributed_engines

        # ── 1. Pause generation and flush KV cache (rank 0 only) ────────────
        # vLLM colocated engines: sleep(level=0) suspends generation and frees
        # KV cache in one call, matching the rlhf_ipc.py reference pattern.
        # Distributed (non-vLLM) engines keep the sglang-style pause+flush API.
        if rank == 0:
            if self._colocated_engines:
                ray.get([engine.sleep.remote(level=0) for engine in self._colocated_engines])
            if self._distributed_engines:
                ray.get([engine.pause_generation.remote() for engine in self._distributed_engines])
                ray.get([engine.flush_cache.remote() for engine in self._distributed_engines])
            if self.quantization_config and self.quantization_config.get("quant_method") in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=all_engines,
                )
        dist.barrier(group=get_gloo_group())

        # ── 2. One-time IPC weight transfer engine init (rank 0 only) ───────
        if rank == 0 and self._colocated_engines and not self._ipc_initialized:
            for engine in self._colocated_engines:
                ray.get(engine.init_weight_transfer_engine.remote(dict(init_info=dict())))
            self._ipc_initialized = True
        dist.barrier(group=get_gloo_group())

        # ── 3. Signal colocated vLLM engines to enter weight-update mode ─────
        if rank == 0 and self._colocated_engines:
            ray.get(
                [engine.start_weight_update.remote(is_checkpoint_format=True) for engine in self._colocated_engines]
            )
        dist.barrier(group=get_gloo_group())

        # Required so vLLM can deserialize CUDA IPC handle payloads.
        os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

        from vllm.distributed.weight_transfer.ipc_engine import (  # noqa: PLC0415
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        # ── 4. Iterate HF weight chunks and send ─────────────────────────────
        megatron_local_weights = self.weights_getter()
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            # Colocated path: all ranks must call trainer_send_weights so that
            # _all_gather_and_merge_handles can collect every GPU's IPC handle.
            # Rank 0 then delivers the merged dict to all colocated engine actors.
            if self._colocated_engines:
                trainer_args = IPCTrainerSendWeightsArgs(
                    send_mode="ray",
                    llm_handle=self._colocated_engines,
                    packed=False,
                )
                IPCWeightTransferEngine.trainer_send_weights(
                    iterator=iter(hf_named_tensors),
                    trainer_args=trainer_args,
                )

            # Distributed overflow path (only the designated src rank).
            if self._distributed_engines and self._is_distributed_src_rank:
                refs = update_weights_from_distributed(
                    self._group_name,
                    self._model_update_groups,
                    self.weight_version,
                    self._distributed_engines,
                    hf_named_tensors,
                    packed=False,
                )
                if refs:
                    ray.get(refs)

        # ── 5. Signal colocated engines to exit weight-update mode ───────────
        if rank == 0 and self._colocated_engines:
            ray.get([engine.finish_weight_update.remote() for engine in self._colocated_engines])
        dist.barrier(group=get_gloo_group())

        # ── 6. Post-process quantization (if needed) and resume ───────────────
        # vLLM colocated engines: wake_up(tags=["scheduling"]) matches the
        # rlhf_ipc.py reference pattern.
        # Distributed engines use the sglang-style continue_generation.
        if rank == 0:
            if self.quantization_config and self.quantization_config.get("quant_method") in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=all_engines,
                )
            if self._colocated_engines:
                ray.get([engine.wake_up.remote(tags=["scheduling"]) for engine in self._colocated_engines])
            if self._distributed_engines:
                ray.get([engine.continue_generation.remote() for engine in self._distributed_engines])
        dist.barrier(group=get_gloo_group())
