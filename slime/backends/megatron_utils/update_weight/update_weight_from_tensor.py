"""
Colocated vLLM weight sync (trainer + worker)
=============================================

Trainer: ``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC (Ray).

Worker: ``vLLMColocateWorkerExtension`` — passed to ``vllm serve`` via
``--worker-extension-cls``; patches IPC receive before handle deserialisation.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/

The flow for colocated engines:
1. Megatron params → HF conversion (via HfWeightIteratorBase)
2. All trainer ranks call ``IPCWeightTransferEngine.trainer_send_weights()``
   with ``send_mode="ray"`` pointing at the colocated vLLM engine actor on the
   same GPU slot.  Each rank creates a CUDA IPC handle for its GPU; the engine
   collects all handles via ``_all_gather_and_merge_handles`` so every vLLM
   worker can pick the handle belonging to its physical GPU UUID.

For non-colocated overflow engines the existing NCCL distributed broadcast
(``update_weights_from_distributed``) is used unchanged.
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


def _apply_monkey_patch_torch_reductions() -> None:
    """CUDA IPC tensor rebuild uses GPU UUIDs; patch torch reductions before IPC."""
    from slime.backends.megatron_utils.sglang import monkey_patch_torch_reductions

    monkey_patch_torch_reductions()


class UpdateWeightFromTensor:
    """
    Update colocated vLLM engines from tensors via CUDA IPC (Ray send mode).

    Colocated path:
        Megatron weights → HF conversion → CUDA IPC to vLLM engine actors via
        ``IPCWeightTransferEngine.trainer_send_weights(send_mode="ray")``.
        Each trainer rank sends to the colocated engine on its GPU slot.

    Distributed overflow path (optional):
        Falls back to NCCL distributed broadcast via
        ``update_weights_from_distributed`` for engines whose GPUs lie outside
        the actor GPU range.

    Engine lifecycle per ``update_weights`` call::

        colocated:   release_memory_occupation(level=0) (rank 0)
        distributed: pause_generation / flush_cache      (rank 0)
        init_weight_transfer_engine                      (rank 0, colocated, first call only)
        start_weight_update                              (each rank, its colocated engine)
        [for each HF chunk]
          trainer_send_weights                           (rank with _ipc_engine)
          update_weights_from_distributed                (src rank, distributed)
          barrier                                        (all ranks)
        finish_weight_update                             (each rank, its colocated engine)
        colocated:   resume_memory_occupation(tags=["weights", "kv_cache"]) (rank 0)
        distributed: continue_generation                           (rank 0)
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
        # vLLM 0.21 IPC (mode=ray): one Ray actor per GPU slot; this rank's engine.
        self._ipc_engine: ActorHandle | None = None
        self._distributed_engines: list[ActorHandle] = []
        self._model_update_groups = None
        self._is_distributed_src_rank: bool = False
        self._group_name = "slime"
        # IPC weight transfer engine is initialized once per set of colocated
        # engines (not per update call).
        self._ipc_initialized: bool = False
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
        Split engines into colocated (IPC) vs distributed (NCCL) buckets.

        Colocated engines are those whose GPU range fits entirely within the
        trainer actor GPU range.  The remainder are treated as distributed and
        receive weights via NCCL broadcast.
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

        # Map this trainer rank to the colocated vLLM engine on the same GPU slot.
        # vLLM 0.21 ``trainer_send_weights(mode="ray")`` expects a single ``llm_handle``,
        # not a list (list fan-out is only in newer vLLM with ``send_mode="ray"``).
        self._ipc_engine = None
        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]
        for i, engine in enumerate(self._colocated_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            if start <= dist.get_rank() < end:
                self._ipc_engine = engine

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

        Colocated engines receive weights via CUDA IPC (per-rank engine RPC).
        Distributed overflow engines receive weights via NCCL broadcast (source rank only).
        """
        self.weight_version += 1
        rank = dist.get_rank()
        all_engines = self._colocated_engines + self._distributed_engines

        # ── 1. Pause generation and flush KV cache (rank 0 only) ────────────
        if rank == 0:
            if self._colocated_engines:
                ray.get([engine.release_memory_occupation.remote(level=0) for engine in self._colocated_engines])
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
            ray.get(
                [engine.init_weight_transfer_engine.remote({"init_info": {}}) for engine in self._colocated_engines]
            )
            self._ipc_initialized = True
        dist.barrier(group=get_gloo_group())

        # ── 3. Enter weight-update mode (vLLM #39212: /start_weight_update) ───
        if self._ipc_engine is not None:
            ray.get(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
        dist.barrier(group=get_gloo_group())

        from vllm.distributed.weight_transfer.ipc_engine import (  # noqa: PLC0415
            IPCTrainerSendWeightsArgs,
            IPCWeightTransferEngine,
        )

        if self._colocated_engines:
            _apply_monkey_patch_torch_reductions()

        # ── 4. Iterate HF weight chunks and send ─────────────────────────────
        megatron_local_weights = self.weights_getter()
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            if self._ipc_engine is not None:
                trainer_args = IPCTrainerSendWeightsArgs(
                    mode="ray",
                    llm_handle=self._ipc_engine,
                )
                IPCWeightTransferEngine.trainer_send_weights(
                    iterator=iter(hf_named_tensors),
                    trainer_args=trainer_args,
                )

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

            dist.barrier(group=get_gloo_group())

        # ── 5. Signal colocated engines to exit weight-update mode ───────────
        if self._ipc_engine is not None:
            ray.get(self._ipc_engine.finish_weight_update.remote())
        dist.barrier(group=get_gloo_group())

        # ── 6. Post-process quantization (if needed) and resume ───────────────
        if rank == 0:
            if self.quantization_config and self.quantization_config.get("quant_method") in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=all_engines,
                )
            if self._colocated_engines:
                ray.get(
                    [
                        engine.resume_memory_occupation.remote(tags=["weights", "kv_cache"])
                        for engine in self._colocated_engines
                    ]
                )
            if self._distributed_engines:
                ray.get([engine.continue_generation.remote() for engine in self._distributed_engines])
        dist.barrier(group=get_gloo_group())


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded by ``--worker-extension-cls`` in colocate mode)
# ---------------------------------------------------------------------------


class _VLLMHijack:
    """Monkey-patch vLLM IPC receive so CUDA IPC handles deserialize on the correct GPU."""

    @staticmethod
    def hijack() -> None:
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

        if getattr(IPCWeightTransferEngine, "_slime_receive_patched", False):
            return

        _orig = IPCWeightTransferEngine.receive_weights

        def _slime_receive_weights(self, update_info, load_weights, _orig=_orig):
            _apply_monkey_patch_torch_reductions()
            _orig(self, update_info, load_weights)

        IPCWeightTransferEngine.receive_weights = _slime_receive_weights
        IPCWeightTransferEngine._slime_receive_patched = True  # type: ignore[attr-defined]


class vLLMColocateWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for colocated IPC weight sync."""

    def __new__(cls, **kwargs):
        _VLLMHijack.hijack()
        return super().__new__(cls)
