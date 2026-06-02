"""
Colocated vLLM weight sync (trainer + worker)
=============================================

Trainer: ``UpdateWeightFromTensor`` — Megatron → HF chunks → CUDA IPC (Ray).

Worker: ``vLLMColocateWorkerExtension`` — passed to ``vllm serve`` via
``--worker-extension-cls``; patches IPC receive before handle deserialisation.

https://docs.vllm.ai/en/stable/examples/rl/rlhf_ipc/
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable, Iterable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
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
    """Update colocated vLLM engines via CUDA IPC, with NCCL fallback for
    non-colocated overflow engines. See the module docstring for the
    high-level design (why we dispatch via ``update_weights_from_tensor``
    directly instead of vLLM's ``trainer_send_weights``).

    Engine lifecycle per ``update_weights`` call::

        colocated:   release_memory_occupation(level=0) (rank 0)
        distributed: pause_generation / flush_cache      (rank 0)
        init_weight_transfer_engine                      (rank 0, colocated, first call only)
        start_weight_update                              (coordinator rank per engine only)
        [for each HF chunk]
          update_weights_from_tensor                     (per-rank or coordinator-merged)
          update_weights_from_distributed                (src rank, distributed)
          barrier                                        (all ranks)
        finish_weight_update                             (coordinator rank per engine only)
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
        # First trainer rank in each engine GPU range issues start/finish (TP ranks share one engine).
        self._ipc_engine_coordinator: bool = False
        self._ipc_engine_slot_start: int | None = None
        self._ipc_engine_slot_end: int | None = None
        self._distributed_engines: list[ActorHandle] = []
        self._model_update_groups = None
        self._is_distributed_src_rank: bool = False
        self._group_name = "vime"
        # IPC weight transfer engine is initialized once per set of colocated
        # engines (not per update call).
        self._ipc_initialized: bool = False
        # Per-engine-slot process group for IPC payload gather (created in connect_rollout_engines).
        self._ipc_slot_group = None
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
        self._ipc_engine_coordinator = False
        self._ipc_engine_slot_start = None
        self._ipc_engine_slot_end = None
        # Build per-slot process groups so IPC payload gather covers all ranks in the
        # engine's GPU slot — Megatron TP group does NOT cover the slot when Megatron
        # TP != rollout-num-gpus-per-engine (e.g. Megatron TP=1 + rollout TP=2 in
        # parallel-check). Every trainer rank must enter dist.new_group collectively.
        self._ipc_slot_group = None
        rank_for_slot = dist.get_rank()
        colocate_gpu_offsets = engine_gpu_offsets[:colocate_engine_nums]
        colocate_gpu_counts = engine_gpu_counts[:colocate_engine_nums]
        # First pass: create per-slot process groups collectively (every rank must call new_group).
        for i in range(colocate_engine_nums):
            slot_start = colocate_gpu_offsets[i]
            slot_end = slot_start + colocate_gpu_counts[i]
            slot_ranks = list(range(slot_start, slot_end))
            grp = dist.new_group(ranks=slot_ranks, backend="gloo")
            if slot_start <= rank_for_slot < slot_end:
                self._ipc_slot_group = grp
        # Second pass: bind this rank to its engine + decide coordinator.
        for i, engine in enumerate(self._colocated_engines):
            start = colocate_gpu_offsets[i]
            end = start + colocate_gpu_counts[i]
            rank = dist.get_rank()
            if start <= rank < end:
                self._ipc_engine = engine
                self._ipc_engine_slot_start = start
                self._ipc_engine_slot_end = end
                # Slot leader (lowest trainer rank in the engine GPU range) issues start/finish.
                if rank == start:
                    self._ipc_engine_coordinator = True

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
        if self._ipc_engine_coordinator:
            ray.get(self._ipc_engine.start_weight_update.remote(is_checkpoint_format=True))
        dist.barrier(group=get_gloo_group())

        # ── 4. Iterate HF weight chunks and send ─────────────────────────────
        megatron_local_weights = self.weights_getter()
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            if self._ipc_engine is not None:
                self._send_hf_chunk_via_ipc(hf_named_tensors)

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
        # State-machine bookend only; ``_weight_version`` is recorded inside
        # ``update_weights_from_tensor`` (step 4) when the data RPC succeeds —
        # matches vime's single-RPC version-with-data semantics.
        if self._ipc_engine_coordinator:
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

    def _send_hf_chunk_via_ipc(self, hf_named_tensors: Sequence[tuple[str, torch.Tensor]]) -> None:
        """Send one HF chunk to the colocated vLLM engine via CUDA IPC (Ray → HTTP).

        ``slot_size == 1``: this rank ships its IPC payload directly.
        ``slot_size > 1`` (vLLM TP): every rank in the slot builds its handle;
        the coordinator gathers them, merges UUIDs, and issues one RPC for the
        slot. Both paths dispatch the same RPC — ``update_weights_from_tensor`` —
        with ``weight_version`` alongside the data (see module docstring).
        """
        assert self._ipc_engine is not None
        assert self._ipc_engine_slot_start is not None
        assert self._ipc_engine_slot_end is not None

        slot_size = self._ipc_engine_slot_end - self._ipc_engine_slot_start
        if slot_size <= 1:
            local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
            ray.get(
                self._ipc_engine.update_weights_from_tensor.remote(
                    **local_info,
                    weight_version=str(self.weight_version),
                )
            )
            # Keep CUDA IPC producer tensors alive until ray.get() returns
            # (the HTTP weight update completes inside the engine actor); then release.
            del weight_refs
            return

        local_info, weight_refs = _build_ipc_update_info_from_named_tensors(hf_named_tensors)
        payload = _serialize_ipc_update_info(local_info)

        slot_group = self._ipc_slot_group
        slot_ranks = list(range(self._ipc_engine_slot_start, self._ipc_engine_slot_end))

        # Gather IPC payloads over the engine slot ranks (NOT Megatron TP group — see
        # connect_rollout_engines for why). all_gather_object is monkey-patched for
        # ReloadableProcessGroup; gather_object is not (fails after Megatron reload).
        gathered_payloads: list[str | None] = [None] * slot_size
        dist.all_gather_object(gathered_payloads, payload, group=slot_group)
        if self._ipc_engine_coordinator:
            if any(p is None for p in gathered_payloads):
                raise RuntimeError(f"Missing IPC payloads on slot ranks {slot_ranks}; " f"got {gathered_payloads!r}")
            slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
            merged = _merge_ipc_update_infos(slot_infos)
            ray.get(
                self._ipc_engine.update_weights_from_tensor.remote(
                    **merged,
                    weight_version=str(self.weight_version),
                )
            )

        dist.barrier(group=slot_group)
        # Keep CUDA IPC producer tensors alive until every TP worker has opened
        # the handles and the coordinator's HTTP update has completed.
        del weight_refs


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded by ``--worker-extension-cls`` in colocate mode)
# ---------------------------------------------------------------------------


class _VLLMHijack:
    """Monkey-patch vLLM IPC receive so CUDA IPC handles deserialize on the correct GPU."""

    @staticmethod
    def hijack() -> None:
        from vllm.distributed.weight_transfer.ipc_engine import IPCWeightTransferEngine

        if getattr(IPCWeightTransferEngine, "_vime_receive_patched", False):
            return

        _orig = IPCWeightTransferEngine.receive_weights

        def _vime_receive_weights(self, update_info, load_weights, _orig=_orig):
            _orig(self, update_info, load_weights)

        IPCWeightTransferEngine.receive_weights = _vime_receive_weights
        IPCWeightTransferEngine._vime_receive_patched = True  # type: ignore[attr-defined]


class vLLMColocateWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for colocated IPC weight sync."""

    def __new__(cls, **kwargs):
        _VLLMHijack.hijack()
        return super().__new__(cls)
