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
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle

from vime.utils.common import is_npu
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
        rebuild_func, ipc_args = reduce_tensor(weight)
        ipc_handles.append({gpu_uuid: (rebuild_func, ipc_args)})

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

    # all_gather_object is monkey-patched for ReloadableProcessGroup; gather_object
    # is not (it fails after a Megatron reload).
    gathered_payloads = [None] * slot_size
    dist.all_gather_object(gathered_payloads, payload, group=ipc_gather_group)

    refs = []
    if dist.get_rank() == ipc_gather_src:
        if any(p is None for p in gathered_payloads):
            raise RuntimeError(f"Missing IPC payloads in slot {ipc_gather_src}; got {gathered_payloads!r}")
        slot_infos = [_deserialize_ipc_update_info(p) for p in gathered_payloads]
        merged = _merge_ipc_update_infos(slot_infos)
        refs.append(ipc_engine.update_weights_from_tensor.remote(**merged, weight_version=str(weight_version)))

    return refs, weight_refs


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded by ``--worker-extension-cls``)
# ---------------------------------------------------------------------------


class _VLLMHijack:
    """Monkey-patch vLLM IPC receive so CUDA IPC handles deserialize on the correct GPU.

    On NPU only:
    - Patches NPUWorker.load_model and NPUWorker.start_weight_update to fix
      MoE weight_loader missing on EP (a vLLM bug where w13_weight/w2_weight
      params lack weight_loader attr when EP is enabled).
    - Patches ApplyRotaryEmb.__init__ to skip flash_attn import
      (mindspeed/megatron backends introduce flash_attn as a dummy module,
      but vllm_ascend does not use it).
    """

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

    @staticmethod
    def _patch_npu_worker() -> None:
        from vllm_ascend.worker.worker import NPUWorker

        if getattr(NPUWorker, "_npu_worker_patched", False):
            return

        _VLLMHijack._patch_one_worker(NPUWorker)
        NPUWorker._npu_worker_patched = True

    @staticmethod
    def _patch_one_worker(worker_cls: type) -> None:
        import inspect

        _orig_load_model = worker_cls.load_model
        _orig_start_weight_update = worker_cls.start_weight_update
        has_dummy_kw = "load_dummy_weights" in inspect.signature(_orig_load_model).parameters

        if has_dummy_kw:

            def _patched_load_model(self, *, load_dummy_weights: bool = False, _orig=_orig_load_model) -> None:
                _orig(self, load_dummy_weights=load_dummy_weights)
                _VLLMHijack.patch_moe_weight_loader(self.model_runner.model)

        else:

            def _patched_load_model(self, _orig=_orig_load_model) -> None:
                _orig(self)
                _VLLMHijack.patch_moe_weight_loader(self.model_runner.model)

        def _patched_start_weight_update(
            self, is_checkpoint_format: bool = True, _orig=_orig_start_weight_update
        ) -> None:
            _VLLMHijack.patch_moe_weight_loader(self.model_runner.model)
            _orig(self, is_checkpoint_format=is_checkpoint_format)

        worker_cls.load_model = _patched_load_model  # type: ignore[attr-defined]
        worker_cls.start_weight_update = _patched_start_weight_update  # type: ignore[attr-defined]

    @staticmethod
    def patch_moe_weight_loader(model: torch.nn.Module) -> None:
        inner_model = getattr(model, "model", None) or getattr(model, "language_model", None)
        if inner_model is None:
            return
        if not hasattr(inner_model, "layers"):
            inner_model = getattr(inner_model, "model", None)
            if inner_model is None or not hasattr(inner_model, "layers"):
                return

        for layer in inner_model.layers:
            mlp = getattr(layer, "mlp", None) or getattr(layer, "block_sparse_moe", None)
            if mlp is None:
                continue
            experts = getattr(mlp, "experts", None)
            if experts is None or not hasattr(experts, "weight_loader"):
                continue
            for name, param in mlp.named_parameters():
                if "w13_weight" in name or "w2_weight" in name:
                    if not hasattr(param, "weight_loader"):
                        param.weight_loader = experts.weight_loader  # type: ignore[attr-defined]

    @staticmethod
    def _patch_npu_rotary_emb() -> None:
        from vllm.model_executor.layers.rotary_embedding.common import ApplyRotaryEmb

        if getattr(ApplyRotaryEmb, "_npu_rotary_patched", False):
            return

        def _npu_rotary_emb_init(
            self,
            enforce_enable: bool = False,
            is_neox_style: bool = True,
            enable_fp32_compute: bool = False,
        ) -> None:
            super(ApplyRotaryEmb, self).__init__(enforce_enable=enforce_enable)
            self.is_neox_style = is_neox_style
            self.enable_fp32_compute = enable_fp32_compute
            self.apply_rotary_emb_flash_attn = None

        ApplyRotaryEmb.__init__ = _npu_rotary_emb_init  # type: ignore[attr-defined]
        ApplyRotaryEmb._npu_rotary_patched = True


class vLLMColocateWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for colocated IPC weight sync."""

    def __new__(cls, **kwargs):
        _VLLMHijack.hijack()
        return super().__new__(cls)

    # ── Three-phase weight update protocol ────────────────────────────────────
    # Mirrors SkyRL's NewInferenceWorkerWrap. Callable via /collective_rpc from
    # VLLMEngine.update_weights_chunk / update_weights_chunk on the trainer side.

    def update_weights_chunk(self, update_info: dict) -> None:
        """Receive and load a single chunk of weights via CUDA IPC.

        Accepts the ``update_info`` dict produced by
        ``VLLMEngine.update_weights`` / ``update_weights``, which
        carries ``ipc_handles_pickled`` (cloudpickle + base64 serialised CUDA
        IPC handles assembled by the trainer's
        ``IPCWeightTransferEngine.trainer_send_weights``).

        Deserialises IPC handles inline (the same pattern as SkyRL's
        NewInferenceWorkerWrap) and reconstructs each weight tensor before
        loading into the model — no dependency on
        ``weight_transfer_engine.receive_weights``.

        Args:
            update_info: Dict with keys:
                - names: list[str]
                - dtype_names: list[str]
                - shapes: list[list[int]]
                - ipc_handles_pickled: base64(cloudpickle({gpu_uuid: (func, args)}))
        """
        if not getattr(self, "_weight_update_active", False):
            raise RuntimeError("start_weight_update must be called before update_weights.")

        import base64

        import cloudpickle

        # Deserialise cloudpickle+b64 encoded IPC handles back to raw callables.
        inner = dict(update_info)
        if "ipc_handles_pickled" in inner:
            inner["ipc_handles"] = cloudpickle.loads(base64.b64decode(inner.pop("ipc_handles_pickled")))

        names: list[str] = inner["names"]
        shapes: list[list[int]] = inner["shapes"]
        ipc_handles: list[dict] = inner["ipc_handles"]

        device_index = torch.cuda.current_device()
        physical_gpu_id = str(torch.cuda.get_device_properties(device_index).uuid)

        # Reconstruct weights from per-tensor IPC handles (one handle per
        # parameter — the vLLM IPCWeightTransferEngine.trainer_send_weights
        # convention, which differs from SkyRL's single-packed-buffer approach).
        weights: list[tuple[str, torch.Tensor]] = []
        for name, _shape, ipc_handle in zip(names, shapes, ipc_handles, strict=True):
            if physical_gpu_id not in ipc_handle:
                raise ValueError(
                    f"IPC handle not found for GPU UUID {physical_gpu_id}. "
                    f"Available UUIDs: {list(ipc_handle.keys())}"
                )
            func, args = ipc_handle[physical_gpu_id]
            # Index 6 is the device_index in torch's rebuild_cuda_tensor tuple.
            # Remap to the local (receiver-side) device index.
            list_args = list(args)
            list_args[6] = device_index
            weight: torch.Tensor = func(*list_args)
            weights.append((name, weight))

        # Load weights into the model.
        from vllm.config import set_current_vllm_config

        model = self.model_runner.model
        with set_current_vllm_config(self.vllm_config), torch.device(self.device):
            if self._is_checkpoint_format:
                model.load_weights(weights=iter(weights))
            else:
                for name, weight in weights:
                    param = model.get_parameter(name)
                    param.copy_(weight)

        # Ensure the receiver has finished consuming the IPC tensors before
        # the sender drops its reference on the next barrier.
        torch.accelerator.synchronize()


class vLLMWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for general bugfix."""

    def __new__(cls, **kwargs):
        if is_npu():
            _VLLMHijack._patch_npu_worker()
            _VLLMHijack._patch_npu_rotary_emb()
        return super().__new__(cls)
