"""
Colocated vLLM weight sync using native IPC transfer engines.
"""

from __future__ import annotations

import os
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
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

        self._model_update_groups = None
        # vLLM #39212 IPC transfer-engine init runs once per set of colocated engines.
        self._ipc_initialized = False
        # vLLM IPC handle payloads are pickled on the Ray/HTTP bridge.
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

        # Enter the native vLLM weight-update state machine on every colocated engine.
        if rank == 0:
            ray.get([engine.start_weight_update.remote(is_checkpoint_format=True) for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

        megatron_local_weights = self.weights_getter()

        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            refs = self._send_hf_params(hf_named_tensors)
            ray.get(refs)
            # Free chunk tensors so the caching allocator can reuse the blocks.
            del hf_named_tensors
            if is_npu():
                torch.npu.synchronize()
            else:
                torch.cuda.ipc_collect()

        dist.barrier(group=get_gloo_group())
        # After the barrier all engines have returned, so every rank's last-chunk
        # IPC handles are now released by the consumers.  Clean them up.
        if is_npu():
            torch.npu.synchronize()
        else:
            torch.cuda.ipc_collect()

        # Exit the native vLLM weight-update state machine.
        if rank == 0:
            ray.get([engine.finish_weight_update.remote() for engine in self.rollout_engines])
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

    def _send_hf_params(self, hf_named_tensors) -> list[object]:
        all_refs: list[object] = []

        _send_to_colocated_engine(
            hf_named_tensors,
            rollout_engines=self.rollout_engines,
            weight_version=self.weight_version,
        )

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

        return all_refs


def _send_to_colocated_engine(
    hf_named_tensors: list[tuple[str, torch.Tensor]],
    *,
    rollout_engines: Sequence[ActorHandle],
    weight_version: int,
) -> None:
    if not rollout_engines:
        return

    def send_to_vllm(update_info) -> None:
        request = {"update_info": asdict(update_info)}
        ray.get(
            [engine.update_weights.remote(request, weight_version=str(weight_version)) for engine in rollout_engines]
        )

    if is_npu():
        from vllm_ascend.distributed.weight_transfer.npu_ipc_engine import (
            NPUIPCTrainerSendWeightsArgs,
            NPUIPCWeightTransferEngine,
        )

        trainer_args = NPUIPCTrainerSendWeightsArgs(send_mode=send_to_vllm, packed=False)
        NPUIPCWeightTransferEngine.trainer_send_weights(iter(hf_named_tensors), trainer_args)
    else:
        from vllm.distributed.weight_transfer.ipc_engine import IPCTrainerSendWeightsArgs, IPCWeightTransferEngine

        trainer_args = IPCTrainerSendWeightsArgs(send_mode=send_to_vllm, packed=False)
        IPCWeightTransferEngine.trainer_send_weights(iter(hf_named_tensors), trainer_args)


# ---------------------------------------------------------------------------
# vLLM worker extension (loaded by ``--worker-extension-cls``)
# ---------------------------------------------------------------------------


class _VLLMHijack:
    """vLLM worker extension helpers.

    On NPU:
    - Patches NPUWorker.load_model and NPUWorker.start_weight_update to fix
      MoE weight_loader missing on EP (a vLLM bug where w13_weight/w2_weight
      params lack weight_loader attr when EP is enabled).
    - Patches ApplyRotaryEmb.__init__ to skip flash_attn import
      (Megatron/NPU backends introduce flash_attn as a dummy module,
      but vllm_ascend does not use it).
    """

    @staticmethod
    def _patch_npu_worker() -> None:
        from vllm_ascend.worker.worker import NPUWorker

        if getattr(NPUWorker, "_npu_worker_patched", False):
            return

        _VLLMHijack._patch_one_worker(NPUWorker)
        NPUWorker._npu_worker_patched = True

    @staticmethod
    def _patch_a3_moe_alltoall_expert_ids() -> None:
        """Restore the ALLTOALL expert-ID template after colocated memory reuse."""
        from vllm_ascend.utils import AscendDeviceType, get_ascend_device_type

        if get_ascend_device_type() != AscendDeviceType.A3:
            return

        from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV

        if getattr(TokenDispatcherWithAll2AllV, "_vime_expert_ids_patched", False):
            return

        original_dispatch_preprocess = TokenDispatcherWithAll2AllV._dispatch_preprocess
        TokenDispatcherWithAll2AllV._vime_expert_ids_generation = 0

        def _patched_dispatch_preprocess(self, hidden_states, topk_ids):
            generation = TokenDispatcherWithAll2AllV._vime_expert_ids_generation
            if self.num_local_experts > 1 and getattr(self, "_vime_seen_expert_ids_generation", -1) != generation:
                expert_ids = self.expert_ids_per_ep_rank
                self.expert_ids_per_ep_rank = torch.arange(
                    self.num_experts,
                    device=expert_ids.device,
                    dtype=expert_ids.dtype,
                ).remainder(self.num_local_experts)
                self._vime_seen_expert_ids_generation = generation
            return original_dispatch_preprocess(self, hidden_states, topk_ids)

        TokenDispatcherWithAll2AllV._dispatch_preprocess = _patched_dispatch_preprocess
        TokenDispatcherWithAll2AllV._vime_expert_ids_patched = True

    @staticmethod
    def _invalidate_moe_alltoall_expert_ids() -> None:
        try:
            from vllm_ascend.ops.fused_moe.token_dispatcher import TokenDispatcherWithAll2AllV
        except ImportError:
            return

        if getattr(TokenDispatcherWithAll2AllV, "_vime_expert_ids_patched", False):
            TokenDispatcherWithAll2AllV._vime_expert_ids_generation += 1

    @staticmethod
    def _patch_one_worker(worker_cls: type) -> None:
        import inspect

        _orig_load_model = worker_cls.load_model
        _orig_start_weight_update = worker_cls.start_weight_update
        _orig_wake_up = worker_cls.wake_up
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
            _VLLMHijack._invalidate_moe_alltoall_expert_ids()

        def _patched_wake_up(self, tags=None, _orig=_orig_wake_up) -> None:
            quant_config = self.vllm_config.quant_config
            if quant_config is not None:
                _orig(self, tags=tags)
                _VLLMHijack._invalidate_moe_alltoall_expert_ids()
                return

            # vllm-ascend transposes unquantized w13_weight/w2_weight in
            # wake_up(). Keep the native allocator and buffer restoration, but
            # skip that branch: layerwise reload owns the final runtime layout.
            self.vllm_config.quant_config = object()
            try:
                _orig(self, tags=tags)
            finally:
                self.vllm_config.quant_config = quant_config
            _VLLMHijack._invalidate_moe_alltoall_expert_ids()

        worker_cls.load_model = _patched_load_model  # type: ignore[attr-defined]
        worker_cls.start_weight_update = _patched_start_weight_update  # type: ignore[attr-defined]
        worker_cls.wake_up = _patched_wake_up  # type: ignore[attr-defined]

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
    """vLLM ``--worker-extension-cls`` entry for colocated rollout workers."""

    def __new__(cls, **kwargs):
        if is_npu():
            _VLLMHijack._patch_a3_moe_alltoall_expert_ids()
            _VLLMHijack._patch_npu_worker()
            _VLLMHijack._patch_npu_rotary_emb()
        return super().__new__(cls)


class vLLMWorkerExtension:
    """vLLM ``--worker-extension-cls`` entry for general bugfix."""

    def __new__(cls, **kwargs):
        if is_npu():
            _VLLMHijack._patch_npu_worker()
            _VLLMHijack._patch_npu_rotary_emb()
        return super().__new__(cls)
