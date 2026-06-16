from __future__ import annotations

import logging
import os
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine
from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLTrainerSendWeightsArgs, HCCLWeightTransferEngine

from vime.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_param, named_params_and_buffers
from .hf_weight_iterator_base import HfWeightIteratorBase

from vime.utils.common import is_npu

logger = logging.getLogger(__name__)


def _begin_vllm_weight_update_session(rollout_engines: Sequence[ActorHandle]) -> None:
    if dist.get_rank() == 0:
        logger.info("vLLM weight update: start_weight_update")
        ray.get([engine.start_weight_update.remote(is_checkpoint_format=True) for engine in rollout_engines])
    dist.barrier(group=get_gloo_group())


def _end_vllm_weight_update_session(rollout_engines: Sequence[ActorHandle]) -> None:
    if dist.get_rank() == 0:
        logger.info("vLLM weight update: finish_weight_update")
        ray.get([engine.finish_weight_update.remote() for engine in rollout_engines])
    dist.barrier(group=get_gloo_group())


class UpdateWeightFromDistributed:
    """
    Update distributed engines via NCCL. Each PP rank: group "vime-pp_{pp_rank}",
    only DP=TP=0 broadcasts. Non-expert (TP) and expert (EP) params separate.
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
        Initialize. Groups created in connect_rollout_engines.
        """
        self.args = args
        self.model = model
        self.weights_getter = weights_getter
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None
        self._hf_weight_iterator = (
            HfWeightIteratorBase.create(
                args=args,
                model=model,
                model_name=model_name,
                quantization_config=quantization_config,
            )
            if args.megatron_to_hf_mode == "bridge"
            else None
        )

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        """
        Create NCCL "vime-pp_{pp_rank}" if PP source (DP=TP=0). Lock prevents concurrent broadcasts.
        """
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        # For TP:
        #   1. AllGather parameters to rank 0
        #   2. Broadcast parameters from rank 0 to all vLLM engines
        self._is_pp_src_rank = (
            mpu.get_data_parallel_rank(with_context_parallel=True) == 0 and mpu.get_tensor_model_parallel_rank() == 0
        )
        pp_rank = mpu.get_pipeline_model_parallel_rank()
        if self._is_pp_src_rank:
            self._group_name = f"vime-pp_{pp_rank}"

        if self._is_pp_src_rank:
            if self._model_update_groups is not None:
                disconnect_rollout_engines_from_distributed(
                    self.args, self._group_name, self._model_update_groups, self.rollout_engines
                )
            self._model_update_groups = connect_rollout_engines_from_distributed(
                self.args,
                self._group_name,
                rollout_engines,
                engine_gpu_counts=engine_gpu_counts,
            )

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_pp_src_rank", False) or self._model_update_groups is None:
            return
        disconnect_rollout_engines_from_distributed(
            self.args, self._group_name, self._model_update_groups, self.rollout_engines
        )
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        """
        Pause → flush → _send_weights → continue. Progress on PP source.
        """
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            # int4/fp4 pre_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None
        _begin_vllm_weight_update_session(self.rollout_engines)
        try:
            self._send_weights(pbar)
            if self._is_pp_src_rank:
                torch.cuda.synchronize()
        finally:
            _end_vllm_weight_update_session(self.rollout_engines)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            # int4/fp4 post_process
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _send_weights(self, pbar: tqdm | None) -> None:
        """
        Non-expert (TP) pass → barrier → expert (EP) pass → barrier. Each iterator
        yields broadcast-ready chunks (bucketing happens internally).
        """
        use_vllm_packed = self._use_vllm_packed()
        if self._hf_weight_iterator is not None:
            self._sync_bridge_weights_to_rollout_engines(pbar, use_vllm_packed=use_vllm_packed)
            return

        if use_vllm_packed and self._is_pp_src_rank:
            logger.info("Using vLLM packed weight sync (bucketed; metadata + trainer_send_weights per bucket)")

        for hf_chunk in self._iter_non_expert_chunks():
            self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar, packed=use_vllm_packed)
        dist.barrier(group=get_gloo_group())

        if not use_vllm_packed:
            for hf_chunk in self._iter_expert_chunks():
                self._update_bucket_weights_from_distributed(hf_chunk, pbar=pbar, packed=False)
            dist.barrier(group=get_gloo_group())

    def _sync_bridge_weights_to_rollout_engines(self, pbar: tqdm | None, *, use_vllm_packed: bool) -> None:
        """
        Export HF weights through Megatron-Bridge, then send each exported chunk
        over the same NCCL non-colocate path used by the raw converter.
        """
        if self._is_pp_src_rank:
            logger.info("Using Megatron-Bridge HF weight export for non-colocate vLLM weight sync")

        megatron_local_weights = self.weights_getter()
        for hf_named_tensors in self._hf_weight_iterator.get_hf_weight_chunks(megatron_local_weights):
            if self._is_pp_src_rank:
                hf_named_tensors = list(hf_named_tensors)
                self._update_bucket_weights_from_distributed(hf_named_tensors, pbar=pbar, packed=use_vllm_packed)

        dist.barrier(group=get_gloo_group())

    def _use_vllm_packed(self) -> bool:
        """Use vLLM packed weight transfer (one-shot metadata + trainer_send_weights)."""
        if not getattr(self.args, "vllm_weight_sync_packed", True):
            return False
        if any(".experts." in name for name, _ in named_params_and_buffers(self.args, self.model)):
            return False
        if self.quantization_config and self.quantization_config.get("quant_method") == "compressed-tensors":
            return False
        return True

    def _iter_non_expert_chunks(self) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield broadcast-sized HF chunks of non-expert params: TP all-gather +
        HF convert per param, then bucket up to ``--update-weight-buffer-size``.
        Empty on non-PP-src ranks (they still join all_gather_param).
        """
        buffer_size = 0
        buffer: list[tuple[str, torch.Tensor]] = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if not self._is_pp_src_rank:
                continue
            hf_chunk = convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
            chunk_bytes = sum(t.numel() * t.element_size() for _, t in hf_chunk)
            if buffer and buffer_size + chunk_bytes > self.args.update_weight_buffer_size:
                yield buffer
                buffer = []
                buffer_size = 0
            buffer.extend(hf_chunk)
            buffer_size += chunk_bytes
        if buffer:
            yield buffer

    def _iter_expert_chunks(
        self,
        params: Iterator[tuple[str, torch.Tensor]] | None = None,
    ) -> Iterator[list[tuple[str, torch.Tensor]]]:
        """
        Yield one HF chunk per EP-weighted batch of expert params: TP gather +
        buffer until threshold, then EP gather + HF convert.
        """
        if params is None:
            params = ((n, p) for n, p in named_params_and_buffers(self.args, self.model) if ".experts." in n)

        buffer_size = 0
        batch: list[tuple[str, torch.Tensor]] = []
        for name, param in params:
            param = all_gather_param(name, param)
            param_size = param.numel() * param.element_size()
            if (
                buffer_size + param_size
            ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
                hf_chunk = self._ep_gather_and_convert(batch)
                if hf_chunk:
                    yield hf_chunk
                batch = []
                buffer_size = 0

            batch.append((name, param))
            buffer_size += param_size

        if batch:
            hf_chunk = self._ep_gather_and_convert(batch)
            if hf_chunk:
                yield hf_chunk

    def _ep_gather_and_convert(self, named_tensors: list[tuple[str, torch.Tensor]]) -> list[tuple[str, torch.Tensor]]:
        """
        EP all-gather a buffered batch + HF convert on PP source. Returns HF tensors on
        PP source, [] elsewhere. Clears ``named_tensors``.
        """
        names = [name for name, _ in named_tensors]
        all_names = [None] * mpu.get_expert_model_parallel_world_size()
        dist.all_gather_object(all_names, names, group=mpu.get_expert_model_parallel_group())

        for names in all_names:
            assert len(named_tensors) == len(names), f"mismatch names length: {len(named_tensors)} != {len(names)}"

        all_gathered_params = [[] for _ in range(mpu.get_expert_model_parallel_world_size())]
        handles = []
        for i, (_name, param) in enumerate(named_tensors):
            params = [
                torch.empty_like(param.data, device=torch.cuda.current_device())
                for _ in range(mpu.get_expert_model_parallel_world_size())
            ]
            handle = dist.all_gather(params, param.data, group=mpu.get_expert_model_parallel_group(), async_op=True)
            handles.append(handle)
            for ep_rank, names in enumerate(all_names):
                all_gathered_params[ep_rank].append((names[i], params[ep_rank]))
        for handle in handles:
            handle.wait()

        named_tensors.clear()
        if not self._is_pp_src_rank:
            return []

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        return converted_hf_tensors

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        pbar: tqdm | None = None,
        *,
        packed: bool = False,
    ) -> None:
        """
        Lock → broadcast → clear → unlock → pbar++. Lock prevents NCCL deadlock.
        """
        # lock the rollout engines to prevent dead lock on broadcast.
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
            packed=packed,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
) -> Any:
    """
    Create NCCL group: training rank 0 + all engine GPUs. Blocks until joined.

    ``engine_gpu_counts`` gives the number of GPUs per engine.  When engines
    have heterogeneous TP sizes (e.g. prefill TP=2, decode TP=4), each engine
    occupies a different number of ranks in the NCCL group.

    Trainer rank 0 uses ``NCCLWeightTransferEngine.trainer_init``
    in-process (StatelessProcessGroup + PyNcclCommunicator).
    """
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()
    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]
    world_size = sum(engine_gpu_counts) + 1  # +1 for training rank 0

    cumulative = [0]
    for c in engine_gpu_counts:
        cumulative.append(cumulative[-1] + c)

    backend = "hccl" if is_npu() else "nccl"
    refs = [
        engine.init_weights_update_group.remote(
            master_address=master_address,
            master_port=master_port,
            rank_offset=cumulative[i] + 1,
            world_size=world_size,
            group_name=group_name,
            backend=backend,
        )
        for i, engine in enumerate(rollout_engines)
    ]

    if is_npu():
        torch.npu.synchronize()
        torch.npu.empty_cache()
        device = torch.npu.current_device()
        logger.info(
            "vLLM in-process weight transfer: addr=%s port=%d world_size=%d device=%d CVD=%s",
            master_address,
            master_port,
            world_size,
            device,
            os.environ.get("ASCEND_RT_VISIBLE_DEVICES", ""),
        )
        # 使用HCCLWeightTransferEngine
        from vllm_ascend.distributed.weight_transfer.hccl_engine import HCCLWeightTransferEngine
        model_update_groups = HCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )
    else:
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        device = torch.cuda.current_device()
        logger.info(
            "vLLM in-process weight transfer: addr=%s port=%d world_size=%d device=%d CVD=%s",
            master_address,
            master_port,
            world_size,
            device,
            os.environ.get("CUDA_VISIBLE_DEVICES", ""),
        )
        model_update_groups = NCCLWeightTransferEngine.trainer_init(
            {
                "master_address": master_address,
                "master_port": master_port,
                "world_size": world_size,
            }
        )

    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    model_update_groups: Any,
    rollout_engines: Sequence[ActorHandle],
) -> None:
    """
    Tear down the weight-update NCCL group on the rollout engines.

    ``model_update_groups`` is a vLLM ``PyNcclCommunicator`` returned by
    ``NCCLWeightTransferEngine.trainer_init`` (built on a ``StatelessProcessGroup``),
    NOT a torch c10d ``ProcessGroup``. It is deliberately not registered in
    torch.distributed's global registry, so ``dist.destroy_process_group`` on it
    raises ``ValueError: Invalid process group specified`` (see #127 regression).

    We therefore do not tear the trainer-side communicator down here; this matches
    the pre-#127 behavior. (Note ``engine.destroy_weights_update_group`` is itself
    a no-op on the engine side.) An explicit ``model_update_groups.destroy()`` would
    abort the NCCL comm, but that changes long-standing behavior and risks the
    CUDA-graph-capture self-deadlock documented in ``PyNcclCommunicator.destroy``;
    leave it out of this fix.
    """
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    ray.get(refs)


def update_weights_from_distributed(
    group_name: str,
    group: Any,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    packed: bool = False,
) -> list[ObjectRef]:
    """
    Send metadata (Ray), broadcast tensors (NCCL rank 0 → engines).

    The *group* is a vLLM ``PyNcclCommunicator`` from ``trainer_init``
    in the Megatron trainer process.
    """
    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in converted_named_tensors],
            dtypes=[param.dtype for _, param in converted_named_tensors],
            shapes=[param.shape for _, param in converted_named_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            packed=packed,
        )
        for engine in rollout_engines
    ]

    named_gpu_iter = (
        (name, (param.data if hasattr(param, "data") else param).contiguous())
        for name, param in converted_named_tensors
    )
    if is_npu():
        HCCLWeightTransferEngine.trainer_send_weights(
            named_gpu_iter,
            HCCLTrainerSendWeightsArgs(group=group, packed=packed),
        )
    else:
        NCCLWeightTransferEngine.trainer_send_weights(
            named_gpu_iter,
            NCCLTrainerSendWeightsArgs(group=group, packed=packed),
        )

    return refs


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    """
    Trigger post-process for int4/fp4 quantization on all rollout engines.
    """
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
