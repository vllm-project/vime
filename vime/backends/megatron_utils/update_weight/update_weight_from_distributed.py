from __future__ import annotations

import logging
import os
import socket
import time
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import ray
import torch
import torch.distributed as dist
from megatron.core import mpu
from ray import ObjectRef
from ray.actor import ActorHandle
from tqdm import tqdm
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine

from vime.utils.distributed_utils import get_gloo_group

from ..megatron_to_hf import convert_to_hf, convert_to_hf_shard
from .common import all_gather_param, named_params_and_buffers

logger = logging.getLogger(__name__)


def _begin_vllm_weight_update_session(rollout_engines: Sequence[ActorHandle], is_checkpoint_format: bool = True) -> None:
    if dist.get_rank() == 0:
        logger.info("vLLM weight update: start_weight_update (checkpoint_format=%s)", is_checkpoint_format)
        ray.get([engine.start_weight_update.remote(is_checkpoint_format=is_checkpoint_format) for engine in rollout_engines])
    dist.barrier(group=get_gloo_group())


def _end_vllm_weight_update_session(rollout_engines: Sequence[ActorHandle]) -> None:
    if dist.get_rank() == 0:
        logger.info("vLLM weight update: finish_weight_update")
        ray.get([engine.finish_weight_update.remote() for engine in rollout_engines])
    dist.barrier(group=get_gloo_group())


class UpdateWeightFromDistributed:
    """Shard-level P2P weight transfer without all_gather.

    Each DP=0 TP rank converts only its own shard via convert_to_hf_shard
    and broadcasts it to the corresponding vLLM inference ranks via its own
    NCCL group. This eliminates the all_gather memory bottleneck.

    Optimized: each TP rank's NCCL group includes only matching vLLM workers,
    eliminating wasted bandwidth from inactive broadcast participants.
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
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.weight_version = 0
        self._model_update_groups = None

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self.rollout_engines = rollout_engines
        self.rollout_engine_lock = rollout_engine_lock
        self._engine_gpu_counts = engine_gpu_counts

        dp_rank = mpu.get_data_parallel_rank(with_context_parallel=True)
        tp_rank = mpu.get_tensor_model_parallel_rank()
        tp_size = mpu.get_tensor_model_parallel_world_size()
        pp_rank = mpu.get_pipeline_model_parallel_rank()

        self._is_pp_src_rank = dp_rank == 0 and tp_rank == 0
        self._is_dp0 = dp_rank == 0
        self._tp_rank = tp_rank
        self._tp_size = tp_size
        self._pp_rank = pp_rank

        if self._is_dp0:
            self._group_name = f"vime-pp_{pp_rank}_tp{tp_rank}"
            if self._model_update_groups is not None:
                logger.info("NCCL group %s already connected, skipping reconnection", self._group_name)
                return
            tp_rank_for_group = self._tp_rank if self._use_shard_conversion() else None
            while not ray.get(self.rollout_engine_lock.acquire.remote()):
                time.sleep(0.1)
            try:
                self._model_update_groups = connect_rollout_engines_from_distributed(
                    self.args,
                    self._group_name,
                    rollout_engines,
                    engine_gpu_counts=engine_gpu_counts,
                    target_tp_rank=tp_rank_for_group,
                )
            finally:
                ray.get(self.rollout_engine_lock.release.remote())

    def disconnect_rollout_engines(self) -> None:
        if not getattr(self, "_is_dp0", False) or self._model_update_groups is None:
            return
        logger.info("NCCL group %s kept alive (persistent connection)", self._group_name)

    def shutdown_rollout_engines(self) -> None:
        if not getattr(self, "_is_dp0", False) or self._model_update_groups is None:
            return
        disconnect_rollout_engines_from_distributed(
            self.args, self._group_name, self._model_update_groups, self.rollout_engines
        )
        self._model_update_groups = None

    @torch.no_grad()
    def update_weights(self) -> None:
        self.weight_version += 1

        if dist.get_rank() == 0:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])

            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )
        dist.barrier(group=get_gloo_group())

        use_shard = self._use_shard_conversion()
        is_checkpoint_format = not use_shard

        _begin_vllm_weight_update_session(self.rollout_engines, is_checkpoint_format=is_checkpoint_format)
        try:
            self._sync_weights_to_rollout_engines()
        finally:
            _end_vllm_weight_update_session(self.rollout_engines)

        dist.barrier(group=get_gloo_group())
        if dist.get_rank() == 0:
            if self.quantization_config and self.quantization_config["quant_method"] in ["compressed-tensors"]:
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )
            ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
        dist.barrier(group=get_gloo_group())

    def _sync_weights_to_rollout_engines(self) -> None:
        use_vllm_packed = self._use_vllm_packed()
        use_shard = self._use_shard_conversion()

        if use_shard and use_vllm_packed and self._is_dp0:
            logger.info("Using shard-level P2P weight sync (no all_gather)")

        if use_shard and use_vllm_packed:
            self._sync_weights_shard_packed()
        elif use_vllm_packed:
            self._sync_weights_full_packed()
        else:
            self._sync_weights_full_nonpacked()

        if self._is_dp0:
            torch.cuda.synchronize()

    def _use_shard_conversion(self) -> bool:
        if self.quantization_config and self.quantization_config.get("quant_method") == "compressed-tensors":
            return False
        if any(".experts." in name for name, _ in named_params_and_buffers(self.args, self.model)):
            return False
        if self._engine_gpu_counts and any(c != self._tp_size for c in self._engine_gpu_counts):
            return False
        return True

    def _sync_weights_shard_packed(self) -> None:
        """Shard-level P2P: each TP rank converts its own shard without all_gather.

        For embedding/output_layer, the Megatron shard layout (padded vocab) doesn't
        align with vLLM's shard layout (unpadded vocab). These are handled with a
        small all_gather + remove_padding + split, which is cheap since they're small.
        """
        from vime.backends.megatron_utils.megatron_to_hf import remove_padding
        from vime.backends.megatron_utils.misc_utils import strip_param_name_prefix

        if self._is_dp0:
            converted_named_tensors: list[tuple[str, torch.Tensor]] = []
            pbar = (
                tqdm(desc=f"[{self._group_name}] Shard P2P update", total=0)
                if self._is_pp_src_rank
                else None
            )
            buffer_size = 0

            # Phase 1: Handle embedding/output_layer with all_gather (small params)
            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." in name:
                    continue
                stripped = strip_param_name_prefix(name)
                if stripped not in {"embedding.word_embeddings.weight", "output_layer.weight"}:
                    continue
                full_param = all_gather_param(name, param)
                full_param = remove_padding(name, full_param, self.args.vocab_size)
                converted = convert_to_hf(
                    self.args, self.model_name, name, full_param,
                    self.quantization_config,
                )
                if not converted:
                    continue
                for hf_name, hf_param in converted:
                    if hf_param.shape[0] % self._tp_size != 0:
                        logger.warning("Cannot split %s (shape %s) by tp_size=%d, skipping shard split",
                                       hf_name, hf_param.shape, self._tp_size)
                        continue
                    shard_size = hf_param.shape[0] // self._tp_size
                    my_shard = hf_param[self._tp_rank * shard_size:(self._tp_rank + 1) * shard_size]
                    param_size = my_shard.numel() * my_shard.element_size()
                    if buffer_size + param_size > self.args.update_weight_buffer_size:
                        if converted_named_tensors:
                            self._update_weights_shard_packed(converted_named_tensors)
                            converted_named_tensors = []
                            if pbar is not None:
                                pbar.update(1)
                        buffer_size = 0
                    converted_named_tensors.append((hf_name, my_shard))
                    buffer_size += param_size

            # Phase 2: Handle all other params with shard-level conversion (no all_gather)
            for name, param in named_params_and_buffers(self.args, self.model):
                if ".experts." in name:
                    continue
                stripped = strip_param_name_prefix(name)
                if stripped in {"embedding.word_embeddings.weight", "output_layer.weight"}:
                    continue
                shard_converted = convert_to_hf_shard(
                    self.args, self.model_name, name, param.data,
                    self._tp_rank, self._tp_size, self.quantization_config,
                )
                if not shard_converted:
                    continue
                for hf_name, hf_param in shard_converted:
                    param_size = hf_param.numel() * hf_param.element_size()
                    if buffer_size + param_size > self.args.update_weight_buffer_size:
                        if converted_named_tensors:
                            self._update_weights_shard_packed(converted_named_tensors)
                            converted_named_tensors = []
                            if pbar is not None:
                                pbar.update(1)
                        buffer_size = 0
                    converted_named_tensors.append((hf_name, hf_param))
                    buffer_size += param_size
            if converted_named_tensors:
                self._update_weights_shard_packed(converted_named_tensors)
                if pbar is not None:
                    pbar.update(1)

        dist.barrier(group=get_gloo_group())

    def _sync_weights_full_packed(self) -> None:
        """Original full all_gather + convert + broadcast path."""
        gathered_params: list[tuple[str, torch.Tensor]] = []
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            param = all_gather_param(name, param)
            if self._is_dp0:
                gathered_params.append((name, param))

        dist.barrier(group=get_gloo_group())

        if self._is_dp0:
            converted_named_tensors: list[tuple[str, torch.Tensor]] = []
            pbar = (
                tqdm(desc=f"[{self._group_name}] Update weights (vLLM packed)", total=0)
                if self._is_pp_src_rank
                else None
            )
            buffer_size = 0
            for name, param in gathered_params:
                param_size = param.numel() * param.element_size()
                if buffer_size + param_size > self.args.update_weight_buffer_size:
                    if converted_named_tensors:
                        self._update_weights_vllm_packed(converted_named_tensors)
                        converted_named_tensors.clear()
                        if pbar is not None:
                            pbar.update(1)
                    buffer_size = 0
                converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
                buffer_size += param_size
            if converted_named_tensors:
                self._update_weights_vllm_packed(converted_named_tensors)
                if pbar is not None:
                    pbar.update(1)

        dist.barrier(group=get_gloo_group())

    def _sync_weights_full_nonpacked(self) -> None:
        buffer_size = 0
        converted_named_tensors = []
        pbar = tqdm(desc=f"[{self._group_name}] Update weights", total=0) if self._is_pp_src_rank else None

        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." in name:
                continue
            buffer_size = self._update_weight_from_distributed(
                name, param, converted_named_tensors, buffer_size, pbar=pbar
            )

        if converted_named_tensors:
            self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)

        dist.barrier(group=get_gloo_group())

        buffer_size = 0
        named_tensors = []
        pbar = (
            tqdm(desc=f"[{self._group_name}] Update weights (experts)", total=0) if self._is_pp_src_rank else None
        )
        for name, param in named_params_and_buffers(self.args, self.model):
            if ".experts." not in name:
                continue
            buffer_size = self._update_expert_weight_from_distributed(
                name, param, named_tensors, buffer_size, pbar=pbar
            )

        if named_tensors:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)

    def _use_vllm_packed(self) -> bool:
        if not getattr(self.args, "vllm_weight_sync_packed", True):
            return False
        if any(".experts." in name for name, _ in named_params_and_buffers(self.args, self.model)):
            return False
        if self.quantization_config and self.quantization_config.get("quant_method") == "compressed-tensors":
            return False
        return True

    def _update_weights_shard_packed(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            refs = update_weights_from_distributed_shard(
                self._group_name,
                self._model_update_groups,
                self.weight_version,
                self.rollout_engines,
                converted_named_tensors,
                packed=True,
            )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())

    def _update_weights_vllm_packed(self, converted_named_tensors: list[tuple[str, torch.Tensor]]) -> None:
        use_shard = self._use_shard_conversion()
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        try:
            if use_shard:
                refs = update_weights_from_distributed_p2p(
                    self._group_name,
                    self._model_update_groups,
                    self.weight_version,
                    self.rollout_engines,
                    converted_named_tensors,
                    self._tp_rank,
                    self._tp_size,
                    packed=True,
                )
            else:
                refs = update_weights_from_distributed(
                    self._group_name,
                    self._model_update_groups,
                    self.weight_version,
                    self.rollout_engines,
                    converted_named_tensors,
                    packed=True,
                )
            ray.get(refs)
        finally:
            ray.get(self.rollout_engine_lock.release.remote())

    def _update_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        converted_named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
        *,
        flush_packed: bool = False,
    ) -> int:
        param = all_gather_param(name, param)
        if not self._is_pp_src_rank:
            return buffer_size

        param_size = param.numel() * param.element_size()
        if buffer_size + param_size > self.args.update_weight_buffer_size:
            if converted_named_tensors:
                if flush_packed:
                    self._update_weights_vllm_packed(converted_named_tensors)
                    converted_named_tensors.clear()
                    if pbar is not None:
                        pbar.update(1)
                else:
                    self._update_bucket_weights_from_distributed(converted_named_tensors, pbar=pbar)
            buffer_size = 0
        converted_named_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)
        buffer_size += param_size
        return buffer_size

    def _update_expert_weight_from_distributed(
        self,
        name: str,
        param: torch.nn.Parameter,
        named_tensors: list[tuple[str, torch.Tensor]],
        buffer_size: int,
        pbar: tqdm | None = None,
    ) -> int:
        param = all_gather_param(name, param)

        param_size = param.numel() * param.element_size()
        if (
            buffer_size + param_size
        ) * mpu.get_expert_model_parallel_world_size() > self.args.update_weight_buffer_size:
            self._update_expert_bucket_weights_from_distributed(named_tensors, pbar=pbar)
            buffer_size = 0

        named_tensors.append((name, param))
        buffer_size += param_size
        return buffer_size

    def _update_expert_bucket_weights_from_distributed(
        self, named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
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
            return

        all_gathered_params = sum(all_gathered_params, [])
        converted_hf_tensors = []
        for name, param in all_gathered_params:
            converted_hf_tensors += convert_to_hf(self.args, self.model_name, name, param, self.quantization_config)

        self._update_bucket_weights_from_distributed(converted_hf_tensors, pbar)

    def _update_bucket_weights_from_distributed(
        self, converted_named_tensors: list[tuple[str, torch.Tensor]], pbar: tqdm | None = None
    ) -> None:
        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)

        refs = update_weights_from_distributed_p2p(
            self._group_name,
            self._model_update_groups,
            self.weight_version,
            self.rollout_engines,
            converted_named_tensors,
            self._tp_rank,
            self._tp_size,
            packed=False,
        )

        ray.get(refs)
        converted_named_tensors.clear()
        ray.get(self.rollout_engine_lock.release.remote())
        if pbar is not None:
            pbar.update(1)


def connect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    rollout_engines: Sequence[ActorHandle],
    engine_gpu_counts: Sequence[int] | None = None,
    target_tp_rank: int | None = None,
) -> dist.ProcessGroup:
    if engine_gpu_counts is None:
        engine_gpu_counts = [args.rollout_num_gpus_per_engine] * len(rollout_engines)

    master_address = ray._private.services.get_node_ip_address()

    with socket.socket() as sock:
        sock.bind(("", 0))
        master_port = sock.getsockname()[1]

    num_engines = len(rollout_engines)

    if target_tp_rank is not None:
        world_size = 1 + num_engines
    else:
        world_size = sum(engine_gpu_counts) + 1

    init_kwargs = dict(
        master_address=master_address,
        master_port=master_port,
        world_size=world_size,
        group_name=group_name,
        backend="nccl",
    )
    if target_tp_rank is not None:
        init_kwargs["target_tp_rank"] = target_tp_rank

    if target_tp_rank is not None:
        refs = [
            engine.init_weights_update_group.remote(
                rank_offset=0,
                shard_rank=1 + i,
                **init_kwargs,
            )
            for i, engine in enumerate(rollout_engines)
        ]
    else:
        cumulative = [1]
        for c in engine_gpu_counts:
            cumulative.append(cumulative[-1] + c)
        refs = [
            engine.init_weights_update_group.remote(
                rank_offset=cumulative[i],
                **init_kwargs,
            )
            for i, engine in enumerate(rollout_engines)
        ]

    torch.cuda.synchronize()
    torch.cuda.empty_cache()

    device = torch.cuda.current_device()
    logger.info(
        "vLLM P2P weight transfer: group=%s addr=%s port=%d world_size=%d device=%d shard=%s",
        group_name, master_address, master_port, world_size, device,
        target_tp_rank is not None,
    )
    model_update_groups = NCCLWeightTransferEngine.trainer_init(
        {
            "master_address": master_address,
            "master_port": master_port,
            "world_size": world_size,
            "rank": 0,
        }
    )

    ray.get(refs)
    return model_update_groups


def disconnect_rollout_engines_from_distributed(
    args: Namespace,
    group_name: str,
    model_update_groups: dist.ProcessGroup,
    rollout_engines: Sequence[ActorHandle],
) -> None:
    refs = [engine.destroy_weights_update_group.remote(group_name) for engine in rollout_engines]
    del model_update_groups
    ray.get(refs)


def update_weights_from_distributed_shard(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    packed: bool = False,
) -> list[ObjectRef]:
    """Broadcast all shard-converted tensors (no TP splitting - each rank already has its own shard)."""
    if not converted_named_tensors:
        return []

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
    NCCLWeightTransferEngine.trainer_send_weights(
        named_gpu_iter,
        NCCLTrainerSendWeightsArgs(group=group, packed=packed),
    )

    return refs


def update_weights_from_distributed_p2p(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    tp_rank: int,
    tp_size: int,
    *,
    packed: bool = False,
) -> list[ObjectRef]:
    n = len(converted_named_tensors)
    chunk_size = (n + tp_size - 1) // tp_size
    start = tp_rank * chunk_size
    end = min(start + chunk_size, n)
    my_tensors = list(converted_named_tensors[start:end])

    if not my_tensors:
        return []

    refs = [
        engine.update_weights_from_distributed.remote(
            names=[name for name, _ in my_tensors],
            dtypes=[param.dtype for _, param in my_tensors],
            shapes=[param.shape for _, param in my_tensors],
            group_name=group_name,
            weight_version=str(weight_version),
            packed=packed,
        )
        for engine in rollout_engines
    ]

    named_gpu_iter = (
        (name, (param.data if hasattr(param, "data") else param).contiguous())
        for name, param in my_tensors
    )
    NCCLWeightTransferEngine.trainer_send_weights(
        named_gpu_iter,
        NCCLTrainerSendWeightsArgs(group=group, packed=packed),
    )

    return refs


def update_weights_from_distributed(
    group_name: str,
    group: dist.ProcessGroup,
    weight_version: int,
    rollout_engines: Sequence[ActorHandle],
    converted_named_tensors: Sequence[tuple[str, torch.Tensor]],
    *,
    packed: bool = False,
) -> list[ObjectRef]:
    return update_weights_from_distributed_p2p(
        group_name, group, weight_version, rollout_engines,
        converted_named_tensors, tp_rank=0, tp_size=1, packed=packed,
    )


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
):
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )
