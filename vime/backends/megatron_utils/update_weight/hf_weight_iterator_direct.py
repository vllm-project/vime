import dataclasses
from argparse import Namespace
from collections.abc import Callable, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from tqdm import tqdm

from vime.utils import accelerator
from vime.utils.distributed_utils import get_gloo_group
from vime.utils.types import ParamInfo

from ..megatron_to_hf import convert_to_hf
from .common import all_gather_params_async, named_params_and_buffers


class HfWeightIteratorDirect:
    def __init__(self, args, model, model_name, quantization_config, transform_ue8m0=False):
        self.args = args
        self.model = model
        self.model_name = model_name
        self.quantization_config = quantization_config
        self.transform_ue8m0 = transform_ue8m0
        self.megatron_local_param_info_buckets = _get_megatron_local_param_info_buckets(self.args, self.model)
        self.ep_broadcast_src_rank_map = _get_ep_broadcast_src_rank_map()

    def get_hf_weight_chunks(
        self,
        megatron_local_weights,
        progress_desc: str = "Update weights",
        should_convert_chunk: Callable[[int], bool] | None = None,
        param_info_buckets: Sequence[Sequence[ParamInfo]] | None = None,
    ):
        rank = dist.get_rank()
        param_info_buckets = (
            self.megatron_local_param_info_buckets if param_info_buckets is None else param_info_buckets
        )

        for chunk_idx, megatron_local_param_infos in enumerate(
            tqdm(param_info_buckets, disable=rank != 0, desc=progress_desc)
        ):
            megatron_full_params = _get_megatron_full_params(
                megatron_local_param_infos,
                megatron_local_weights,
                self.args.update_weight_buffer_size,
                self.ep_broadcast_src_rank_map,
            )
            if should_convert_chunk is None or should_convert_chunk(chunk_idx):
                hf_named_tensors = self._convert_to_hf_named_tensors(
                    megatron_full_params,
                    megatron_local_param_infos,
                )
            else:
                hf_named_tensors = []
            try:
                yield hf_named_tensors
            finally:
                del hf_named_tensors, megatron_full_params

    def _convert_to_hf_named_tensors(
        self,
        megatron_full_params: Sequence[torch.Tensor],
        param_infos: Sequence[ParamInfo],
    ):
        hf_named_tensors = []
        for info, param in zip(param_infos, megatron_full_params, strict=False):
            # Re-apply attrs from ParamInfo onto fresh tensors returned by
            # all_gather_params_async (e.g. _in_proj_split_sizes_full, needed
            # by convert_to_hf to split merged in_proj.weight back to 4 keys).
            for key, value in info.attrs.items():
                if value is not None and not hasattr(param, key):
                    setattr(param, key, value)
            hf_named_tensors.extend(
                convert_to_hf(
                    self.args,
                    self.model_name,
                    info.name,
                    param,
                    self.quantization_config,
                    transform_ue8m0=self.transform_ue8m0,
                )
            )
        return hf_named_tensors


def _get_megatron_full_params(
    megatron_local_param_infos: Sequence[ParamInfo],
    megatron_local_weights,
    broadcast_buffer_size: int,
    ep_broadcast_src_rank_map: dict[int, int],
) -> Sequence[torch.Tensor]:
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()
    # init params:
    params = []
    for info in megatron_local_param_infos:
        if dist.get_rank() == info.src_rank:
            params.append(
                torch.nn.Parameter(
                    megatron_local_weights[info.name].to(device=accelerator.current_device(), non_blocking=True),
                    requires_grad=False,
                )
            )
        else:
            params.append(torch.empty(info.shape, dtype=info.dtype, device=accelerator.current_device()))
    accelerator.synchronize()
    # broadcast params across pp ranks
    if pp_size > 1:
        handles = []
        for info, param in zip(megatron_local_param_infos, params, strict=False):
            if info.src_rank in dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group()):
                handles.append(
                    torch.distributed.broadcast(
                        param, src=info.src_rank, group=mpu.get_pipeline_model_parallel_group(), async_op=True
                    )
                )
        for handle in handles:
            handle.wait()

    # broadcast params across ep ranks
    if ep_size > 1:
        _broadcast_expert_params(megatron_local_param_infos, params, broadcast_buffer_size, ep_broadcast_src_rank_map)

    # Set tp attrs for all params
    for info, param in zip(megatron_local_param_infos, params, strict=False):
        for key, value in info.attrs.items():
            setattr(param, key, value)

    # Batch async all_gather for all parameters
    gathered_params = all_gather_params_async(list(zip(megatron_local_param_infos, params, strict=False)))

    return gathered_params


def _broadcast_expert_params(
    param_infos: Sequence[ParamInfo],
    params: Sequence[torch.Tensor],
    buffer_size: int,
    src_rank_map: dict[int, int],
) -> None:
    ep_group = mpu.get_expert_model_parallel_group()
    params_by_src: dict[int, list[torch.Tensor]] = {}
    for info, param in zip(param_infos, params, strict=False):
        if ".experts." not in info.name:
            continue
        params_by_src.setdefault(src_rank_map[info.src_rank], []).append(param)

    for src_rank, expert_params in params_by_src.items():
        dist._broadcast_coalesced(ep_group, expert_params, buffer_size, src=src_rank)


def _get_ep_broadcast_src_rank_map() -> dict[int, int]:
    ep_group = mpu.get_expert_model_parallel_group()
    ep_size = mpu.get_expert_model_parallel_world_size()
    if ep_size == 1:
        return {dist.get_rank(): 0}

    pp_group_ranks = dist.get_process_group_ranks(mpu.get_pipeline_model_parallel_group())
    pp_groups: list[list[int] | None] = [None] * ep_size
    dist.all_gather_object(pp_groups, pp_group_ranks, group=ep_group)

    src_rank_map = {}
    for ep_rank, pp_group in enumerate(pp_groups):
        assert pp_group is not None
        for global_rank in pp_group:
            src_rank_map[global_rank] = ep_rank
    return src_rank_map


def _get_megatron_local_param_info_buckets(args: Namespace, model: Sequence[torch.nn.Module]) -> list[list[ParamInfo]]:
    """
    Partition params into buckets ≤ update_weight_buffer_size (with TP replication).
    """
    param_infos = _get_megatron_local_param_infos(args, model)
    return pack_param_info_buckets(param_infos, args.update_weight_buffer_size)


def pack_param_info_buckets(
    param_infos: Sequence[ParamInfo],
    update_weight_buffer_size: int,
) -> list[list[ParamInfo]]:
    param_info_buckets = [[]]  # Start with one empty bucket
    buffer_size = 0  # Track current bucket size in bytes

    for info in param_infos:
        # Expert params use expert-TP size, others use regular-TP size
        if ".experts." in info.name:
            tp_size = mpu.get_expert_tensor_parallel_world_size()
        else:
            tp_size = mpu.get_tensor_model_parallel_world_size()

        # Full param size = shard size × TP replicas (all-gather will reconstruct full param)
        param_size = info.size * tp_size

        # If adding this param exceeds limit AND current bucket has params: start new bucket
        if buffer_size + param_size > update_weight_buffer_size and len(param_info_buckets[-1]) > 0:
            param_info_buckets.append([])
            buffer_size = 0

        # Add param to current bucket and update size
        param_info_buckets[-1].append(info)
        buffer_size += param_size

    return param_info_buckets


def _get_megatron_local_param_infos(args: Namespace, model: Sequence[torch.nn.Module]) -> list[ParamInfo]:
    """
    Build global param metadata: collect → exchange PP/EP → resolve duplicates (MTP virtual PP)
    by min src_rank → validate. Returns sorted ParamInfo identical across all ranks.
    """
    pp_size = mpu.get_pipeline_model_parallel_world_size()
    ep_size = mpu.get_expert_model_parallel_world_size()

    param_infos = {}
    rank = dist.get_rank()
    # Pre-scan model modules for _in_proj_split_sizes_full (set on the GDN
    # module in qwen3_5.py).  DDP wrapping can drop it from the parameter
    # object that named_params_and_buffers yields, so build a name→sizes map
    # here to recover it for ParamInfo.attrs.
    _in_proj_split_sizes_by_name: dict[str, list[int]] = {}
    _conv1d_split_sizes_by_name: dict[str, list[int]] = {}
    for vp_stage, model_module in enumerate(model):
        for module_name, module in model_module.named_modules():
            if hasattr(module, "_in_proj_split_sizes_full") and hasattr(module, "in_proj"):
                _in_proj_split_sizes_by_name[module_name + ".in_proj.weight"] = module._in_proj_split_sizes_full
            if hasattr(module, "_conv1d_split_sizes_full") and hasattr(module, "conv1d"):
                _conv1d_split_sizes_by_name[module_name + ".conv1d.weight"] = module._conv1d_split_sizes_full
    for name, param in named_params_and_buffers(args, model):
        # DSpark draft params are sent through the draft weight source.
        if ".draft_model." in name:
            continue
        # Recover _in_proj_split_sizes_full from the pre-scan if the parameter
        # lost the attribute during DDP wrapping.
        in_proj_split_sizes = getattr(param, "_in_proj_split_sizes_full", None)
        if in_proj_split_sizes is None and "linear_attn.in_proj.weight" in name:
            for key, sizes in _in_proj_split_sizes_by_name.items():
                if name.endswith(key):
                    in_proj_split_sizes = sizes
                    break
        # Recover _conv1d_split_sizes_full the same way for conv1d.weight.
        conv1d_split_sizes = getattr(param, "_conv1d_split_sizes_full", None)
        if conv1d_split_sizes is None and "linear_attn.conv1d.weight" in name:
            for key, sizes in _conv1d_split_sizes_by_name.items():
                if name.endswith(key):
                    conv1d_split_sizes = sizes
                    break
        param_infos[name] = ParamInfo(
            name=name,
            dtype=param.dtype,
            shape=param.shape,
            attrs={
                "tensor_model_parallel": getattr(param, "tensor_model_parallel", False),
                "partition_dim": getattr(param, "partition_dim", -1),
                "partition_stride": getattr(param, "partition_stride", 1),
                "parallel_mode": getattr(param, "parallel_mode", None),
                # Merged GDN in_proj.weight carries its per-sub-matrix split sizes
                # so all_gather_params_async can gather each sub-matrix independently
                # (each was TP-sharded by head) and convert_to_hf can split it back.
                "_in_proj_split_sizes_full": in_proj_split_sizes,
                # GDN conv1d.weight carries its per-sub-matrix split sizes for the
                # same reason (q/k/v must be gathered independently by head).
                "_conv1d_split_sizes_full": conv1d_split_sizes,
            },
            size=param.numel() * param.element_size(),
            src_rank=rank,
        )

    if pp_size > 1:
        param_infos_list = [None] * pp_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_pipeline_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            if src_rank == rank:
                continue
            for name, info in infos.items():
                if name in param_infos:
                    old_info = param_infos[name]
                    if old_info.src_rank > src_rank:
                        param_infos[name] = info
                else:
                    param_infos[name] = info

    if ep_size > 1:
        param_infos_list = [None] * ep_size
        dist.all_gather_object(
            obj=(rank, param_infos), object_list=param_infos_list, group=mpu.get_expert_model_parallel_group()
        )
        for src_rank, infos in param_infos_list:
            for name, info in infos.items():
                if name not in param_infos:
                    # here we need to set the src_rank to the rank within the expert model parallel group
                    info = dataclasses.replace(info, src_rank=src_rank)
                    param_infos[name] = info

    param_infos = list(param_infos.values())
    param_infos = sorted(param_infos, key=lambda info: info.name)

    # Check all ranks has the same parameter info
    all_param_info_list = [None] * dist.get_world_size()
    dist.all_gather_object(
        obj=param_infos,
        object_list=all_param_info_list,
        group=get_gloo_group(),
    )
    for i, param_info in enumerate(param_infos):
        for infos in all_param_info_list:
            assert infos[i].name == param_info.name, f"Parameter name mismatch: {infos[i].name} != {param_info.name}"
            assert (
                infos[i].shape == param_info.shape
            ), f"Parameter shape mismatch: {infos[i].shape} != {param_info.shape}"
            assert (
                infos[i].dtype == param_info.dtype
            ), f"Parameter dtype mismatch: {infos[i].dtype} != {param_info.dtype}"

    return param_infos
