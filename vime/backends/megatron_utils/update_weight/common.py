import inspect
import re
from argparse import Namespace
from collections.abc import Iterator, Sequence

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from vime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from vime.utils.types import ParamInfo


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param, non-TP/duplicated→param.data.
    Uses expert-TP for ".experts.", else regular-TP. linear_fc1 rechunked (GLU), linear_fc2 dim fix.
    """
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()
        tp_group = mpu.get_tensor_model_parallel_group()

    if tp_size == 1:
        return param.data

    param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
    dist.all_gather(param_partitions, param.data, group=tp_group)
    partition_dim = param.partition_dim
    assert param.partition_stride == 1 or (
        param.partition_stride == 2 and "linear_fc1" in name
    ), "partition_stride != 1 is not supported"
    # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
    # TODO: check only GLU is used.
    if "linear_fc1.weight" in name or "linear_fc1.bias" in name:
        param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
        param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
    # this is bug in megatron's grouped moe.
    if "linear_fc2.weight" in name:
        if partition_dim == 0:
            partition_dim = 1
    param = torch.cat(param_partitions, dim=partition_dim)
    return param


def all_gather_params_async(
    param_infos_and_params: list[tuple[ParamInfo, torch.Tensor]],
) -> list[torch.Tensor]:
    """
    Coalesce TP-sharded params by process group and dtype, then reconstruct
    their original layouts after one all-gather per bucket.
    """
    gathered_params: list[torch.Tensor | None] = [None] * len(param_infos_and_params)
    grouped_params: dict[tuple[bool, torch.dtype], list[tuple[int, ParamInfo, torch.Tensor]]] = {}

    for index, (info, param) in enumerate(param_infos_and_params):
        if "expert_bias" in info.name:
            gathered_params[index] = param
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gathered_params[index] = param.data
        else:
            is_expert = ".experts." in info.name
            tp_size = (
                mpu.get_expert_tensor_parallel_world_size()
                if is_expert
                else mpu.get_tensor_model_parallel_world_size()
            )
            if tp_size == 1:
                gathered_params[index] = param.data
            else:
                grouped_params.setdefault((is_expert, param.dtype), []).append((index, info, param))

    gather_tasks = []
    for (is_expert, _dtype), entries in grouped_params.items():
        tp_size = (
            mpu.get_expert_tensor_parallel_world_size() if is_expert else mpu.get_tensor_model_parallel_world_size()
        )
        tp_group = mpu.get_expert_tensor_parallel_group() if is_expert else mpu.get_tensor_model_parallel_group()
        local_flat = torch.cat([param.data.reshape(-1) for _, _, param in entries])
        gathered_flat = torch.empty(tp_size * local_flat.numel(), dtype=local_flat.dtype, device=local_flat.device)
        handle = dist.all_gather_into_tensor(gathered_flat, local_flat, group=tp_group, async_op=True)
        gather_tasks.append((handle, gathered_flat, local_flat.numel(), entries, tp_size))

    for handle, gathered_flat, rank_stride, entries, tp_size in gather_tasks:
        handle.wait()
        offset = 0
        for index, info, param in entries:
            numel = param.numel()
            param_partitions = [
                gathered_flat.narrow(0, rank * rank_stride + offset, numel).view_as(param) for rank in range(tp_size)
            ]
            partition_dim = param.partition_dim
            assert param.partition_stride == 1 or (
                param.partition_stride == 2 and "linear_fc1" in info.name
            ), "partition_stride != 1 is not supported"
            if "linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name:
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
            if "linear_fc2.weight" in info.name and partition_dim == 0:
                partition_dim = 1
            gathered_params[index] = torch.cat(param_partitions, dim=partition_dim)
            offset += numel

    assert all(param is not None for param in gathered_params)
    return [param for param in gathered_params if param is not None]


def named_params_and_buffers(
    args: Namespace,
    model: Sequence[torch.nn.Module],
    convert_to_global_name: bool = True,
    translate_gpu_to_cpu: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if convert_to_global_name:
        ans = _named_params_and_buffers_global(args, model)
    else:
        ans = _named_params_and_buffers_vanilla(model)

    if translate_gpu_to_cpu:
        ans = ((name, _maybe_get_cpu_backup(tensor)) for name, tensor in ans)

    return ans


def _maybe_get_cpu_backup(x: torch.Tensor):
    from torch_memory_saver import torch_memory_saver

    if (cpu_tensor := torch_memory_saver.get_cpu_backup(x, zero_copy=True)) is not None:
        return cpu_tensor

    return x


def _named_params_and_buffers_vanilla(model: Sequence[torch.nn.Module]) -> Iterator[tuple[str, torch.Tensor]]:
    for vp_stage, model_module in enumerate(model):

        def _compute_fqn(name, vp_stage=vp_stage):
            return f"vp_stages.{vp_stage}.{strip_param_name_prefix(name)}"

        for name, param in model_module.named_parameters():
            yield _compute_fqn(name), param

        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            yield _compute_fqn(name), buffer


def _named_params_and_buffers_global(
    args: Namespace, model: Sequence[torch.nn.Module]
) -> Iterator[tuple[str, torch.Tensor]]:
    """
    Yield (global_name, param/buffer) with consistent names across PP/EP. Adjusts indices for
    virtual PP + EP offsets. Handles decoder.layers, mtp.layers (Multi-Token Prediction), expert_bias.
    """
    ep_size = mpu.get_expert_model_parallel_world_size()
    ep_rank = mpu.get_expert_model_parallel_rank()
    if args.num_experts:
        expert_offset = ep_rank * args.num_experts // ep_size

    sig = inspect.signature(get_transformer_layer_offset)
    need_vp_stage = "vp_stage" in sig.parameters

    for vp_stage, model_module in enumerate(model):
        if need_vp_stage:
            layer_offset = get_transformer_layer_offset(model_module.config, vp_stage)
        else:
            layer_offset = get_transformer_layer_offset(model_module.config)
        for name, param in model_module.named_parameters():
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
                match = re.match(mtp_layers_pattern, name)
                if not match:
                    yield name, param
                    continue

                # MTP layer indices start from 0
                layer_idx, rest = match.groups()
                expert_pattern = r"transformer_layer\.mlp\.experts\.(.+)\.(weight|bias)(\d+)"
                match = re.match(expert_pattern, rest)
                if not match:
                    yield name, param
                    continue

                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.{param_type}{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"module.module.decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param
            else:
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers like Megatron Bridge
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name

            decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"module.module.decoder.layers.{layer_idx}.{rest}", buffer
