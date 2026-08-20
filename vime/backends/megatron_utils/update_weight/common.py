import inspect
import re
import socket
from argparse import Namespace
from collections.abc import Callable, Iterator, Mapping, Sequence
from typing import Any

import torch
import torch.distributed as dist
from megatron.core import mpu
from megatron.core.transformer.transformer_layer import get_transformer_layer_offset

from vime.backends.megatron_utils.misc_utils import strip_param_name_prefix
from vime.utils.distributed_utils import get_gloo_group
from vime.utils.types import ParamInfo


def all_gather_param(name: str, param: torch.nn.Parameter) -> torch.Tensor:
    """
    All-gather TP-sharded param to full tensor. expert_bias→param,
    non-TP/duplicated/TP-size-1→param.data.
    Uses expert-TP for ".experts.", else regular-TP. linear_fc1 rechunked (GLU), linear_fc2 dim fix.
    """
    if "expert_bias" in name:
        return param

    assert hasattr(param, "tensor_model_parallel"), f"{name} does not have tensor_model_parallel attribute"
    if not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
        return param.data

    if ".experts." in name:
        tp_size = mpu.get_expert_tensor_parallel_world_size()
    else:
        tp_size = mpu.get_tensor_model_parallel_world_size()

    if tp_size == 1:
        return param.data

    if ".experts." in name:
        tp_group = mpu.get_expert_tensor_parallel_group()
    else:
        tp_group = mpu.get_tensor_model_parallel_group()

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
    Parallel TP all-gather for multiple params. Loop 1: for each TP param, allocate buffers +
    dist.all_gather(async_op=True) on expert-TP/regular-TP group
    (skip expert_bias/non-TP/duplicated/TP-size-1).
    Loop 2: wait all NCCL handles (enables overlap). Loop 3: concat partitions + apply GLU rechunk/MoE dim fix.
    """
    # Phase 1: Start all async all_gather operations
    gather_tasks = []
    handles = []

    for info, param in param_infos_and_params:
        # Prepare async all_gather
        if "expert_bias" in info.name:
            gather_tasks.append((info, param, None, None, None))
        elif not param.tensor_model_parallel or getattr(param, "parallel_mode", None) == "duplicated":
            gather_tasks.append((info, param.data, None, None, None))
        else:
            # Start async all_gather
            if ".experts." in info.name:
                tp_size = mpu.get_expert_tensor_parallel_world_size()
            else:
                tp_size = mpu.get_tensor_model_parallel_world_size()

            if tp_size == 1:
                gather_tasks.append((info, param.data, None, None, None))
                continue

            if ".experts." in info.name:
                tp_group = mpu.get_expert_tensor_parallel_group()
            else:
                tp_group = mpu.get_tensor_model_parallel_group()

            param_partitions = [torch.empty_like(param.data) for _ in range(tp_size)]
            handle = dist.all_gather(param_partitions, param.data, group=tp_group, async_op=True)
            gather_tasks.append((info, None, handle, param_partitions, param.partition_dim))
            handles.append(handle)

    # Phase 2: Wait for ALL async operations to complete at once
    # This ensures maximum parallelism by not blocking on individual operations
    for handle in handles:
        handle.wait()

    # Phase 3: Process all results after all communications are done
    gathered_params = []
    for info, direct_param, handle, param_partitions, partition_dim in gather_tasks:
        if handle is None:
            # No all_gather needed
            param = direct_param
        else:
            # Process the gathered partitions (same logic as original all_gather_param)
            assert partition_dim is not None, "partition_stride != 1 is not supported"
            # TODO: here we did an extra copy during concat, maybe merge this with convert_to_hf is better?
            # TODO: check only GLU is used.
            if "linear_fc1.weight" in info.name or "linear_fc1.bias" in info.name:
                param_partitions = [p.chunk(2, dim=0) for p in param_partitions]
                param_partitions = [p[0] for p in param_partitions] + [p[1] for p in param_partitions]
            # this is bug in megatron's grouped moe.
            if "linear_fc2.weight" in info.name:
                if partition_dim == 0:
                    partition_dim = 1
            param = torch.cat(param_partitions, dim=partition_dim)

        gathered_params.append(param)

    return gathered_params


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
            # TODO shall we handle (almost) all buffers
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
            prefix = "module.module.language_model." if ".language_model." in name else "module.module."

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                # MTP (Multi-Token Prediction) layers for speculative decoding
                mtp_layers_pattern = r"module\.module\.(?:language_model\.)?mtp\.layers\.(\d+)\.(.+)"
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
                yield f"{prefix}mtp.layers.{layer_idx}.transformer_layer.mlp.experts.{rest}.{param_type}{expert_idx}", param
                continue

            layer_idx, rest = match.groups()
            layer_idx = int(layer_idx) + layer_offset

            # this is hardcoded for te grouped matmul
            expert_pattern = r"mlp\.experts\.(.+)\.(weight|bias)(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, param_type, expert_idx = match.groups()
                expert_idx = int(expert_idx) + expert_offset
                yield f"{prefix}decoder.layers.{layer_idx}.mlp.experts.{rest}.{param_type}{expert_idx}", param
            else:
                yield f"{prefix}decoder.layers.{layer_idx}.{rest}", param

        # treat expert bias as normal parameters
        for name, buffer in model_module.named_buffers():
            # TODO shall we handle (almost) all buffers
            if "expert_bias" not in name:
                continue
            # for model without ddp wrap
            if not name.startswith("module.module."):
                name = "module." + name
            prefix = "module.module.language_model." if ".language_model." in name else "module.module."

            decoder_layers_pattern = r"module\.module\.(?:language_model\.)?decoder\.layers\.(\d+)\.(.+)"
            match = re.match(decoder_layers_pattern, name)
            if not match:
                yield name, buffer
            else:
                layer_idx, rest = match.groups()
                layer_idx = int(layer_idx) + layer_offset
                yield f"{prefix}decoder.layers.{layer_idx}.{rest}", buffer


class HfWeightSource:
    def __init__(self, iterator, weights_getter: Callable[[], Mapping[str, torch.Tensor]]) -> None:
        self.iterator = iterator
        self.weights_getter = weights_getter
        self._metadata = None

    def metadata(self):
        if self._metadata is None:
            from vllm.distributed.weight_transfer.base import ParamMeta

            self._metadata = [ParamMeta(name, tensor.dtype, tuple(tensor.shape)) for name, tensor in self]
        return self._metadata

    def __iter__(self):
        for chunk in self.iterator.get_hf_weight_chunks(self.weights_getter()):
            yield from chunk


class VimeRayWeightSyncClient:
    def __init__(
        self,
        engines: Sequence[Any],
        version_getter: Callable[[], int],
        engine_gpu_counts: Sequence[int] | None = None,
    ) -> None:
        self.engines = list(engines)
        self.version_getter = version_getter
        self.engine_gpu_counts = engine_gpu_counts
        self.draft = False

    def init_weight_transfer_engine(self, init_info: dict[str, Any]) -> None:
        import ray

        refs = []
        rank_offset = 1
        for index, engine in enumerate(self.engines):
            engine_info = dict(init_info)
            if self.engine_gpu_counts is not None:
                engine_info["rank_offset"] = rank_offset
                rank_offset += self.engine_gpu_counts[index]
            refs.append(engine.init_weight_transfer_engine.remote({"init_info": engine_info}))
        ray.get(refs)

    def start_weight_update(self) -> None:
        import ray

        method = "start_draft_weight_update" if self.draft else "start_weight_update"
        ray.get([getattr(engine, method).remote() for engine in self.engines])

    def update_weights(self, update_info: dict[str, Any] | list[dict[str, Any] | None]) -> None:
        import ray

        ray.get([engine.update_weights.remote(update_info) for engine in self.engines])

    def finish_weight_update(self, weight_version: str | None = None) -> None:
        import ray

        version = str(self.version_getter()) if weight_version is None else str(weight_version)
        ray.get([engine.finish_weight_update.remote(weight_version=version) for engine in self.engines])


def create_nccl_trainer(
    client: VimeRayWeightSyncClient,
    source: HfWeightSource,
    engine_gpu_counts: Sequence[int],
):
    import ray
    from vllm.distributed.weight_transfer.factory import WeightTransferTrainerFactory
    from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerInitInfo

    rendezvous = [None]
    if dist.get_rank() == 0:
        with socket.socket() as sock:
            sock.bind(("", 0))
            rendezvous[0] = (ray._private.services.get_node_ip_address(), sock.getsockname()[1])
    dist.broadcast_object_list(rendezvous, src=0, group=get_gloo_group())
    master_address, master_port = rendezvous[0]
    return WeightTransferTrainerFactory.trainer_init(
        NCCLTrainerInitInfo(
            master_address=master_address,
            master_port=master_port,
            world_size=sum(engine_gpu_counts) + 1,
            rank=dist.get_rank(),
            packed_num_buffers=1,
        ),
        client=client,
        source=source,
    )
