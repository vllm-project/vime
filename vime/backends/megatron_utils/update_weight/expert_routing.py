import logging
import re
from argparse import Namespace
from collections import defaultdict
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

import torch.distributed as dist

from vime.utils.distributed_utils import get_gloo_group
from vime.utils.types import ParamInfo

__all__ = ["configure_expert_routing"]


logger = logging.getLogger(__name__)

_ROUTED_EXPERT = re.compile(r"module\.module\.decoder\.layers\.(\d+)\.mlp\.experts\.linear_fc([12])\.weight(\d+)")


@dataclass(frozen=True)
class _ExpertParam:
    info: ParamInfo
    layer: int
    expert: int
    target_ranks: tuple[int, ...]


@dataclass(frozen=True)
class _ExpertTransfer:
    source_rank: int
    target_ranks: tuple[int, ...]
    params: tuple[_ExpertParam, ...]


_ExpertTransferBatch = tuple[_ExpertTransfer, ...]
_ExpertTransferGroup = tuple[_ExpertTransferBatch, ...]


@dataclass(frozen=True)
class _VLLMMoeTopology:
    tp_size: int
    pp_size: int
    pcp_size: int
    dp_size: int
    enable_expert_parallel: bool
    ep_size: int


def _config_value(
    parallel_config: Mapping[str, Any] | None,
    key: str,
    default: Any,
) -> Any:
    if parallel_config is None:
        return default
    return parallel_config.get(key, parallel_config.get(key.replace("_", "-"), default))


def _get_vllm_moe_topology(
    args: Namespace,
    engine_gpu_count: int,
    parallel_config: Mapping[str, Any] | None = None,
) -> _VLLMMoeTopology:
    pp_size = int(_config_value(parallel_config, "pp_size", getattr(args, "vllm_pp_size", 1)) or 1)
    pcp_size = int(
        _config_value(
            parallel_config,
            "pcp_size",
            getattr(args, "vllm_prefill_context_parallel_size", 1),
        )
        or 1
    )
    dp_size = int(_config_value(parallel_config, "dp_size", getattr(args, "vllm_dp_size", 1)) or 1)
    parallel_divisor = pp_size * pcp_size * dp_size
    if engine_gpu_count % parallel_divisor:
        raise ValueError(
            f"VLLM engine GPU count {engine_gpu_count} is not divisible by PP*PCP*DP "
            f"({pp_size}*{pcp_size}*{dp_size})"
        )
    default_tp_size = engine_gpu_count // parallel_divisor
    tp_size = int(_config_value(parallel_config, "tp_size", default_tp_size) or default_tp_size)
    if tp_size * parallel_divisor != engine_gpu_count:
        raise ValueError(
            f"VLLM engine GPU count {engine_gpu_count} does not match TP*PP*PCP*DP "
            f"({tp_size}*{pp_size}*{pcp_size}*{dp_size})"
        )
    enable_expert_parallel = bool(
        _config_value(
            parallel_config,
            "enable_expert_parallel",
            getattr(args, "vllm_enable_expert_parallel", False),
        )
    )
    ep_size = tp_size * pcp_size * dp_size if enable_expert_parallel else 1

    return _VLLMMoeTopology(
        tp_size=tp_size,
        pp_size=pp_size,
        pcp_size=pcp_size,
        dp_size=dp_size,
        enable_expert_parallel=enable_expert_parallel,
        ep_size=ep_size,
    )


def _vllm_topology_signature(topology: _VLLMMoeTopology) -> tuple[int, int, int, int, bool, int]:
    return (
        topology.tp_size,
        topology.pp_size,
        topology.pcp_size,
        topology.dp_size,
        topology.enable_expert_parallel,
        topology.ep_size,
    )


def _get_homogeneous_vllm_moe_topology(
    args: Namespace,
    engine_gpu_counts: Sequence[int],
    engine_parallel_configs: Sequence[Mapping[str, Any]] | None,
) -> _VLLMMoeTopology:
    if engine_parallel_configs is None:
        return _get_vllm_moe_topology(args, engine_gpu_count=engine_gpu_counts[0])
    if len(engine_parallel_configs) != len(engine_gpu_counts):
        raise ValueError(
            f"VLLM engine parallel config count {len(engine_parallel_configs)} "
            f"!= engine count {len(engine_gpu_counts)}"
        )

    topologies = [
        _get_vllm_moe_topology(args, engine_gpu_count=gpu_count, parallel_config=parallel_config)
        for gpu_count, parallel_config in zip(engine_gpu_counts, engine_parallel_configs, strict=True)
    ]
    signatures = {_vllm_topology_signature(topology) for topology in topologies}
    if len(signatures) != 1:
        raise ValueError(f"VLLM engines have heterogeneous parallel topology: {sorted(signatures)}")
    return topologies[0]


def _can_route_experts(
    args: Namespace,
    vllm_moe_topology: _VLLMMoeTopology,
    engine_gpu_counts: Sequence[int],
) -> bool:
    from megatron.core import mpu

    eplb_config = getattr(args, "vllm_eplb_config", None)
    if isinstance(eplb_config, Mapping):
        num_redundant_experts = eplb_config.get("num_redundant_experts", 0)
    else:
        num_redundant_experts = getattr(eplb_config, "num_redundant_experts", 0)

    return (
        vllm_moe_topology.pp_size == 1
        and vllm_moe_topology.enable_expert_parallel
        and vllm_moe_topology.ep_size > 1
        and not getattr(args, "vllm_enable_eplb", False)
        and num_redundant_experts == 0
        and getattr(args, "vllm_expert_placement_strategy", "linear") == "linear"
        and not getattr(args, "vllm_enable_elastic_ep", False)
        and mpu.get_expert_tensor_parallel_world_size() == 1
        and _vllm_moe_tp_is_one(engine_gpu_counts, vllm_moe_topology)
    )


def _vllm_moe_tp_is_one(
    engine_gpu_counts: Sequence[int],
    topology: _VLLMMoeTopology,
) -> bool:
    """Return whether each VLLM engine has no tensor parallelism inside experts."""
    if topology.pp_size != 1:
        return False
    expected_size = topology.pp_size * topology.ep_size
    return all(gpu_count == expected_size for gpu_count in engine_gpu_counts)


def _get_expert_target_ranks(
    engine_gpu_counts: Sequence[int],
    engine_gpu_offsets: Sequence[int],
    *,
    ep_size: int,
    world_size: int,
) -> tuple[tuple[int, ...], ...]:
    """Map each EP shard to the corresponding colocated rank."""
    expected_size = ep_size
    targets = [[] for _ in range(ep_size)]
    for gpu_count, gpu_offset in zip(engine_gpu_counts, engine_gpu_offsets, strict=True):
        if gpu_count != expected_size:
            raise ValueError(f"VLLM MoE TP must be 1, got engine_size={gpu_count}, EP={ep_size}")
        if gpu_offset < 0 or gpu_offset + gpu_count > world_size:
            raise ValueError("VLLM engine is outside the Megatron world")
        for ep_rank in range(ep_size):
            targets[ep_rank].append(gpu_offset + ep_rank)
    return tuple(tuple(ranks) for ranks in targets)


def _build_expert_params(
    infos: Sequence[ParamInfo],
    target_ranks: Sequence[Sequence[int]],
    *,
    num_experts: int,
) -> list[_ExpertParam]:
    ep_size = len(target_ranks)
    if num_experts % ep_size:
        raise ValueError("num_experts must be divisible by VLLM EP")
    experts_per_rank = num_experts // ep_size
    coverage: dict[int, set[tuple[int, int]]] = defaultdict(set)
    params = []
    for info in infos:
        layer, projection, expert = map(int, _ROUTED_EXPERT.fullmatch(info.name).groups())
        if not 0 <= expert < num_experts:
            raise ValueError(f"invalid expert id {expert} in {info.name}")
        ep_rank = expert // experts_per_rank
        coverage[layer].add((expert, projection))
        params.append(
            _ExpertParam(
                info=info,
                layer=layer,
                expert=expert,
                target_ranks=tuple(target_ranks[ep_rank]),
            )
        )

    expected = {(expert, projection) for expert in range(num_experts) for projection in (1, 2)}
    if not coverage or any(found != expected for found in coverage.values()):
        raise ValueError("routed-expert metadata is incomplete")
    return sorted(params, key=lambda param: (param.layer, param.info.name))


def _set_expert_source_ranks(
    infos: Sequence[ParamInfo],
    local_names_by_rank: Sequence[Sequence[str]],
) -> list[ParamInfo]:
    owners = {}
    for rank, names in enumerate(local_names_by_rank):
        for name in names:
            owners.setdefault(name, rank)
    missing = [info.name for info in infos if info.name not in owners]
    if missing:
        raise ValueError(f"no physical owner for {missing[0]}")
    return [replace(info, src_rank=owners[info.name]) for info in infos]


def _resolve_expert_source_ranks(
    infos: Sequence[ParamInfo],
    get_local_weight_names: Callable[[], Iterable[str]],
) -> list[ParamInfo]:
    local_expert_names = tuple(name for name in get_local_weight_names() if _ROUTED_EXPERT.fullmatch(name))
    local_names_by_rank = [None] * dist.get_world_size()
    dist.all_gather_object(local_names_by_rank, local_expert_names, group=get_gloo_group())
    return _set_expert_source_ranks(infos, local_names_by_rank)


def _build_expert_transfer_plan(
    params: Sequence[_ExpertParam],
    buffer_size: int,
) -> list[_ExpertTransferGroup]:
    """Build expert transfer groups with pre-packed, rank-bounded transfer batches."""
    if buffer_size <= 0:
        raise ValueError("update_weight_buffer_size must be positive")

    params_by_transfer: dict[tuple[int, int, tuple[int, ...], int], list[_ExpertParam]] = defaultdict(list)
    for param in params:
        params_by_transfer[(param.layer, param.expert, param.target_ranks, param.info.src_rank)].append(param)

    by_layer: dict[int, list[_ExpertTransfer]] = defaultdict(list)
    for (layer, _expert, target_ranks, source_rank), transfer_params in params_by_transfer.items():
        transfer = _ExpertTransfer(
            source_rank=source_rank,
            target_ranks=target_ranks,
            params=tuple(sorted(transfer_params, key=lambda param: (param.expert, param.info.name))),
        )
        by_layer[layer].append(transfer)

    transfer_plan = []
    for layer in sorted(by_layer):
        transfer_group = tuple(
            sorted(by_layer[layer], key=lambda transfer: (transfer.target_ranks, transfer.source_rank))
        )
        transfer_plan.append(tuple(_pack_expert_transfer_batches(transfer_group, buffer_size)))
    return transfer_plan


def _expert_transfer_size(transfer: _ExpertTransfer) -> int:
    return sum(param.info.size for param in transfer.params)


def _pack_expert_transfer_batches(
    transfers: Sequence[_ExpertTransfer],
    buffer_size: int,
) -> list[_ExpertTransferBatch]:
    """First-fit transfers while capping per-rank staging bytes."""
    sized_transfers = sorted(
        ((_expert_transfer_size(transfer), transfer) for transfer in transfers),
        key=lambda item: (-item[0], item[1].target_ranks, item[1].source_rank),
    )
    if buffer_size < sized_transfers[0][0]:
        raise ValueError("one source-to-target expert transfer bundle exceeds update_weight_buffer_size")

    batches: list[list[_ExpertTransfer]] = []
    batch_costs: list[dict[int, int]] = []
    for size, transfer in sized_transfers:
        participants = set(transfer.target_ranks) | {transfer.source_rank}
        candidates = [
            index
            for index, costs in enumerate(batch_costs)
            if all(costs.get(rank, 0) + size <= buffer_size for rank in participants)
        ]
        if candidates:
            batch_index = min(candidates, key=lambda index: (sum(batch_costs[index].values()), index))
        else:
            batch_index = len(batches)
            batches.append([])
            batch_costs.append({})

        batches[batch_index].append(transfer)
        for rank in participants:
            batch_costs[batch_index][rank] = batch_costs[batch_index].get(rank, 0) + size

    return [tuple(batch) for batch in batches]


def _log_disabled_expert_routing(reason: str) -> None:
    if dist.get_rank() == 0:
        logger.info("Disable rank-local expert update: %s", reason)


def configure_expert_routing(
    *,
    args: Namespace,
    full_param_info_buckets: Sequence[Sequence[ParamInfo]] | None,
    get_local_weight_names: Callable[[], Iterable[str]],
    engine_gpu_counts: Sequence[int],
    engine_gpu_offsets: Sequence[int],
    engine_parallel_configs: Sequence[Mapping[str, Any]] | None,
    use_distribute: bool,
) -> tuple[list[list[ParamInfo]] | None, list[_ExpertTransferGroup]]:
    if full_param_info_buckets is None:
        return None, []

    if use_distribute:
        _log_disabled_expert_routing("distributed VLLM engines are present")
        return None, []
    if not engine_gpu_counts:
        _log_disabled_expert_routing("no colocated VLLM engines")
        return None, []

    try:
        vllm_moe_topology = _get_homogeneous_vllm_moe_topology(
            args,
            engine_gpu_counts,
            engine_parallel_configs,
        )
    except (AttributeError, TypeError, ValueError) as exc:
        _log_disabled_expert_routing(str(exc))
        return None, []

    if not _can_route_experts(
        args,
        vllm_moe_topology,
        engine_gpu_counts=engine_gpu_counts,
    ):
        _log_disabled_expert_routing("VLLM/Megatron expert topology is not eligible")
        return None, []
    dense_infos = []
    expert_infos = []
    for bucket in full_param_info_buckets:
        for info in bucket:
            (expert_infos if _ROUTED_EXPERT.fullmatch(info.name) else dense_infos).append(info)
    if not expert_infos:
        return None, []

    try:
        from megatron.core import mpu

        from .hf_weight_iterator_direct import pack_param_info_buckets

        expert_infos = _resolve_expert_source_ranks(expert_infos, get_local_weight_names)
        target_ranks = _get_expert_target_ranks(
            engine_gpu_counts,
            engine_gpu_offsets,
            ep_size=vllm_moe_topology.ep_size,
            world_size=dist.get_world_size(),
        )
        expert_params = _build_expert_params(
            expert_infos,
            target_ranks,
            num_experts=args.num_experts,
        )
        buffer_size = args.update_weight_buffer_size
        expert_transfer_plan = _build_expert_transfer_plan(expert_params, buffer_size)
        expert_transfer_batches = sum(len(group) for group in expert_transfer_plan)
        dense_buckets = pack_param_info_buckets(dense_infos, buffer_size)
    except (AttributeError, TypeError, ValueError) as exc:
        _log_disabled_expert_routing(str(exc))
        return None, []

    if dist.get_rank() == 0:
        logger.info(
            "Enabled rank-local expert update: Megatron PP=%d EP=%d, VLLM EP=%d, "
            "%d -> %d transfer groups (%d dense + %d expert, %d expert transfer batches)",
            mpu.get_pipeline_model_parallel_world_size(),
            mpu.get_expert_model_parallel_world_size(),
            vllm_moe_topology.ep_size,
            len(full_param_info_buckets),
            len(dense_buckets) + len(expert_transfer_plan),
            len(dense_buckets),
            len(expert_transfer_plan),
            expert_transfer_batches,
        )
    return dense_buckets, expert_transfer_plan
