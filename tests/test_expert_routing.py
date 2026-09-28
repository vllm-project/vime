import sys
import types
from argparse import Namespace
from dataclasses import replace

import pytest
import torch

from vime.backends.megatron_utils.update_weight.expert_routing import (
    _build_expert_transfer_plan,
    _can_route_experts,
    _expert_transfer_size,
    _ExpertParam,
    _get_expert_target_ranks,
    _get_vllm_moe_topology,
)
from vime.utils.types import ParamInfo

NUM_GPUS = 0


def _topology_args(**overrides):
    values = {
        "vllm_pp_size": 1,
        "vllm_prefill_context_parallel_size": 1,
        "vllm_dp_size": 1,
        "vllm_enable_expert_parallel": False,
        "vllm_enable_eplb": False,
        "vllm_eplb_config": Namespace(num_redundant_experts=0),
        "vllm_expert_placement_strategy": "linear",
        "vllm_enable_elastic_ep": False,
    }
    values.update(overrides)
    return Namespace(**values)


def test_vllm_moe_topology_derives_ep_from_data_parallelism():
    topology = _get_vllm_moe_topology(
        _topology_args(vllm_dp_size=64, vllm_enable_expert_parallel=True),
        engine_gpu_count=64,
    )

    assert (topology.tp_size, topology.pp_size, topology.pcp_size, topology.dp_size) == (1, 1, 1, 64)
    assert topology.enable_expert_parallel is True
    assert topology.ep_size == 64


def test_vllm_moe_topology_includes_pcp_and_tp_in_ep_group():
    topology = _get_vllm_moe_topology(
        _topology_args(vllm_prefill_context_parallel_size=2, vllm_enable_expert_parallel=True),
        engine_gpu_count=4,
    )

    assert (topology.tp_size, topology.pcp_size, topology.dp_size) == (2, 2, 1)
    assert topology.ep_size == 4


def test_vllm_moe_topology_disables_expert_sharding_without_ep():
    topology = _get_vllm_moe_topology(_topology_args(), engine_gpu_count=4)

    assert topology.tp_size == 4
    assert topology.enable_expert_parallel is False
    assert topology.ep_size == 1


def test_expert_target_ranks_map_each_ep_shard_to_colocated_rank():
    assert _get_expert_target_ranks([4], [2], ep_size=4, world_size=8) == ((2,), (3,), (4,), (5,))


def _can_route_with_args(monkeypatch, **overrides):
    mpu = types.SimpleNamespace(get_expert_tensor_parallel_world_size=lambda: 1)
    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron.core = megatron_core
    megatron_core.mpu = mpu
    monkeypatch.setitem(sys.modules, "megatron", megatron)
    monkeypatch.setitem(sys.modules, "megatron.core", megatron_core)
    args = _topology_args(vllm_enable_expert_parallel=True, **overrides)
    topology = _get_vllm_moe_topology(args, engine_gpu_count=2)
    return _can_route_experts(args, topology, engine_gpu_counts=[2])


def test_vllm_rank_local_expert_routing_accepts_linear_static_ep(monkeypatch):
    assert _can_route_with_args(monkeypatch)


@pytest.mark.parametrize(
    "overrides",
    [
        {"vllm_expert_placement_strategy": "round_robin"},
        {"vllm_enable_eplb": True},
        {"vllm_eplb_config": Namespace(num_redundant_experts=1)},
        {"vllm_enable_elastic_ep": True},
    ],
    ids=["round-robin", "eplb", "redundant-experts", "elastic-ep"],
)
def test_vllm_rank_local_expert_routing_rejects_non_static_placement(monkeypatch, overrides):
    assert not _can_route_with_args(monkeypatch, **overrides)


def _param(*, expert: int, projection: int, source_rank: int, target_rank: int, size: int) -> _ExpertParam:
    info = ParamInfo(
        name=f"module.module.decoder.layers.3.mlp.experts.linear_fc{projection}.weight{expert}",
        dtype=torch.bfloat16,
        shape=torch.Size([size // 2]),
        attrs={},
        size=size,
        src_rank=source_rank,
    )
    return _ExpertParam(
        info=info,
        layer=3,
        expert=expert,
        target_ranks=(target_rank,),
    )


def test_transfer_plan_splits_same_rank_experts_at_expert_boundaries():
    params = []
    for expert, rank in ((0, 0), (1, 0), (2, 1), (3, 1)):
        params.extend(
            [
                _param(expert=expert, projection=1, source_rank=rank, target_rank=rank, size=60),
                _param(expert=expert, projection=2, source_rank=rank, target_rank=rank, size=60),
            ]
        )

    plan = _build_expert_transfer_plan(params, buffer_size=150)

    assert len(plan) == 1
    assert len(plan[0]) == 2
    transfers = [transfer for batch in plan[0] for transfer in batch]
    assert len(transfers) == 4
    assert all(_expert_transfer_size(transfer) == 120 for transfer in transfers)
    assert all(len({param.expert for param in transfer.params}) == 1 for transfer in transfers)
    assert {(param.expert, param.info.name) for transfer in transfers for param in transfer.params} == {
        (param.expert, param.info.name) for param in params
    }


def test_transfer_plan_rejects_one_expert_larger_than_buffer():
    first = _param(expert=0, projection=1, source_rank=0, target_rank=0, size=80)
    second = replace(
        first,
        info=replace(
            first.info,
            name="module.module.decoder.layers.3.mlp.experts.linear_fc2.weight0",
            size=80,
        ),
    )

    with pytest.raises(ValueError, match="exceeds update_weight_buffer_size"):
        _build_expert_transfer_plan([first, second], buffer_size=150)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
