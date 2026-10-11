"""Weight-gather layout and asynchronous completion contracts on real CPU tensors."""

import importlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

_tests_root = Path(__file__).resolve().parents[1]
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs

NUM_GPUS = 0
MODULE = "vime.backends.megatron_utils.update_weight.common"


@pytest.fixture(scope="module")
def common_module():
    names = (
        "megatron",
        "megatron.core",
        "megatron.core.parallel_state",
        "megatron.core.transformer",
        "megatron.core.transformer.transformer_layer",
        "ray",
        "ray.actor",
        "vime.utils.distributed_utils",
        MODULE,
    )
    saved = _unit_stubs.save_sys_modules(names)
    for name in names:
        sys.modules.pop(name, None)
    _unit_stubs.install_megatron_mpu_stub()
    _unit_stubs.install_ray_stub()
    _unit_stubs.install_vime_distributed_utils_stub()
    try:
        yield importlib.import_module(MODULE)
    finally:
        _unit_stubs.restore_sys_modules(saved)


def _parameter(data, *, is_cuda=True, partition_dim=0, partition_stride=1, parallel=True, duplicated=False):
    # Only the CUDA dispatch flag is synthetic; layout and values are real CPU
    # tensors. Real NCCL numerical and timing checks use the separate GPU run.
    return SimpleNamespace(
        data=data,
        shape=data.shape,
        is_cuda=is_cuda,
        is_contiguous=data.is_contiguous,
        partition_dim=partition_dim,
        partition_stride=partition_stride,
        tensor_model_parallel=parallel,
        parallel_mode="duplicated" if duplicated else None,
    )


@pytest.fixture
def collectives(common_module, monkeypatch):
    state = SimpleNamespace(calls=[], waits=[], backend="nccl", tp_size=2, etp_size=3)
    groups = {"tp": 2, "etp": 3}
    monkeypatch.setattr(
        common_module,
        "mpu",
        SimpleNamespace(
            get_tensor_model_parallel_world_size=lambda: state.tp_size,
            get_expert_tensor_parallel_world_size=lambda: state.etp_size,
            get_tensor_model_parallel_group=lambda: "tp",
            get_expert_tensor_parallel_group=lambda: "etp",
        ),
    )
    monkeypatch.setattr(common_module.dist, "get_backend", lambda group: state.backend)

    def complete(kind, output, data, group, async_op):
        state.calls.append((kind, group, async_op))

        def copy():
            for rank, tensor in enumerate(output):
                tensor.copy_(data + rank * 100)

        if async_op:
            return SimpleNamespace(wait=lambda: (state.waits.append(kind), copy()))
        copy()
        return None

    def into(output, data, *, group, async_op=False):
        return complete("into", output.chunk(groups[group], dim=0), data, group, async_op)

    def listed(output, data, *, group, async_op=False):
        return complete("list", output, data, group, async_op)

    monkeypatch.setattr(common_module.dist, "all_gather_into_tensor", into)
    monkeypatch.setattr(common_module.dist, "all_gather", listed)
    return state


@pytest.mark.unit
@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize(
    "name", ["embedding.weight", "attention.linear_qkv.weight", "attention.bias", "mlp.experts.projection.weight"]
)
def test_row_gather_preserves_rank_order_and_expert_group(common_module, collectives, name, async_op):
    data = torch.arange(24, dtype=torch.bfloat16).reshape(6, 4)
    parameter = _parameter(data)
    if async_op:
        (result,) = common_module.all_gather_params_async([(SimpleNamespace(name=name), parameter)])
    else:
        result = common_module.all_gather_param(name, parameter)
    size, group = (3, "etp") if ".experts." in name else (2, "tp")
    expected = torch.cat([data + rank * 100 for rank in range(size)], dim=0)
    assert torch.equal(result, expected)
    assert result.dtype == data.dtype and result.is_contiguous()
    assert collectives.calls == [("into", group, async_op)]
    assert collectives.waits == (["into"] if async_op else [])


@pytest.mark.unit
@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize(
    "case", ["cpu", "gloo", "noncontiguous", "column", "glu_weight", "glu_bias", "fc2", "strided_glu"]
)
def test_existing_layouts_keep_their_gather_and_reordering(common_module, collectives, case, async_op):
    name = "attention.projection.weight"
    data = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    options = {}
    dim, glu = 0, False
    if case == "cpu":
        options["is_cuda"] = False
    elif case == "gloo":
        collectives.backend = "gloo"
    elif case == "noncontiguous":
        data = data.T
    elif case == "column":
        options["partition_dim"] = dim = 1
    elif case in ("glu_weight", "glu_bias", "strided_glu"):
        name = "mlp.linear_fc1.bias" if case == "glu_bias" else "mlp.linear_fc1.weight"
        glu = True
        if case == "strided_glu":
            options["partition_stride"] = 2
    elif case == "fc2":
        name, dim = "mlp.experts.linear_fc2.weight", 1
    parameter = _parameter(data, **options)
    if async_op:
        (result,) = common_module.all_gather_params_async([(SimpleNamespace(name=name), parameter)])
    else:
        result = common_module.all_gather_param(name, parameter)
    size, group = (3, "etp") if ".experts." in name else (2, "tp")
    shards = [data + rank * 100 for rank in range(size)]
    if glu:
        pairs = [shard.chunk(2, dim=0) for shard in shards]
        shards = [pair[0] for pair in pairs] + [pair[1] for pair in pairs]
    assert torch.equal(result, torch.cat(shards, dim=dim))
    assert collectives.calls == [("list", group, async_op)]


@pytest.mark.unit
@pytest.mark.parametrize("async_op", [False, True])
@pytest.mark.parametrize("case", ["replica", "duplicated", "tp_one", "expert_bias"])
def test_replicated_parameters_do_not_allocate_or_communicate(common_module, collectives, case, async_op):
    data = torch.arange(12).reshape(3, 4)
    parameter = _parameter(data, parallel=case != "replica", duplicated=case == "duplicated")
    name = "mlp.expert_bias" if case == "expert_bias" else "attention.weight"
    if case == "tp_one":
        collectives.tp_size = 1
    if async_op:
        (result,) = common_module.all_gather_params_async([(SimpleNamespace(name=name), parameter)])
    else:
        result = common_module.all_gather_param(name, parameter)
    assert result is (parameter if case == "expert_bias" else data)
    assert collectives.calls == []


@pytest.mark.unit
def test_mixed_async_bucket_waits_for_every_output_and_preserves_order(common_module, collectives):
    data = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    names = ["attention.weight", "mlp.linear_fc1.weight", "norm.weight"]
    parameters = [_parameter(data), _parameter(data), _parameter(data, parallel=False)]
    results = common_module.all_gather_params_async(
        [(SimpleNamespace(name=name), parameter) for name, parameter in zip(names, parameters, strict=True)]
    )
    torch.testing.assert_close(results[0], torch.cat([data, data + 100]), rtol=0, atol=0)
    halves = [shard.chunk(2, dim=0) for shard in [data, data + 100]]
    torch.testing.assert_close(
        results[1], torch.cat([halves[0][0], halves[1][0], halves[0][1], halves[1][1]]), rtol=0, atol=0
    )
    assert results[2] is data
    assert collectives.waits == ["into", "list"]
