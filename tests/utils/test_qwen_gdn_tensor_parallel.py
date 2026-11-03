"""CPU layout/gradient contracts; fused NPU kernels are covered by the NPU E2E."""

import importlib
import sys
from types import ModuleType, SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from vime.utils.gdn_tp import merge_gdn_qkv, shard_gdn_qkv


@pytest.mark.parametrize("world", [1, 2, 4])
@pytest.mark.parametrize("shape", [(40,), (40, 7), (40, 1, 4)])
def test_qkv_shards_preserve_unequal_segment_order(world, shape):
    tensor = torch.arange(torch.tensor(shape).prod()).reshape(shape)
    sizes = (8, 8, 24)
    shards = [shard_gdn_qkv(tensor, sizes, world, rank) for rank in range(world)]
    for rank, shard in enumerate(shards):
        expected = torch.cat(
            [
                tensor[start + rank * size // world : start + (rank + 1) * size // world]
                for start, size in zip((0, 8, 16), sizes, strict=True)
            ]
        )
        assert torch.equal(shard, expected)
    assert torch.equal(merge_gdn_qkv(shards, sizes), tensor)


def test_qkv_shards_reject_indivisible_heads():
    with pytest.raises(ValueError, match="divide TP4"):
        shard_gdn_qkv(torch.zeros(14, 2), (4, 4, 6), 4, 0)


class _Convolution(nn.Module):
    def __init__(self, hidden_size, kernel_size, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(hidden_size, kernel_size))

    def forward(self, x, cu_seqlens):
        parts = []
        for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
            part = F.conv1d(
                x[:, start:end].transpose(1, 2),
                self.weight[:, None],
                padding=self.weight.shape[-1] - 1,
                groups=self.weight.shape[0],
            )[:, :, : end - start]
            parts.append(F.silu(part.transpose(1, 2)))
        return torch.cat(parts, dim=1), None


class _GatedNorm(nn.Module):
    def __init__(self, hidden_size, eps, **kwargs):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x, z):
        return x * torch.rsqrt(x.square().mean(-1, keepdim=True) + self.eps) * self.weight * F.silu(z)


def _delta_rule(q, k, v, *, g, beta, cu_seqlens, **kwargs):
    """Small differentiable, head-local recurrence, independent of the FLA kernel."""
    q, k = F.normalize(q, dim=-1), F.normalize(k, dim=-1)
    outputs = []
    for start, end in zip(cu_seqlens[:-1], cu_seqlens[1:], strict=True):
        state = q.new_zeros(q.shape[0], q.shape[2], q.shape[3], v.shape[3])
        for i in range(start, end):
            state = state * g[:, i].exp()[..., None, None]
            residual = (v[:, i] - torch.einsum("bhkv,bhk->bhv", state, k[:, i])) * beta[:, i, :, None]
            state = state + k[:, i, :, :, None] * residual[:, :, None, :]
            outputs.append(torch.einsum("bhk,bhkv->bhv", q[:, i], state))
    return torch.stack(outputs, dim=1), None


@pytest.fixture
def model_module(monkeypatch):
    state = {"world": 2, "rank": 0, "cp": 1, "calls": []}

    def module(name, **attrs):
        value = ModuleType(name)
        value.__path__ = []
        value.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, value)
        return value

    class MegatronModule(nn.Module):
        def __init__(self, config):
            super().__init__()
            self.config = config

    mpu = SimpleNamespace(
        get_tensor_model_parallel_world_size=lambda: state["world"],
        get_tensor_model_parallel_rank=lambda: state["rank"],
        get_tensor_model_parallel_group=lambda: None,
        get_context_parallel_world_size=lambda: state["cp"],
        get_data_parallel_group=lambda **kwargs: None,
    )
    tp = SimpleNamespace()
    for name in (
        "gather_from_sequence_parallel_region",
        "scatter_to_sequence_parallel_region",
        "reduce_scatter_to_sequence_parallel_region",
        "copy_to_tensor_model_parallel_region",
        "reduce_from_tensor_model_parallel_region",
    ):

        def record(x, _name=name, **kwargs):
            state["calls"].append((_name, kwargs))
            return x

        setattr(tp, name, record)
    for name in [
        "megatron",
        "megatron.core.models",
        "megatron.core.models.gpt",
        "megatron.core.inference",
        "megatron.core.transformer",
    ]:
        module(name)
    module("megatron.core", mpu=mpu, tensor_parallel=tp)
    module("megatron.core.models.gpt.gpt_layer_specs", get_gpt_decoder_block_spec=lambda **kwargs: None)
    module("megatron.core.inference.contexts", BaseInferenceContext=object)
    module("megatron.core.packed_seq_params", PackedSeqParams=object)
    module("megatron.core.transformer.module", MegatronModule=MegatronModule)
    module("megatron.core.transformer.spec_utils", ModuleSpec=object)
    module("megatron.core.transformer.transformer_block", get_num_layers_to_build=lambda *args, **kwargs: 0)
    module("megatron.core.transformer.transformer_layer", get_transformer_layer_offset=lambda *args, **kwargs: 0)
    module("transformers")
    module("transformers.activations", ACT2FN={"silu": F.silu})
    # Exercise the existing native torch RMSNorm fallback without importing HF kernels.
    monkeypatch.setitem(sys.modules, "transformers.models.qwen3_next.modeling_qwen3_next", None)
    package = importlib.import_module("vime_plugins.models")
    for name in ["vime_plugins.models.hf_attention", "vime_plugins.models.qwen3_5"]:
        monkeypatch.setattr(package, name.rsplit(".", 1)[-1], None, raising=False)
        monkeypatch.delitem(sys.modules, name, raising=False)
    attention = importlib.import_module("vime_plugins.models.hf_attention")
    model = importlib.import_module("vime_plugins.models.qwen3_5")
    config = SimpleNamespace(
        hidden_size=16,
        linear_num_key_heads=4,
        linear_num_value_heads=8,
        linear_key_head_dim=2,
        linear_value_head_dim=3,
        linear_conv_kernel_dim=3,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        dtype=torch.float64,
    )
    monkeypatch.setattr(attention, "_load_hf_config", lambda path: config)
    monkeypatch.setattr(model, "ShortConvolution", _Convolution, raising=False)
    monkeypatch.setattr(model, "FusedRMSNormGated", _GatedNorm, raising=False)
    monkeypatch.setattr(model, "get_chunk_gated_delta_rule", lambda backend: _delta_rule)
    monkeypatch.setattr(model.accelerator, "current_device", lambda: "cpu")
    yield model, config, state
    # Remove newly imported modules before restoring the original optional stack.
    sys.modules.pop("vime_plugins.models.qwen3_5", None)
    sys.modules.pop("vime_plugins.models.hf_attention", None)


def _shard(parameter, full, world, rank):
    sizes = getattr(parameter, "tensor_parallel_split_sizes", None)
    if sizes is not None:
        return shard_gdn_qkv(full, sizes, world, rank)
    if getattr(parameter, "tensor_model_parallel", False):
        return full.chunk(world, dim=parameter.partition_dim)[rank]
    return full


@pytest.mark.parametrize("sp", [False, True])
def test_head_tp_matches_replicated_forward_and_all_parameter_gradients(model_module, monkeypatch, sp):
    model, config, state = model_module
    torch.manual_seed(123)
    args = SimpleNamespace(hf_checkpoint="unused", qwen_gdn_tensor_parallel=False, sequence_parallel=sp)
    reference = model.Attention(args, config, layer_number=1).double()
    expected = dict(reference.named_parameters())
    candidates = []
    for rank in range(state["world"]):
        state["rank"] = rank
        args = SimpleNamespace(hf_checkpoint="unused", qwen_gdn_tensor_parallel=True, sequence_parallel=sp)
        candidate = model.Attention(args, config, layer_number=1).double()
        with torch.no_grad():
            for name, parameter in candidate.named_parameters():
                parameter.copy_(_shard(parameter, expected[name], state["world"], rank))
        candidates.append(candidate)

    hook_calls = []
    monkeypatch.setattr(torch.distributed, "all_reduce", lambda grad, **kwargs: hook_calls.append(grad.shape))
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 3, 7], dtype=torch.int32))
    x = torch.randn(1, 7, config.hidden_size, dtype=torch.float64)
    reference_x = x.clone().requires_grad_()
    candidate_x = x.clone().requires_grad_()
    expected_output = reference.hf_forward(reference_x, packed)
    # Sum the head-local row-projection results, as the TP output collective does.
    output = sum(candidate.hf_forward(candidate_x, packed) for candidate in candidates)
    torch.testing.assert_close(output, expected_output, atol=1e-6, rtol=1e-5)
    gradient = torch.randn_like(output)
    expected_output.backward(gradient)
    output.backward(gradient)
    torch.testing.assert_close(candidate_x.grad, reference_x.grad, atol=1e-6, rtol=1e-5)

    shared = ("input_layernorm.weight", "linear_attn.norm.weight")
    for name in shared:
        parameters = [dict(candidate.named_parameters())[name] for candidate in candidates]
        # This is the shared-parameter sum performed by Megatron's SP finalizer
        # (or by the registered non-SP all-reduce hooks).
        reduced = sum(parameter.grad for parameter in parameters)
        torch.testing.assert_close(reduced, expected[name].grad, atol=1e-6, rtol=1e-5)
        assert all(getattr(parameter, "sequence_parallel", False) == sp for parameter in parameters)
    assert len(hook_calls) == (0 if sp else 2 * state["world"])
    for rank, candidate in enumerate(candidates):
        for name, parameter in candidate.named_parameters():
            if name not in shared:
                torch.testing.assert_close(
                    parameter.grad,
                    _shard(parameter, expected[name].grad, state["world"], rank),
                    atol=1e-6,
                    rtol=1e-5,
                )


@pytest.mark.parametrize("sp", [False, True])
@pytest.mark.parametrize("enabled", [False, True])
def test_wrapper_selects_sharded_or_duplicated_gradient_collectives(model_module, sp, enabled):
    model, config, state = model_module
    args = SimpleNamespace(hf_checkpoint="unused", qwen_gdn_tensor_parallel=enabled, sequence_parallel=sp)
    layer = model.Attention(args, config, layer_number=1)
    packed = SimpleNamespace(cu_seqlens_q=torch.tensor([0, 7], dtype=torch.int32))
    output, bias = layer(torch.randn(7, 1, config.hidden_size), None, packed_seq_params=packed)
    assert output.shape == (7, 1, config.hidden_size) and bias is None
    if sp:
        assert state["calls"][0][1]["tensor_parallel_output_grad"] is enabled
        last = "reduce_scatter_to_sequence_parallel_region" if enabled else "scatter_to_sequence_parallel_region"
        assert state["calls"][-1][0] == last
    else:
        assert [name for name, _ in state["calls"]] == (
            ["copy_to_tensor_model_parallel_region", "reduce_from_tensor_model_parallel_region"] if enabled else []
        )


def test_head_tp_rejects_unvalidated_context_parallel_layout(model_module):
    model, config, state = model_module
    state["cp"] = 2
    with pytest.raises(ValueError, match="context parallel size 1"):
        model.Qwen3_5GatedDeltaNet(config, 0, args=SimpleNamespace(qwen_gdn_tensor_parallel=True))


def test_checkpoint_factory_keeps_qkv_segments_and_pipeline_offsets(model_module, monkeypatch):
    model, config, state = model_module
    exports = {}

    def make_shards(tensors, prefix, axis_map, offsets, **kwargs):
        exports["axes"] = axis_map
        return {
            prefix + key: SimpleNamespace(key=prefix + key, data=value, replica_id=0) for key, value in tensors.items()
        }

    class Factory:
        def __init__(self, key, data, build, merge, replica_id):
            self.key, self.data, self.build, self.merge, self.replica_id = key, data, build, merge, replica_id

    class Tensor:
        @staticmethod
        def from_rank_offsets(key, data, *offsets, **kwargs):
            return SimpleNamespace(key=key, data=data, offsets=offsets, **kwargs)

    for name, attrs in {
        "megatron.core.dist_checkpointing": {},
        "megatron.core.dist_checkpointing.mapping": {"ShardedTensor": Tensor, "ShardedTensorFactory": Factory},
        "megatron.core.transformer.utils": {"make_sharded_tensors_for_checkpoint": make_shards},
    }.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    layer = model.Qwen3_5GatedDeltaNet(
        config, 0, args=SimpleNamespace(qwen_gdn_tensor_parallel=True, sequence_parallel=True)
    )
    offsets = ((0, 1, 2),)
    saved = layer.sharded_state_dict("layer.", offsets, metadata={"dp_cp_group": None})
    assert exports["axes"]["out_proj.weight"] == 1
    assert "norm.weight" not in exports["axes"]
    for name in ["in_proj_qkv.weight", "conv1d.weight"]:
        factory = saved["layer." + name]
        parts = factory.build(factory.key, factory.data, factory.replica_id, None)
        assert [part.key for part in parts] == ["layer." + name + "." + label for label in ["q", "k", "v"]]
        assert [part.data.shape[0] for part in parts] == [4, 4, 12]
        assert all(part.offsets == (*offsets, (1, state["rank"], state["world"])) for part in parts)
        assert torch.equal(factory.merge([part.data for part in parts]), factory.data)
        with pytest.raises(ValueError, match="flattened GDN checkpoint shards"):
            factory.build(factory.key, factory.data, 0, slice(0, 1))
