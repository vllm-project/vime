import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.hf_to_megatron import _LOADERS
from vime.backends.megatron_utils.hf_to_megatron.gemma4 import gemma4_hf_tensor
from vime.backends.megatron_utils.megatron_to_hf.gemma4 import _config_cache, _expert_buffers, convert_gemma4_to_hf

NUM_GPUS = 0

try:
    _has_megatron = importlib.util.find_spec("megatron.core") is not None
except ModuleNotFoundError:
    _has_megatron = False

requires_megatron = pytest.mark.skipif(not _has_megatron, reason="requires the Megatron runtime")


class Reader(dict):
    def get_tensor(self, name):
        return self[name]


def conversion_config(config):
    return {
        "global_attn_layers": {
            index for index, layer_type in enumerate(config.layer_types) if layer_type == "full_attention"
        },
        "local_head_dim": config.head_dim,
        "global_head_dim": config.global_head_dim,
        "num_attention_heads": config.num_attention_heads,
        "local_num_kv_heads": config.num_key_value_heads,
        "global_num_kv_heads": config.num_global_key_value_heads,
        "hidden_size": config.hidden_size,
        "num_experts": config.num_experts,
    }


@pytest.fixture
def config():
    return SimpleNamespace(
        model_type="gemma4_text",
        enable_moe_block=True,
        layer_types=["sliding_attention", "full_attention"],
        num_attention_heads=4,
        num_key_value_heads=2,
        num_global_key_value_heads=1,
        head_dim=2,
        global_head_dim=4,
        hidden_size=3,
        num_experts=2,
    )


@pytest.mark.unit
def test_registers_native_hf_loader():
    assert _LOADERS["gemma4_text"] is gemma4_hf_tensor


@pytest.mark.unit
def test_native_model_does_not_import_bridge():
    source = (Path(__file__).resolve().parents[2] / "vime_plugins/models/gemma4.py").read_text()

    assert "megatron.bridge" not in source
    assert "mbridge" not in source


@pytest.mark.unit
@requires_megatron
def test_native_model_extends_transformer_config():
    from megatron.core.transformer.transformer_config import TransformerConfig

    from vime_plugins.models.gemma4 import Gemma4TransformerConfig

    assert issubclass(Gemma4TransformerConfig, TransformerConfig)


@pytest.mark.unit
@requires_megatron
def test_marks_experts_for_direct_hf_loading():
    from vime_plugins.models.gemma4_provider import _mark_expert_weights_for_direct_hf_loading

    fc1 = torch.nn.Parameter(torch.zeros(1))
    fc2 = torch.nn.Parameter(torch.zeros(1))
    dense = torch.nn.Parameter(torch.zeros(1))
    model = SimpleNamespace(
        named_parameters=lambda: iter(
            [
                ("decoder.layers.0.mlp.experts.linear_fc1.weight0", fc1),
                ("decoder.layers.0.mlp.experts.linear_fc2.weight0", fc2),
                ("decoder.layers.0.self_attention.linear_proj.weight", dense),
            ]
        )
    )

    _mark_expert_weights_for_direct_hf_loading(model)

    assert (fc1.tensor_model_parallel, fc1.partition_dim, fc1.partition_stride) == (True, 0, 1)
    assert (fc2.tensor_model_parallel, fc2.partition_dim, fc2.partition_stride) == (True, 1, 1)
    assert not hasattr(dense, "tensor_model_parallel")


@pytest.mark.unit
@requires_megatron
def test_promotes_layer_scalars_for_direct_hf_loading():
    from vime_plugins.models.gemma4_provider import _promote_layer_scalars

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.register_buffer("layer_scalar", torch.tensor([0.5]))

    _promote_layer_scalars(model)

    assert torch.equal(dict(model.named_parameters())["layer.layer_scalar"], torch.tensor([0.5]))
    assert not model.layer.layer_scalar.requires_grad


@pytest.mark.unit
def test_tied_output_weight_is_not_exported(config):
    args = SimpleNamespace(hf_checkpoint="checkpoint")
    _config_cache[args.hf_checkpoint] = conversion_config(config)

    assert convert_gemma4_to_hf(args, "module.module.output_layer.weight", torch.empty(1)) == []


@pytest.mark.unit
@pytest.mark.parametrize("layer", [0, 1])
def test_qkv_round_trip(config, layer):
    attention_config = (
        config
        if layer == 0
        else SimpleNamespace(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_global_key_value_heads,
            head_dim=config.global_head_dim,
            hidden_size=config.hidden_size,
        )
    )
    q = torch.arange(
        attention_config.num_attention_heads * attention_config.head_dim * config.hidden_size,
        dtype=torch.float32,
    ).reshape(-1, config.hidden_size)
    k = torch.arange(
        attention_config.num_key_value_heads * attention_config.head_dim * config.hidden_size,
        dtype=torch.float32,
    ).reshape(-1, config.hidden_size)
    v = k if layer == 1 else k + 1000
    prefix = f"model.language_model.layers.{layer}.self_attn"
    reader = Reader(
        {
            f"{prefix}.q_proj.weight": q,
            f"{prefix}.k_proj.weight": k,
            **({} if layer == 1 else {f"{prefix}.v_proj.weight": v}),
        }
    )

    megatron = gemma4_hf_tensor(
        f"decoder.layers.{layer}.self_attention.linear_qkv.weight",
        reader,
        config,
    )
    args = SimpleNamespace(hf_checkpoint="checkpoint")
    _config_cache[args.hf_checkpoint] = conversion_config(config)
    converted = dict(
        convert_gemma4_to_hf(
            args,
            f"module.module.decoder.layers.{layer}.self_attention.linear_qkv.weight",
            megatron,
        )
    )

    assert torch.equal(converted[f"{prefix}.q_proj.weight"], q)
    assert torch.equal(converted[f"{prefix}.k_proj.weight"], k)
    if layer == 0:
        assert torch.equal(converted[f"{prefix}.v_proj.weight"], v)
    else:
        assert f"{prefix}.v_proj.weight" not in converted


@pytest.mark.unit
def test_expert_weights_round_trip_as_packed_hf_tensor(config):
    args = SimpleNamespace(hf_checkpoint="checkpoint")
    _config_cache[args.hf_checkpoint] = conversion_config(config)
    _expert_buffers.clear()
    packed = torch.arange(2 * 6 * 3, dtype=torch.float32).reshape(2, 6, 3)
    reader = Reader({"model.language_model.layers.0.experts.gate_up_proj": packed})

    first = gemma4_hf_tensor("decoder.layers.0.mlp.experts.linear_fc1.weight0", reader, config)
    second = gemma4_hf_tensor("decoder.layers.0.mlp.experts.linear_fc1.weight1", reader, config)
    assert (
        convert_gemma4_to_hf(
            args,
            "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight0",
            first,
        )
        == []
    )
    converted = convert_gemma4_to_hf(
        args,
        "module.module.decoder.layers.0.mlp.experts.linear_fc1.weight1",
        second,
    )

    assert converted[0][0] == "model.language_model.layers.0.experts.gate_up_proj"
    assert torch.equal(converted[0][1], packed)
    assert not _expert_buffers


@pytest.mark.unit
def test_dense_mlp_and_router_scale(config):
    prefix = "model.language_model.layers.0"
    gate = torch.arange(18, dtype=torch.float32).reshape(6, 3)
    up = gate + 100
    router_scale = torch.arange(3, dtype=torch.float32)
    reader = Reader(
        {
            f"{prefix}.mlp.gate_proj.weight": gate,
            f"{prefix}.mlp.up_proj.weight": up,
            f"{prefix}.router.scale": router_scale,
        }
    )

    assert torch.equal(
        gemma4_hf_tensor("decoder.layers.0.dense_mlp.linear_fc1.weight", reader, config),
        torch.cat((gate, up)),
    )
    assert torch.equal(gemma4_hf_tensor("decoder.layers.0.mlp.router.scale", reader, config), router_scale)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
