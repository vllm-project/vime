import argparse
import json
import os
import re
import subprocess
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from safetensors import safe_open
from safetensors.torch import save_file

from vime.backends.megatron_utils.deepseek_v4_weights import (
    _PENDING_ALPHAS,
    convert_deepseek_v4_to_hf,
    deepseek_v4_hf_tensor,
    quantize_deepseek_v4_weights,
)
from vime.backends.megatron_utils.hf_to_megatron.common import SafetensorReader
from vime.backends.megatron_utils.update_weight import common as update_weight_common


class Reader:
    def __init__(self, tensors):
        self.tensors = tensors

    def get_tensor(self, name):
        return self.tensors[name]


@pytest.mark.parametrize(
    "mcore,checkpoint",
    [
        ("embedding.word_embeddings.weight", "embed.weight"),
        ("decoder.hc_head_fn", "hc_head_fn"),
        ("decoder.layers.0.self_attention.linear_o_group_proj", "layers.0.attn.wo_a.weight"),
        ("decoder.layers.2.self_attention.core_attention.compressor.ape", "layers.2.attn.compressor.ape"),
        (
            "decoder.layers.3.self_attention.core_attention.indexer.compressor.linear_wkv.weight",
            "layers.3.attn.indexer.compressor.wkv.weight",
        ),
        ("decoder.layers.42.mlp.router.expert_bias", "layers.42.ffn.gate.bias"),
    ],
)
def test_native_v4_names_round_trip(mcore, checkpoint):
    tensor = torch.randn(2, 3)
    name = f"module.module.{mcore}"
    exported = convert_deepseek_v4_to_hf(None, name, tensor)
    assert len(exported) == 1 and exported[0][0] == checkpoint
    assert torch.equal(exported[0][1], tensor)
    assert torch.equal(deepseek_v4_hf_tensor(name, Reader({checkpoint: tensor}), None), tensor)


def test_expert_gate_up_order_round_trip():
    gate, up = torch.randn(2, 3), torch.randn(2, 3)
    prefix = "layers.3.ffn.experts.17."
    name = "module.module.decoder.layers.3.mlp.experts.linear_fc1.weight17"
    reader = Reader({prefix + "w1.weight": gate, prefix + "w3.weight": up})
    merged = deepseek_v4_hf_tensor(name, reader, None)
    assert torch.equal(merged, torch.cat((gate, up)))
    exported = convert_deepseek_v4_to_hf(None, name, merged)
    assert [name for name, _ in exported] == [prefix + "w1.weight", prefix + "w3.weight"]
    assert torch.equal(exported[0][1], gate)
    assert torch.equal(exported[1][1], up)


def test_hyper_connection_alpha_pack_is_order_independent():
    _PENDING_ALPHAS.clear()
    outputs = []
    for suffix, value in (("alpha_res", 3), ("alpha_pre", 1), ("alpha_post", 2)):
        outputs += convert_deepseek_v4_to_hf(
            None,
            f"module.module.decoder.layers.2.self_attention_hyper_connection.{suffix}",
            torch.tensor([value], dtype=torch.float32),
        )
    assert len(outputs) == 1
    assert outputs[0][0] == "layers.2.hc_attn_scale"
    assert torch.equal(outputs[0][1], torch.tensor([1.0, 2.0, 3.0]))
    for index, suffix in enumerate(("alpha_pre", "alpha_post", "alpha_res")):
        loaded = deepseek_v4_hf_tensor(
            f"decoder.layers.2.self_attention_hyper_connection.{suffix}",
            Reader(dict(outputs)),
            None,
        )
        assert loaded.item() == index + 1
    assert not _PENDING_ALPHAS


def test_native_v4_fp8_scale_is_dequantized(tmp_path):
    weight = torch.tensor([[2.0, -4.0]], dtype=torch.float8_e4m3fn)
    scale = torch.tensor([[0.25]], dtype=torch.float32)
    save_file({"layers.0.attn.wq_a.weight": weight, "layers.0.attn.wq_a.scale": scale}, tmp_path / "model.safetensors")
    loaded = SafetensorReader(tmp_path).get_tensor("layers.0.attn.wq_a.weight")
    assert loaded.dtype == torch.bfloat16
    assert torch.equal(loaded, torch.tensor([[0.5, -1.0]], dtype=torch.bfloat16))


def test_native_v4_rejects_packed_non_fp8_weights(tmp_path):
    weight = torch.tensor([[0x12, 0x34]], dtype=torch.uint8)
    scale = torch.ones((1, 1), dtype=torch.float32)
    save_file(
        {"layers.0.ffn.experts.0.w1.weight": weight, "layers.0.ffn.experts.0.w1.scale": scale},
        tmp_path / "model.safetensors",
    )
    with pytest.raises(TypeError, match="requires FP8 E4M3"):
        SafetensorReader(tmp_path).get_tensor("layers.0.ffn.experts.0.w1.weight")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 refit needs CUDA")
def test_online_fp8_scale_matches_v4_checkpoint_format():
    pytest.importorskip("vllm")
    weight = torch.randn((128, 128), device="cuda", dtype=torch.bfloat16)
    config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    exported = dict(quantize_deepseek_v4_weights([("layers.0.attn.wq_a.weight", weight)], config))
    scale = exported["layers.0.attn.wq_a.scale"]
    assert exported["layers.0.attn.wq_a.weight"].dtype == torch.float8_e4m3fn
    assert torch.all(torch.log2(scale.float()) == torch.log2(scale.float()).round())


@pytest.mark.skipif(not torch.cuda.is_available(), reason="FP8 refit needs CUDA")
def test_online_fp8_export_covers_flash_checkpoint_scales():
    pytest.importorskip("vllm")
    checkpoint = os.environ.get("DSV4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set DSV4_CHECKPOINT to check the local Flash checkpoint")
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    scale_names = {
        re.sub(r"experts\.\d+", "experts.0", re.sub(r"^layers\.\d+", "layers.0", name))
        for name in index
        if name.startswith("layers.") and name.endswith(".scale")
    }
    weight = torch.ones((128, 128), device="cuda", dtype=torch.bfloat16)
    config = {"quant_method": "fp8", "weight_block_size": [128, 128]}
    for scale_name in scale_names:
        weight_name = scale_name.removesuffix(".scale") + ".weight"
        exported = dict(quantize_deepseek_v4_weights([(weight_name, weight)], config))
        assert scale_name in exported, f"Missing online FP8 scale for {weight_name}"
        assert exported[scale_name].shape == (1, 1)
        assert exported[weight_name].dtype == torch.float8_e4m3fn


def test_unknown_parameter_is_rejected():
    with pytest.raises(KeyError, match="Unsupported DeepSeek-V4"):
        deepseek_v4_hf_tensor("decoder.layers.0.unknown", Reader({}), SimpleNamespace())


def test_hash_router_table_is_exported(monkeypatch):
    monkeypatch.setattr(update_weight_common.mpu, "get_expert_model_parallel_world_size", lambda: 1)
    monkeypatch.setattr(update_weight_common.mpu, "get_expert_model_parallel_rank", lambda: 0)
    monkeypatch.setattr(update_weight_common, "get_transformer_layer_offset", lambda config: 0)

    model = torch.nn.Module()
    model.config = SimpleNamespace()
    decoder = torch.nn.Module()
    layers = torch.nn.ModuleList([torch.nn.Module()])
    mlp = torch.nn.Module()
    router = torch.nn.Module()
    table = torch.tensor([[1, 2], [3, 4]], dtype=torch.int32)
    router.register_buffer("tid2eid", table)
    mlp.add_module("router", router)
    layers[0].add_module("mlp", mlp)
    decoder.add_module("layers", layers)
    model.add_module("decoder", decoder)
    inner = torch.nn.Module()
    inner.add_module("module", model)
    wrapped = torch.nn.Module()
    wrapped.config = model.config
    wrapped.add_module("module", inner)

    exported = dict(update_weight_common.named_params_and_buffers(SimpleNamespace(num_experts=256), [wrapped]))
    name = "module.module.decoder.layers.0.mlp.router.tid2eid"
    assert torch.equal(exported[name], table)
    assert convert_deepseek_v4_to_hf(None, name, exported[name])[0][0] == "layers.0.ffn.gate.tid2eid"


def test_flash_recipe_uses_recognized_megatron_args():
    from megatron.training.arguments import add_megatron_arguments

    root = Path(__file__).resolve().parents[1]
    result = subprocess.run(
        ["bash", "-c", 'source scripts/models/deepseek-v4-flash.sh; printf "%s\\n" "${MODEL_ARGS[@]}"'],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    )
    parser = add_megatron_arguments(argparse.ArgumentParser(allow_abbrev=False))
    parser.add_argument("--model-name")  # vime-owned option
    args = parser.parse_args(result.stdout.splitlines())
    assert args.experimental_attention_variant == "dsv4_hybrid"
    assert args.output_projection_groups == 8
    assert args.output_projection_lora_rank == 1024
    assert args.moe_num_hash_layers == 3
    assert args.mhc_num_residual_streams == 4
    assert args.csa_compress_ratios[:4] == [0, 0, 4, 128]


def test_local_four_layer_fixture_coverage():
    checkpoint = os.environ.get("DSV4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set DSV4_CHECKPOINT to validate the local Flash checkpoint")
    path = Path(checkpoint)
    config = json.loads((path / "config.json").read_text())
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    assert config["num_hidden_layers"] == 43
    assert config["compress_ratios"][:4] == [0, 0, 4, 128]
    assert len(config["compress_ratios"]) == 44  # Last ratio belongs to disabled MTP.
    for layer in (0, 1):
        assert f"layers.{layer}.attn.compressor.ape" not in index
    for layer in (2, 3):
        assert f"layers.{layer}.attn.compressor.ape" in index
    assert "layers.2.attn.indexer.wq_b.weight" in index
    assert "layers.3.attn.indexer.wq_b.weight" not in index
    assert "layers.0.ffn.gate.tid2eid" in index


def test_local_fp8_weight_decoding():
    checkpoint = os.environ.get("DSV4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set DSV4_CHECKPOINT to validate the local Flash checkpoint")
    path = Path(checkpoint)
    index = json.loads((path / "model.safetensors.index.json").read_text())["weight_map"]
    name = "layers.0.attn.wq_a.weight"
    with safe_open(path / index[name], framework="pt", device="cpu") as tensors:
        raw = tensors.get_tensor(name)
        scale = tensors.get_tensor("layers.0.attn.wq_a.scale")
    loaded = SafetensorReader(path).get_tensor(name)
    expected = raw[:4, :4].float() * scale[0, 0]
    assert loaded.dtype == torch.bfloat16
    assert torch.allclose(loaded[:4, :4].float(), expected, rtol=0.01, atol=1e-4)
    expert_name = "layers.3.ffn.experts.0.w1.weight"
    with safe_open(path / index[expert_name], framework="pt", device="cpu") as tensors:
        assert tensors.get_slice(expert_name).get_dtype() == "F8_E4M3"
        assert tensors.get_slice(expert_name.removesuffix(".weight") + ".scale").get_dtype() == "F32"
