"""DeepSeek-V4-Flash checkpoint names shared by load and rollout weight sync."""

from __future__ import annotations

import re

import torch


_ROOT = {
    "embedding.word_embeddings.weight": "embed.weight",
    "decoder.final_layernorm.weight": "norm.weight",
    "output_layer.weight": "head.weight",
    "decoder.hc_head_fn": "hc_head_fn",
    "decoder.hc_head_base": "hc_head_base",
    "decoder.hc_head_scale": "hc_head_scale",
}

_LAYER = {
    "input_layernorm.weight": "attn_norm.weight",
    "pre_mlp_layernorm.weight": "ffn_norm.weight",
    "self_attention.linear_q_down_proj.weight": "attn.wq_a.weight",
    "self_attention.q_layernorm.weight": "attn.q_norm.weight",
    "self_attention.linear_q_up_proj.weight": "attn.wq_b.weight",
    "self_attention.linear_kv_proj.weight": "attn.wkv.weight",
    "self_attention.kv_layernorm.weight": "attn.kv_norm.weight",
    "self_attention.linear_o_group_proj": "attn.wo_a.weight",
    "self_attention.linear_proj.weight": "attn.wo_b.weight",
    "self_attention.core_attention.attn_sink": "attn.attn_sink",
    "self_attention.core_attention.compressor.ape": "attn.compressor.ape",
    "self_attention.core_attention.compressor.linear_wkv.weight": "attn.compressor.wkv.weight",
    "self_attention.core_attention.compressor.linear_wgate.weight": "attn.compressor.wgate.weight",
    "self_attention.core_attention.compressor.norm.weight": "attn.compressor.norm.weight",
    "self_attention.core_attention.indexer.linear_wq_b.weight": "attn.indexer.wq_b.weight",
    "self_attention.core_attention.indexer.linear_weights_proj.weight": "attn.indexer.weights_proj.weight",
    "self_attention.core_attention.indexer.compressor.ape": "attn.indexer.compressor.ape",
    "self_attention.core_attention.indexer.compressor.linear_wkv.weight": "attn.indexer.compressor.wkv.weight",
    "self_attention.core_attention.indexer.compressor.linear_wgate.weight": "attn.indexer.compressor.wgate.weight",
    "self_attention.core_attention.indexer.compressor.norm.weight": "attn.indexer.compressor.norm.weight",
    "mlp.router.weight": "ffn.gate.weight",
    "mlp.router.expert_bias": "ffn.gate.bias",
    "mlp.router.tid2eid": "ffn.gate.tid2eid",
    "self_attention_hyper_connection.mapping_proj.weight": "hc_attn_fn",
    "self_attention_hyper_connection.bias": "hc_attn_base",
    "mlp_hyper_connection.mapping_proj.weight": "hc_ffn_fn",
    "mlp_hyper_connection.bias": "hc_ffn_base",
}

_ALPHAS = {"alpha_pre": 0, "alpha_post": 1, "alpha_res": 2}
_HC_SCALE = {
    "self_attention_hyper_connection": "hc_attn_scale",
    "mlp_hyper_connection": "hc_ffn_scale",
}
_PENDING_ALPHAS: dict[str, dict[int, torch.Tensor]] = {}


def _strip(name: str) -> str:
    while name.startswith("module."):
        name = name.removeprefix("module.")
    return name.removeprefix("language_model.")


def _layer_parts(name: str) -> tuple[str, str]:
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if match is None:
        raise KeyError(f"Unsupported DeepSeek-V4 parameter {name!r}")
    layer, rest = match.groups()
    return f"layers.{layer}.", rest


def _expert_names(rest: str) -> tuple[str, ...] | None:
    match = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if match:
        projection, expert = match.groups()
        prefix = f"ffn.experts.{expert}."
    else:
        match = re.fullmatch(r"mlp\.shared_experts\.linear_fc([12])\.weight", rest)
        if match is None:
            return None
        projection = match.group(1)
        prefix = "ffn.shared_experts."
    return (f"{prefix}w1.weight", f"{prefix}w3.weight") if projection == "1" else (f"{prefix}w2.weight",)


def _alpha_name(rest: str) -> tuple[str, int] | None:
    for stem, checkpoint_name in _HC_SCALE.items():
        for suffix, index in _ALPHAS.items():
            if rest == f"{stem}.{suffix}":
                return checkpoint_name, index
    return None


def deepseek_v4_hf_tensor(name: str, reader, config) -> torch.Tensor:
    name = _strip(name)
    if name in _ROOT:
        return reader.get_tensor(_ROOT[name])
    prefix, rest = _layer_parts(name)
    if alpha := _alpha_name(rest):
        checkpoint_name, index = alpha
        return reader.get_tensor(prefix + checkpoint_name).reshape(-1)[index : index + 1]
    if expert_names := _expert_names(rest):
        parts = [reader.get_tensor(prefix + key) for key in expert_names]
        return torch.cat(parts, dim=0) if len(parts) == 2 else parts[0]
    if rest not in _LAYER:
        raise KeyError(f"Unsupported DeepSeek-V4 parameter {name!r}")
    return reader.get_tensor(prefix + _LAYER[rest])


def convert_deepseek_v4_to_hf(args, name: str, parameter: torch.Tensor) -> list[tuple[str, torch.Tensor]]:
    name = _strip(name)
    if name in _ROOT:
        return [(_ROOT[name], parameter)]
    prefix, rest = _layer_parts(name)
    if alpha := _alpha_name(rest):
        checkpoint_name, index = alpha
        key = prefix + checkpoint_name
        pieces = _PENDING_ALPHAS.setdefault(key, {})
        pieces[index] = parameter.reshape(1)
        if len(pieces) != 3:
            return []
        del _PENDING_ALPHAS[key]
        return [(key, torch.cat([pieces[i] for i in range(3)]))]
    if expert_names := _expert_names(rest):
        parts = parameter.chunk(2, dim=0) if len(expert_names) == 2 else (parameter,)
        return [(prefix + key, part) for key, part in zip(expert_names, parts, strict=True)]
    if rest not in _LAYER:
        raise KeyError(f"Unsupported DeepSeek-V4 parameter {name!r}")
    return [(prefix + _LAYER[rest], parameter)]


def quantize_deepseek_v4_weights(named_tensors, quantization_config):
    if not quantization_config or quantization_config.get("quant_method") not in {"fp8", "deepseek_v4_fp8"}:
        return named_tensors
    from vime.backends.megatron_utils.vllm import quant_weight_ue8m0

    block_size = quantization_config["weight_block_size"]
    result = []
    for name, tensor in named_tensors:
        quantized = name.endswith(".weight") and (
            any(f".attn.{part}.weight" in name for part in ("wq_a", "wq_b", "wkv", "wo_a", "wo_b", "indexer.wq_b"))
            or ".ffn.experts." in name
            or ".ffn.shared_experts." in name
        )
        if quantized:
            weight, scale = quant_weight_ue8m0(tensor, block_size)
            result.extend(((name, weight), (name.removesuffix(".weight") + ".scale", scale)))
        else:
            result.append((name, tensor))
    return result
