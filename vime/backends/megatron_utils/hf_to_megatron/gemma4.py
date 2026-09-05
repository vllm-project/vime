import re
from types import SimpleNamespace

import torch

from .common import SafetensorReader, merge_gate_up, merge_qkv, strip_mcore_wrappers, text_config


def _layer(name: str) -> tuple[int, str]:
    match = re.fullmatch(r"decoder\.layers\.(\d+)\.(.+)", name)
    if not match:
        raise KeyError(f"Unsupported Gemma4 Megatron parameter {name!r}")
    return int(match.group(1)), match.group(2)


def _attention_config(config, layer: int):
    config = text_config(config)
    if hasattr(config, "per_layer_config"):
        config = config.per_layer_config[layer]
        return SimpleNamespace(
            num_attention_heads=config.num_attention_heads,
            num_key_value_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            hidden_size=config.hidden_size,
        )
    if config.layer_types[layer] != "full_attention":
        return config
    return SimpleNamespace(
        num_attention_heads=config.num_attention_heads,
        num_key_value_heads=config.num_global_key_value_heads,
        head_dim=config.global_head_dim,
        hidden_size=config.hidden_size,
    )


def gemma4_hf_tensor(name: str, reader: SafetensorReader, config) -> torch.Tensor:
    name = strip_mcore_wrappers(name)
    direct = {
        "embedding.word_embeddings.weight": "model.language_model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.language_model.norm.weight",
        "output_layer.weight": "model.language_model.embed_tokens.weight",
    }
    if name in direct:
        return reader.get_tensor(direct[name])

    layer, rest = _layer(name)
    prefix = f"model.language_model.layers.{layer}"
    mapping = {
        "self_attention.linear_qkv.layer_norm_weight": "input_layernorm.weight",
        "self_attention.q_layernorm.weight": "self_attn.q_norm.weight",
        "self_attention.k_layernorm.weight": "self_attn.k_norm.weight",
        "self_attention.linear_proj.weight": "self_attn.o_proj.weight",
        "pre_mlp_layernorm.weight": "pre_feedforward_layernorm.weight",
        "dense_mlp.linear_fc1.layer_norm_weight": "pre_feedforward_layernorm.weight",
        "post_attention_layernorm.weight": "post_attention_layernorm.weight",
        "post_feedforward_layernorm.weight": "post_feedforward_layernorm.weight",
        "mlp.pre_feedforward_layernorm_2.weight": "pre_feedforward_layernorm_2.weight",
        "pre_feedforward_layernorm_2.weight": "pre_feedforward_layernorm_2.weight",
        "post_feedforward_layernorm_1.weight": "post_feedforward_layernorm_1.weight",
        "post_feedforward_layernorm_2.weight": "post_feedforward_layernorm_2.weight",
        "mlp.router.proj.weight": "router.proj.weight",
        "mlp.router.scale": "router.scale",
        "mlp.router.per_expert_scale": "router.per_expert_scale",
        "layer_scalar": "layer_scalar",
    }
    if rest in mapping:
        return reader.get_tensor(f"{prefix}.{mapping[rest]}")

    if rest == "self_attention.linear_qkv.weight":
        attention_config = _attention_config(config, layer)
        q = reader.get_tensor(f"{prefix}.self_attn.q_proj.weight")
        k = reader.get_tensor(f"{prefix}.self_attn.k_proj.weight")
        v_name = f"{prefix}.self_attn.v_proj.weight"
        v = reader.get_tensor(v_name) if v_name in reader else k
        return merge_qkv(q, k, v, attention_config)

    if rest in {"mlp.linear_fc1.weight", "dense_mlp.linear_fc1.weight"}:
        return merge_gate_up(
            reader.get_tensor(f"{prefix}.mlp.gate_proj.weight"),
            reader.get_tensor(f"{prefix}.mlp.up_proj.weight"),
        )
    if rest in {"mlp.linear_fc2.weight", "dense_mlp.linear_fc2.weight"}:
        return reader.get_tensor(f"{prefix}.mlp.down_proj.weight")

    match = re.fullmatch(r"mlp\.experts\.linear_fc([12])\.weight(\d+)", rest)
    if match:
        projection, expert = map(int, match.groups())
        packed_name = "gate_up_proj" if projection == 1 else "down_proj"
        return reader.get_tensor(f"{prefix}.experts.{packed_name}")[expert].contiguous()

    raise KeyError(f"Unsupported Gemma4 Megatron parameter {name!r}")
