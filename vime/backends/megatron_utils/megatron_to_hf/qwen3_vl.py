import re
import torch


def convert_qwen3vl_to_hf(args, name, param):
    if name.startswith("module.module.language_model."):
        name = "module.module." + name[len("module.module.language_model.") :]

    # (Optional safety) if you ever see extra "module." prefixes
    while name.startswith("module.module.module."):
        name = name.replace("module.module.module.", "module.module.", 1)

    if name.startswith("module.module.vision_model."):
        vision_name = name[len("module.module.vision_model.") :]

        match = re.match(r"decoder\.layers\.(\d+)\.(.+)", vision_name)
        if match:
            layer_idx, rest = match.groups()
            rest = rest.replace("self_attention.linear_proj.", "attn.proj.", 1)
            rest = rest.replace("self_attention.linear_qkv.", "attn.qkv.", 1)
            rest = rest.replace("attn.qkv.layer_norm_", "norm1.", 1)
            rest = rest.replace("mlp.linear_fc1.layer_norm_", "norm2.", 1)
            if rest in {
                "attn.proj.weight",
                "attn.proj.bias",
                "attn.qkv.weight",
                "attn.qkv.bias",
                "norm1.weight",
                "norm1.bias",
                "norm2.weight",
                "norm2.bias",
                "mlp.linear_fc1.weight",
                "mlp.linear_fc1.bias",
                "mlp.linear_fc2.weight",
                "mlp.linear_fc2.bias",
            }:
                return [(f"model.visual.blocks.{layer_idx}.{rest}", param)]

        match = re.match(r"decoder\.deepstack_merger_list\.(\d+)\.(.+)", vision_name)
        if match:
            layer_idx, rest = match.groups()
            rest = rest.replace("patch_norm.", "norm.", 1)
            if rest in {
                "norm.weight",
                "norm.bias",
                "linear_fc1.weight",
                "linear_fc1.bias",
                "linear_fc2.weight",
                "linear_fc2.bias",
            }:
                return [(f"model.visual.deepstack_merger_list.{layer_idx}.{rest}", param)]

        if vision_name.startswith("merger.patch_norm."):
            return [(f"model.visual.{vision_name.replace('patch_norm.', 'norm.', 1)}", param)]
        if re.match(r"merger\.linear_fc[12]\.(weight|bias)$", vision_name):
            return [(f"model.visual.{vision_name}", param)]
        if vision_name == "pos_embed.weight":
            return [("model.visual.pos_embed.weight", param)]
        if vision_name.startswith("patch_embed.proj."):
            return [(f"model.visual.{vision_name}", param)]

    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.language_model.embed_tokens.weight", param)]

    if name == "module.module.output_layer.weight":
        # Your key list has lm_head.weight at top-level
        return [("lm_head.weight", param)]

    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.language_model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        # Everything goes under model.language_model.layers.{i}.*
        base = f"model.language_model.layers.{layer_idx}"

        if rest == "self_attention.linear_proj.weight":
            return [(f"{base}.self_attn.o_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.weight":
            # Keep your original split logic -> q_proj/k_proj/v_proj
            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"{base}.self_attn.q_proj.weight", q_param),
                (f"{base}.self_attn.k_proj.weight", k_param),
                (f"{base}.self_attn.v_proj.weight", v_param),
            ]

        elif rest == "self_attention.linear_qkv.bias":
            # Keep your original split logic -> q_proj/k_proj/v_proj bias
            param = param.view(args.num_query_groups, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"{base}.self_attn.q_proj.bias", q_bias),
                (f"{base}.self_attn.k_proj.bias", k_bias),
                (f"{base}.self_attn.v_proj.bias", v_bias),
            ]

        elif rest == "mlp.linear_fc1.weight":
            # Keep your original split logic -> gate_proj/up_proj
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"{base}.mlp.gate_proj.weight", gate_weight),
                (f"{base}.mlp.up_proj.weight", up_weight),
            ]

        elif rest == "mlp.linear_fc2.weight":
            return [(f"{base}.mlp.down_proj.weight", param)]

        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"{base}.input_layernorm.weight", param)]

        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"{base}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"{base}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"{base}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
