import re
import torch


def convert_qwen2_to_hf(args, name, param):
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.weight":

            param = param.view(args.num_query_groups, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1)
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.weight", q_param),
                (f"model.layers.{layer_idx}.self_attn.k_proj.weight", k_param),
                (f"model.layers.{layer_idx}.self_attn.v_proj.weight", v_param),
            ]
        elif rest == "self_attention.linear_qkv.bias":
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
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]
        elif rest == "mlp.linear_fc1.weight":
            gate_weight, up_weight = param.chunk(2, dim=0)
            return [
                (f"model.layers.{layer_idx}.mlp.gate_proj.weight", gate_weight),
                (f"model.layers.{layer_idx}.mlp.up_proj.weight", up_weight),
            ]
        elif rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]
        elif rest == "self_attention.linear_qkv.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")


def convert_qwen2_to_hf_shard(args, name, param, tp_rank, tp_size):
    """Shard-level HF conversion: operates on a single TP shard without all_gather.

    For Qwen2/3 with GQA, Megatron shards by query groups, which maps directly
    to vLLM's head-based sharding after QKV split. Each TP rank converts its
    own shard and sends to the corresponding vLLM TP rank.

    Args:
        args: Model config (num_attention_heads, num_query_groups, etc.)
        name: Megatron parameter name
        param: TP-sharded parameter (this rank's shard only)
        tp_rank: Current tensor model parallel rank
        tp_size: Tensor model parallel world size

    Returns:
        List of (hf_name, shard_tensor) tuples. Empty list for duplicated
        params on non-rank-0 (only rank 0 sends duplicated params).
    """
    try:
        head_dim = args.kv_channels if args.kv_channels is not None else args.hidden_size // args.num_attention_heads
    except AttributeError:
        head_dim = args.hidden_size // args.num_attention_heads
    value_num_per_group = args.num_attention_heads // args.num_query_groups

    # Duplicated params: every TP rank sends these (each group needs them)
    if name == "module.module.decoder.final_layernorm.weight":
        return [("model.norm.weight", param)]

    decoder_layers_pattern = r"module\.module\.decoder\.layers\.(\d+)\.(.+)"
    match = re.match(decoder_layers_pattern, name)
    if match:
        layer_idx, rest = match.groups()

        # Duplicated params: layernorms, qk norm - every rank sends in its group
        _duplicated_map = {
            "self_attention.linear_qkv.layer_norm_weight": "input_layernorm",
            "mlp.linear_fc1.layer_norm_weight": "post_attention_layernorm",
            "self_attention.q_layernorm.weight": "self_attn.q_norm",
            "self_attention.k_layernorm.weight": "self_attn.k_norm",
        }
        if rest in _duplicated_map:
            hf_name = f"model.layers.{layer_idx}.{_duplicated_map[rest]}.weight"
            return [(hf_name, param)]

        # TP-sharded params: each rank converts its own shard
        if rest == "self_attention.linear_qkv.weight":
            groups_per_rank = args.num_query_groups // tp_size
            param = param.view(groups_per_rank, -1, head_dim, args.hidden_size)
            q_param, k_param, v_param = torch.split(
                param, split_size_or_sections=[value_num_per_group, 1, 1], dim=1
            )
            q_param = q_param.reshape(-1, args.hidden_size)
            k_param = k_param.reshape(-1, args.hidden_size)
            v_param = v_param.reshape(-1, args.hidden_size)
            # For is_checkpoint_format=False: produce combined qkv_proj matching vLLM layout
            qkv_param = torch.cat([q_param, k_param, v_param], dim=0)
            return [
                (f"model.layers.{layer_idx}.self_attn.qkv_proj.weight", qkv_param),
            ]

        if rest == "self_attention.linear_qkv.bias":
            groups_per_rank = args.num_query_groups // tp_size
            param = param.view(groups_per_rank, -1)
            q_bias, k_bias, v_bias = torch.split(
                param,
                split_size_or_sections=[value_num_per_group * head_dim, head_dim, head_dim],
                dim=1,
            )
            q_bias = q_bias.contiguous().flatten()
            k_bias = k_bias.contiguous().flatten()
            v_bias = v_bias.contiguous().flatten()
            return [
                (f"model.layers.{layer_idx}.self_attn.q_proj.bias", q_bias),
                (f"model.layers.{layer_idx}.self_attn.k_proj.bias", k_bias),
                (f"model.layers.{layer_idx}.self_attn.v_proj.bias", v_bias),
            ]

        if rest == "self_attention.linear_proj.weight":
            return [(f"model.layers.{layer_idx}.self_attn.o_proj.weight", param)]

        if rest == "mlp.linear_fc1.weight":
            # For is_checkpoint_format=False: produce combined gate_up_proj matching vLLM layout
            return [
                (f"model.layers.{layer_idx}.mlp.gate_up_proj.weight", param),
            ]

        if rest == "mlp.linear_fc2.weight":
            return [(f"model.layers.{layer_idx}.mlp.down_proj.weight", param)]

    # Embedding and output layer: TP-sharded along vocab dim
    if name == "module.module.embedding.word_embeddings.weight":
        return [("model.embed_tokens.weight", param)]
    if name == "module.module.output_layer.weight":
        return [("lm_head.weight", param)]

    raise ValueError(f"Unknown parameter name: {name}")
