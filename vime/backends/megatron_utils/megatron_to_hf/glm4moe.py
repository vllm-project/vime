import re

import torch

from vime.utils.common import is_npu


def convert_glm4moe_to_hf(args, name, param):
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

        if is_npu():
            npu_outputs = _convert_npu_experts_and_mla(args, name, param, layer_idx, rest)
            if npu_outputs is not None:
                return npu_outputs
        else:
            # Standard Megatron: one set of weights per expert
            expert_pattern = r"mlp.experts\.(.+)\.weight(\d+)"
            match = re.match(expert_pattern, rest)
            if match:
                rest, expert_idx = match.groups()
                if rest == "linear_fc1":
                    gate_weight, up_weight = param.chunk(2, dim=0)
                    outputs = [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                    ]
                    return outputs
                elif rest == "linear_fc2":
                    outputs = [
                        (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                    ]
                    return outputs
                else:
                    raise ValueError(f"Unknown expert parameter name: {name}")

        # shared expert
        shared_expert_pattern = r"mlp.shared_experts\.(.+)"
        match = re.match(shared_expert_pattern, rest)
        if match:
            rest = match.groups()[0]
            if rest == "linear_fc1.weight":
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.shared_experts.up_proj.weight", up_weight),
                ]
            elif rest == "linear_fc2.weight":
                return [
                    (f"model.layers.{layer_idx}.mlp.shared_experts.down_proj.weight", param),
                ]
            else:
                raise ValueError(f"Unknown shared expert parameter name: {name}")

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
        elif rest == "self_attention.linear_qkv.layer_norm_weight" or (is_npu() and rest == "input_layernorm.weight"):
            return [(f"model.layers.{layer_idx}.input_layernorm.weight", param)]
        elif rest == "mlp.linear_fc1.layer_norm_weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "post_self_attn_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_self_attn_layernorm.weight", param)]
        elif rest == "post_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_mlp_layernorm.weight", param)]
        elif rest == "pre_mlp_layernorm.weight":
            return [(f"model.layers.{layer_idx}.post_attention_layernorm.weight", param)]
        elif rest == "mlp.router.weight":
            return [(f"model.layers.{layer_idx}.mlp.gate.weight", param)]
        elif rest == "mlp.router.expert_bias":
            return [(f"model.layers.{layer_idx}.mlp.gate.e_score_correction_bias", param)]

        # qk norm
        elif rest == "self_attention.q_layernorm.weight":
            if is_npu():
                # MindSpeed MLA
                return [(f"model.layers.{layer_idx}.self_attn.q_a_layernorm.weight", param)]
            return [(f"model.layers.{layer_idx}.self_attn.q_norm.weight", param)]
        elif rest == "self_attention.k_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.k_norm.weight", param)]
        elif is_npu() and rest == "self_attention.kv_layernorm.weight":
            return [(f"model.layers.{layer_idx}.self_attn.kv_a_layernorm.weight", param)]

    mtp_layer_pattern = r"module\.module\.mtp\.layers\.(\d+)\.(.+)"
    match = re.match(mtp_layer_pattern, name)
    if match:
        layer_idx, rest = match.groups()
        layer_idx = int(layer_idx) + args.num_layers
        if rest == "eh_proj.weight":
            return [(f"model.layers.{layer_idx}.eh_proj.weight", param)]
        elif rest == "enorm.weight":
            return [(f"model.layers.{layer_idx}.enorm.weight", param)]
        elif rest == "hnorm.weight":
            return [(f"model.layers.{layer_idx}.hnorm.weight", param)]
        elif rest == "final_layernorm.weight":
            return [(f"model.layers.{layer_idx}.shared_head.norm.weight", param)]
        else:
            name = f"module.module.decoder.layers.{layer_idx}.{rest}"
            name = name.replace("transformer_layer.", "")
            return convert_glm4moe_to_hf(args, name, param)

    raise ValueError(f"Unknown parameter name: {name}")


def _convert_npu_experts_and_mla(args, name, param, layer_idx, rest):
    """MindSpeed GroupedGemm / MLA mappings used only on NPU."""
    # MindSpeed GmmExpertsImpl: "mlp.experts.experts.linear_fc{1,2}.weight"
    # Standard Megatron: "mlp.experts.linear_fc{1,2}.weight{N}"
    expert_pattern = r"mlp.experts\.(.+)\.weight(\d*)$"
    match = re.match(expert_pattern, rest)
    if match:
        fc_name, expert_idx = match.groups()
        # Handle double "experts" in MindSpeed naming
        if fc_name.startswith("experts."):
            fc_name = fc_name[len("experts.") :]
        if fc_name == "linear_fc1":
            if expert_idx:
                # Standard Megatron: one expert per param
                gate_weight, up_weight = param.chunk(2, dim=0)
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight", gate_weight),
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight", up_weight),
                ]
            # MindSpeed GroupedGemm: all experts packed in one 3D param
            # Shape: [num_experts, fc1_output, hidden_size]
            # fc1_output = 2 * moe_ffn_hidden_size (gate+up packed)
            # vLLM expects gate_proj/up_proj: [intermediate, hidden_size]
            num_experts = args.num_experts
            if param.dim() == 3:
                # 3D: [num_experts, fc1_output, hidden_size] - slice on dim=1
                gate_weight, up_weight = param.chunk(2, dim=1)
                outputs = []
                for i in range(num_experts):
                    outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight", gate_weight[i]))
                    outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight", up_weight[i]))
            else:
                # 2D: [hidden_size, fc1_output * num_experts] - old format
                gate_up = param.view(num_experts, args.hidden_size, -1)
                gate_weight, up_weight = gate_up.chunk(2, dim=2)
                outputs = []
                for i in range(num_experts):
                    outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight", gate_weight[i].t()))
                    outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight", up_weight[i].t()))
            return outputs
        elif fc_name == "linear_fc2":
            if expert_idx:
                # Standard Megatron: one expert per param
                return [
                    (f"model.layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight", param),
                ]
            # MindSpeed GroupedGemm: all experts packed in one 3D param
            # Shape: [num_experts, hidden_size, fc2_input]
            # vLLM expects down_proj: [hidden_size, intermediate]
            num_experts = args.num_experts
            if param.dim() == 3:
                # 3D: [num_experts, hidden_size, fc2_input] - use directly
                outputs = []
                for i in range(num_experts):
                    outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight", param[i]))
            else:
                # 2D: [fc2_input * num_experts, hidden_size] - old format
                down = param.view(num_experts, -1, args.hidden_size)
                outputs = []
                for i in range(num_experts):
                    outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight", down[i].t()))
            return outputs
        else:
            raise ValueError(f"Unknown expert parameter name: {name}")

    # GroupedGemm format: weight1/weight2 (all experts packed together)
    if rest == "mlp.experts.weight1":
        # 3D: [num_experts, fc1_output, hidden_size] (MindSpeed GmmExpertsImpl)
        # 2D: [hidden_size, fc1_output * num_experts] (after EP all-gather + concat)
        num_experts = args.num_experts
        if param.dim() == 3:
            gate_weight, up_weight = param.chunk(2, dim=1)
            outputs = []
            for i in range(num_experts):
                outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight", gate_weight[i]))
                outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight", up_weight[i]))
        else:
            gate_up = param.view(num_experts, args.hidden_size, -1)
            gate_weight, up_weight = gate_up.chunk(2, dim=2)
            outputs = []
            for i in range(num_experts):
                outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.gate_proj.weight", gate_weight[i].t()))
                outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.up_proj.weight", up_weight[i].t()))
        return outputs
    elif rest == "mlp.experts.weight2":
        # 3D: [num_experts, hidden_size, fc2_input] (MindSpeed GmmExpertsImpl)
        # 2D: [fc2_input * num_experts, hidden_size] (after EP all-gather + concat)
        num_experts = args.num_experts
        if param.dim() == 3:
            outputs = []
            for i in range(num_experts):
                outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight", param[i]))
        else:
            down = param.view(num_experts, -1, args.hidden_size)
            outputs = []
            for i in range(num_experts):
                outputs.append((f"model.layers.{layer_idx}.mlp.experts.{i}.down_proj.weight", down[i].t()))
        return outputs

    # MindSpeed MLA attention: separate q/kv projections
    if rest == "self_attention.linear_q_down_proj.weight":
        return [(f"model.layers.{layer_idx}.self_attn.q_a_proj.weight", param)]
    elif rest == "self_attention.linear_q_up_proj.weight":
        return [(f"model.layers.{layer_idx}.self_attn.q_b_proj.weight", param)]
    elif rest == "self_attention.linear_kv_down_proj.weight":
        return [(f"model.layers.{layer_idx}.self_attn.kv_a_proj_with_mqa.weight", param)]
    elif rest == "self_attention.linear_kv_up_proj.weight":
        return [(f"model.layers.{layer_idx}.self_attn.kv_b_proj.weight", param)]

    return None
