"""Load one full-size V4-Flash Megatron layer from HF weights and train it once."""

import argparse
import importlib.util
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _build_dsv4_moe_layer,
    _make_config,
)

from vime.backends.megatron_utils.deepseek_v4_weights import deepseek_v4_hf_tensor


def checkpoint_reader(path):
    source = Path(__file__).resolve().parents[1] / "vime/backends/megatron_utils/hf_to_megatron/common.py"
    spec = importlib.util.spec_from_file_location("dsv4_safetensor_reader", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.SafetensorReader(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("--layer-index", type=int, default=2)
    args = parser.parse_args()
    hf_config = json.loads((args.checkpoint / "config.json").read_text())
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    try:
        config = _make_config(
            num_layers=hf_config["num_hidden_layers"],
            hidden_size=hf_config["hidden_size"],
            num_attention_heads=hf_config["num_attention_heads"],
            v_head_dim=hf_config["head_dim"],
            qk_pos_emb_head_dim=hf_config["qk_rope_head_dim"],
            q_lora_rank=hf_config["q_lora_rank"],
            output_projection_groups=hf_config["o_groups"],
            output_projection_lora_rank=hf_config["o_lora_rank"],
            csa_compress_ratios=hf_config["compress_ratios"][:43],
            csa_window_size=hf_config["sliding_window"],
            dsa_indexer_n_heads=hf_config["index_n_heads"],
            dsa_indexer_head_dim=hf_config["index_head_dim"],
            dsa_indexer_topk=hf_config["index_topk"],
            num_moe_experts=hf_config["n_routed_experts"],
            moe_ffn_hidden_size=hf_config["moe_intermediate_size"],
            moe_layer_freq=1,
            moe_router_topk=hf_config["num_experts_per_tok"],
            moe_router_score_function=hf_config["scoring_func"],
            moe_router_dtype="fp32",
            moe_router_enable_expert_bias=True,
            moe_n_hash_layers=hf_config["num_hash_layers"],
            actual_vocab_size=hf_config["vocab_size"],
            activation_func=F.silu,
            gated_linear_unit=True,
            activation_func_clamp_value=hf_config["swiglu_limit"],
            moe_grouped_gemm=True,
            qk_layernorm=True,
            enable_hyper_connections=True,
            mhc_num_residual_streams=hf_config["hc_mult"],
            mhc_sinkhorn_iterations=hf_config["hc_sinkhorn_iters"],
            layernorm_epsilon=hf_config["rms_norm_eps"],
            attention_latent_norm_epsilon=hf_config["rms_norm_eps"],
            csa_compress_rotary_base=hf_config["compress_rope_theta"],
            rotary_scaling_factor=hf_config["rope_scaling"]["factor"],
            original_max_position_embeddings=hf_config["rope_scaling"]["original_max_position_embeddings"],
        )
        layer_id = args.layer_index
        pg = ProcessGroupCollection.use_mpu_process_groups()
        layer = _build_dsv4_moe_layer(config, layer_id + 1, pg).cuda()
        reader = checkpoint_reader(args.checkpoint)
        loaded = 0
        with torch.no_grad():
            for name, parameter in layer.named_parameters():
                key = f"decoder.layers.{layer_id}.{name}"
                tensor = deepseek_v4_hf_tensor(key, reader, hf_config)
                if tensor.shape != parameter.shape:
                    raise ValueError(f"{key}: checkpoint {tuple(tensor.shape)}, Megatron {tuple(parameter.shape)}")
                parameter.copy_(tensor.to(device="cuda", dtype=parameter.dtype))
                loaded += 1
            for name, buffer in layer.named_buffers():
                if name.endswith(("router.expert_bias", "router.tid2eid")):
                    tensor = deepseek_v4_hf_tensor(f"decoder.layers.{layer_id}.{name}", reader, hf_config)
                    if tensor.shape != buffer.shape:
                        raise ValueError(f"{name}: checkpoint {tuple(tensor.shape)}, Megatron {tuple(buffer.shape)}")
                    buffer.copy_(tensor.to(device="cuda", dtype=buffer.dtype))
                    loaded += 1
        reader._files.clear()
        torch.cuda.empty_cache()
        print(f"loaded={loaded}", flush=True)
        hidden = torch.randn(256, 1, config.hidden_size * hf_config["hc_mult"], device="cuda", dtype=torch.bfloat16, requires_grad=True)
        input_ids = torch.randint(0, hf_config["vocab_size"], (1, 256), device="cuda")
        output, _ = layer(hidden_states=hidden, attention_mask=None, input_ids=input_ids)
        loss = output.float().square().mean()
        loss.backward()
        grad_count = sum(p.grad is not None and torch.isfinite(p.grad).all().item() for p in layer.parameters())
        assert torch.isfinite(loss) and grad_count > 0
        print(f"loaded={loaded} loss={loss.item():.6g} finite_gradient_tensors={grad_count}")
    finally:
        Utils.destroy_model_parallel()


if __name__ == "__main__":
    main()
