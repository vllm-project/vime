"""GLM-4.7-Flash (``glm4_moe_lite``) bridge for megatron.bridge.

Registers ``Glm4MoeLiteForCausalLM`` so that ``AutoBridge.from_hf_pretrained``
recognises GLM-4.7-Flash checkpoints and can provide a Megatron-compatible model +
weight mappings, with Multi-Token Prediction (MTP) support on the Ascend 910B NPU.

Architecture:
  MLA (Multi-Head Latent Attention, DeepSeek-V3-style)
  + GLM-style MoE (64 routed experts + 1 shared expert, sigmoid router with
  expert bias), subclassing ``GLM45Bridge`` to reuse its MTP plumbing.
"""

import logging
from functools import partial

import torch
from megatron.bridge.models.conversion.mapping_registry import MegatronMappingRegistry
from megatron.bridge.models.conversion.model_bridge import MegatronModelBridge
from megatron.bridge.models.conversion.param_mapping import AutoMapping, GatedMLPMapping
from megatron.bridge.models.glm.glm45_bridge import GLM45Bridge
from megatron.bridge.models.glm.glm_moe_mappings import GLMExpertDownProjMapping, GLMExpertGateUpProjMapping
from megatron.bridge.models.hf_pretrained.causal_lm import PreTrainedCausalLM
from megatron.bridge.models.mla_provider import MLAModelProvider
from megatron.core.models.gpt import GPTModel
from megatron.core.models.gpt.gpt_layer_specs import get_gpt_decoder_block_spec
from transformers import Glm4MoeLiteForCausalLM

try:
    import transformer_engine  # noqa: F401

    HAVE_TE = True
except (ImportError, ModuleNotFoundError):
    HAVE_TE = False


logger = logging.getLogger(__name__)


@MegatronModelBridge.register_bridge(
    source=Glm4MoeLiteForCausalLM,
    target=GPTModel,
    model_type="glm4_moe_lite",
)
class GLM47MTPBridge(GLM45Bridge):
    """Megatron bridge for GLM-4.7-Flash (glm4_moe_lite) with MTP support.

    GLM-4.7-Flash is an MLA model (``q_lora_rank`` in its config), so it needs
    the MLAModelProvider and the MLA weight mappings.  GLM45Bridge's stock
    provider_bridge / mapping_registry only handle non-MLA GLM-4.5 (fused QKV);
    we override both with the MLA-aware versions.  Everything else
    (``build_conversion_tasks``, MTP loop, fused-expert handling) is inherited.
    """

    def provider_bridge(self, hf_pretrained: PreTrainedCausalLM):
        """Convert HuggingFace config to MLAModelProvider."""
        provider_kwargs = self.hf_config_to_provider_kwargs(hf_pretrained.config)
        mla_rope = provider_kwargs.pop("_mla_rope_params", None)
        provider_class = self.PROVIDER_CLASS if self.PROVIDER_CLASS is not None else MLAModelProvider
        provider = provider_class(**provider_kwargs)

        # Set rope type
        hf_rope_scaling = getattr(hf_pretrained.config, "rope_scaling", None)
        rope_type = None
        if hf_rope_scaling:
            rope_type = hf_rope_scaling.get("type") or hf_rope_scaling.get("rope_type")
        if rope_type != "yarn":
            provider.position_embedding_type = "rope"

        # Match vLLM defaults (no scaling, mscale=1.0) when HF config has no explicit rope params.
        if not mla_rope:
            mla_rope = {"rotary_scaling_factor": 1.0, "mscale_all_dim": 1.0}

        if mla_rope:
            for key, value in mla_rope.items():
                setattr(provider, key, value)
        hf_config = hf_pretrained.config

        # Use decoder block spec to properly handle moe_layer_freq (mixed dense/MoE layers)
        provider.transformer_layer_spec = partial(get_gpt_decoder_block_spec, use_transformer_engine=HAVE_TE)
        provider.normalization = "RMSNorm"
        provider.gated_linear_unit = True
        provider.add_bias_linear = False
        provider.share_embeddings_and_output_weights = False
        provider.multi_latent_attention = True
        provider.qk_layernorm = True

        provider.moe_shared_expert_overlap = True
        provider.moe_token_dispatcher_type = "alltoall"
        provider.moe_router_load_balancing_type = "seq_aux_loss"
        provider.moe_router_pre_softmax = True
        provider.moe_grouped_gemm = True
        provider.moe_router_score_function = "sigmoid"
        provider.moe_permute_fusion = True
        provider.moe_router_enable_expert_bias = True
        provider.moe_router_dtype = "fp32"
        provider.moe_router_bias_update_rate = 0
        provider.moe_aux_loss_coeff = 0.001

        provider.persist_layer_norm = True
        provider.bias_activation_fusion = True
        provider.bias_dropout_fusion = True
        provider.hidden_dropout = 0.0
        provider.autocast_dtype = torch.bfloat16
        provider.mtp_loss_scaling_factor = 0.3
        provider.moe_shared_expert_intermediate_size = hf_config.moe_intermediate_size * int(
            getattr(hf_config, "n_shared_experts", 1)
        )

        provider.moe_layer_freq = [0] * hf_config.first_k_dense_replace + [1] * (
            hf_config.num_hidden_layers - hf_config.first_k_dense_replace
        )

        return provider

    def mapping_registry(self) -> MegatronMappingRegistry:
        mapping_list = []
        use_fused_experts = self._uses_fused_experts()
        gate_up_suffix = self._hf_expert_suffix("mlp.experts.gate_up_proj")
        down_suffix = self._hf_expert_suffix("mlp.experts.down_proj")

        param_mappings = {
            # Embed
            "embedding.word_embeddings.weight": "model.embed_tokens.weight",
            # LM Head
            "decoder.final_layernorm.weight": "model.norm.weight",
            "output_layer.weight": "lm_head.weight",
        }

        layer_specific_mappings = {
            # Attention shared by all GLM variants
            "decoder.layers.*.input_layernorm.weight": "model.layers.*.input_layernorm.weight",
            "decoder.layers.*.self_attention.linear_proj.weight": "model.layers.*.self_attn.o_proj.weight",
            "decoder.layers.*.pre_mlp_layernorm.weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.self_attention.q_layernorm.weight": "model.layers.*.self_attn.q_a_layernorm.weight",
            "decoder.layers.*.self_attention.k_layernorm.weight": "model.layers.*.self_attn.k_norm.weight",
            # MLA-specific layernorm
            "decoder.layers.*.self_attention.kv_layernorm.weight": "model.layers.*.self_attn.kv_a_layernorm.weight",
            # MLP
            "decoder.layers.*.mlp.linear_fc2.weight": "model.layers.*.mlp.down_proj.weight",
            "decoder.layers.*.mlp.linear_fc1.layer_norm_weight": "model.layers.*.post_attention_layernorm.weight",
            "decoder.layers.*.mlp.shared_experts.linear_fc2.weight": "model.layers.*.mlp.shared_experts.down_proj.weight",
            "decoder.layers.*.mlp.shared_experts.router.weight": "model.layers.*.mlp.shared_experts.gate.weight",
            "decoder.layers.*.mlp.router.weight": "model.layers.*.mlp.gate.weight",
            "decoder.layers.*.mlp.router.expert_bias": "model.layers.*.mlp.gate.e_score_correction_bias",
        }

        for megatron_param, hf_param in param_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        for megatron_param, hf_param in layer_specific_mappings.items():
            mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

        # Add special mappings that require parameter concatenation/transformation
        mapping_list.extend(
            [
                # MLA attention: individual Q/KV down/up projections (for GLM-4.7-Flash)
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_q_down_proj.weight",
                    hf_param="model.layers.*.self_attn.q_a_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_q_up_proj.weight",
                    hf_param="model.layers.*.self_attn.q_b_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_kv_down_proj.weight",
                    hf_param="model.layers.*.self_attn.kv_a_proj_with_mqa.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_kv_up_proj.weight",
                    hf_param="model.layers.*.self_attn.kv_b_proj.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_q_up_proj.layer_norm_weight",
                    hf_param="model.layers.*.self_attn.q_a_layernorm.weight",
                ),
                AutoMapping(
                    megatron_param="decoder.layers.*.self_attention.linear_kv_up_proj.layer_norm_weight",
                    hf_param="model.layers.*.self_attn.kv_a_layernorm.weight",
                ),
                # Gated MLP: Combine gate and up projection matrices into single FC1 matrix
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.linear_fc1.weight",
                    gate="model.layers.*.mlp.gate_proj.weight",
                    up="model.layers.*.mlp.up_proj.weight",
                ),
                GatedMLPMapping(
                    megatron_param="decoder.layers.*.mlp.shared_experts.linear_fc1.weight",
                    gate="model.layers.*.mlp.shared_experts.gate_proj.weight",
                    up="model.layers.*.mlp.shared_experts.up_proj.weight",
                ),
            ]
        )
        if use_fused_experts:
            mapping_list.extend(
                [
                    GLMExpertGateUpProjMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        hf_param=f"model.layers.*.mlp.experts.gate_up_proj{gate_up_suffix}",
                    ),
                    GLMExpertDownProjMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param=f"model.layers.*.mlp.experts.down_proj{down_suffix}",
                    ),
                ]
            )
        else:
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc1.weight*",
                        gate="model.layers.*.mlp.experts.*.gate_proj.weight",
                        up="model.layers.*.mlp.experts.*.up_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param="decoder.layers.*.mlp.experts.linear_fc2.weight*",
                        hf_param="model.layers.*.mlp.experts.*.down_proj.weight",
                    ),
                ]
            )
        # optionally add MTP mappings
        if not hasattr(self, "_hf_config"):
            logger.warning("No HF config found, skipping MTP mappings.")
            return MegatronMappingRegistry(*mapping_list)
        hf_config = self._hf_config
        num_mtp_layers = getattr(hf_config, "num_nextn_predict_layers", 0)
        num_transformer_layers = hf_config.num_hidden_layers
        for mtp_layer in range(num_mtp_layers):
            for megatron_param, hf_param in layer_specific_mappings.items():
                megatron_param = (
                    megatron_param.replace(".*", ".*.transformer_layer")
                    .replace("decoder", "mtp")
                    .replace(".*", f".{mtp_layer}")
                )
                hf_param = hf_param.replace("layers.*", f"layers.{mtp_layer + num_transformer_layers}")
                mapping_list.append(AutoMapping(megatron_param=megatron_param, hf_param=hf_param))

            # MTP specific mappings
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.enorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.enorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.hnorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.hnorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.eh_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.eh_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.final_layernorm.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.shared_head.norm.weight",
                    ),
                ]
            )
            # MTP transformer layer reuses the last normal layer spec (MLA), so map
            # the individual Q/KV down/up projections instead of a fused QKV.
            mapping_list.extend(
                [
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_down_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_a_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_up_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_b_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_kv_down_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.kv_a_proj_with_mqa.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_kv_up_proj.weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.kv_b_proj.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_q_up_proj.layer_norm_weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.q_a_layernorm.weight",
                    ),
                    AutoMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.self_attention.linear_kv_up_proj.layer_norm_weight",
                        hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.self_attn.kv_a_layernorm.weight",
                    ),
                ]
            )
            # MTP transformer layer MLP mappings
            mapping_list.extend(
                [
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.gate.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.linear_fc1.up.weight",
                    ),
                    GatedMLPMapping(
                        megatron_param=f"mtp.layers.{mtp_layer}.transformer_layer.mlp.shared_experts.linear_fc1.weight",
                        gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.gate_proj.weight",
                        up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.shared_experts.up_proj.weight",
                    ),
                ]
            )
            if use_fused_experts:
                mapping_list.extend(
                    [
                        GLMExpertGateUpProjMapping(
                            megatron_param=(
                                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc1.weight*"
                            ),
                            hf_param=(
                                f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.gate_up_proj"
                                f"{gate_up_suffix}"
                            ),
                        ),
                        GLMExpertDownProjMapping(
                            megatron_param=(
                                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc2.weight*"
                            ),
                            hf_param=(
                                f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.down_proj{down_suffix}"
                            ),
                        ),
                    ]
                )
            else:
                mapping_list.extend(
                    [
                        GatedMLPMapping(
                            megatron_param=(
                                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc1.weight*"
                            ),
                            gate=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.gate_proj.weight",
                            up=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.up_proj.weight",
                        ),
                        AutoMapping(
                            megatron_param=(
                                f"mtp.layers.{mtp_layer}.transformer_layer.mlp.experts.linear_fc2.weight*"
                            ),
                            hf_param=f"model.layers.{mtp_layer + num_transformer_layers}.mlp.experts.*.down_proj.weight",
                        ),
                    ]
                )

        return MegatronMappingRegistry(*mapping_list)


def _register_mindspeed_te_module_types():
    """Register MindSpeed TE module types for weight-mapping parallelism detection."""
    try:
        from megatron.bridge.models.conversion.param_mapping import AutoMapping
    except ImportError:
        return

    for module_name, parallelism_type in {
        "MindSpeedTEColumnParallelLinear": "column",
        "MindSpeedTELayerNormColumnParallelLinear": "column",
        "MindSpeedTEColumnParallelGroupedLinear": "column",
        "MindSpeedTEGroupedLinear": "column",
        "MindSpeedTEGroupedLinearGMM": "column",
        "MindSpeedTEDotProductAttention": "column",
        "MindSpeedTERowParallelGroupedLinear": "row",
        "MindSpeedTELayernorm": "replicated",
        "MindSpeedTELinear": "replicated",
    }.items():
        AutoMapping.register_module_type(module_name, parallelism_type)


_register_mindspeed_te_module_types()
