"""Use Megatron's native DeepSeek-V4 hybrid attention implementation."""

from megatron.core.models.gpt.experimental_attention_variant_module_specs import (
    get_transformer_block_with_experimental_attention_variant_spec,
)


def get_dsv4_spec(args, config, vp_stage=None):
    if config.experimental_attention_variant != "dsv4_hybrid":
        raise ValueError("DeepSeek-V4 requires --experimental-attention-variant dsv4_hybrid")
    if args.mtp_num_layers:
        raise ValueError("DeepSeek-V4 MTP is not supported by this recipe")
    return get_transformer_block_with_experimental_attention_variant_spec(config, vp_stage=vp_stage)
