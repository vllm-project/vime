"""Load checkpoint configurations also supported by the rollout backend."""

from transformers import AutoConfig, PretrainedConfig


def _normalize_glm5_next_gate(config_dict):
    linear = config_dict.get("text_config", config_dict).get("linear_attn_config", {})
    # Older checkpoints encode sigmoid gates bounded by -5 as safe_gate=True.
    # Recent SGLang kernels select that formula using gate_lower_bound instead.
    if "safe_gate" in linear and "gate_lower_bound" not in linear:
        linear["gate_lower_bound"] = -5.0 if linear["safe_gate"] else None


def load_hf_config(path, **kwargs):
    config_dict, _ = PretrainedConfig.get_config_dict(path, **kwargs)
    if config_dict.get("model_type") == "glm5_next":
        # Reuse the Transformers config schema used by vLLM for actor and rollout.
        from transformers import Glm5NextConfig

        _normalize_glm5_next_gate(config_dict)
        return Glm5NextConfig.from_dict(config_dict)
    return AutoConfig.from_pretrained(path, **kwargs)


def vllm_model_config_overrides(path, overrides):
    """Translate legacy checkpoint fields while preserving explicit server overrides."""
    config_dict, _ = PretrainedConfig.get_config_dict(path)
    if config_dict.get("model_type") != "glm5_next":
        return overrides
    values = dict(overrides or {})
    text = config_dict.get("text_config", config_dict)
    text_override = values.get("text_config", {})
    linear = {**text.get("linear_attn_config", {}), **text_override.get("linear_attn_config", {})}
    if not linear:
        return overrides
    _normalize_glm5_next_gate({"linear_attn_config": linear})
    values["text_config"] = {
        "linear_lower_bound": linear.get("gate_lower_bound"),
        **text_override,
        "linear_attn_config": linear,
    }
    return values
