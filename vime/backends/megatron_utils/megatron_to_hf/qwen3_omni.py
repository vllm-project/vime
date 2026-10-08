from .qwen3moe import convert_qwen3moe_to_hf


def convert_qwen3_omni_to_hf(args, name, param):
    prefixes = {
        "module.module.audio_model.": "thinker.audio_tower.",
        "module.module.vision_model.": "thinker.visual.",
    }
    for model_prefix, hf_prefix in prefixes.items():
        if name.startswith(model_prefix):
            return [(hf_prefix + name.removeprefix(model_prefix), param)]

    name = name.replace("module.module.language_model.", "module.module.", 1)
    return [("thinker." + hf_name, tensor) for hf_name, tensor in convert_qwen3moe_to_hf(args, name, param)]
