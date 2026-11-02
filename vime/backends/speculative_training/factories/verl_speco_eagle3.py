from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig


def build_model(args, device: torch.device) -> torch.nn.Module:
    """Load a vLLM-style EAGLE3 checkpoint with verl-SpeCo's trainable model.

    Public EAGLE3 checkpoints usually declare ``Eagle3LlamaForCausalLM`` but do
    not ship Transformers ``auto_map`` code.  Their state-dict layout matches
    verl-SpeCo's ``LlamaForCausalLMEagle3`` (``midlayer.*``, ``fc``, ``norm``
    and ``lm_head``), so construct that model explicitly and leave the missing
    Target embedding to VIME's normal embedding initializer.
    """

    from verl_speco.models.eagle import llama_eagle

    # verl-SpeCo decorates RMSNorm with torch.compile.  The current Ascend
    # torch_npu/Triton stack can generate an invalid vector kernel for this very
    # small normalization.  Restore the eager implementation; it is negligible
    # beside the decoder layer and is stable on both CUDA and NPU.
    def eager(callable_):
        while hasattr(callable_, "__wrapped__"):
            callable_ = callable_.__wrapped__
        return callable_

    llama_eagle.apply_rotary_pos_emb = eager(llama_eagle.apply_rotary_pos_emb)
    llama_eagle.LlamaRotaryEmbedding.forward = eager(llama_eagle.LlamaRotaryEmbedding.forward)
    llama_eagle.LlamaRMSNorm.forward = eager(llama_eagle.LlamaRMSNorm.forward)
    LlamaForCausalLMEagle3 = llama_eagle.LlamaForCausalLMEagle3

    checkpoint_dir = Path(args.draft_model_path)
    config_path = checkpoint_dir / "config.json"
    weights_path = checkpoint_dir / "model.safetensors"
    if not config_path.is_file() or not weights_path.is_file():
        raise FileNotFoundError(f"Expected config.json and model.safetensors under {checkpoint_dir}")

    with config_path.open(encoding="utf-8") as handle:
        config_dict = json.load(handle)
    layer_ids = [int(value) for value in args.draft_feature_layer_ids]
    config_dict["architectures"] = ["LlamaForCausalLMEagle3"]
    config_dict["target_hidden_size"] = int(args.hidden_size)
    config_dict["num_aux_hidden_states"] = len(layer_ids)
    config_dict["eagle_aux_hidden_state_layer_ids"] = layer_ids
    config_dict["target_hidden_layer_ids"] = layer_ids
    config_dict["tie_word_embeddings"] = False
    config_dict.setdefault("pad_token_id", 0)
    config = LlamaConfig.from_dict(config_dict)

    model = LlamaForCausalLMEagle3(config, attention_backend="sdpa")
    state_dict = load_file(str(weights_path), device="cpu")
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    allowed_missing = {"embed_tokens.weight"}
    unexpected = set(incompatible.unexpected_keys)
    if missing - allowed_missing or unexpected:
        raise RuntimeError(
            "EAGLE3 checkpoint does not match the verl-SpeCo training model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return model.to(device=device, dtype=torch.bfloat16)
