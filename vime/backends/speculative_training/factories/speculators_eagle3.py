"""Built-in Speculators factory for EAGLE3 online draft training."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from safetensors.torch import load_file
from transformers import LlamaConfig


from ..config import load_draft_checkpoint_config

# vLLM Eagle3LlamaForCausalLM checkpoints name the single draft decoder
# layer "midlayer.*" while Speculators (and vLLM after its own load-time
# remap) name it "layers.0.*".  The remaining tensors (fc, norm, lm_head,
# t2d, d2t) already share names across both layouts.
_VLLM_LAYER_PREFIX = "midlayer."
_SPECULATORS_LAYER_PREFIX = "layers.0."

# Keys the Speculators model expects but the EAGLE3 checkpoint does not
# ship: the embedding and verifier projection are loaded from the Target,
# and the vocab mappings go through their dedicated API below.
_TARGET_OWNED_KEYS = (
    "embed_tokens.weight",
    "verifier_lm_head.weight",
    "verifier_norm.weight",
    "t2d",
    "d2t",
)


def _load_speculators_types():
    try:
        from speculators.config import SpeculatorsConfig, VerifierConfig
        from speculators.models.eagle3.config import Eagle3SpeculatorConfig
        from speculators.models.eagle3.core import Eagle3DraftModel
        from speculators.proposals.greedy import GreedyTokenProposalConfig
    except ImportError as exc:
        raise ImportError("EAGLE3 online training requires a compatible speculators package") from exc
    return (
        Eagle3SpeculatorConfig,
        Eagle3DraftModel,
        SpeculatorsConfig,
        GreedyTokenProposalConfig,
        VerifierConfig,
    )


def _build_config(
    args: Any,
    config_type: type,
    speculators_config_type: type,
    proposal_config_type: type,
    verifier_config_type: type,
) -> tuple[Any, str]:
    """Translate a vLLM-layout EAGLE3 checkpoint config for Speculators."""

    config_dict = load_draft_checkpoint_config(args)
    model_type = str(config_dict.get("model_type", "")).lower()
    if model_type != "llama":
        raise ValueError(
            "Speculators EAGLE3 supports vLLM Llama-family EAGLE3 "
            f"checkpoints; --draft-model-path declares model_type={model_type!r}"
        )
    if config_dict.get("draft_vocab_size") is None:
        raise ValueError("Speculators EAGLE3 requires a draft_vocab_size entry in the " "draft checkpoint config")
    layer_ids = [int(value) for value in args.draft_feature_layer_ids]
    checkpoint_layer_ids = [int(value) for value in (config_dict.get("eagle_aux_hidden_state_layer_ids") or [])]
    if not checkpoint_layer_ids:
        raise ValueError(
            "EAGLE3 checkpoint config must declare eagle_aux_hidden_state_layer_ids: vLLM "
            "silently serves with its fallback layers [2, num_layers // 2, num_layers - 3] "
            "when the field is missing, so the engine and Draft training would consume "
            "different layers than the fc weight was trained on"
        )
    if checkpoint_layer_ids != layer_ids:
        raise ValueError(
            "EAGLE3 checkpoint eagle_aux_hidden_state_layer_ids "
            f"{checkpoint_layer_ids} do not match the feature collection "
            f"configuration {layer_ids}"
        )
    target_path = str(getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None))
    if not target_path:
        raise ValueError(
            "EAGLE3 draft training requires a Target model path " "(--draft-target-embedding-path or --hf-checkpoint)"
        )

    # Speculators consumes the checkpoint config as a plain LlamaConfig for
    # the single draft decoder layer; a None implementation lets the model
    # pick its own default (the factory pins eager on NPU below).
    transformer_config = LlamaConfig.from_dict(dict(config_dict))
    transformer_config._attn_implementation = None

    target_hidden_size = int(getattr(args, "hidden_size", 0) or 0)
    if target_hidden_size and target_hidden_size != transformer_config.hidden_size:
        raise ValueError(
            "Speculators EAGLE3 requires matching hidden sizes, got "
            f"target={target_hidden_size} and draft={transformer_config.hidden_size}"
        )

    config = config_type(
        transformer_layer_config=transformer_config,
        draft_vocab_size=int(config_dict["draft_vocab_size"]),
        target_hidden_size=target_hidden_size or transformer_config.hidden_size,
        eagle_aux_hidden_state_layer_ids=layer_ids,
        speculators_config=speculators_config_type(
            algorithm="eagle3",
            proposal_methods=[proposal_config_type(speculative_tokens=int(args.draft_ttt_length))],
            default_proposal_method="greedy",
            verifier=verifier_config_type.from_pretrained(target_path),
        ),
    )
    return config, target_path


def _load_weights(model: Any, checkpoint_dir: Path) -> tuple[Any, Any]:
    """Load a vLLM-layout EAGLE3 checkpoint into the Speculators model."""

    weights_path = checkpoint_dir / "model.safetensors"
    if not weights_path.is_file():
        raise FileNotFoundError(f"Speculators EAGLE3 expects model.safetensors under {checkpoint_dir}")
    state_dict = load_file(str(weights_path), device="cpu")
    t2d = state_dict.pop("t2d", None)
    d2t = state_dict.pop("d2t", None)
    state_dict = {
        (
            _SPECULATORS_LAYER_PREFIX + key[len(_VLLM_LAYER_PREFIX) :] if key.startswith(_VLLM_LAYER_PREFIX) else key
        ): value
        for key, value in state_dict.items()
    }
    incompatible = model.load_state_dict(state_dict, strict=False)
    missing = set(incompatible.missing_keys)
    unexpected = set(incompatible.unexpected_keys)
    if missing - set(_TARGET_OWNED_KEYS) or unexpected:
        raise RuntimeError(
            "EAGLE3 checkpoint does not match the Speculators Eagle3 model: "
            f"missing={sorted(missing)}, unexpected={sorted(unexpected)}"
        )
    return t2d, d2t


def build_model(args: Any, device: torch.device) -> torch.nn.Module:
    """Load a vLLM-style EAGLE3 checkpoint with the canonical Speculators model."""

    (
        config_type,
        model_type,
        speculators_config_type,
        proposal_config_type,
        verifier_config_type,
    ) = _load_speculators_types()
    config, _ = _build_config(args, config_type, speculators_config_type, proposal_config_type, verifier_config_type)

    if device.type == "npu":
        # Speculators defaults to flex attention; the eager mask path is the
        # stable one on Ascend.
        config.transformer_layer_config._attn_implementation = "eager"

    model = model_type(config)
    t2d, d2t = _load_weights(model, Path(str(args.draft_model_path)))
    if t2d is not None or d2t is not None:
        model.load_vocab_mappings(
            t2d.to(torch.bool) if torch.is_tensor(t2d) else t2d,
            d2t.to(torch.long) if torch.is_tensor(d2t) else d2t,
        )
    model.load_verifier_weights()
    # VIME collects the already-normalized tensor that feeds the Target LM
    # head, so the Speculators verifier norm must be an identity.
    model.verifier_norm = torch.nn.Identity()
    # The checkpoint ships a trained draft head; keep it trainable so
    # --draft-publish sends it.  The verifier projection stays frozen and is
    # re-synced from the Target on every weight version.
    model.lm_head.weight.requires_grad_(True)
    return model.to(device=device, dtype=torch.bfloat16)
