from __future__ import annotations

from copy import deepcopy

import torch

from ..config import load_draft_checkpoint_config


_TARGET_OWNED_PREFIXES = (
    "embed_tokens.",
    "lm_head.",
    "verifier_lm_head.",
    "verifier_norm.",
)


def _load_speculators_types():
    try:
        from speculators.models.dspark.config import DSparkSpeculatorConfig
        from speculators.models.dspark.core import DSparkDraftModel
    except ImportError as exc:
        raise ImportError(
            "Qwen DSpark online training requires compatible speculators and hs_connectors packages"
        ) from exc
    return DSparkSpeculatorConfig, DSparkDraftModel


def _build_config(args, config_type):
    config_dict = deepcopy(load_draft_checkpoint_config(args))
    checkpoint_type = str(config_dict.get("speculators_model_type", "")).lower()
    architectures = config_dict.get("architectures") or []
    if checkpoint_type != "dspark" and "Qwen3DSparkModel" not in architectures:
        raise ValueError("--draft-model-path must point to a DSpark checkpoint")

    # vLLM stores the Qwen architecture at the top level, while Speculators
    # expects the same fields under transformer_layer_config. Without this
    # translation, Speculators silently constructs its default Qwen3Config
    # (hidden_size=4096) instead of using the checkpoint's real layout.
    if str(config_dict.get("model_type", "")).lower() == "qwen3":
        speculator_fields = set(config_type.model_fields)
        config_dict["transformer_layer_config"] = deepcopy(
            {
                name: value
                for name, value in config_dict.items()
                if name not in speculator_fields and name != "target_layer_ids"
            }
        )
        if config_dict.get("aux_hidden_state_layer_ids") is None and config_dict.get("target_layer_ids") is not None:
            config_dict["aux_hidden_state_layer_ids"] = [
                int(layer_id) + 1 for layer_id in config_dict["target_layer_ids"]
            ]
        if config_dict.get("draft_vocab_size") is None and config_dict.get("vocab_size") is not None:
            config_dict["draft_vocab_size"] = int(config_dict["vocab_size"])

    config_dict["speculators_model_type"] = "dspark"
    if config_dict.get("speculators_config") is None:
        config_dict.pop("speculators_config", None)

    config = config_type.from_dict(config_dict)
    expected_layers = tuple(int(value) for value in args.draft_feature_layer_ids)
    configured_layers = getattr(config, "aux_hidden_state_layer_ids", None)
    if configured_layers is None:
        config.aux_hidden_state_layer_ids = list(expected_layers)
    elif tuple(int(value) for value in configured_layers) != expected_layers:
        raise ValueError(
            "DSpark checkpoint aux_hidden_state_layer_ids do not match feature collection: "
            f"{tuple(configured_layers)} != {expected_layers}"
        )
    return config


def _validate_config(args, config) -> None:
    transformer = config.transformer_layer_config
    if str(getattr(transformer, "model_type", "")).lower() != "qwen3":
        raise ValueError("VIME supports only dense Qwen3 DSpark checkpoints")
    if str(getattr(config, "markov_head_type", "vanilla")).lower() != "vanilla":
        raise ValueError("vLLM Qwen3 DSpark serving supports only markov_head_type='vanilla'")

    draft_hidden_size = int(transformer.hidden_size)
    target_hidden_size = int(getattr(config, "target_hidden_size", None) or draft_hidden_size)
    configured_target_size = int(getattr(args, "hidden_size", 0) or 0)
    if target_hidden_size != draft_hidden_size or (
        configured_target_size and target_hidden_size != configured_target_size
    ):
        raise ValueError(
            "DSpark, its verifier config, and the Megatron Target must use the same hidden size: "
            f"draft={draft_hidden_size}, verifier={target_hidden_size}, target={configured_target_size or 'unknown'}"
        )

    mask_token_id = getattr(config, "mask_token_id", None)
    if mask_token_id is None or not 0 <= int(mask_token_id) < int(transformer.vocab_size):
        raise ValueError("DSpark mask_token_id must be inside the verifier vocabulary")
    if (
        bool(getattr(config, "enable_confidence_head", False))
        and bool(getattr(config, "confidence_head_with_markov", False))
        and int(getattr(config, "markov_rank", 0)) <= 0
    ):
        raise ValueError("confidence_head_with_markov requires markov_rank > 0")


def _load_model(model_type, args, config):
    model, loading_info = model_type.from_pretrained(
        args.draft_model_path,
        config=config,
        output_loading_info=True,
    )
    missing = [name for name in loading_info.get("missing_keys", []) if not name.startswith(_TARGET_OWNED_PREFIXES)]
    mismatched = loading_info.get("mismatched_keys", [])
    if missing or mismatched:
        raise ValueError(
            "DSpark checkpoint is incomplete or incompatible with its config: "
            f"missing={missing}, mismatched={mismatched}"
        )
    return model


def build_model(args, device: torch.device) -> torch.nn.Module:
    """Load the canonical Speculators DSpark training model."""

    config_type, model_type = _load_speculators_types()
    config = _build_config(args, config_type)
    _validate_config(args, config)

    if device.type == "npu":
        config.transformer_layer_config._attn_implementation = "eager"

    target_path = getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None)
    verifier = getattr(getattr(config, "speculators_config", None), "verifier", None)
    if verifier is not None and target_path:
        verifier.name_or_path = str(target_path)

    model = _load_model(model_type, args, config)
    expected_layers = tuple(int(value) for value in args.draft_feature_layer_ids)
    if tuple(int(value) for value in model.target_layer_ids) != expected_layers:
        raise ValueError("Loaded DSpark model uses different Target feature layers")
    if int(model.block_size) != int(args.draft_dspark_block_size):
        raise ValueError("Loaded DSpark model uses a different block size")

    # VIME captures the already-normalized tensor entering the Target LM head.
    model.verifier_norm = torch.nn.Identity()
    for head_name in ("lm_head", "verifier_lm_head"):
        head = getattr(model, head_name, None)
        if head is not None:
            head.requires_grad_(False)

    model = model.to(device=device, dtype=torch.bfloat16)
    confidence_head = getattr(model, "confidence_head", None)
    if confidence_head is not None:
        confidence_head.to(device=device, dtype=torch.float32)
    return model
