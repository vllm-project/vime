from __future__ import annotations

import json
from argparse import Namespace
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

_DRAFT_CONFIG_CACHE_ATTR = "_vime_draft_checkpoint_config"
_DRAFT_CAPTURE_LAYER_ID_KEYS = (
    "aux_hidden_state_layer_ids",
    "eagle_aux_hidden_state_layer_ids",
    "target_hidden_layer_ids",
)


def external_draft_enabled(args: Namespace) -> bool:
    return bool(getattr(args, "enable_external_draft_training", False))


def parse_int_list(value: object) -> list[int] | None:
    if value is None:
        return None
    if isinstance(value, int):
        return [int(value)]
    if isinstance(value, str):
        value = value.strip()
        if not value:
            return None
        if value.startswith("["):
            value = json.loads(value)
        else:
            value = [item.strip() for item in value.split(",") if item.strip()]
    if not isinstance(value, Sequence):
        raise TypeError(f"Expected a list of integers, got {type(value).__name__}")
    result = [int(item) for item in value]
    if len(set(result)) != len(result):
        raise ValueError(f"Draft feature layer ids must be unique, got {result}")
    return result


def _looks_like_local_path(value: str, path: Path) -> bool:
    return path.exists() or path.is_absolute() or PurePosixPath(value).is_absolute() or value.startswith(("./", "../"))


def load_draft_checkpoint_config(args: Namespace) -> dict:
    cached = getattr(args, _DRAFT_CONFIG_CACHE_ATTR, None)
    if isinstance(cached, dict):
        return cached

    model_path = getattr(args, "draft_model_path", None)
    if not model_path:
        return {}
    model_id = str(model_path)
    local_path = Path(model_id).expanduser()
    if _looks_like_local_path(model_id, local_path):
        if not local_path.exists():
            raise ValueError(
                f"DSpark checkpoint path {model_id!r} does not exist on the VIME driver. "
                "Mount the checkpoint at the same path used by the rollout workers, or use a Hugging Face model ID."
            )
        if not local_path.is_dir():
            raise ValueError(f"DSpark checkpoint path {model_id!r} must be a directory")
        config_path = local_path / "config.json"
        if not config_path.is_file():
            raise ValueError(f"DSpark checkpoint directory {model_id!r} does not contain config.json")
        try:
            with config_path.open(encoding="utf-8") as handle:
                value = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Unable to read DSpark checkpoint config {config_path}") from exc
    else:
        try:
            from transformers import PretrainedConfig

            value, _ = PretrainedConfig.get_config_dict(model_id)
        except Exception as exc:
            raise ValueError(
                f"Unable to load config.json for DSpark checkpoint {model_id!r}. "
                "If this is a local checkpoint, pass an existing path visible to the VIME driver."
            ) from exc
    if not isinstance(value, dict):
        raise TypeError(f"Draft config for {model_id!r} must contain a JSON object")
    setattr(args, _DRAFT_CONFIG_CACHE_ATTR, value)
    return value


def _draft_config_value(config: dict, keys: Sequence[str]) -> object | None:
    """Read a DSpark field across Speculators and vLLM checkpoint schemas."""

    for key in keys:
        if config.get(key) is not None:
            return config[key]
    for container_key in ("hf_config", "draft_model_config", "eagle_config"):
        nested = config.get(container_key)
        if isinstance(nested, dict):
            for key in keys:
                if nested.get(key) is not None:
                    return nested[key]
    return None


def resolve_feature_layer_ids(args: Namespace) -> list[int]:
    explicit = parse_int_list(getattr(args, "draft_feature_layer_ids", None))
    num_layers = int(getattr(args, "num_layers", 0) or 0)
    algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if explicit is None:
        if algorithm == "dspark":
            draft_config = load_draft_checkpoint_config(args)
            explicit = parse_int_list(_draft_config_value(draft_config, _DRAFT_CAPTURE_LAYER_ID_KEYS))
            if explicit is None:
                dense_target_layer_ids = parse_int_list(_draft_config_value(draft_config, ("target_layer_ids",)))
                if dense_target_layer_ids is not None:
                    # Dense vLLM/DeepSpec DSpark checkpoints store the decoder
                    # layer preceding each captured hidden state.
                    explicit = [layer_id + 1 for layer_id in dense_target_layer_ids]
            if explicit is None:
                if num_layers < 5:
                    raise ValueError(
                        f"DSpark checkpoint {args.draft_model_path!r} config.json does not define any of "
                        f"{[*_DRAFT_CAPTURE_LAYER_ID_KEYS, 'target_layer_ids']}, and the Target layer count "
                        "is unavailable; "
                        "pass --draft-feature-layer-ids explicitly"
                    )
                # Match Speculators' resolve_target_layer_ids default for
                # checkpoints created without an explicit target layer list.
                explicit = [2, num_layers // 2, num_layers - 3]
        elif num_layers < 5:
            raise ValueError(
                "--draft-feature-layer-ids is required when the Target layer count cannot "
                "safely derive the EAGLE3 default [2, num_layers//2, num_layers-3]."
            )
        else:
            explicit = [2, num_layers // 2, num_layers - 3]
    normalized = []
    for layer_id in explicit:
        if layer_id < 0:
            if num_layers <= 0:
                raise ValueError("Negative Draft layer ids require --num-layers")
            layer_id += num_layers
        if layer_id < 0 or (num_layers > 0 and layer_id >= num_layers):
            raise ValueError(f"Draft feature layer id {layer_id} is outside Target depth {num_layers}")
        normalized.append(layer_id)
    return normalized


def resolve_dspark_block_size(args: Namespace) -> int:
    value = getattr(args, "draft_dspark_block_size", None)
    if value is None:
        value = _draft_config_value(load_draft_checkpoint_config(args), ("block_size",))
    if value is None:
        raise ValueError(
            f"DSpark checkpoint {args.draft_model_path!r} config.json does not define block_size; "
            "pass --draft-dspark-block-size explicitly"
        )
    value = int(value)
    if value < 2:
        raise ValueError("DSpark block size must be at least 2")
    return value


def should_run_draft_interval(rollout_id: int, interval: int | None) -> bool:
    if interval is None or int(interval) <= 0:
        return False
    return (int(rollout_id) + 1) % int(interval) == 0


def _speculative_config(args: Namespace) -> dict:
    value = getattr(args, "vllm_speculative_config", None)
    if isinstance(value, str):
        value = json.loads(value)
    return value if isinstance(value, dict) else {}


def _activate_dspark_training_for_export(args: Namespace) -> None:
    """Turn a DSpark inference export request into an explicit training setup."""

    if not getattr(args, "draft_save_hf", None):
        return
    spec_config = _speculative_config(args)
    method = str(spec_config.get("method", "")).strip().lower()
    if method != "dspark":
        raise ValueError("--draft-save-hf requires DSpark inference (--vllm-speculative-config method=dspark)")

    args.enable_external_draft_training = True
    args.draft_algorithm = "dspark"
    if not getattr(args, "draft_model_path", None):
        configured_model = spec_config.get("model")
        if not configured_model:
            raise ValueError(
                "--draft-save-hf with DSpark inference requires a model in --vllm-speculative-config "
                "or --draft-model-path"
            )
        args.draft_model_path = str(configured_model)


def validate_external_draft_args(args: Namespace) -> None:
    _activate_dspark_training_for_export(args)
    if not external_draft_enabled(args):
        return

    algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
    if algorithm not in {"eagle3", "dspark"}:
        raise ValueError("External Draft training supports --draft-algorithm=eagle3 or dspark")
    if not getattr(args, "draft_model_path", None):
        raise ValueError("--enable-external-draft-training requires --draft-model-path")
    if not (getattr(args, "draft_target_embedding_path", None) or getattr(args, "hf_checkpoint", None)):
        raise ValueError("External Draft training requires --hf-checkpoint or --draft-target-embedding-path")
    if not str(getattr(args, "draft_target_embedding_key", "") or "").strip():
        raise ValueError("--draft-target-embedding-key must be non-empty")
    if str(getattr(args, "train_backend", "megatron")) != "megatron":
        raise ValueError("External Draft feature collection currently requires --train-backend=megatron")
    if bool(getattr(args, "debug_rollout_only", False)):
        raise ValueError("External Draft training is unavailable with --debug-rollout-only")
    if bool(getattr(args, "colocate", False)):
        raise ValueError("The external Draft MVP requires a disaggregated rollout; --colocate is not supported")
    if bool(getattr(args, "release_train", False)):
        raise ValueError("The external Draft MVP does not support --release-train")
    if bool(getattr(args, "keep_old_actor", False)):
        raise ValueError(
            "The external Draft MVP does not support --keep-old-actor because hidden states and the "
            "supervising LM Head must come from the same model copy"
        )
    if bool(getattr(args, "enable_mtp_training", False)):
        raise ValueError("External Draft training and inline MTP training cannot be enabled together")
    if bool(getattr(args, "use_routing_replay", False)) or bool(getattr(args, "use_rollout_routing_replay", False)):
        raise ValueError("The external Draft MVP does not yet support MoE routing replay during feature capture")
    if str(getattr(args, "update_weight_mode", "full")) != "full":
        raise ValueError("External Draft publication requires --update-weight-mode=full")
    if str(getattr(args, "update_weight_transport", "nccl")) != "nccl":
        raise ValueError("External Draft publication currently requires --update-weight-transport=nccl")
    if int(getattr(args, "pipeline_model_parallel_size", 1) or 1) != 1:
        raise ValueError("The external Draft MVP currently requires pipeline model parallel size 1")
    if int(getattr(args, "context_parallel_size", 1) or 1) != 1:
        raise ValueError("The external Draft MVP currently requires context parallel size 1")
    if int(getattr(args, "virtual_pipeline_model_parallel_size", 1) or 1) != 1:
        raise ValueError("The external Draft MVP currently requires virtual pipeline parallel size 1")

    spec_config = _speculative_config(args)
    if not spec_config:
        raise ValueError("External Draft training requires --vllm-speculative-config")
    method = str(spec_config.get("method", "")).strip().lower()
    expected_methods = {"eagle", "eagle3"} if algorithm == "eagle3" else {"dspark"}
    if method not in expected_methods:
        expected = "'eagle' or 'eagle3'" if algorithm == "eagle3" else "'dspark'"
        raise ValueError(f"External {algorithm} training requires vLLM speculative method {expected}")
    configured_model = spec_config.get("model")
    if configured_model and str(configured_model) != str(args.draft_model_path):
        raise ValueError(
            "vLLM speculative model and --draft-model-path must identify the same checkpoint: "
            f"{configured_model!r} != {args.draft_model_path!r}"
        )
    acceptance_method = str(spec_config.get("acceptance_method", "")).strip().lower()
    if acceptance_method in {"typical_acceptance_sampler", "typical", "topk"}:
        raise ValueError("External Draft RL rollout requires a lossless speculative acceptance method")

    layer_ids = resolve_feature_layer_ids(args)
    args.draft_feature_layer_ids = layer_ids
    if algorithm == "dspark":
        args.draft_dspark_block_size = resolve_dspark_block_size(args)
    draft_save_hf = getattr(args, "draft_save_hf", None)
    if draft_save_hf:
        try:
            str(draft_save_hf).format(rollout_id=0)
        except (IndexError, KeyError, ValueError) as exc:
            raise ValueError("--draft-save-hf must be a valid path template using only {rollout_id}") from exc
        export_path = Path(str(draft_save_hf)).expanduser()
        if not export_path.is_absolute():
            export_path = Path.cwd() / export_path
        args.draft_save_hf = str(export_path)
    positive_names = (
        "draft_collect_interval",
        "draft_train_interval",
        "draft_publish_interval",
        "draft_train_steps_per_trigger",
        "draft_batch_size_per_gpu",
        "draft_hidden_window_tokens",
    )
    if algorithm == "dspark":
        positive_names += ("draft_dspark_max_anchors",)
    for name in positive_names:
        if int(getattr(args, name, 0) or 0) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    rate = float(getattr(args, "draft_collection_sample_rate", 1.0))
    if rate <= 0 or rate > 1:
        raise ValueError("--draft-collection-sample-rate must be in (0, 1]")
    if int(getattr(args, "draft_lr_warmup_steps", 0)) < 0 or int(getattr(args, "draft_lr_total_steps", 0)) < 0:
        raise ValueError("Draft LR warmup and total steps must be non-negative")
