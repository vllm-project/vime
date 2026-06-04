from __future__ import annotations

import importlib
import importlib.util
import json
import sys
import uuid
from pathlib import Path
from typing import Any

import torch
from PIL import Image

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from vime.rollout.vllm_rollout import (
    GenerateState,
    _apply_vllm_routed_experts,
    _build_inference_sampling_params,
    _coerce_flat_int_token_ids,
    _inference_generate_tokens_and_logprobs,
    _mm_render_response_to_generate_body,
)
from vime.utils.http_utils import post
from vime.utils.processing_utils import encode_image_for_rollout_engine
from vime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.geo3k_vlm_multi_turn.env_geo3k"


def _load_env_module(env_path: str | None):
    """Load the interaction environment module from a module path or a file path."""
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix == ".py" and module_path.exists():
        spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot import environment module from {module_path}")
        module = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = module
        spec.loader.exec_module(module)
        return module
    return importlib.import_module(target)


def _image_to_render_url(image: Any) -> str:
    """Convert a dataset/processor image object to the image_url payload accepted by vLLM render."""
    if isinstance(image, str):
        if image.startswith("data:"):
            # Some prepared Geo3K rows use data:image/None; vLLM expects a concrete media type.
            return image.replace("data:image/None;", "data:image/png;", 1)
        if image.startswith(("http://", "https://")):
            return image
        image_path = Path(image).expanduser()
        if image_path.exists():
            with Image.open(image_path) as loaded_image:
                return encode_image_for_rollout_engine(loaded_image)
        raise ValueError(f"Unsupported image string for vLLM render: {image!r}")
    return encode_image_for_rollout_engine(image)


def _build_initial_messages(sample: Sample) -> list[dict]:
    """Build the initial conversation from the dataset prompt.

    The preferred path mirrors SkyRL: keep the untemplated conversation as the source of truth
    and let vLLM render/chat-template it on every turn. If an old pre-rendered string prompt is
    supplied, fall back to pairing it with sample.multimodal_inputs while removing literal
    <image> placeholders to avoid double image tokens.
    """
    if isinstance(sample.prompt, list):
        return [dict(message) for message in sample.prompt]

    content: list[dict] = []
    images = (sample.multimodal_inputs or {}).get("images") or []
    for image in images:
        content.append({"type": "image", "image": image})
    # Backward-compatible fallback for old runs that pass a raw text prompt containing <image>.
    text_prompt = str(sample.prompt).replace("<image>", "").lstrip()
    content.append({"type": "text", "text": text_prompt})
    return [{"role": "user", "content": content}]


def _messages_for_render(messages: list[dict]) -> list[dict]:
    """Normalize per-turn messages to the render-route shape (image → image_url)."""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if not isinstance(content, list):
            out.append(msg)
            continue

        rendered_content: list[dict] = []
        for part in content:
            if part.get("type") == "image" and part.get("image") is not None:
                rendered_content.append(
                    {"type": "image_url", "image_url": {"url": _image_to_render_url(part["image"])}}
                )
            else:
                rendered_content.append(part)
        out.append({"role": msg["role"], "content": rendered_content})
    return out


def _multimodal_train_inputs_from_features(features: Any) -> dict | None:
    """Decode the latest vLLM render features into train-side multimodal tensors."""
    if not features:
        return None
    if isinstance(features, str):
        try:
            features = json.loads(features)
        except json.JSONDecodeError:
            return None
    if not isinstance(features, dict):
        return None
    kwargs_data = features.get("kwargs_data")
    if not isinstance(kwargs_data, dict):
        return None
    if "image" not in kwargs_data:
        return None

    from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item as vllm_decode

    parts_by_key: dict[str, list[torch.Tensor]] = {}
    for encoded in kwargs_data["image"]:
        item = vllm_decode(encoded)
        for key, value in item.get_data().items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "image_grid_thw" and value.dim() == 1:
                value = value.reshape(1, -1)
            parts_by_key.setdefault(key, []).append(value)

    return {
        key: torch.cat(values, dim=0) if len(values) > 1 else values[0] for key, values in parts_by_key.items()
    } or None


def _validate_multimodal_train_inputs(sample: Sample, tokenizer: Any, processor: Any, mm_inputs: dict | None) -> None:
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        raise RuntimeError("Tokenizer does not define <|image_pad|>.")

    image_tokens = sample.tokens.count(int(image_token_id))
    if image_tokens == 0:
        return

    grid = None if not mm_inputs else mm_inputs.get("image_grid_thw")
    if grid is None:
        raise RuntimeError(f"Found {image_tokens} image tokens, but multimodal render features are missing.")

    grid = grid.reshape(-1, 3)
    image_processor = getattr(processor, "image_processor", processor)
    merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
    expected = int((grid.prod(dim=1) // (merge_size * merge_size)).sum().item())
    if image_tokens != expected:
        raise RuntimeError(
            "Image token count does not match multimodal render features: "
            f"image_tokens={image_tokens}, expected_image_tokens={expected}, image_grid_thw={grid.tolist()}"
        )


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment via the vLLM render route."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    if args.max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"

    env_module = _load_env_module(args.rollout_interaction_env_path)
    sample.metadata = sample.metadata or {}
    headers = None
    if getattr(args, "router_policy", None) == "consistent_hash":
        sample.session_id = sample.session_id or str(uuid.uuid4())
        headers = {"x-session-id": sample.session_id}

    build_env = env_module.build_env
    if not callable(build_env):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    env = build_env(sample=sample, args=args)

    messages = _build_initial_messages(sample)
    response_tokens: list[int] = []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.tokens = list(sample.tokens) if sample.tokens else []

    sampling_params = sampling_params.copy()
    inference_sampling_params = _build_inference_sampling_params(sampling_params)

    max_response_budget = sampling_params.get("max_new_tokens")

    def remaining_budget() -> int | None:
        return None if max_response_budget is None else max_response_budget - sample.response_length

    async def render() -> dict:
        payload = {"model": args.hf_checkpoint, "messages": _messages_for_render(messages)}
        render_data = await post(f"{base_url}/v1/chat/completions/render", payload, headers=headers)
        return _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)

    def append_response_window(
        token_ids: list[int],
        loss_mask: list[int],
        log_probs: list[float] | None = None,
    ) -> None:
        if not token_ids:
            return
        if len(loss_mask) != len(token_ids):
            raise ValueError(f"loss_mask length {len(loss_mask)} != token_ids length {len(token_ids)}")
        sample.tokens.extend(token_ids)
        sample.loss_mask.extend(loss_mask)
        sample.rollout_log_probs.extend(log_probs if log_probs is not None else [0.0] * len(token_ids))
        sample.response_length += len(token_ids)

    def sampling_params_for_turn() -> dict | None:
        params = dict(inference_sampling_params)
        max_tokens = remaining_budget()
        if max_tokens is None:
            return params
        if max_tokens <= 0:
            return None
        params["max_tokens"] = max_tokens
        return params

    try:
        env.reset()
        latest_features = None
        pending_obs_offset: int | None = None
        rendered_body = await render()
        prompt_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
        if not sample.tokens:
            sample.tokens = list(prompt_ids)
        if args.rollout_max_context_len is not None:
            max_response_budget = max(0, args.rollout_max_context_len - len(sample.tokens))

        for turn_idx in range(args.max_turns):
            input_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
            latest_features = rendered_body.get("features")

            if pending_obs_offset is not None:
                obs_tokens = input_ids[pending_obs_offset:]
                remaining = remaining_budget()
                if remaining is not None and len(obs_tokens) > remaining:
                    append_response_window(obs_tokens[: max(remaining, 0)], [0] * max(remaining, 0))
                    sample.status = Sample.Status.TRUNCATED
                    break
                append_response_window(obs_tokens, [0] * len(obs_tokens))
                pending_obs_offset = None

            current_sampling_params = sampling_params_for_turn()
            if current_sampling_params is None:
                sample.status = Sample.Status.TRUNCATED
                break

            body = dict(rendered_body)
            body["sampling_params"] = current_sampling_params
            output = await post(f"{base_url}/inference/v1/generate", body, headers=headers)
            choice = output["choices"][0]
            finish_reason = choice.get("finish_reason") or "stop"
            new_tokens, new_logprobs = _inference_generate_tokens_and_logprobs(choice)

            if not new_tokens:
                if finish_reason in ("abort", "cancelled"):
                    sample.status = Sample.Status.ABORTED
                    break

            response_text = state.tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens else ""
            train_tokens = list(new_tokens)
            train_logprobs = list(new_logprobs)
            train_loss_mask = [1] * len(train_tokens)
            stop = current_sampling_params.get("stop")
            eos_token_id = getattr(state.tokenizer, "eos_token_id", None)
            append_stop_eos = (
                stop
                and eos_token_id is not None
                and getattr(args, "append_eos_token_after_stop_str_in_multi_turn", True)
            )
            if append_stop_eos:
                stop_strings = (stop,) if isinstance(stop, str) else tuple(stop)
                already_has_eos = bool(train_tokens and train_tokens[-1] == eos_token_id)
                if stop_strings and response_text.endswith(stop_strings) and not already_has_eos:
                    if getattr(args, "use_rollout_routing_replay", False):
                        raise RuntimeError(
                            "Routing replay is not supported when appending an artificial EOS after a stop string, "
                            "because vLLM does not return routed experts for that extra token."
                        )
                    train_tokens.append(int(eos_token_id))
                    train_logprobs.append(0.0)
                    train_loss_mask.append(0)

            response_tokens.extend(new_tokens)
            append_response_window(train_tokens, train_loss_mask, train_logprobs)
            _apply_vllm_routed_experts(args, sample, choice)

            messages.append({"role": "assistant", "content": response_text})

            if finish_reason == "length":
                sample.status = Sample.Status.TRUNCATED
                break
            if finish_reason in ("abort", "cancelled"):
                sample.status = Sample.Status.ABORTED
                break

            observation, done, _ = env.step(response_text)
            if done:
                sample.status = Sample.Status.COMPLETED
                break

            if turn_idx + 1 >= args.max_turns:
                sample.status = Sample.Status.TRUNCATED
                break

            next_user_message = env.format_observation(observation)
            messages.append(next_user_message)
            render_prefix_len = len(input_ids) + len(new_tokens)
            pending_obs_offset = render_prefix_len
            rendered_body = await render()
            rendered_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
            is_prefix_stable = rendered_ids[:pending_obs_offset] == sample.tokens[:pending_obs_offset]
            sample.metadata["multiturn_render"] = {
                "prefix_stable": is_prefix_stable,
                "prefix_len": pending_obs_offset,
                "sample_len": len(sample.tokens),
                "rendered_len": len(rendered_ids),
            }
            if not is_prefix_stable:
                raise RuntimeError(
                    "Full conversation render is not prefix-stable with the generated token stream: "
                    f"{sample.metadata['multiturn_render']}"
                )

        multimodal_train_inputs = _multimodal_train_inputs_from_features(latest_features)
        _validate_multimodal_train_inputs(sample, state.tokenizer, state.processor, multimodal_train_inputs)
        sample.multimodal_train_inputs = multimodal_train_inputs
        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(sample.loss_mask)
        if sample.status == Sample.Status.PENDING:
            sample.status = Sample.Status.COMPLETED
        return sample
    finally:
        try:
            env.close()
        except Exception:
            pass
