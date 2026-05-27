from __future__ import annotations

import importlib
import importlib.util
import sys
from pathlib import Path
from typing import Any

import torch

# When executed as a module: python -m examples.vlm_multi_turn.rollout
from slime.rollout.vllm_rollout import (
    GenerateState,
    _build_inference_sampling_params,
    _inference_generate_tokens_and_logprobs,
    _mm_render_response_to_generate_body,
)
from slime.utils.http_utils import post
from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.vlm_multi_turn.env_geo3k"


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


def _build_env(env_module, sample: Sample, args: Any):
    """Instantiate the interaction environment using the provided module."""
    build_fn = env_module.build_env
    if not callable(build_fn):
        raise ValueError("Environment module must expose a callable `build_env(sample, args)`.")
    try:
        return build_fn(sample=sample, args=args)
    except TypeError:
        return build_fn(sample, args)


def _content_to_render_format(content: list[dict]) -> list[dict]:
    """Convert env-style message content (image objects) to render-route format (image_url data URLs)."""
    out: list[dict] = []
    for part in content:
        ptype = part.get("type")
        if ptype == "image" and part.get("image") is not None:
            data_url = encode_image_for_rollout_engine(part["image"])
            out.append({"type": "image_url", "image_url": {"url": data_url}})
        else:
            out.append(part)
    return out


def _build_initial_user_message(sample: Sample) -> dict:
    """Build the initial user message from sample.prompt + sample.multimodal_inputs.images."""
    content: list[dict] = []
    images = (sample.multimodal_inputs or {}).get("images") or []
    for image in images:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": sample.prompt})
    return {"role": "user", "content": content}


def _messages_for_render(messages: list[dict]) -> list[dict]:
    """Normalize per-turn messages to the render-route shape (image → image_url)."""
    out: list[dict] = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            out.append({"role": msg["role"], "content": _content_to_render_format(content)})
        else:
            out.append(msg)
    return out


def _processor_features_from_message(processor, tokenizer, message: dict) -> dict | None:
    """Run the HF processor on a single user message containing images, returning train-side multimodal inputs."""
    if processor is None:
        return None
    content = message.get("content")
    if not isinstance(content, list):
        return None

    from qwen_vl_utils import process_vision_info

    images, videos = process_vision_info([message])
    if not images and not videos:
        return None
    # We only need processor-side features here; tokens are produced by the render route.
    formatted_prompt = tokenizer.apply_chat_template([message], tokenize=False, add_generation_prompt=False)
    processor_output = processor(text=formatted_prompt, images=images, videos=videos)
    return {k: v for k, v in processor_output.items() if k not in ("input_ids", "attention_mask")} or None


def _merge_multimodal_train_inputs(chunks: list[dict | None]) -> dict | None:
    """Concatenate per-turn processor outputs along dim 0 for torch tensor fields."""
    if not chunks:
        return None
    values_by_key: dict[str, list] = {}
    for chunk in chunks:
        if not chunk:
            continue
        for key, val in chunk.items():
            if val is None:
                continue
            values_by_key.setdefault(key, []).append(val)
    merged: dict = {}
    for key, values in values_by_key.items():
        if all(isinstance(v, torch.Tensor) for v in values):
            merged[key] = torch.cat(values, dim=0)
    return merged


async def _render_and_generate(
    args,
    base_url: str,
    messages: list[dict],
    sampling_params: dict,
) -> dict:
    """Render messages to engine-prompt body, then call /inference/v1/generate."""
    render_payload = {"model": args.hf_checkpoint, "messages": _messages_for_render(messages)}
    render_data = await post(f"{base_url}/v1/chat/completions/render", render_payload)
    body = _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)
    body["sampling_params"] = sampling_params
    return await post(f"{base_url}/inference/v1/generate", body)


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    """Custom multi-turn rollout that interacts with a pluggable environment via the vLLM render route."""
    assert not args.partial_rollout, "Partial rollout is not supported for interaction rollouts."

    if args.max_turns is None:
        raise ValueError("max_turns must be set via --custom-config-path in the custom config file.")
    state = GenerateState(args)
    base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"

    env_module = _load_env_module(args.rollout_interaction_env_path)
    sample.metadata = sample.metadata or {}
    env = _build_env(env_module, sample, args)

    messages: list[dict] = [_build_initial_user_message(sample)]
    response_tokens: list[int] = []
    sample.loss_mask = sample.loss_mask or []
    sample.rollout_log_probs = sample.rollout_log_probs or []
    sample.tokens = list(sample.tokens) if sample.tokens else []
    multimodal_train_inputs_buffer: list[dict | None] = []

    initial_train_feats = _processor_features_from_message(state.processor, state.tokenizer, messages[0])
    if initial_train_feats:
        multimodal_train_inputs_buffer.append(initial_train_feats)

    sampling_params = sampling_params.copy()
    inference_sampling_params = _build_inference_sampling_params(sampling_params)

    budget = None
    if args.rollout_max_context_len is not None:
        budget = args.rollout_max_context_len - len(sample.tokens)
    elif sampling_params.get("max_new_tokens") is not None:
        budget = sampling_params["max_new_tokens"] - len(sample.tokens)

    try:
        env.reset()
        for turn_idx in range(args.max_turns):
            if budget is not None and budget <= 0:
                sample.status = Sample.Status.TRUNCATED
                break
            if budget is not None:
                inference_sampling_params["max_tokens"] = budget

            output = await _render_and_generate(args, base_url, messages, inference_sampling_params)
            choice = output["choices"][0]
            finish_reason = choice.get("finish_reason") or "stop"
            new_tokens, new_logprobs = _inference_generate_tokens_and_logprobs(choice)

            if not new_tokens:
                if finish_reason in ("abort", "cancelled"):
                    sample.status = Sample.Status.ABORTED
                    break

            response_text = state.tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens else ""
            sample.tokens.extend(new_tokens)
            response_tokens.extend(new_tokens)
            sample.loss_mask.extend([1] * len(new_tokens))
            sample.rollout_log_probs.extend(new_logprobs)
            sample.response_length = len(response_tokens)
            if budget is not None:
                budget -= len(new_tokens)

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

            next_user_message = env.format_observation(observation)
            messages.append(next_user_message)

            obs_train_feats = _processor_features_from_message(state.processor, state.tokenizer, next_user_message)
            if obs_train_feats:
                multimodal_train_inputs_buffer.append(obs_train_feats)

            if turn_idx + 1 >= args.max_turns:
                sample.status = Sample.Status.COMPLETED
                break

        sample.multimodal_train_inputs = _merge_multimodal_train_inputs(multimodal_train_inputs_buffer)
        sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
        sample.response_length = len(response_tokens)
        if sample.status is None:
            sample.status = Sample.Status.COMPLETED
        return sample
    finally:
        try:
            env.close()
        except Exception:
            pass
