from __future__ import annotations

import base64
import importlib
import importlib.util
import io
import json
import sys
import uuid
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from vime.rollout.vllm_rollout import (
    GenerateState,
    _build_inference_sampling_params,
    _coerce_flat_int_token_ids,
    _mm_render_response_to_generate_body,
)
from vime.utils.http_utils import post
from vime.utils.processing_utils import encode_image_for_rollout_engine
from vime.utils.types import Sample

DEFAULT_ENV_MODULE = "examples.geo3k_vlm_multi_turn.env_geo3k"
DUMMY_MESSAGES = (
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "I am a user."},
    {"role": "assistant", "content": "I am an assistant."},
)
IMAGE_GRID_DIMENSIONS = 3


def _load_env_module(env_path: str | None):
    target = env_path or DEFAULT_ENV_MODULE
    module_path = Path(target)
    if module_path.suffix != ".py" or not module_path.exists():
        return importlib.import_module(target)

    spec = importlib.util.spec_from_file_location(f"rollout_env_{module_path.stem}", module_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot import environment module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _image_to_render_url(image: Any) -> str:
    if not isinstance(image, str):
        return encode_image_for_rollout_engine(image)
    if image.startswith("data:"):
        return image.replace("data:image/None;", "data:image/png;", 1)
    if image.startswith(("http://", "https://")):
        return image

    image_path = Path(image).expanduser()
    if not image_path.exists():
        raise ValueError(f"Unsupported image string for vLLM render: {image!r}")
    with Image.open(image_path) as loaded_image:
        return encode_image_for_rollout_engine(loaded_image)


def _build_initial_messages(sample: Sample) -> list[dict[str, Any]]:
    if isinstance(sample.prompt, list):
        return [dict(message) for message in sample.prompt]

    content: list[dict[str, Any]] = []
    for image in (sample.multimodal_inputs or {}).get("images") or []:
        content.append({"type": "image", "image": image})
    content.append({"type": "text", "text": str(sample.prompt).replace("<image>", "").lstrip()})
    return [{"role": "user", "content": content}]


def _messages_for_render(messages: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rendered: list[dict[str, Any]] = []
    for message in messages:
        content = message.get("content")
        if not isinstance(content, list):
            rendered.append(dict(message))
            continue
        parts: list[dict[str, Any]] = []
        for part in content:
            if part.get("type") == "image" and part.get("image") is not None:
                parts.append({"type": "image_url", "image_url": {"url": _image_to_render_url(part["image"])}})
            else:
                parts.append(dict(part))
        rendered.append({"role": message["role"], "content": parts})
    return rendered


def _multimodal_train_inputs_from_features(features: Any) -> dict[str, torch.Tensor] | None:
    if not features:
        return None
    decoded = json.loads(features) if isinstance(features, str) else features
    if not isinstance(decoded, dict):
        raise TypeError(f"vLLM features must decode to a dictionary, got {type(decoded).__name__}")
    kwargs_data = decoded.get("kwargs_data")
    if not isinstance(kwargs_data, dict) or "image" not in kwargs_data:
        return None
    encoded_images = kwargs_data["image"]
    if not isinstance(encoded_images, list):
        raise TypeError("vLLM features.kwargs_data.image must be a list")

    from vllm.entrypoints.serve.disagg.mm_serde import decode_mm_kwargs_item as vllm_decode

    parts_by_key: dict[str, list[torch.Tensor]] = {}
    for encoded in encoded_images:
        item = vllm_decode(encoded)
        for key, value in item.get_data().items():
            if not isinstance(value, torch.Tensor):
                continue
            if key == "image_grid_thw" and value.dim() == 1:
                value = value.reshape(1, -1)
            parts_by_key.setdefault(key, []).append(value)
    return {key: values[0] if len(values) == 1 else torch.cat(values, dim=0) for key, values in parts_by_key.items()}


def _validate_multimodal_train_inputs(
    sample: Sample,
    tokenizer: Any,
    processor: Any,
    *,
    mm_inputs: dict[str, torch.Tensor] | None,
) -> None:
    image_token_id = tokenizer.convert_tokens_to_ids("<|image_pad|>")
    if image_token_id is None:
        raise RuntimeError("Tokenizer does not define <|image_pad|>.")
    image_tokens = sample.tokens.count(int(image_token_id))
    if image_tokens == 0:
        return

    grid = None if not mm_inputs else mm_inputs.get("image_grid_thw")
    if grid is None:
        raise RuntimeError(f"Found {image_tokens} image tokens, but multimodal render features are missing.")
    grid = grid.reshape(-1, IMAGE_GRID_DIMENSIONS)
    image_processor = getattr(processor, "image_processor", processor)
    merge_size = int(getattr(image_processor, "merge_size", 1) or 1)
    expected = int((grid.prod(dim=1) // (merge_size * merge_size)).sum().item())
    if image_tokens != expected:
        raise RuntimeError(
            "Image token count does not match multimodal render features: "
            f"image_tokens={image_tokens}, expected_image_tokens={expected}, image_grid_thw={grid.tolist()}"
        )


def _require_token_ids(value: Any, *, field: str) -> list[int]:
    if not isinstance(value, list) or not all(type(token) is int for token in value):
        raise TypeError(f"{field} must be a list of integer token ids")
    return list(value)


def _parse_log_probs(choice: dict[str, Any], token_count: int) -> list[float]:
    logprobs = choice.get("logprobs")
    content = logprobs.get("content") if isinstance(logprobs, dict) else None
    if not isinstance(content, list):
        raise TypeError("choice.logprobs.content must be a list")
    if len(content) != token_count:
        raise ValueError(f"token/logprob count mismatch: tokens={token_count}, logprobs={len(content)}")
    values: list[float] = []
    for index, item in enumerate(content):
        if not isinstance(item, dict) or not isinstance(item.get("logprob"), int | float):
            raise TypeError(f"choice.logprobs.content[{index}].logprob must be numeric")
        values.append(float(item["logprob"]))
    return values


def _parse_choice(choice: dict[str, Any]) -> tuple[str, list[int], list[float]]:
    finish = choice.get("finish_reason")
    finish = finish.get("type") if isinstance(finish, dict) else finish
    if finish in {"abort", "cancelled"}:
        return "abort", [], []
    if finish not in {"length", "stop"}:
        raise ValueError(f"Unsupported vLLM finish_reason: {finish!r}")

    token_ids = _require_token_ids(choice.get("token_ids"), field="choice.token_ids")
    return str(finish), token_ids, _parse_log_probs(choice, len(token_ids))


def _template_token_ids(value: Any, *, field: str) -> list[int]:
    if isinstance(value, Mapping):
        value = value.get("input_ids")
    return _require_token_ids(value, field=field)


def _validate_text_observation(message: dict[str, Any]) -> None:
    content = message.get("content")
    if isinstance(content, str):
        return
    if not isinstance(content, list):
        raise TypeError("Geo3K observation content must be text or a list of text parts")
    for index, part in enumerate(content):
        if not isinstance(part, dict) or part.get("type") != "text" or not isinstance(part.get("text"), str):
            raise ValueError(
                "Geo3K suffix-only rollout supports text observations only; " f"content[{index}]={part!r}"
            )


def _observation_token_ids(
    tokenizer: Any,
    message: dict[str, Any],
    canonical_ids: list[int],
    *,
    tools: list[dict[str, Any]] | None,
    template_kwargs: dict[str, Any] | None,
) -> list[int]:
    """Encode only the next user-turn boundary without re-rendering generated IDs."""
    kwargs = dict(template_kwargs or {})
    prefix = tokenizer.apply_chat_template(
        list(DUMMY_MESSAGES),
        tools=tools,
        tokenize=True,
        add_generation_prompt=False,
        **kwargs,
    )
    rendered = tokenizer.apply_chat_template(
        [*DUMMY_MESSAGES, message],
        tools=tools,
        tokenize=True,
        add_generation_prompt=True,
        **kwargs,
    )
    prefix_ids = _template_token_ids(prefix, field="dummy observation prefix")
    rendered_ids = _template_token_ids(rendered, field="observation template")
    if rendered_ids[: len(prefix_ids)] != prefix_ids:
        raise ValueError("Observation template is not prefix-stable")

    eos_token_id = getattr(tokenizer, "eos_token_id", None)
    if not isinstance(eos_token_id, int):
        raise ValueError("tokenizer.eos_token_id must be an integer")
    try:
        eos_index = len(prefix_ids) - 1 - prefix_ids[::-1].index(eos_token_id)
    except ValueError as exc:
        raise ValueError(f"Dummy observation prefix lacks eos_token_id={eos_token_id}") from exc
    boundary = prefix_ids[eos_index:] + rendered_ids[len(prefix_ids) :]
    return boundary[1:] if canonical_ids and canonical_ids[-1] == eos_token_id else boundary


def _decode_routing_metadata(args: Any, choice: dict[str, Any], *, expected_transitions: int) -> dict[str, Any] | None:
    routed_experts = choice.get("routed_experts")
    if routed_experts is None:
        if getattr(args, "use_rollout_routing_replay", False):
            raise RuntimeError("vLLM routing replay response is missing choices[0].routed_experts")
        return None
    if not isinstance(routed_experts, str):
        raise TypeError("choice.routed_experts must be a base64 string")
    raw = base64.b64decode(routed_experts.encode("ascii"), validate=True)
    decoded = np.load(io.BytesIO(raw), allow_pickle=False)
    if getattr(args, "use_rollout_routing_replay", False):
        expected_size = expected_transitions * args.num_layers * args.moe_router_topk
        if int(decoded.size) != expected_size:
            raise ValueError(
                "vLLM routed experts shape does not match the generated sequence: "
                f"actual_size={decoded.size}, expected_size={expected_size}"
            )
    return {"routed_experts": decoded}


def _response_budget(sampling_params: dict[str, Any], context_limit: int | None, prompt_length: int) -> int | None:
    budget = sampling_params.get("max_new_tokens")
    if budget is not None and budget < 0:
        raise ValueError(f"max_new_tokens must be non-negative, got {budget}")
    if context_limit is None:
        return budget
    if context_limit < 0:
        raise ValueError(f"rollout_max_context_len must be non-negative, got {context_limit}")
    context_budget = max(0, context_limit - prompt_length)
    return context_budget if budget is None else min(budget, context_budget)


def _stop_eos_token_id(
    text: str,
    token_ids: list[int],
    *,
    finish: str,
    stop: Any,
    eos_token_id: Any,
) -> int | None:
    if finish != "stop" or not stop or not isinstance(eos_token_id, int):
        return None
    stop_strings = (stop,) if isinstance(stop, str) else tuple(stop)
    if not text.endswith(stop_strings):
        return None
    if token_ids and token_ids[-1] == eos_token_id:
        return None
    return eos_token_id


def _validate_rollout_request(args: Any, sample: Sample) -> None:
    if args.partial_rollout:
        raise ValueError("Partial rollout is not supported for interaction rollouts.")
    if not isinstance(args.max_turns, int) or args.max_turns <= 0:
        raise ValueError("max_turns must be a positive integer in the custom config file.")
    if sample.status != Sample.Status.PENDING:
        raise ValueError(f"Geo3K rollout requires a pending sample, got {sample.status.value}")
    if sample.response or sample.response_length or sample.loss_mask or sample.rollout_log_probs:
        raise ValueError("Geo3K rollout does not accept pre-existing response state")


@dataclass(frozen=True, kw_only=True)
class _Turn:
    choice: dict[str, Any]
    tokens: list[int]
    log_probs: list[float]
    text: str
    finish: str


class _Geo3kRollout:
    def __init__(self, args: Any, sample: Sample, sampling_params: dict[str, Any]) -> None:
        _validate_rollout_request(args, sample)
        self.args = args
        self.sample = sample
        self.sample.metadata = dict(sample.metadata or {})
        self.sampling_params = dict(sampling_params)
        self.state = GenerateState(args)
        self.base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
        self.headers: dict[str, str] | None = None
        if getattr(args, "router_policy", None) == "consistent_hash":
            sample.session_id = sample.session_id or str(uuid.uuid4())
            self.headers = {"x-session-id": sample.session_id}
        self.inference_params = _build_inference_sampling_params(self.sampling_params)
        self.render_body: dict[str, Any] = {}
        self.max_response_budget: int | None = None
        self.response_tokens: list[int] = []

    @property
    def _remaining_budget(self) -> int | None:
        if self.max_response_budget is None:
            return None
        return self.max_response_budget - self.sample.response_length

    async def _initialize_prompt(self) -> None:
        payload: dict[str, Any] = {
            "model": self.args.hf_checkpoint,
            "messages": _messages_for_render(_build_initial_messages(self.sample)),
        }
        tools = self.sample.metadata.get("tools")
        if tools is not None:
            payload["tools"] = tools
        template_kwargs = getattr(self.args, "apply_chat_template_kwargs", None)
        if template_kwargs:
            payload["chat_template_kwargs"] = dict(template_kwargs)

        render_data = await post(
            f"{self.base_url}/v1/chat/completions/render",
            payload,
            headers=self.headers,
        )
        body = _mm_render_response_to_generate_body(render_data, self.args.hf_checkpoint)
        prompt_ids = _coerce_flat_int_token_ids(body.get("token_ids"))
        if not prompt_ids:
            raise ValueError("vLLM render returned empty token_ids")
        if self.sample.tokens and self.sample.tokens != prompt_ids:
            raise ValueError("Initial render token_ids differ from sample.tokens")

        self.sample.tokens = list(prompt_ids)
        self.sample.loss_mask = []
        self.sample.rollout_log_probs = []
        self.sample.response_length = 0
        self.render_body = body
        self.max_response_budget = _response_budget(
            self.sampling_params,
            self.args.rollout_max_context_len,
            len(prompt_ids),
        )

    async def _generate_turn(self, sampling_params: dict[str, Any]) -> _Turn:
        body = dict(self.render_body)
        body.pop("request_id", None)
        body["token_ids"] = list(self.sample.tokens)
        body["sampling_params"] = sampling_params
        output = await post(
            f"{self.base_url}/inference/v1/generate",
            body,
            headers=self.headers,
        )
        choices = output.get("choices") if isinstance(output, dict) else None
        if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
            raise ValueError("vLLM generate response must contain choices[0]")

        choice = choices[0]
        finish, tokens, log_probs = _parse_choice(choice)
        text = self.state.tokenizer.decode(tokens, skip_special_tokens=False) if tokens else ""
        return _Turn(
            choice=choice,
            tokens=tokens,
            log_probs=log_probs,
            text=text,
            finish=finish,
        )

    def _append_generated(self, turn: _Turn) -> bool:
        remaining = self._remaining_budget
        if remaining is not None and len(turn.tokens) > remaining:
            raise ValueError(
                "vLLM generated more tokens than requested: "
                f"generated_tokens={len(turn.tokens)}, remaining_budget={remaining}"
            )
        meta = _decode_routing_metadata(
            self.args,
            turn.choice,
            expected_transitions=len(self.sample.tokens) + len(turn.tokens) - 1,
        )
        eos_token_id = None
        if getattr(self.args, "append_eos_token_after_stop_str_in_multi_turn", True):
            eos_token_id = _stop_eos_token_id(
                turn.text,
                turn.tokens,
                finish=turn.finish,
                stop=self.inference_params.get("stop"),
                eos_token_id=getattr(self.state.tokenizer, "eos_token_id", None),
            )
        if eos_token_id is not None and getattr(self.args, "use_rollout_routing_replay", False):
            raise RuntimeError("Routing replay cannot append an artificial EOS after a stop string")

        self.sample.append_response_tokens(
            self.args,
            tokens=turn.tokens,
            log_probs=turn.log_probs,
            trainable=True,
            meta_info=meta,
            update_terminal_info=False,
        )
        self.response_tokens.extend(turn.tokens)
        if eos_token_id is None:
            return False
        remaining = self._remaining_budget
        if remaining is not None and remaining <= 0:
            self.sample.status = Sample.Status.TRUNCATED
            self.sample.metadata["multiturn_truncation"] = {
                "reason": "insufficient_budget_for_stop_eos",
                "remaining_budget": remaining,
            }
            return True
        self.sample.append_response_tokens(tokens=[eos_token_id], trainable=False)
        return False

    def _advance_environment(self, env: Any, turn: _Turn, turn_index: int) -> bool:
        observation, done, _ = env.step(turn.text)
        if done:
            self.sample.status = Sample.Status.COMPLETED
            return True
        if turn_index + 1 >= self.args.max_turns:
            self.sample.status = Sample.Status.TRUNCATED
            return True

        message = env.format_observation(observation)
        _validate_text_observation(message)
        token_ids = _observation_token_ids(
            self.state.tokenizer,
            message,
            self.sample.tokens,
            tools=self.sample.metadata.get("tools"),
            template_kwargs=getattr(self.args, "apply_chat_template_kwargs", None),
        )
        remaining = self._remaining_budget
        if remaining is None or len(token_ids) < remaining:
            self.sample.append_response_tokens(tokens=token_ids, trainable=False)
            return False
        self.sample.status = Sample.Status.TRUNCATED
        self.sample.metadata["multiturn_truncation"] = {
            "reason": "insufficient_budget_for_next_turn",
            "observation_tokens": len(token_ids),
            "remaining_budget": remaining,
        }
        return True

    def _finalize(self) -> Sample:
        mm_inputs = _multimodal_train_inputs_from_features(self.render_body.get("features"))
        _validate_multimodal_train_inputs(
            self.sample,
            self.state.tokenizer,
            self.state.processor,
            mm_inputs=mm_inputs,
        )
        self.sample.multimodal_train_inputs = mm_inputs
        self.sample.response = self.state.tokenizer.decode(self.response_tokens, skip_special_tokens=False)
        return self.sample

    async def _run_turns(self, env: Any) -> None:
        for turn_index in range(self.args.max_turns):
            remaining = self._remaining_budget
            if remaining is not None and remaining <= 0:
                self.sample.status = Sample.Status.TRUNCATED
                break
            sampling_params = dict(self.inference_params)
            if remaining is not None:
                sampling_params["max_tokens"] = remaining

            turn = await self._generate_turn(sampling_params)
            if turn.finish == "abort":
                self.sample.status = Sample.Status.ABORTED
                break
            if self._append_generated(turn):
                break
            if turn.finish == "length":
                self.sample.status = Sample.Status.TRUNCATED
                break
            if self._advance_environment(env, turn, turn_index):
                break

    async def run(self) -> Sample:
        env_module = _load_env_module(self.args.rollout_interaction_env_path)
        build_env = getattr(env_module, "build_env", None)
        if not callable(build_env):
            raise ValueError("Environment module must expose a callable build_env(sample, args).")
        env = build_env(sample=self.sample, args=self.args)
        active_error: BaseException | None = None
        try:
            env.reset()
            await self._initialize_prompt()
            await self._run_turns(env)
            return self._finalize()
        except BaseException as error:
            active_error = error
            raise
        finally:
            try:
                env.close()
            except BaseException as close_error:
                if active_error is None:
                    raise
                raise active_error from close_error


async def generate(args: Any, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    return await _Geo3kRollout(args, sample, sampling_params).run()
