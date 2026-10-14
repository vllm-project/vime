from __future__ import annotations

import base64
import binascii
import io
from typing import Any
from urllib.parse import urlsplit, urlunsplit

import numpy as np
import torch

from vime.utils.http_utils import post
from vime.utils.types import Sample


def _teacher_base_url(rm_url: str) -> str:
    """Strip the path off ``--rm-url`` to get the teacher server base (scheme://host:port)."""
    parts = urlsplit(rm_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _response_start(sample: Sample) -> int:
    length = sample.response_length
    if length < 0 or length > len(sample.tokens) or (length and length == len(sample.tokens)):
        raise ValueError("OPD response_length must leave at least one context token for a nonempty response")
    return len(sample.tokens) - length


def _use_per_position(args, sample: Sample) -> bool:
    return getattr(args, "opd_teacher_scoring", "prompt-logprobs") == "per-position" and not (
        sample.multimodal_inputs or {}
    ).get("images")


async def reward_func(args, sample, **kwargs):
    response_start = _response_start(sample)
    teacher_model = getattr(args, "opd_teacher_model", None)
    sampling_params = {
        "max_tokens": 1,
        "temperature": args.rollout_temperature,
        "prompt_logprobs": 1,
        "skip_special_tokens": False,
    }
    base = _teacher_base_url(args.rm_url)
    mm = sample.multimodal_inputs or {}
    images = mm.get("images")
    # Only image multimodal is wired here (mirrors vllm_rollout.generate, which renders
    # images only). Fail loud on other modalities rather than silently scoring text-only
    # without their context.
    unsupported = [k for k, v in mm.items() if k != "images" and v]
    if unsupported and not images:
        raise NotImplementedError(
            "OPD teacher scoring over /inference/v1/generate supports only image multimodal; "
            f"got unsupported modalities: {unsupported}"
        )

    if sample.response_length == 0:
        return {}

    if images:
        # Multimodal: render (preprocess images) to get token_ids + features, then
        # score the student's full prompt+response token sequence with those features
        # attached — mirrors vime.rollout.vllm_rollout.generate's disaggregated MM flow.
        # Lazy import: these helpers live in vllm_rollout, which pulls in the rollout/
        # router stack the text-only reward path never needs — import only when an
        # image sample is actually scored.
        from vime.rollout.vllm_rollout import (
            _align_mm_feature_placeholders_to_tokens,
            _coerce_flat_int_token_ids,
            _mm_render_response_to_generate_body,
        )
        from vime.utils.processing_utils import encode_image_for_rollout_engine

        content: list[dict[str, Any]] = [{"type": "text", "text": sample.prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": encode_image_for_rollout_engine(image)}})
        render_payload: dict[str, Any] = {"messages": [{"role": "user", "content": content}]}
        if teacher_model:
            render_payload["model"] = teacher_model
        render_data = await post(f"{base}/v1/chat/completions/render", render_payload)

        body = _mm_render_response_to_generate_body(render_data, teacher_model or "")
        canonical = _coerce_flat_int_token_ids(sample.tokens)
        # Align the rendered feature placeholders to VIME's canonical ids BEFORE
        # overriding token_ids (the aligner reads the render's token_ids first).
        _align_mm_feature_placeholders_to_tokens(body, canonical)
        body["token_ids"] = canonical
        body["sampling_params"] = sampling_params
        if teacher_model:
            body["model"] = teacher_model
        else:
            body.pop("model", None)
        return await post(args.rm_url, body)

    if _use_per_position(args, sample):
        sampling_params.pop("prompt_logprobs")
        # Each logit row predicts the next token, including the first response token.
        sampling_params["prompt_logprob_start"] = response_start - 1
        sampling_params["prompt_logprob_token_ids"] = [[token] for token in sample.tokens[response_start:]]

    payload: dict[str, Any] = {"token_ids": sample.tokens, "sampling_params": sampling_params}
    if teacher_model:
        payload["model"] = teacher_model
    return await post(args.rm_url, payload)


def _logprob_for_token(pos_entry: dict | None, token_id: int) -> float:
    """Pull the teacher's logprob for ``token_id`` out of one position's logprob dict.

    Raises on a missing entry: vLLM always includes the actual prompt token in
    ``prompt_logprobs`` (even when only top-1 is requested), so a missing token
    means a malformed/misaligned response that must not be papered over with 0.0.
    JSON serializes integer dict keys as strings, so we accept both. Each value
    is either a dict (``{"logprob": float, ...}``) or a flattened float.
    """
    if pos_entry is None:
        raise ValueError("teacher prompt_logprobs has a None entry at a scored position")
    entry = pos_entry.get(token_id)
    if entry is None:
        entry = pos_entry.get(str(token_id))
    if entry is None:
        raise ValueError(f"teacher prompt_logprobs missing logprob for token_id={token_id}")
    if isinstance(entry, dict):
        return float(entry["logprob"])
    if isinstance(entry, (int, float)):
        return float(entry)
    return float(entry.logprob)


def _per_position_logprobs(reward: dict, response_length: int) -> torch.Tensor:
    encoded = reward.get("prompt_token_id_logprobs")
    if not isinstance(encoded, str):
        raise ValueError(
            "teacher response missing prompt_token_id_logprobs; per-position scoring requires "
            "a vLLM teacher supporting per-row candidates with VLLM_USE_V2_MODEL_RUNNER=1"
        )
    try:
        raw = base64.b64decode(encoded, validate=True)
        if not raw.startswith(b"\x93NUMPY"):
            raise ValueError("expected a NumPy array")
        scores = np.load(io.BytesIO(raw), allow_pickle=False)
    except (binascii.Error, ValueError, EOFError, OSError) as exc:
        raise ValueError("teacher prompt_token_id_logprobs must be a base64-encoded NumPy array") from exc
    if scores.dtype != np.float32 or scores.shape != (response_length, 1):
        raise ValueError(
            "teacher prompt_token_id_logprobs must be float32 with shape "
            f"({response_length}, 1), got {scores.dtype} {scores.shape}"
        )
    if not np.isfinite(scores).all() or np.any(scores > 1e-5):
        raise ValueError("teacher scores must be finite log probabilities; configure --logprobs-mode raw_logprobs")
    return torch.from_numpy(scores[:, 0].copy())


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Store response-token teacher scores, preserving the learner's OPD contract."""
    teacher_log_probs: list[torch.Tensor] = []
    for sample in samples:
        _response_start(sample)
        if sample.response_length == 0:
            teacher_log_probs.append(torch.empty(0, dtype=torch.float32))
            continue
        reward = sample.get_reward_value(args)
        if _use_per_position(args, sample):
            teacher_log_probs.append(_per_position_logprobs(reward, sample.response_length))
            continue
        plp = reward.get("prompt_logprobs")
        assert plp is not None, "teacher response missing top-level prompt_logprobs"
        if len(plp) != len(sample.tokens):
            raise ValueError("teacher prompt_logprobs must align with the submitted token_ids")
        # plp[i] scores sample.tokens[i]; position 0 has no prior context.
        per_pos = [_logprob_for_token(plp[i], sample.tokens[i]) for i in range(1, len(sample.tokens))]
        teacher_log_probs.append(torch.tensor(per_pos[-sample.response_length :], dtype=torch.float32))

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Pure on-policy distillation: task reward is 0; KL penalty carries the signal.
    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
