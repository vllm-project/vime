"""On-Policy Distillation (OPD): teacher logprobs via vLLM `/v1/completions`.

The reward function pings an external vLLM teacher server and requests
prompt-level logprobs for the (prompt + student response) sequence; the
post-process step pulls the per-token logprobs out of the teacher
response and stores them on each sample for the OPD KL penalty.

Endpoint contract (vLLM 0.21+):

- Request body (OpenAI-compatible `POST /v1/completions`)::

    {
        "model": <teacher model name>,
        "prompt_token_ids": [...],     # full prompt+response token ids
        "max_tokens": 1,               # vLLM requires >=1; we ignore the
                                       # single generated token
        "temperature": 0,
        "prompt_logprobs": 1,          # ask vLLM to score each prompt token
        "logprobs": 0,                 # we don't need the output token's logprob
        "skip_special_tokens": False,
    }

- Response body (truncated)::

    {
        "choices": [{
            "index": 0,
            "text": "<one generated token>",
            "prompt_logprobs": [
                None,                  # first position: no prior context
                {<chosen_token_id>: {"logprob": -3.21, "rank": 1,
                                     "decoded_token": "..."},  ...},
                ...
            ],
            ...
        }],
        ...
    }

`prompt_logprobs[i]` is a dict `{token_id -> Logprob}`. We pick out the
entry whose key matches `sample.tokens[i]` (the actual token at position
i). JSON serializes dict keys as strings, so we look up by both int and
str. See vllm/entrypoints/openai/completion/protocol.py:487 and
vllm/logprobs.py:Logprob.
"""

from __future__ import annotations

from typing import Any

import aiohttp
import torch

from slime.utils.processing_utils import encode_image_for_rollout_engine
from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    payload: dict[str, Any] = {
        "model": getattr(args, "opd_teacher_model", None) or args.hf_checkpoint,
        "prompt_token_ids": sample.tokens,
        "max_tokens": 1,
        "temperature": 0,
        "prompt_logprobs": 1,
        "logprobs": 0,
        "skip_special_tokens": False,
    }

    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        image_data = sample.multimodal_inputs["images"]
        # vLLM accepts multimodal inputs via the chat completions endpoint
        # only; for /v1/completions we fall back to the rollout-engine
        # helper to encode images alongside the token sequence.
        payload["image_data"] = [encode_image_for_rollout_engine(image) for image in image_data]

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


def _logprob_for_token(pos_entry: dict | None, token_id: int) -> float:
    """Pull the teacher's logprob for `token_id` out of one position's logprob dict.

    `pos_entry` is `None` at the very first prompt position (no prior context).
    JSON serializes integer dict keys as strings, so we accept both. Each value
    is itself either a dict (`{"logprob": float, ...}`) or already a flattened
    float (depending on the server's serialization toggle).
    """
    if pos_entry is None:
        return 0.0
    entry = pos_entry.get(token_id)
    if entry is None:
        entry = pos_entry.get(str(token_id))
    if entry is None:
        return 0.0
    if isinstance(entry, dict):
        return float(entry.get("logprob", 0.0))
    if isinstance(entry, (int, float)):
        return float(entry)
    return float(getattr(entry, "logprob", 0.0))


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Process rewards from teacher model and extract teacher log probabilities.

    This function:
    1. Extracts teacher log-probs from the teacher's `/v1/completions`
       `prompt_logprobs` payload.
    2. Trims them to match the response length.
    3. Stores them in `sample.teacher_log_probs` for OPD KL penalty computation.
    4. Returns scalar rewards (0.0 for pure distillation) compatible with GRPO/PPO.

    For pure on-policy distillation without task rewards we return 0.0 for each
    sample — the learning signal comes entirely from the OPD KL penalty applied
    in `compute_advantages_and_returns`.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs: list[torch.Tensor] = []
    for reward, sample in zip(raw_rewards, samples, strict=False):
        choice = reward["choices"][0]
        plp = choice.get("prompt_logprobs") or []
        # Skip position 0 (always None — no prior context). Align the
        # remaining positions with sample.tokens[1:].
        per_pos = [_logprob_for_token(plp[i], sample.tokens[i]) for i in range(1, min(len(plp), len(sample.tokens)))]
        teacher_log_probs.append(torch.tensor(per_pos, dtype=torch.float32))

    teacher_log_probs = [
        t_log_prob[-response_length:]
        for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=False)
    ]

    for sample, t_log_probs in zip(samples, teacher_log_probs, strict=False):
        sample.teacher_log_probs = t_log_probs

    # Pure on-policy distillation: task reward is 0; KL penalty carries the signal.
    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
