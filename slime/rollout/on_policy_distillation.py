"""On-Policy Distillation (OPD): teacher logprobs via vLLM ``/inference/v1/generate``.

The reward function sends the full (prompt + student response) token sequence to
an external vLLM teacher server's ``/inference/v1/generate`` endpoint with
``prompt_logprobs`` enabled; the post-process step reads the per-token teacher
logprobs out of the response and stores them on each sample for the OPD KL
penalty.

Endpoint contract (vime vLLM disaggregated ``/inference/v1/generate``):

- Request body::

    {
        "token_ids": [...],            # full prompt+response token ids
        "model": <teacher model>,      # OPTIONAL on this endpoint; only sent when
                                       # --opd-teacher-model is set (single-model
                                       # teacher servers use their loaded model)
        "sampling_params": {
            "max_tokens": 1,           # endpoint requires >=1; the generated
                                       # token is ignored
            "temperature": 0,
            "prompt_logprobs": 1,      # score every prompt token
            "skip_special_tokens": False,
        },
    }

- Response body (``GenerateResponse``)::

    {
        "choices": [{"token_ids": [...], "logprobs": {...}, ...}],
        "prompt_logprobs": [           # TOP-LEVEL, aligned with the input token_ids
            None,                      # position 0: no prior context
            {<token_id>: {"logprob": -3.21, "rank": 1, "decoded_token": "..."}, ...},
            ...
        ],
    }

``prompt_logprobs[i]`` is a dict ``{token_id -> Logprob}`` for the token at
position ``i``. JSON serializes integer dict keys as strings, so we look up by
both ``int`` and ``str``. See ``vllm/entrypoints/serve/disagg/protocol.py``
(``GenerateRequest`` / ``GenerateResponse``).
"""

from __future__ import annotations

from typing import Any

import aiohttp
import torch

from slime.utils.types import Sample


async def reward_func(args, sample, **kwargs):
    if sample.multimodal_inputs and sample.multimodal_inputs.get("images"):
        # ``/inference/v1/generate`` is token-only; a multimodal teacher requires the
        # render -> generate flow (``/v1/chat/completions/render`` then attach
        # ``features``), as in ``slime.rollout.vllm_rollout.generate``. Not yet wired
        # for OPD — fail loudly rather than silently scoring a text-only sequence.
        raise NotImplementedError(
            "OPD multimodal teacher scoring over /inference/v1/generate is not implemented; "
            "wire the /v1/chat/completions/render -> /inference/v1/generate flow first."
        )

    payload: dict[str, Any] = {
        "token_ids": sample.tokens,
        "sampling_params": {
            "max_tokens": 1,
            "temperature": 0,
            "prompt_logprobs": 1,
            "skip_special_tokens": False,
        },
    }
    # ``model`` is optional on /inference/v1/generate. Only send it when the teacher
    # model is explicitly named; never fall back to the *student* checkpoint
    # (args.hf_checkpoint), which would mis-name a teacher!=student server.
    teacher_model = getattr(args, "opd_teacher_model", None)
    if teacher_model:
        payload["model"] = teacher_model

    async with aiohttp.ClientSession() as session:
        async with session.post(args.rm_url, json=payload) as resp:
            resp.raise_for_status()
            return await resp.json()


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


def post_process_rewards(args, samples: list[Sample], **kwargs):
    """Extract teacher log-probs from the ``/inference/v1/generate`` responses.

    1. Read top-level ``prompt_logprobs`` (aligned with the submitted token_ids).
    2. Pick out each actual token's logprob, skipping position 0 (always None).
    3. Trim to the response length and store on ``sample.teacher_log_probs``.
    4. Return scalar rewards (0.0 for pure distillation); the learning signal is
       the OPD KL penalty applied in ``compute_advantages_and_returns``.
    """
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    response_lengths = [sample.response_length for sample in samples]

    teacher_log_probs: list[torch.Tensor] = []
    for reward, sample in zip(raw_rewards, samples, strict=True):
        plp = reward.get("prompt_logprobs")
        assert plp is not None, "teacher response missing top-level prompt_logprobs"
        assert len(plp) == len(
            sample.tokens
        ), f"prompt_logprobs length {len(plp)} != token_ids length {len(sample.tokens)}"
        # plp[i] scores sample.tokens[i]; position 0 has no prior context.
        per_pos = [_logprob_for_token(plp[i], sample.tokens[i]) for i in range(1, len(sample.tokens))]
        teacher_log_probs.append(torch.tensor(per_pos, dtype=torch.float32))

    trimmed: list[torch.Tensor] = []
    for t_log_prob, response_length in zip(teacher_log_probs, response_lengths, strict=True):
        assert (
            len(t_log_prob) >= response_length
        ), f"teacher logprobs ({len(t_log_prob)}) shorter than response_length ({response_length})"
        trimmed.append(t_log_prob[-response_length:])

    for sample, t_log_probs in zip(samples, trimmed, strict=True):
        sample.teacher_log_probs = t_log_probs

    # Pure on-policy distillation: task reward is 0; KL penalty carries the signal.
    scalar_rewards = [0.0] * len(samples)
    return scalar_rewards, scalar_rewards
