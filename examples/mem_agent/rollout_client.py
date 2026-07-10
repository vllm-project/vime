"""Lightweight vLLM router client for MemAgent multi-turn rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from vime.rollout.vllm_rollout import (
    _align_engine_tokens_and_logprobs,
    _build_inference_sampling_params,
    _inference_generate_tokens_and_logprobs,
)
from vime.utils.http_utils import post


@dataclass
class GenerationOutput:
    prompt_text: str
    prompt_token_ids: list[int]
    response: str
    token_ids: list[int]
    log_probs: list[float]
    finish_reason: str


class MemAgentRolloutClient:
    """Wrap vLLM ``/inference/v1/generate`` for chat-style MemAgent turns."""

    def __init__(self, args: Any, tokenizer: Any, sampling_params: dict):
        self.base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
        self.model = args.hf_checkpoint
        self.tokenizer = tokenizer
        self.sampling_params = dict(sampling_params)

    async def generate(
        self,
        messages: list[dict[str, str]],
        sampling_params: dict | None = None,
    ) -> GenerationOutput:
        params = dict(self.sampling_params)
        if sampling_params:
            params.update(sampling_params)

        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        prompt_token_ids = self.tokenizer(prompt_text, add_special_tokens=False)["input_ids"]

        payload = {
            "model": self.model,
            "token_ids": prompt_token_ids,
            "sampling_params": _build_inference_sampling_params(params),
        }
        output = await post(f"{self.base}/inference/v1/generate", payload)
        choice = output["choices"][0]

        skip_sp = params.get("skip_special_tokens")
        skip_decode = True if skip_sp is None else bool(skip_sp)
        out_ids = choice.get("token_ids") or []
        text = (
            self.tokenizer.decode(out_ids, skip_special_tokens=skip_decode)
            if isinstance(out_ids, list) and out_ids
            else ""
        )

        token_ids, log_probs = _inference_generate_tokens_and_logprobs(choice)
        token_ids, log_probs = _align_engine_tokens_and_logprobs(token_ids, log_probs)

        fr = choice.get("finish_reason") or "stop"
        if isinstance(fr, dict):
            finish_reason = fr.get("type", "stop")
        else:
            finish_reason = "length" if fr == "length" else "stop"

        return GenerationOutput(
            prompt_text=prompt_text,
            prompt_token_ids=prompt_token_ids,
            response=text,
            token_ids=token_ids,
            log_probs=log_probs,
            finish_reason=finish_reason,
        )
