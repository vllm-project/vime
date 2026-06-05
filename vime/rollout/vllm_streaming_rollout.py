"""Streaming vLLM rollout (example).

Drop-in alternative to :func:`vime.rollout.vllm_rollout.generate` that consumes
vLLM's ``/inference/v1/generate`` SSE stream incrementally instead of awaiting
one final JSON response. The win is on **abort**: every chunk we receive lands
directly on ``sample`` (tokens, response text, log-probs), so when a
partial-rollout recycling or weight-update abort fires mid-generation, the
partial state is already on the sample — we don't depend on the engine
returning the collected text.

Wire it in as the per-sample generate function::

    --rollout-function-path vime.rollout.vllm_rollout.generate_rollout \\
    --custom-generate-function-path vime.rollout.vllm_streaming_rollout.generate_streaming

The outer rollout loop (semaphore, dp_rank balancing, abort orchestration,
partial-rollout buffer hand-off) is still owned by ``vllm_rollout``; this file
only replaces the inner HTTP call.

vime/vLLM counterpart of ``slime.rollout.sglang_streaming_rollout``. The
behavioural difference from sglang matters here: sglang's streamed
``meta_info.output_token_logprobs`` is **cumulative** (every chunk references
the full list-so-far), whereas vLLM's ``/inference/v1/generate`` SSE chunks
carry **delta** ``token_ids`` + ``logprobs`` per
``GenerateResponseStreamChoice`` — so we *accumulate* the per-chunk deltas
(``+=``) rather than overwriting from each chunk. Each delta choice has the
same shape as the non-streaming choice, so :func:`_inference_generate_tokens_and_logprobs`
parses it unchanged.
"""

import json
import logging
from argparse import Namespace
from typing import Any

from vime.rollout.vllm_rollout import (
    GenerateState,
    _align_engine_tokens_and_logprobs,
    _align_mm_feature_placeholders_to_tokens,
    _apply_vllm_routed_experts,
    _base_dataset_prompt_ids,
    _build_inference_sampling_params,
    _coerce_flat_int_token_ids,
    _inference_generate_tokens_and_logprobs,
    _mm_render_response_to_generate_body,
    _prepare_prompt_ids,
    _vllm_meta_from_generate_choice,
)
from vime.utils import http_utils
from vime.utils.processing_utils import encode_image_for_rollout_engine
from vime.utils.trace_utils import build_vllm_meta_trace_attrs, trace_span
from vime.utils.types import Sample

__all__ = ["generate_streaming"]

logger = logging.getLogger(__name__)


async def generate_streaming(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Streaming counterpart to :func:`vime.rollout.vllm_rollout.generate`.

    Writes the accumulated state from each SSE chunk onto ``sample`` so an abort
    that cuts the stream still leaves a coherent partial sample behind.
    """
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
    url = f"{base}/inference/v1/generate"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)
    base_prompt_ids = _base_dataset_prompt_ids(sample, state.tokenizer, state.processor)

    # Multimodal samples use the same render-dance as the non-streaming text
    # path (/v1/chat/completions/render → features), then stream the generate
    # call. Streaming only changes how output is returned (SSE deltas vs one
    # JSON); the image render (input prep) is identical. Built below once
    # sampling params + token_ids are resolved.
    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None

    params = dict(sampling_params)
    if len(sample.response) > 0:
        params["max_new_tokens"] -= len(sample.tokens) - len(base_prompt_ids)

    assert params["max_new_tokens"] >= 0, (
        f"max_new_tokens: {params['max_new_tokens']} should not be less than 0 "
        f"(after partial continuation adjustment; tokens={len(sample.tokens)}, base_prompt={len(base_prompt_ids)})"
    )
    if params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample
    inference_sampling_params = _build_inference_sampling_params(params)

    if not sample.tokens:
        sample.tokens = prompt_ids

    # vLLM ``/inference/v1/generate`` is token-only. On partial continuation,
    # send the full prompt+response prefix so the engine continues from the
    # current sample state (mirrors the non-streaming text path).
    if len(sample.response) > 0:
        token_ids = _coerce_flat_int_token_ids(sample.tokens)
    else:
        token_ids = prompt_ids

    # Use session_id for consistent_hash routing (vime convention: x-session-id
    # header + policy "consistent_hash"). See vllm_rollout.generate.
    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hash":
        headers = {"x-session-id": sample.session_id}

    payload: dict[str, Any]
    if images:
        # Same render-dance as vllm_rollout.generate's MM path, then stream.
        # mm placeholders live in the (stable) prompt prefix, so re-rendering and
        # re-aligning to the current token_ids holds across partial continuations.
        content: list[dict[str, Any]] = [{"type": "text", "text": sample.prompt}]
        for image in images:
            content.append({"type": "image_url", "image_url": {"url": encode_image_for_rollout_engine(image)}})
        render_payload = {"model": args.hf_checkpoint, "messages": [{"role": "user", "content": content}]}
        with trace_span(sample, "vllm_mm_render", attrs={"model": args.hf_checkpoint}):
            render_data = await http_utils.post(
                f"{base}/v1/chat/completions/render", render_payload, headers=headers
            )
        payload = _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)
        if token_ids:
            _align_mm_feature_placeholders_to_tokens(payload, token_ids)
            payload["token_ids"] = token_ids
        payload["sampling_params"] = inference_sampling_params
        payload["stream"] = True
    else:
        payload = {
            "model": args.hf_checkpoint,
            "token_ids": token_ids,
            "sampling_params": inference_sampling_params,
            "stream": True,
        }

    # Snapshot pre-call sample state. vLLM's SSE chunks are *deltas* within this
    # call; on each chunk we append the delta and rebuild the post-call view of
    # the sample = prior state + accumulated deltas. A mid-stream break leaves
    # the sample exactly at the boundary of the last chunk we observed.
    base_tokens = list(sample.tokens)
    base_response = sample.response or ""
    base_response_length = sample.response_length
    base_log_probs = list(sample.rollout_log_probs or [])
    base_loss_mask = list(sample.loss_mask) if sample.loss_mask is not None else None

    skip_sp = params.get("skip_special_tokens")
    skip_decode = True if skip_sp is None else bool(skip_sp)

    call_tokens: list[int] = []
    call_log_probs: list[float] = []
    last_choice: dict[str, Any] | None = None
    last_usage: dict[str, Any] | None = None
    finish_reason: Any = None

    client = http_utils._http_client
    assert client is not None, "http client not initialized; call init_http_client first"

    with trace_span(
        sample, "vllm_inference_generate_stream", attrs={"max_new_tokens": params["max_new_tokens"]}
    ) as span:
        async with client.stream("POST", url, json=payload, headers=headers) as response:
            response.raise_for_status()
            async for raw_line in response.aiter_lines():
                if not raw_line or not raw_line.startswith("data:"):
                    continue
                data_str = raw_line[len("data:") :].strip()
                if not data_str or data_str == "[DONE]":
                    continue
                try:
                    chunk = json.loads(data_str)
                except json.JSONDecodeError:
                    logger.warning("vllm_streaming: skipping non-JSON chunk: %r", data_str[:120])
                    continue

                choices = chunk.get("choices") or []
                if not choices:
                    # usage-only / keepalive chunk
                    if chunk.get("usage"):
                        last_usage = chunk["usage"]
                    continue
                choice = choices[0]
                last_choice = choice
                if chunk.get("usage"):
                    last_usage = chunk["usage"]
                if choice.get("finish_reason"):
                    finish_reason = choice["finish_reason"]

                # Each streamed choice carries only this chunk's *delta* tokens
                # (GenerateResponseStreamChoice), so accumulate.
                delta_tokens, delta_log_probs = _inference_generate_tokens_and_logprobs(choice)
                if delta_tokens:
                    call_tokens += delta_tokens
                    call_log_probs += delta_log_probs

                # Surface partial state on the sample immediately. If the outer
                # abort path cuts us, whatever we've written so far is what
                # survives. Decode the *accumulated* tokens (not the per-chunk
                # delta) so multi-token characters straddling a chunk boundary
                # decode correctly.
                sample.tokens = base_tokens + call_tokens
                sample.response = base_response + (
                    state.tokenizer.decode(call_tokens, skip_special_tokens=skip_decode) if call_tokens else ""
                )
                sample.response_length = base_response_length + len(call_tokens)
                sample.rollout_log_probs = base_log_probs + call_log_probs
                if base_loss_mask is not None:
                    assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
                    sample.loss_mask = base_loss_mask + [1] * len(call_tokens)

                if state.aborted:
                    break

        if finish_reason and last_choice is not None:
            span.update(build_vllm_meta_trace_attrs({"choices": [last_choice], "usage": last_usage}))

    if finish_reason and last_choice is not None:
        # Finalize exactly like the non-streaming path: align logprobs to tokens,
        # rebuild meta + output_token_logprobs, then let Sample own status.
        new_response_tokens, new_response_log_probs = _align_engine_tokens_and_logprobs(call_tokens, call_log_probs)

        meta = _vllm_meta_from_generate_choice(args, last_choice, last_usage)
        if new_response_tokens:
            meta["output_token_logprobs"] = [
                [float(lp), int(tid)] for lp, tid in zip(new_response_log_probs, new_response_tokens, strict=True)
            ]

        sample.tokens = base_tokens + new_response_tokens
        sample.response = base_response + (
            state.tokenizer.decode(new_response_tokens, skip_special_tokens=skip_decode) if new_response_tokens else ""
        )
        sample.response_length = base_response_length + len(new_response_tokens)
        sample.rollout_log_probs = base_log_probs + new_response_log_probs
        if base_loss_mask is not None:
            assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
            sample.loss_mask = base_loss_mask + [1] * len(new_response_tokens)

        sample.update_from_meta_info(args, meta)
        # MoE routing replay (when requested) ships on the terminal choice.
        _apply_vllm_routed_experts(args, sample, last_choice)
    elif state.aborted:
        sample.status = Sample.Status.ABORTED

    return sample
