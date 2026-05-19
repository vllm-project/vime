import asyncio
import copy
import inspect
import json
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import vllm_router  # noqa: F401 — same side-effect as ``import sglang_router`` in sglang rollout
from tqdm import tqdm

from slime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from slime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from slime.utils.async_utils import run
from slime.utils.data import Dataset
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.http_utils import get, post
from slime.utils.misc import SingletonMeta, load_function
from slime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from slime.utils.trace_utils import trace_function, trace_span
from slime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}


def _coerce_flat_int_token_ids(ids: Any) -> list[int]:
    """Turn tokenizer / processor output into a flat ``list[int]`` for vLLM ``/v1/completions`` JSON.

    vLLM deserializes ``prompt`` as ``StringOrArray`` (Rust): a JSON string or a JSON array of integers.
    Nested lists, numpy/torch scalars, or non-int elements cause ``422`` deserialization errors.
    """
    if ids is None:
        return []
    if isinstance(ids, str):
        raise TypeError("token ids must not be a str; use the string ``prompt`` field for text prompts")
    x = ids
    if hasattr(x, "tolist") and not isinstance(x, (list, tuple, str, bytes)):
        x = x.tolist()
    if isinstance(x, (list, tuple)):
        out: list[int] = []
        for item in x:
            out.extend(_coerce_flat_int_token_ids(item))
        return out
    return [int(x)]


def _prepare_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    reuse_existing_input_ids = bool(sample.tokens) and (
        sample.multimodal_train_inputs is not None or not has_multimodal_inputs
    )

    if processor and has_multimodal_inputs and not reuse_existing_input_ids:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        if sample.multimodal_train_inputs is None:
            sample.multimodal_train_inputs = {
                k: v for k, v in processor_output.items() if k not in _PROCESSOR_PROMPT_KEYS
            } or None
        return _coerce_flat_int_token_ids(prompt_ids)

    if reuse_existing_input_ids:
        return _coerce_flat_int_token_ids(sample.tokens)

    return _coerce_flat_int_token_ids(tokenizer.encode(sample.prompt, add_special_tokens=False))


def _base_dataset_prompt_ids(sample: Sample, tokenizer, processor: Any) -> list[int]:
    """Token ids for the dataset prompt only (never reuse ``sample.tokens``).

    Used for partial-continuation budgeting to match ``dev_vllm`` ``sglang_rollout``:
    ``max_new_tokens -= len(sample.tokens) - len(base_prompt_ids)`` when ``sample.response`` is non-empty.
    """
    raw_multimodal_inputs = sample.multimodal_inputs or {}
    has_multimodal_inputs = any(value is not None for value in raw_multimodal_inputs.values())
    if processor and has_multimodal_inputs:
        processor_output = processor(text=sample.prompt, **build_processor_kwargs(raw_multimodal_inputs))
        prompt_ids = processor_output["input_ids"][0]
        return _coerce_flat_int_token_ids(prompt_ids)
    return _coerce_flat_int_token_ids(tokenizer.encode(sample.prompt, add_special_tokens=False))


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/v1/completions") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--sglang-config``::

        url = get_model_url(args, "ref", "/v1/completions")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``sglang_model_routers`` is not set.
    """
    routers = getattr(args, "sglang_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.router_ip}:{args.router_port}{endpoint}"


async def _router_worker_urls(args: Namespace) -> list[str]:
    """Resolve worker base URLs from the vLLM router (same HTTP shape as SGLang router)."""
    base = f"http://{args.router_ip}:{args.router_port}"
    try:
        response = await get(f"{base}/workers")
        return [worker["url"] for worker in response["workers"]]
    except Exception:
        response = await get(f"{base}/list_workers")
        return list(response["urls"])


async def _resume_vllm_workers(urls: list[str]) -> None:
    """Call ``POST /resume`` on each worker after ``pause?mode=abort`` so engines accept traffic again."""
    if not urls:
        return
    logger.info("vLLM rollout: resuming workers after abort drain: %s", urls)
    resume_tasks = [post(f"{url.rstrip('/')}/resume", {}, max_retries=3) for url in urls]
    resume_results = await asyncio.gather(*resume_tasks, return_exceptions=True)
    for url, result in zip(urls, resume_results, strict=False):
        if isinstance(result, Exception):
            logger.warning("Failed to resume vLLM worker at %s: %s", url, result)


def _openai_meta_from_completion_choice(args: Namespace, choice: dict, usage: dict | None) -> dict[str, Any]:
    """Build a minimal ``meta_info``-like dict for :meth:`Sample.update_from_meta_info`."""
    fr = choice.get("finish_reason") or "stop"
    if isinstance(fr, dict):
        finish = fr
    else:
        if fr == "length":
            typ = "length"
        elif fr in ("abort", "cancelled"):
            typ = "abort"
        else:
            typ = "stop"
        finish = {"type": typ}
    meta: dict[str, Any] = {"finish_reason": finish}
    if usage:
        meta["prompt_tokens"] = usage.get("prompt_tokens", 0)
        meta["completion_tokens"] = usage.get("completion_tokens", 0)
    return meta


def _apply_vllm_routed_experts(
    args: Namespace,
    sample: Sample,
    _output: dict,
    choice: dict,
) -> None:
    """Populate ``sample.rollout_routed_experts`` from vLLM ``choices[].routed_experts`` when enabled.

    vLLM exposes MoE routing replay on the **per-completion** ``CompletionOutput`` as a single
    ``routed_experts`` ndarray (shape ``[num_positions, num_layers, topk]``); the V1 scheduler
    builds it once per finished request from KV slot indices for ``request.num_tokens - 1``
    positions — there is **no** separate ``prompt_routed_experts`` key on the HTTP completion
    payload (confirmed absent in upstream ``vllm``; see ``CompletionOutput`` in
    ``vllm/outputs.py`` and ``_get_routed_experts`` in ``vllm/v1/core/sched/scheduler.py``).
    When the OpenAI layer forwards it, it appears as an extra field on the choice object
    (Pydantic ``extra="allow"`` on ``CompletionResponseChoice``).
    """
    if not getattr(args, "use_rollout_routing_replay", False):
        return
    gen_re = choice.get("routed_experts")
    if gen_re is None:
        return
    arr = np.asarray(gen_re, dtype=np.int32)
    n_tok = len(sample.tokens)
    expected_rows = max(0, n_tok - 1)
    if arr.ndim != 3:
        logger.warning(f"Unexpected routed_experts ndim={arr.ndim} shape={arr.shape}")
        return
    if arr.shape[0] == n_tok:
        arr = arr[:-1]
    elif arr.shape[0] != expected_rows:
        logger.warning(
            f"routed_experts row count {arr.shape[0]} not in {{{expected_rows}, {n_tok}}}; "
            "skipping rollout_routed_experts assign",
        )
        return
    nl = getattr(args, "num_layers", None)
    mtk = getattr(args, "moe_router_topk", None)
    if nl is not None and mtk is not None and (arr.shape[1] != nl or arr.shape[2] != mtk):
        logger.warning(
            f"routed_experts shape {arr.shape} does not match args (num_layers={nl}, moe_router_topk={mtk})",
        )
        return
    sample.rollout_routed_experts = arr


def _fallback_tokens_from_text_and_completion_logprobs(
    tokenizer, choice: dict, completion_text: str
) -> tuple[list[int], list[float]]:
    """Fallback when vLLM did not return ``token_ids``: approximate from string logprobs or re-tokenize."""
    lp = choice.get("logprobs")
    if not lp or not isinstance(lp, dict):
        toks = tokenizer.encode(completion_text, add_special_tokens=False)
        return toks, [0.0] * len(toks)

    token_logprobs = lp.get("token_logprobs")
    tokens_field = lp.get("tokens")
    if token_logprobs and tokens_field and len(token_logprobs) == len(tokens_field):
        new_toks: list[int] = []
        new_lps: list[float] = []
        for piece, logp in zip(tokens_field, token_logprobs, strict=False):
            if logp is None:
                continue
            ids = tokenizer.encode(piece, add_special_tokens=False)
            if not ids:
                continue
            per = float(logp) / max(len(ids), 1)
            new_toks.extend(ids)
            new_lps.extend([per] * len(ids))
        if new_toks:
            return new_toks, new_lps

    toks = tokenizer.encode(completion_text, add_special_tokens=False)
    return toks, [0.0] * len(toks)


def _vllm_engine_tokens_and_logprobs(
    tokenizer,
    choice: dict[str, Any],
    completion_text: str,
    *,
    is_chat: bool,
) -> tuple[list[int], list[float]]:
    """Parse engine ``token_ids`` and per-token logprobs from an OpenAI choice (requires ``return_token_ids``)."""
    tids_raw = choice.get("token_ids")
    if isinstance(tids_raw, list) and tids_raw and all(isinstance(x, int) for x in tids_raw):
        tids = [int(x) for x in tids_raw]
        lps: list[float] = []
        lp = choice.get("logprobs")
        if not isinstance(lp, dict):
            return tids, [0.0] * len(tids)

        if is_chat:
            content = lp.get("content")
            if isinstance(content, list) and len(content) == len(tids):
                for item in content:
                    if isinstance(item, dict):
                        lps.append(float(item.get("logprob", 0.0)))
                    else:
                        lps.append(0.0)
                return tids, lps
            if isinstance(content, list) and content:
                # Partial content: pad / truncate to token_ids length if possible
                for i in range(len(tids)):
                    if i < len(content) and isinstance(content[i], dict):
                        lps.append(float(content[i].get("logprob", 0.0)))
                    else:
                        lps.append(0.0)
                return tids, lps
            return tids, [0.0] * len(tids)

        token_logprobs = lp.get("token_logprobs")
        if isinstance(token_logprobs, list) and len(token_logprobs) == len(tids):
            for p in token_logprobs:
                lps.append(0.0 if p is None else float(p))
            return tids, lps

        return tids, [0.0] * len(tids)

    if is_chat:
        lp = choice.get("logprobs")
        content = lp.get("content") if isinstance(lp, dict) else None
        if isinstance(content, list) and content:
            tids_ch: list[int] = []
            lps_ch: list[float] = []
            for item in content:
                if not isinstance(item, dict):
                    continue
                tok = item.get("token")
                if not isinstance(tok, str):
                    continue
                ids = tokenizer.encode(tok, add_special_tokens=False)
                if not ids:
                    continue
                lv = float(item.get("logprob", 0.0))
                per = lv / max(len(ids), 1)
                tids_ch.extend(ids)
                lps_ch.extend([per] * len(ids))
            if tids_ch:
                return tids_ch, lps_ch

    return _fallback_tokens_from_text_and_completion_logprobs(tokenizer, choice, completion_text)


def _align_engine_tokens_and_logprobs(
    new_response_tokens: list[int], new_response_log_probs: list[float]
) -> tuple[list[int], list[float]]:
    """Pad or truncate logprobs so ``len(.) == len(new_response_tokens)`` (SGLang always has matched OTL pairs)."""
    n = len(new_response_tokens)
    if n == 0:
        return [], []
    m = len(new_response_log_probs)
    if m == n:
        return new_response_tokens, [float(x) for x in new_response_log_probs]
    if m > n:
        return new_response_tokens, [float(x) for x in new_response_log_probs[:n]]
    return new_response_tokens, [float(x) for x in new_response_log_probs] + [0.0] * (n - m)


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(
            args.vllm_server_concurrency * args.rollout_num_gpus // args.rollout_num_gpus_per_engine
        )
        self.sampling_params: dict[str, Any] = dict(
            temperature=args.rollout_temperature,
            top_p=args.rollout_top_p,
            top_k=args.rollout_top_k,
            max_new_tokens=args.rollout_max_response_len,
            stop=args.rollout_stop,
            stop_token_ids=args.rollout_stop_token_ids,
            skip_special_tokens=args.rollout_skip_special_tokens,
            no_stop_trim=True,
            spaces_between_special_tokens=False,
        )

        if getattr(args, "sglang_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.sglang_dp_size or 1)
        self.dp_rank = 0

        self.reset()

    @contextmanager
    def dp_rank_context(self):
        candidates = [i for i, count in enumerate(self.dp_counts) if count == min(self.dp_counts)]
        dp_rank = int(np.random.choice(candidates))
        self.dp_counts[dp_rank] += 1
        self.dp_rank = dp_rank
        try:
            yield dp_rank
        finally:
            self.dp_counts[dp_rank] -= 1
            assert self.dp_counts[dp_rank] >= 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False

    def submit_generate_tasks(self, samples: list[list[Sample]]) -> None:
        for group in samples:
            self.pendings.add(
                asyncio.create_task(
                    # submit a group of samples as a single task.
                    generate_and_rm_group(
                        self.args,
                        group,
                        sampling_params=self.sampling_params.copy(),
                        evaluation=False,
                    )
                )
            )
        self.remaining_batch_size += len(samples)


def _build_inference_sampling_params(sampling_params: dict[str, Any]) -> dict[str, Any]:
    """Map rollout ``sampling_params`` to vLLM ``/inference/v1/generate`` ``sampling_params`` body."""
    sp: dict[str, Any] = {
        "max_tokens": sampling_params["max_new_tokens"],
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "logprobs": 1,
    }
    tk = sampling_params.get("top_k")
    if tk is not None and tk > 0:
        sp["top_k"] = tk
    if sampling_params.get("stop"):
        sp["stop"] = sampling_params["stop"]
    if sampling_params.get("stop_token_ids"):
        sp["stop_token_ids"] = sampling_params["stop_token_ids"]
    if sampling_params.get("seed") is not None:
        sp["seed"] = sampling_params["seed"]
    if sampling_params.get("skip_special_tokens") is not None:
        sp["skip_special_tokens"] = bool(sampling_params["skip_special_tokens"])
    return sp


def _mm_render_response_to_generate_body(render_data: Any, model: str) -> dict[str, Any]:
    """Turn ``/v1/chat/completions/render`` JSON into a ``/inference/v1/generate`` request body (minus ``sampling_params``).

    vLLM stable docs use a flat dict with ``token_ids`` and optional ``features``; some builds return
    ``[conversation, engine_prompts]`` from the render route — normalize both.
    """
    if isinstance(render_data, dict) and isinstance(render_data.get("token_ids"), list):
        body = copy.deepcopy(render_data)
        body.setdefault("model", model)
        return body

    if isinstance(render_data, list) and len(render_data) >= 2:
        engine_prompts = render_data[1]
        if not isinstance(engine_prompts, list) or not engine_prompts:
            raise ValueError("chat/render: expected non-empty engine_prompts list")
        p = engine_prompts[0]
        if not isinstance(p, dict):
            raise ValueError("chat/render: engine_prompts[0] must be a dict")
        token_ids = p.get("prompt_token_ids") or p.get("token_ids")
        if not isinstance(token_ids, list) or not token_ids:
            raise ValueError("chat/render: missing prompt_token_ids / token_ids on engine prompt")
        body: dict[str, Any] = {"token_ids": [int(x) for x in token_ids], "model": model}
        if p.get("features") is not None:
            body["features"] = p["features"]
        elif isinstance(p.get("multi_modal_data"), dict):
            try:
                body["features"] = json.dumps(p["multi_modal_data"], default=str)
            except TypeError:
                pass
        if p.get("cache_salt") is not None:
            body["cache_salt"] = p["cache_salt"]
        return body

    raise ValueError(
        "chat/render: unexpected JSON shape; expected a dict with token_ids or "
        "[conversation, engine_prompts] list"
    )


def _build_completion_payload(
    args: Namespace, sampling_params: dict[str, Any], prompt_field: str | list[int]
) -> dict[str, Any]:
    """Map shared ``sampling_params`` to vLLM OpenAI ``/v1/completions`` body."""
    body: dict[str, Any] = {
        "model": args.hf_checkpoint,
        "prompt": prompt_field,
        "max_tokens": sampling_params["max_new_tokens"],
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "logprobs": 1,
        "return_token_ids": True,
    }
    tk = sampling_params.get("top_k")
    if tk is not None and tk > 0:
        body["top_k"] = tk
    if sampling_params.get("stop"):
        body["stop"] = sampling_params["stop"]
    if sampling_params.get("stop_token_ids"):
        body["stop_token_ids"] = sampling_params["stop_token_ids"]
    if sampling_params.get("skip_special_tokens"):
        body["skip_special_tokens"] = True
    if sampling_params.get("seed") is not None:
        body["seed"] = sampling_params["seed"]
    return body


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using vLLM OpenAI-compatible HTTP API on the router host/port."""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    base = f"http://{args.router_ip}:{args.router_port}"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)
    base_prompt_ids = _base_dataset_prompt_ids(sample, state.tokenizer, state.processor)

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

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None

    if not sample.tokens:
        sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (SGLang Model Gateway)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hashing":
            headers = {"X-SMG-Routing-Key": sample.session_id}

    if images:
        # Disaggregated MM flow: render (preprocess) then tokens-only generate — see vLLM docs
        # ``examples/online_serving/disaggregated_serving`` (``/v1/chat/completions/render`` +
        # ``/inference/v1/generate``).
        content: list[dict[str, Any]] = [{"type": "text", "text": sample.prompt}]
        for image in images:
            data_url = encode_image_for_rollout_engine(image)
            content.append({"type": "image_url", "image_url": {"url": data_url}})
        render_payload = {
            "model": args.hf_checkpoint,
            "messages": [{"role": "user", "content": content}],
        }
        render_url = f"{base}/v1/chat/completions/render"
        with trace_span(sample, "vllm_mm_render", attrs={"model": args.hf_checkpoint}):
            render_data = await post(render_url, render_payload, headers=headers)
        generate_body = _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)
        generate_body["sampling_params"] = _build_inference_sampling_params(params)
        gen_url = f"{base}/inference/v1/generate"
        with trace_span(sample, "vllm_mm_generate", attrs={"max_tokens": params["max_new_tokens"]}):
            output = await post(gen_url, generate_body, headers=headers)
        choice = output["choices"][0]
        skip_sp = params.get("skip_special_tokens")
        skip_decode = True if skip_sp is None else bool(skip_sp)
        out_ids = choice.get("token_ids") or []
        text = (
            state.tokenizer.decode(out_ids, skip_special_tokens=skip_decode)
            if isinstance(out_ids, list) and out_ids
            else ""
        )
        usage = output.get("usage")
        meta = _openai_meta_from_completion_choice(args, choice, usage)
        new_response_tokens, new_response_log_probs = _vllm_engine_tokens_and_logprobs(
            state.tokenizer, choice, text, is_chat=True
        )
        new_response_tokens, new_response_log_probs = _align_engine_tokens_and_logprobs(
            new_response_tokens, new_response_log_probs
        )
    else:
        url = f"{base}/v1/completions"
        # vLLM OpenAI ``prompt``: JSON string or flat array of integers. On partial continuation, send full
        # ``sample.tokens`` as integer ids (aligned with dev_vllm ``sglang_rollout`` + vLLM backend).
        if len(sample.response) > 0:
            prompt_field = _coerce_flat_int_token_ids(sample.tokens)
        elif isinstance(sample.prompt, str):
            prompt_field = sample.prompt
        else:
            prompt_field = prompt_ids
        payload = _build_completion_payload(args, params, prompt_field)
        with trace_span(sample, "vllm_completions", attrs={"max_new_tokens": params["max_new_tokens"]}):
            output = await post(url, payload, headers=headers)
        choice = output["choices"][0]
        text = choice.get("text") or ""
        usage = output.get("usage")
        meta = _openai_meta_from_completion_choice(args, choice, usage)
        new_response_tokens, new_response_log_probs = _vllm_engine_tokens_and_logprobs(
            state.tokenizer, choice, text, is_chat=False
        )
        new_response_tokens, new_response_log_probs = _align_engine_tokens_and_logprobs(
            new_response_tokens, new_response_log_probs
        )

    if new_response_tokens:
        meta["output_token_logprobs"] = [
            [float(lp), int(tid)] for lp, tid in zip(new_response_log_probs, new_response_tokens, strict=True)
        ]

    # Update sample with tokens directly - avoiding re-tokenization
    sample.tokens = sample.tokens + new_response_tokens
    sample.response_length += len(new_response_tokens)
    sample.response += text

    # When partial rollout and masking off policy is enabled, update the loss mask
    if sample.loss_mask is not None:
        assert args.partial_rollout and args.mask_offpolicy_in_partial_rollout
        sample.loss_mask += [1] * len(new_response_tokens)

    if sample.rollout_log_probs is None:
        sample.rollout_log_probs = []
    sample.rollout_log_probs += new_response_log_probs

    _apply_vllm_routed_experts(args, sample, output, choice)

    sample.update_from_meta_info(args, meta)
    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
    if isinstance(sample, list):
        return await asyncio.gather(
            *[generate_and_rm(args, s, sampling_params, evaluation=evaluation) for s in sample]
        )

    # mask previous off-policy generation for partial rollout
    if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
        sample.loss_mask = [0] * sample.response_length

    # For samples with existing response, check if they're complete
    if sample.status == Sample.Status.COMPLETED or sample.status == Sample.Status.TRUNCATED:
        assert sample.response is not None
        if not args.group_rm:
            assert sample.reward is not None
        return sample

    state = GenerateState(args)

    # generate
    async with state.semaphore:
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        with state.dp_rank_context() as _:
            # Check sample.generate_function_path for per-sample custom_generate_function_path (e.g., from eval dataset config)
            custom_func_path = getattr(sample, "generate_function_path", None) or args.custom_generate_function_path

            if custom_func_path is not None:
                custom_generate_func = load_function(custom_func_path)
                # if signature has evaluation, pass evaluation
                if "evaluation" in inspect.signature(custom_generate_func).parameters:
                    sample = await custom_generate_func(args, sample, sampling_params, evaluation=evaluation)
                else:
                    sample = await custom_generate_func(args, sample, sampling_params)
            else:
                sample = await generate(args, sample, sampling_params)

    # for the rm that need the whole group, we will not do the rm here
    if args.group_rm:
        return sample

    if isinstance(sample, list):
        samples = sample
        if any(sample.status == Sample.Status.ABORTED for sample in samples):
            return samples

        samples_need_reward = [sample for sample in samples if sample.reward is None]
        with trace_span(samples_need_reward, "reward_model"):
            rewards = await batched_async_rm(args, samples_need_reward)
        for sample, reward in zip(samples_need_reward, rewards, strict=False):
            sample.reward = reward
        return samples
    else:
        if sample.status == Sample.Status.ABORTED:
            return sample
        # Some custom generate paths may have already filled the reward.
        if sample.reward is None:
            with trace_span(sample, "reward_model"):
                sample.reward = await async_rm(args, sample)

    return sample


@trace_function(
    "generate_and_rm_group",
    target="group",
    attrs_getter=lambda args, group, sampling_params, evaluation=False: {"group_size": len(group)},
)
async def generate_and_rm_group(
    args: Namespace, group: list[Sample], sampling_params: dict[str, Any], evaluation: bool = False
) -> list[Sample]:
    state = GenerateState(args)

    if state.aborted:
        return group

    # Generate a unique session_id for each sample in the group
    for sample in group:
        if sample.session_id is None:
            sample.session_id = str(uuid.uuid4())

    tasks = []
    for idx, sample in enumerate(group):
        current_sampling_params = sampling_params.copy()
        if getattr(args, "sglang_enable_deterministic_inference", False):
            seed = state.group_sampling_seeds[idx]
            current_sampling_params["seed"] = seed
        tasks.append(
            asyncio.create_task(generate_and_rm(args, sample, current_sampling_params, evaluation=evaluation))
        )

    group = await asyncio.gather(*tasks)

    # for the rm that need the whole group, we will do the rm here
    if not state.aborted and args.group_rm:
        with trace_span(group, "group_reward_model"):
            rewards = await batched_async_rm(args, group)
        for sample, reward in zip(group, rewards, strict=False):
            sample.reward = reward

    return group


async def abort(args: Namespace, rollout_id: int) -> list[list[Sample]]:
    aborted_samples: list[list[Sample]] = []

    state = GenerateState(args)
    assert not state.aborted
    state.aborted = True

    urls: list[str] = []
    paused_workers = False
    if state.pendings:
        urls = await _router_worker_urls(args)
        logger.info("vLLM rollout abort (pause) for workers: %s", urls)
        pause_tasks = [post(f"{url.rstrip('/')}/pause?mode=abort", {}, max_retries=3) for url in urls]
        pause_results = await asyncio.gather(*pause_tasks, return_exceptions=True)
        for url, result in zip(urls, pause_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Failed to pause/abort worker at %s: %s", url, result)
        paused_workers = True

    # make sure all the pending tasks are finished
    count = 0
    while state.pendings:
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)

        if not args.partial_rollout:
            continue

        # for partial rollout, collect the partial samples into the data buffer
        for task in done:
            group = task.result()
            for sample in group:
                if sample.response and "start_rollout_id" not in sample.metadata:
                    sample.metadata["start_rollout_id"] = rollout_id
            aborted_samples.append(group)
            count += len(group)

    if args.partial_rollout:
        logger.info(f"Collected {count} partial samples into the data buffer")

    state.pendings = set()
    if paused_workers:
        await _resume_vllm_workers(urls)
    return aborted_samples


async def generate_rollout_async(
    args: Namespace, rollout_id: int, data_source: Callable[[int], list[list[Sample]]]
) -> tuple[RolloutFnTrainOutput, list[list[Sample]]]:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to fetch

    Returns:
        tuple[RolloutFnTrainOutput, list[list[Sample]]]:
            - data: a list of groups of samples generated by the rollout, length equals `rollout_batch_size`
            - aborted_samples: any partial groups collected during abort when partial_rollout is enabled
    """
    assert args.rollout_global_dataset

    state = GenerateState(args)

    # instantiate data filters
    dynamic_filter = (
        load_function(args.dynamic_sampling_filter_path) if args.dynamic_sampling_filter_path is not None else None
    )

    metric_gatherer = MetricGatherer()

    # target_data_size is the total number of valid samples to get
    target_data_size = args.rollout_batch_size

    data = []
    all_data = []
    do_print = True
    pbar = tqdm(total=target_data_size * args.n_samples_per_prompt, desc="Rollout generation")
    while len(data) < target_data_size:
        while state.remaining_batch_size < target_data_size:
            # get samples from the buffer and submit the generation requests.
            samples = data_source(args.over_sampling_batch_size)
            state.submit_generate_tasks(samples)

        # wait for the generation to finish
        done, state.pendings = await asyncio.wait(state.pendings, return_when=asyncio.FIRST_COMPLETED)
        for task in done:
            group: list[Sample] = task.result()

            if do_print:
                sample = group[0][0] if isinstance(group[0], list) else group[0]
                logger.info(
                    f"First rollout sample: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
                )
                do_print = False

            assert len(group) == args.n_samples_per_prompt
            all_data.append(group)
            dynamic_filter_output = call_dynamic_filter(dynamic_filter, args, group)
            if not dynamic_filter_output.keep:
                metric_gatherer.on_dynamic_filter_drop(reason=dynamic_filter_output.reason)
                state.remaining_batch_size -= 1
                continue

            # add the samples to the data
            # NOTE: here we have not stored all the unused samples back to the data buffer.
            if len(data) < target_data_size:
                data.append(group)
                pbar.update(args.n_samples_per_prompt)

    pbar.close()
    sample = data[-1][0][0] if isinstance(data[-1][0], list) else data[-1][0]
    logger.info(
        f"Finish rollout: {[str(sample.prompt) + sample.response]}, label: {str(sample.label)[:100]}, reward: {sample.reward}",
    )

    # there are still some unfinished requests, abort them
    aborted_samples = await abort(args, rollout_id)

    assert len(data) == args.rollout_batch_size, f"Got {len(data)} samples, expected {args.rollout_batch_size}"
    data = sorted(data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index)
    all_samples = sorted(
        all_data, key=lambda group: group[0][0].index if isinstance(group[0], list) else group[0].index
    )

    # reset the global state to prevent effects on the next rollout or eval.
    state.reset()
    if args.rollout_sample_filter_path is not None:
        filter_func = load_function(args.rollout_sample_filter_path)
        filter_func(args, data)

    # There can be circumstances where users want to process all samples including filtered ones.
    if args.rollout_all_samples_process_path is not None:
        process_func = load_function(args.rollout_all_samples_process_path)
        process_func(args, all_samples, data_source)

    return RolloutFnTrainOutput(samples=data, metrics=metric_gatherer.collect()), aborted_samples


EVAL_PROMPT_DATASET = {}


async def eval_rollout(args: Namespace, rollout_id: int) -> tuple[dict[str, dict[str, list[Any]]], list[list[Sample]]]:
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    coros = []
    for dataset_cfg in getattr(args, "eval_datasets", []) or []:
        coros.append(eval_rollout_single_dataset(args, rollout_id, dataset_cfg))
    results_list = await asyncio.gather(*coros)
    results = {}
    for r in results_list:
        results.update(r)
    return RolloutFnEvalOutput(data=results), []


async def eval_rollout_single_dataset(
    args: Namespace, rollout_id: int, dataset_cfg: EvalDatasetConfig
) -> dict[str, dict[str, list[Any]]]:
    """An example to implement the eval_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        dataset_cfg: configuration of the dataset
    """
    assert not args.group_rm, "Group RM is not supported for eval rollout"

    global EVAL_PROMPT_DATASET

    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    if cache_key not in EVAL_PROMPT_DATASET:
        tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        processor = load_processor(args.hf_checkpoint, trust_remote_code=True)
        EVAL_PROMPT_DATASET[cache_key] = Dataset(
            path=dataset_cfg.path,
            tokenizer=tokenizer,
            processor=processor,
            max_length=args.eval_max_prompt_len,
            prompt_key=dataset_cfg.input_key,
            label_key=dataset_cfg.label_key,
            multimodal_keys=args.multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=args.apply_chat_template,
            apply_chat_template_kwargs=args.apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=args.rollout_skip_special_tokens,
        no_stop_trim=True,
        spaces_between_special_tokens=False,
    )

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "sglang_enable_deterministic_inference", False):
                sampling_params = base_sampling_params.copy()
                sampling_params["seed"] = args.rollout_seed + j
            tasks.append(
                asyncio.create_task(
                    generate_and_rm(
                        args,
                        sample,
                        sampling_params=sampling_params,
                        evaluation=True,
                    )
                )
            )

    data = []
    do_print = True
    pbar = tqdm(total=len(tasks), desc=f"Eval {dataset_cfg.name}", disable=not do_print)
    for coro in asyncio.as_completed(tasks):
        sample = await coro
        if do_print:
            logged_sample = sample[0] if isinstance(sample, list) else sample
            logger.info(
                "eval_rollout_single_dataset example data: "
                f"{[str(logged_sample.prompt) + logged_sample.response]} "
                f"reward={logged_sample.reward}"
            )
            do_print = False
        if isinstance(sample, list):
            data.extend(sample)
        else:
            data.append(sample)
        pbar.update(1)
    pbar.close()

    data.sort(key=lambda sample: sample.index)

    reward_key = args.eval_reward_key or args.reward_key
    return {
        dataset_cfg.name: {
            "rewards": [sample.reward if not reward_key else sample.reward[reward_key] for sample in data],
            "truncated": [sample.status == Sample.Status.TRUNCATED for sample in data],
            "samples": data,
        }
    }


def generate_rollout(
    args: Namespace, rollout_id: int, data_source: Any, evaluation: bool = False
) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
    """An example to implement the generate_rollout function for an rule based rm rollout generation.

    Args:
        args: the whole args
        rollout_id: int, the id of the rollout, used for deterministic data generation
        data_source: the data source to get and store samples
        evaluation: bool, whether the rollout is for evaluation or not

    Returns:
        RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
    """
    assert args.rollout_global_dataset
    if evaluation:
        output, _ = run(eval_rollout(args, rollout_id))
        return output

    output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
    if aborted_samples:
        data_source.add_samples(aborted_samples)
    return output