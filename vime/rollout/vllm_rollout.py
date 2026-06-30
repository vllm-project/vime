import asyncio
import base64
import copy
import inspect
import io
import json
import logging
import uuid
from argparse import Namespace
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any

import numpy as np
import vllm_router  # noqa: F401 — ensures vllm-router is importable on startup
from tqdm import tqdm

from vime.rollout.base_types import RolloutFnEvalOutput, RolloutFnTrainOutput
from vime.rollout.filter_hub.base_types import MetricGatherer, call_dynamic_filter
from vime.utils.async_utils import run
from vime.utils.data import Dataset
from vime.utils.eval_config import EvalDatasetConfig
from vime.utils.http_utils import get, get_rollout_num_engines, post
from vime.utils.misc import SingletonMeta, load_function
from vime.utils.processing_utils import (
    build_processor_kwargs,
    encode_image_for_rollout_engine,
    load_processor,
    load_tokenizer,
)
from vime.utils.trace_utils import build_vllm_meta_trace_attrs, trace_function, trace_span
from vime.utils.types import Sample

from .rm_hub import async_rm, batched_async_rm

__all__ = ["generate_rollout", "get_model_url"]

logger = logging.getLogger(__name__)

_PROCESSOR_PROMPT_KEYS = {"input_ids", "attention_mask"}


def _coerce_flat_int_token_ids(ids: Any) -> list[int]:
    """Flatten tokenizer/processor output into ``list[int]`` for vLLM ``/inference/v1/generate``."""
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


def get_model_url(args: Namespace, model_name: str, endpoint: str = "/inference/v1/generate") -> str:
    """Return the router URL for a named model.

    Use this in custom rollout functions to route requests to a specific
    model when multiple models are deployed via ``--vllm-config``::

        url = get_model_url(args, "ref", "/inference/v1/generate")
        resp = await post(url, json=payload)

    Falls back to the default router if *model_name* is not found or
    ``vllm_model_routers`` is not set.
    """
    routers = getattr(args, "vllm_model_routers", None)
    if routers and model_name in routers:
        ip, port = routers[model_name]
        return f"http://{ip}:{port}{endpoint}"
    return f"http://{args.vllm_router_ip}:{args.vllm_router_port}{endpoint}"


class GenerateState(metaclass=SingletonMeta):
    """
    The global state for the generation process.
    """

    def __init__(self, args: Namespace) -> None:
        # persistent state for the generation process
        self.args = args
        self.tokenizer = load_tokenizer(args.hf_checkpoint, trust_remote_code=True)
        self.processor = load_processor(args.hf_checkpoint, trust_remote_code=True)

        self.semaphore = asyncio.Semaphore(args.vllm_server_concurrency * get_rollout_num_engines(args))
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
        if args.rollout_top_p != 1.0:
            self.sampling_params["custom_params"] = {"return_top_p_token_ids": True}

        if getattr(args, "vllm_enable_deterministic_inference", False):
            sampling_seed_base = args.rollout_seed
            self.group_sampling_seeds = [sampling_seed_base + i for i in range(args.n_samples_per_prompt)]

        # dp rank balancing
        self.dp_counts = [0] * (args.vllm_dp_size or 1)
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
    """Map rollout ``sampling_params`` to vLLM ``/inference/v1/generate`` body."""
    sp: dict[str, Any] = {
        "max_tokens": sampling_params["max_new_tokens"],
        "temperature": sampling_params["temperature"],
        "top_p": sampling_params["top_p"],
        "logprobs": 1,
    }
    tk = sampling_params.get("top_k")
    if tk is not None and (tk > 0 or tk == -1):
        sp["top_k"] = tk
    if sampling_params.get("stop"):
        sp["stop"] = sampling_params["stop"]
        if sampling_params.get("no_stop_trim"):
            sp["include_stop_str_in_output"] = True
    if sampling_params.get("stop_token_ids"):
        sp["stop_token_ids"] = sampling_params["stop_token_ids"]
    if sampling_params.get("seed") is not None:
        sp["seed"] = sampling_params["seed"]
    if sampling_params.get("skip_special_tokens") is not None:
        sp["skip_special_tokens"] = bool(sampling_params["skip_special_tokens"])
    return sp


def _mm_render_response_to_generate_body(render_data: Any, model: str) -> dict[str, Any]:
    """Turn ``/v1/chat/completions/render`` JSON into a ``/inference/v1/generate`` request body."""
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
        "chat/render: unexpected JSON shape; expected a dict with token_ids or [conversation, engine_prompts] list"
    )


def _find_token_subsequence(haystack: list[int], needle: list[int], start: int = 0) -> int:
    if not needle:
        return max(start, 0)
    end = len(haystack) - len(needle) + 1
    for i in range(max(start, 0), max(end, 0)):
        if haystack[i : i + len(needle)] == needle:
            return i
    return -1


def _align_mm_feature_placeholders_to_tokens(generate_body: dict[str, Any], token_ids: list[int]) -> None:
    """Point vLLM-rendered multimodal features at vime's canonical prompt ids."""
    features = generate_body.get("features")
    if not isinstance(features, dict):
        return
    placeholders = features.get("mm_placeholders")
    if not isinstance(placeholders, dict):
        raise ValueError("vLLM multimodal features missing mm_placeholders")

    render_token_ids = _coerce_flat_int_token_ids(generate_body.get("token_ids"))
    if not render_token_ids:
        raise ValueError("Cannot align vLLM multimodal placeholders: render response missing token_ids")

    ordered_entries: list[tuple[int, str, dict[str, Any]]] = []
    for modality, entries in placeholders.items():
        if not entries:
            continue
        if not isinstance(entries, list):
            raise ValueError(f"Cannot align vLLM {modality} placeholders: entries is {type(entries)!r}, expected list")
        for entry in entries:
            if not isinstance(entry, dict):
                raise ValueError(f"Cannot align vLLM {modality} placeholder: entry is {type(entry)!r}, expected dict")
            offset = int(entry.get("offset", -1))
            length = int(entry.get("length", -1))
            if offset < 0 or length <= 0 or offset + length > len(render_token_ids):
                raise ValueError(
                    f"Cannot align vLLM {modality} placeholder: invalid render range "
                    f"offset={offset}, length={length}, render_len={len(render_token_ids)}"
                )
            ordered_entries.append((offset, str(modality), entry))

    search_start = 0
    for render_offset, modality, entry in sorted(ordered_entries, key=lambda item: item[0]):
        length = int(entry["length"])
        placeholder_tokens = render_token_ids[render_offset : render_offset + length]
        offset = _find_token_subsequence(token_ids, placeholder_tokens, search_start)
        if offset < 0:
            raise ValueError(
                f"Cannot align vLLM {modality} placeholder from render offset={render_offset}, length={length}: "
                "placeholder token slice not found in canonical token_ids"
            )
        entry["offset"] = offset
        entry["length"] = len(placeholder_tokens)
        search_start = offset + len(placeholder_tokens)


async def generate(args: Namespace, sample: Sample, sampling_params: dict[str, Any]) -> Sample:
    """Generate using vLLM router with token-based workflow"""
    if args.ci_test:
        assert isinstance(sample.prompt, str)

    state = GenerateState(args)
    base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"

    assert (
        sample.status == Sample.Status.PENDING or sample.status == Sample.Status.ABORTED
    ), f"Sample status is {sample.status}"

    prompt_ids = _prepare_prompt_ids(sample, state.tokenizer, state.processor)

    assert (
        sampling_params["max_new_tokens"] >= 0
    ), f"max_new_tokens: {sampling_params['max_new_tokens']} should not be less than 0"
    if sampling_params["max_new_tokens"] == 0:
        sample.status = Sample.Status.TRUNCATED
        return sample

    inference_sampling_params = _build_inference_sampling_params(sampling_params)

    images = sample.multimodal_inputs.get("images") if sample.multimodal_inputs else None

    if not sample.tokens:
        sample.tokens = prompt_ids

    # Use session_id for consistent hashing routing (vLLM router)
    headers = None
    if sample.session_id:
        if getattr(args, "router_policy", None) == "consistent_hash":
            headers = {"x-session-id": sample.session_id}

    # Prepare payload for vLLM server
    if images:
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
        canonical_token_ids = _coerce_flat_int_token_ids(sample.tokens)
        if canonical_token_ids:
            _align_mm_feature_placeholders_to_tokens(generate_body, canonical_token_ids)
            generate_body["token_ids"] = canonical_token_ids
        generate_body["sampling_params"] = inference_sampling_params
        gen_url = f"{base}/inference/v1/generate"
        with trace_span(sample, "vllm_mm_generate", attrs={"max_tokens": sampling_params["max_new_tokens"]}):
            output = await post(gen_url, generate_body, headers=headers)
    else:
        url = f"{base}/inference/v1/generate"
        payload = {
            "model": args.hf_checkpoint,
            "token_ids": prompt_ids,
            "sampling_params": inference_sampling_params,
        }

        with trace_span(sample, "vllm_generate", attrs={"max_new_tokens": sampling_params["max_new_tokens"]}) as span:
            output = await post(url, payload, headers=headers)
            if hasattr(span, "update"):
                span.update(build_vllm_meta_trace_attrs(output))

    choice = output["choices"][0]

    # Parse token_ids and logprobs from vLLM response
    new_response_tokens = choice.get("token_ids") or []
    new_response_log_probs: list[float] = []
    lp = choice.get("logprobs")
    if isinstance(lp, dict):
        content_items = lp.get("content") or []
        new_response_log_probs = [
            float(item.get("logprob", 0.0)) if isinstance(item, dict) else 0.0 for item in content_items
        ]
    if not new_response_log_probs:
        new_response_log_probs = [0.0] * len(new_response_tokens)

    # Decode text from token_ids
    skip_sp = sampling_params.get("skip_special_tokens")
    skip_decode = True if skip_sp is None else bool(skip_sp)
    text = state.tokenizer.decode(new_response_tokens, skip_special_tokens=skip_decode) if new_response_tokens else ""

    # Build meta_info from the vLLM `choices` response format.
    fr = choice.get("finish_reason") or "stop"
    if isinstance(fr, dict):
        finish = fr
    elif fr == "length":
        finish = {"type": "length"}
    elif fr in ("abort", "cancelled"):
        finish = {"type": "abort"}
    else:
        finish = {"type": "stop"}
    meta: dict[str, Any] = {"finish_reason": finish}
    usage = output.get("usage")
    if usage:
        meta["prompt_tokens"] = usage.get("prompt_tokens", 0)
        meta["completion_tokens"] = usage.get("completion_tokens", 0)

    # MoE routing replay: vLLM ships routed_experts as a base64 .npy blob on the choice;
    # decode here and route through meta_info. #183: guard on value (null when replay off).
    routed_experts = choice.get("routed_experts")
    if routed_experts is not None:
        raw = base64.b64decode(routed_experts.encode("ascii"), validate=True)
        meta["routed_experts"] = np.load(io.BytesIO(raw), allow_pickle=False)

    sample.append_response_tokens(
        args,
        tokens=new_response_tokens,
        log_probs=new_response_log_probs,
        trainable=True,
        meta_info=meta,
        text=text,
    )

    return sample


@trace_function("generate_and_rm", target="sample")
async def generate_and_rm(
    args: Namespace,
    sample: Sample | list[Sample],
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample | list[Sample]:
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
) -> list[Sample] | list[list[Sample]]:
    # ``generate_and_rm`` may return either a ``Sample`` or a ``list[Sample]``
    # depending on whether the ``--custom-generate-function-path`` callable
    # emits one trainable sample or several (e.g. multi-turn agent rollouts
    # that fan out into multiple prefix-chained samples). The asyncio.gather
    # below preserves whichever shape each task produced, so the group is
    # ``list[Sample]`` for plain rollouts and ``list[list[Sample]]`` for
    # the fan-out case.
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
        if getattr(args, "vllm_enable_deterministic_inference", False):
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
        base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
        try:
            response = await get(f"{base}/workers")
            urls = [worker["url"] for worker in response["workers"]]
        except Exception:
            response = await get(f"{base}/list_workers")
            urls = list(response["urls"])

        logger.info(f"Abort request for {urls}")
        pause_tasks = [post(f"{url.rstrip('/')}/pause?mode=abort", {}, max_retries=3) for url in urls]
        pause_results = await asyncio.gather(*pause_tasks, return_exceptions=True)
        for url, result in zip(urls, pause_results, strict=False):
            if isinstance(result, Exception):
                logger.warning(f"Failed to abort worker at {url}: {result}")
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
        logger.info("rollout: resuming workers after abort drain: %s", urls)
        resume_tasks = [post(f"{url.rstrip('/')}/resume", {}, max_retries=3) for url in urls]
        resume_results = await asyncio.gather(*resume_tasks, return_exceptions=True)
        for url, result in zip(urls, resume_results, strict=False):
            if isinstance(result, Exception):
                logger.warning("Failed to resume worker at %s: %s", url, result)

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

    eval_multimodal_keys = (
        dataset_cfg.multimodal_keys if dataset_cfg.multimodal_keys is not None else args.multimodal_keys
    )
    eval_apply_chat_template = (
        dataset_cfg.apply_chat_template if dataset_cfg.apply_chat_template is not None else args.apply_chat_template
    )
    eval_apply_chat_template_kwargs = (
        dataset_cfg.apply_chat_template_kwargs
        if dataset_cfg.apply_chat_template_kwargs is not None
        else args.apply_chat_template_kwargs
    )

    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        eval_apply_chat_template,
        json.dumps(eval_multimodal_keys, sort_keys=True) if eval_multimodal_keys is not None else None,
        (
            json.dumps(eval_apply_chat_template_kwargs, sort_keys=True)
            if eval_apply_chat_template_kwargs is not None
            else None
        ),
    )
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
            multimodal_keys=eval_multimodal_keys,
            metadata_key=dataset_cfg.metadata_key,
            tool_key=dataset_cfg.tool_key,
            apply_chat_template=eval_apply_chat_template,
            apply_chat_template_kwargs=eval_apply_chat_template_kwargs,
        )
    dataset = EVAL_PROMPT_DATASET[cache_key]

    base_sampling_params = dict(
        temperature=dataset_cfg.temperature,
        top_p=dataset_cfg.top_p,
        top_k=dataset_cfg.top_k,
        max_new_tokens=dataset_cfg.max_response_len,
        stop=args.rollout_stop,
        stop_token_ids=args.rollout_stop_token_ids,
        skip_special_tokens=(
            dataset_cfg.skip_special_tokens
            if dataset_cfg.skip_special_tokens is not None
            else args.rollout_skip_special_tokens
        ),
        no_stop_trim=dataset_cfg.no_stop_trim if dataset_cfg.no_stop_trim is not None else True,
        spaces_between_special_tokens=False,
    )
    if dataset_cfg.repetition_penalty is not None:
        base_sampling_params["repetition_penalty"] = dataset_cfg.repetition_penalty

    tasks = []
    # do multiple samples for eval prompts
    sample_index = 0
    for _i, prompt_sample in enumerate(dataset.samples):
        for j in range(dataset_cfg.n_samples_per_eval_prompt):
            # use the same prompt for multiple samples
            sample = copy.deepcopy(prompt_sample)
            sample.index = sample_index
            sample_index += 1
            sample.session_id = str(uuid.uuid4())
            sample.metadata = dataset_cfg.inject_metadata(getattr(sample, "metadata", None))
            sample.custom_rm_path = dataset_cfg.custom_rm_path
            sample.generate_function_path = getattr(dataset_cfg, "custom_generate_function_path", None)
            sampling_params = base_sampling_params
            if getattr(args, "vllm_enable_deterministic_inference", False):
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
