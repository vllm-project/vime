"""
MemAgent rollout for vime (migrated from slime-agentic).

Chunk-by-chunk memory update pipeline:
    for chunk in split(context):
        memory = LLM(problem, memory, chunk)
    answer = LLM(problem, memory)  # final turn with \\boxed{}
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from examples.mem_agent.rollout_client import MemAgentRolloutClient

from vime.rollout.vllm_rollout import GenerateState
from vime.utils.types import Sample

CHUNK_TOKENS = int(os.environ.get("MEM_CHUNK_TOKENS", "2048"))
MAX_MEMORY_TOKENS = int(os.environ.get("MEM_MAX_MEMORY", "1024"))
MAX_FINAL_TOKENS = int(os.environ.get("MEM_MAX_FINAL", "256"))
MAX_CHUNKS = int(os.environ.get("MEM_MAX_CHUNKS", "512"))

_MEMORY_TEMPLATE = """You are presented with a problem, a section of an article that may contain the answer to the problem, and a previous memory. Please read the provided section carefully and update the memory with the new information that helps to answer the problem. Be sure to retain all relevant details from the previous memory while adding any new, useful information.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

<section>
{chunk}
</section>

Updated memory:
"""

_FINAL_TEMPLATE = """You are presented with a problem and a previous memory. Please answer the problem based on the previous memory and put the answer in \\boxed{{}}.

<problem> 
{prompt}
</problem>

<memory>
{memory}
</memory>

Your answer:
"""

_NO_MEMORY = "No previous memory"
_STOP_TOKEN_STRINGS = ["", "<|endoftext|>"]


def _strip_stop_tokens(text: str) -> str:
    for tok in _STOP_TOKEN_STRINGS:
        text = text.replace(tok, "")
    return text.strip()


async def generate(
    args: Any,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    state = GenerateState(args)
    client = MemAgentRolloutClient(args, state.tokenizer, sampling_params)

    if not isinstance(sample.metadata, dict):
        sample.metadata = {}

    question = sample.prompt if isinstance(sample.prompt, str) else sample.prompt[-1]["content"]
    context = sample.metadata.get("context", "")
    if not context:
        sample.status = Sample.Status.ABORTED
        sample.rollout_log_probs = []
        return sample

    try:
        context_ids = state.tokenizer.encode(context, add_special_tokens=False)
        chunks_ids = [context_ids[i : i + CHUNK_TOKENS] for i in range(0, len(context_ids), CHUNK_TOKENS)][:MAX_CHUNKS]

        turns: list[dict] = []
        memory = _NO_MEMORY
        mem_params = {**sampling_params, "max_new_tokens": MAX_MEMORY_TOKENS}

        for chunk_ids in chunks_ids:
            chunk_text = state.tokenizer.decode(chunk_ids, skip_special_tokens=True)
            messages = [
                {
                    "role": "user",
                    "content": _MEMORY_TEMPLATE.format(prompt=question, memory=memory, chunk=chunk_text),
                }
            ]
            out = await client.generate(messages, sampling_params=mem_params)
            memory = _strip_stop_tokens(out.response) or memory
            turns.append(
                {
                    "tokens": out.prompt_token_ids + out.token_ids,
                    "response_length": len(out.token_ids),
                    "loss_mask": [1] * len(out.token_ids),
                    "rollout_log_probs": out.log_probs,
                }
            )

        final_messages = [
            {
                "role": "user",
                "content": _FINAL_TEMPLATE.format(prompt=question, memory=memory),
            }
        ]
        final_out = await client.generate(
            final_messages,
            sampling_params={**sampling_params, "max_new_tokens": MAX_FINAL_TOKENS},
        )
        turns.append(
            {
                "tokens": final_out.prompt_token_ids + final_out.token_ids,
                "response_length": len(final_out.token_ids),
                "loss_mask": [1] * len(final_out.token_ids),
                "rollout_log_probs": final_out.log_probs,
            }
        )

        sample.metadata["final_output"] = _strip_stop_tokens(final_out.response)
        sample.train_metadata = {"turns": turns}

        if os.environ.get("MEM_DEBUG"):
            print(
                f"[MemAgent] {len(chunks_ids)} chunks -> {len(turns)} turns, "
                f"response_lengths={[t['response_length'] for t in turns]}"
            )

        first = turns[0]
        first_prompt_len = len(first["tokens"]) - first["response_length"]
        prompt_token_ids = first["tokens"][:first_prompt_len]

        cat_token_ids: list[int] = []
        cat_loss_mask: list[int] = []
        cat_log_probs: list[float] = []
        for t in turns:
            p_len = len(t["tokens"]) - t["response_length"]
            cat_token_ids += t["tokens"]
            cat_loss_mask += [0] * p_len + t["loss_mask"]
            cat_log_probs += [0.0] * p_len + t["rollout_log_probs"]

        sample.prompt = final_messages[0]["content"]
        sample.response = _strip_stop_tokens(final_out.response)
        sample.tokens = prompt_token_ids + cat_token_ids
        sample.response_length = len(cat_token_ids)
        sample.loss_mask = cat_loss_mask
        sample.rollout_log_probs = cat_log_probs
        sample.status = Sample.Status.TRUNCATED if final_out.finish_reason == "length" else Sample.Status.COMPLETED

    except Exception:
        traceback.print_exc()
        sample.response = ""
        sample.rollout_log_probs = []
        sample.status = Sample.Status.FAILED

    return sample


def _last_boxed_only_string(string: str) -> str | None:
    if "\\boxed " in string:
        return "\\boxed " + string.split("\\boxed ")[-1].split("$")[0]
    idx = string.rfind("\\boxed")
    if idx < 0:
        idx = string.rfind("\\fbox")
        if idx < 0:
            return None
    i, right_brace_idx, num_left_braces_open = idx, None, 0
    while i < len(string):
        if string[i] == "{":
            num_left_braces_open += 1
        if string[i] == "}":
            num_left_braces_open -= 1
            if num_left_braces_open == 0:
                right_brace_idx = i
                break
        i += 1
    return string[idx : right_brace_idx + 1] if right_brace_idx is not None else None


def _remove_boxed(s: str) -> str:
    if s.startswith("\\boxed "):
        return s[len("\\boxed ") :]
    left = "\\boxed{"
    assert s.startswith(left) and s.endswith("}")
    return s[len(left) : -1]


def _extract_boxed(text: str) -> str:
    s = _last_boxed_only_string(text)
    if s is None:
        return ""
    try:
        return _remove_boxed(s).strip()
    except Exception:
        return ""


def _strip_string(string: str) -> str:
    string = string.replace("\n", "")
    string = string.replace("\\!", "")
    string = string.replace("\\\\", "\\")
    string = string.replace("tfrac", "frac")
    string = string.replace("dfrac", "frac")
    string = string.replace("\\left", "")
    string = string.replace("\\right", "")
    string = string.replace("^{\\circ}", "")
    string = string.replace("^\\circ", "")
    string = string.replace("\\$", "")
    string = string.replace("\\%", "")
    string = string.replace(" .", " 0.")
    string = string.replace("{.", "{0.")
    if len(string) == 0:
        return string
    return string.replace(" ", "")


def _is_equiv(str1: str, str2: str) -> bool:
    if str1 is None and str2 is None:
        return True
    if str1 is None or str2 is None:
        return False
    try:
        return _strip_string(str1) == _strip_string(str2)
    except Exception:
        return str1 == str2


async def reward_func(args: Any, sample: Any, **kwargs) -> dict:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    final_output = metadata.get("final_output", "") or sample.response or ""
    ground_truth = metadata.get("ground_truth", [])
    if not ground_truth:
        label = str(sample.label) if sample.label is not None else ""
        ground_truth = [label] if label else []

    solution_str = final_output[-300:]
    pred = _extract_boxed(solution_str) or ""

    score = 0.0
    for gt in ground_truth:
        gt_lower = gt.lower()
        try:
            boxed = _last_boxed_only_string(solution_str)
            if boxed is not None:
                answer = _remove_boxed(boxed)
                if _is_equiv(answer.lower(), gt_lower):
                    score = 1.0
                    break
        except Exception:
            pass

    return {"score": score, "pred": pred, "gt": ground_truth[0] if ground_truth else ""}
