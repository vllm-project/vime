"""Multi-turn agent-loop scaffold for vime, wired via ``--custom-generate-function-path``.

vime already supports multi-turn / agentic rollouts: ``generate_and_rm`` calls a custom generate
function per sample (see ``vime/rollout/vllm_rollout.py``), and the default training loss path
only trains on tokens where ``loss_mask == 1``. This module is the seam to plug in a replacement
for the old Maestro agent loop — NO upstream vime changes required.

Wire it in with::

    --custom-generate-function-path vime_plugins.rollout.multi_turn_agent.generate

and supply your own per-turn agent logic by dotted path::

    --agent-step-path my_pkg.my_agent.agent_step      (or env AGENT_STEP_PATH)

This scaffold owns the vime contract (token accounting, loss masking, status, budgets) so your
``agent_step`` only has to decide what happens each turn. It generates one turn at a time against
the vLLM router's token endpoint (``/inference/v1/generate``), appends the model tokens to the
response (loss_mask = 1), optionally injects an observation/tool result (loss_mask = 0), and loops
until the agent says it is done or a budget is hit.

``agent_step`` contract::

    async def agent_step(args, sample, assistant_text, turn_index) -> dict
        # returns:
        #   {"done": bool,                 # stop the loop?
        #    "observation": str | None,    # text to feed back as the next turn's input (masked)
        #    "reward": float | None}       # optional; if set on the final step, skips async_rm

The default ``agent_step`` ends after one turn (so the scaffold runs as a plain single-turn
rollout until you point ``--agent-step-path`` at your loop).

Return value: a single ``Sample`` (extend to return ``list[Sample]`` for fan-out — give every
sibling the same ``sample.rollout_id`` so loss aggregation averages within the rollout).
"""

from argparse import Namespace
from typing import Any

from vime.rollout.vllm_rollout import GenerateState, _build_inference_sampling_params
from vime.utils.http_utils import post
from vime.utils.misc import load_function
from vime.utils.types import Sample
from vime_plugins.utils.common import cfg as _cfg


async def _default_agent_step(args, sample, assistant_text, turn_index) -> dict[str, Any]:
    """No-op single-turn agent: stop after the first model turn."""
    return {"done": True, "observation": None, "reward": None}


def _resolve_agent_step(args):
    path = _cfg(args, "agent_step_path", "AGENT_STEP_PATH", None)
    return load_function(path) if path else _default_agent_step


def _append_model_tokens(sample: Sample, tokens: list[int], text: str) -> None:
    """Append model-generated tokens to the response region and mark them trainable."""
    sample.tokens = sample.tokens + tokens
    sample.response += text
    sample.response_length += len(tokens)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask += [1] * len(tokens)


def _append_observation_tokens(sample: Sample, tokens: list[int]) -> None:
    """Append injected observation/tool tokens to the response region but mask them out of loss."""
    sample.tokens = sample.tokens + tokens
    sample.response_length += len(tokens)
    if sample.loss_mask is None:
        sample.loss_mask = []
    sample.loss_mask += [0] * len(tokens)


async def generate(
    args: Namespace,
    sample: Sample,
    sampling_params: dict[str, Any],
    evaluation: bool = False,
) -> Sample:
    state = GenerateState(args)
    base = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
    tokenizer = state.tokenizer
    agent_step = _resolve_agent_step(args)

    max_turns = int(_cfg(args, "agent_max_turns", "AGENT_MAX_TURNS", 8))
    max_context_len = args.eval_max_context_len if evaluation else args.rollout_max_context_len

    headers = None
    if sample.session_id and getattr(args, "router_policy", None) == "consistent_hash":
        headers = {"x-session-id": sample.session_id}

    # Initial prompt tokens (the prompt region is not part of response_length / loss_mask).
    if not sample.tokens:
        if isinstance(sample.prompt, list):
            sample.tokens = tokenizer.apply_chat_template(sample.prompt, add_generation_prompt=True, tokenize=True)
        else:
            sample.tokens = tokenizer.encode(sample.prompt, add_special_tokens=False)

    final_reward = None
    for turn_index in range(max_turns):
        if state.aborted:
            sample.status = Sample.Status.ABORTED
            return sample

        remaining = max_context_len - len(sample.tokens)
        if remaining <= 0:
            sample.status = Sample.Status.TRUNCATED
            break
        turn_params = dict(sampling_params)
        turn_params["max_new_tokens"] = min(sampling_params["max_new_tokens"], remaining)

        payload = {
            "model": args.hf_checkpoint,
            "token_ids": sample.tokens,
            "sampling_params": _build_inference_sampling_params(turn_params),
        }
        async with state.semaphore:
            output = await post(f"{base}/inference/v1/generate", payload, headers=headers)

        choice = output["choices"][0]
        new_tokens = choice.get("token_ids") or []
        text = tokenizer.decode(new_tokens, skip_special_tokens=True) if new_tokens else ""
        _append_model_tokens(sample, new_tokens, text)

        finish_reason = choice.get("finish_reason") or "stop"
        step = await agent_step(args, sample, text, turn_index)
        final_reward = step.get("reward", final_reward)

        if step.get("done") or finish_reason == "length" or turn_index == max_turns - 1:
            sample.status = Sample.Status.TRUNCATED if finish_reason == "length" else Sample.Status.COMPLETED
            break

        observation = step.get("observation")
        if observation:
            obs_tokens = tokenizer.encode(observation, add_special_tokens=False)
            _append_observation_tokens(sample, obs_tokens)
    else:
        sample.status = Sample.Status.COMPLETED

    if final_reward is not None:
        sample.reward = final_reward
    return sample
