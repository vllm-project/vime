"""Trainable tau-bench agent for vime vLLM rollout."""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import uuid
from typing import Any

import numpy as np
from openai_tool_adapter import create_openai_adapter
from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
from tau_bench.envs import get_env
from tau_bench.types import Action, RunConfig

from vime.rollout.vllm_rollout import (
    GenerateState,
    _build_inference_sampling_params,
    _coerce_flat_int_token_ids,
    _mm_render_response_to_generate_body,
)
from vime.utils.http_utils import post
from vime.utils.types import Sample

logger = logging.getLogger(__name__)


def patch_tau_user_retries() -> None:
    """Reduce circuit-breaker fatality: more LiteLLM retries with backoff."""
    try:
        import tau_bench.envs.user as user_mod

        user_mod.MAX_RETRIES = int(os.environ.get("TAU_USER_LITELLM_RETRIES", "30"))
        user_mod.RETRY_DELAY_SECONDS = float(os.environ.get("TAU_USER_LITELLM_RETRY_DELAY", "2"))
    except Exception:
        pass


def _parse_choice_tokens_and_logprobs(choice: dict[str, Any]) -> tuple[list[int], list[float]]:
    """Parse token_ids + logprobs from vLLM /inference/v1/generate choice."""
    tids_raw = choice.get("token_ids")
    if not (isinstance(tids_raw, list) and tids_raw and all(isinstance(x, int) for x in tids_raw)):
        return [], []
    tids = [int(x) for x in tids_raw]
    lp = choice.get("logprobs")
    if not isinstance(lp, dict):
        return tids, [0.0] * len(tids)
    content = lp.get("content")
    if isinstance(content, list) and content:
        log_probs = [
            float(content[i].get("logprob", 0.0)) if i < len(content) and isinstance(content[i], dict) else 0.0
            for i in range(len(tids))
        ]
        return tids, log_probs
    return tids, [0.0] * len(tids)


def _maybe_apply_routed_experts(args: Any, sample: Sample, choice: dict[str, Any]) -> None:
    if choice.get("routed_experts") is None:
        return
    raw = base64.b64decode(choice["routed_experts"].encode("ascii"), validate=True)
    arr = np.load(io.BytesIO(raw), allow_pickle=False)
    sample.rollout_routed_experts = np.ascontiguousarray(arr.astype(np.int32, copy=True)).reshape(
        len(sample.tokens) - 1,
        args.num_layers,
        args.moe_router_topk,
    )


class TauBenchEnv:
    def __init__(
        self,
        *,
        tau_config: RunConfig,
        task_index: int | None = None,
        max_turns: int = 30,
    ):
        self.tau_config = tau_config
        self.task_index = task_index
        self.max_turns = max_turns
        self.turn = 0
        self.total_reward = 0.0
        self.info: dict[str, Any] = {}
        self.env = None
        self.openai_adapter = None
        self.successful_tool_calls = 0
        self.total_tool_calls = 0
        self.format_correct_calls = 0

    def reset(self):
        self.turn = 0
        self.total_reward = 0.0
        self.info = {}
        self.successful_tool_calls = 0
        self.total_tool_calls = 0
        self.format_correct_calls = 0

        self.env = get_env(
            env_name=self.tau_config.env,
            user_strategy=self.tau_config.user_strategy,
            user_model=self.tau_config.user_model,
            user_provider=self.tau_config.user_model_provider,
            task_split=self.tau_config.task_split,
            task_index=self.task_index,
        )

        self.openai_adapter = create_openai_adapter(
            tools_info=self.env.tools_info,
            parser_type="qwen25",
        )

        env_reset_res = self.env.reset(task_index=self.task_index) if self.task_index is not None else self.env.reset()
        observation = env_reset_res.observation
        self.info = self._to_dict(env_reset_res.info)

        return {
            "obs_str": observation,
            "role": "user",
            "wiki": self.env.wiki,
            "tools_info": self.env.tools_info,
        }

    def step(self, response_text: str):
        self.turn += 1
        is_final_turn = self.turn >= self.max_turns

        openai_result = self.openai_adapter.parse_response_to_openai_format(response_text)

        if not openai_result["success"]:
            logger.warning(f"Tool parsing failed: {openai_result.get('error')}")
            return (
                {
                    "obs_str": "Failed to parse tool call. Please try again.",
                    "role": "tool",
                },
                is_final_turn,
                {"tool_executed": False, "parse_error": openai_result.get("error")},
            )

        parsed = openai_result["parsed_result"]
        agent_content, calls = parsed["normal_text"], parsed["calls"]

        if calls:
            self.format_correct_calls += 1

        action = self._call_to_action(calls, agent_content)

        is_tool_call = action.name != RESPOND_ACTION_NAME
        if is_tool_call:
            self.total_tool_calls += 1

        try:
            env_response = self.env.step(action)
        except Exception as e:
            logger.warning(f"Environment step failed: {e}")
            return (
                {
                    "obs_str": f"Environment error: {e}",
                    "role": "tool",
                },
                True,
                {"tool_executed": False, "env_error": str(e)},
            )

        self.total_reward = env_response.reward
        self.info.update(self._to_dict(env_response.info))

        obs_lower = env_response.observation.lower() if env_response.observation else ""
        if is_tool_call and not obs_lower.startswith(("error", "failed", "invalid", "not found")):
            self.successful_tool_calls += 1

        if action.name != RESPOND_ACTION_NAME:
            obs_role = "tool"
        else:
            obs_role = "user"
        obs_content = env_response.observation

        done = env_response.done or is_final_turn

        return (
            {
                "obs_str": obs_content,
                "role": obs_role,
                "reward": env_response.reward,
            },
            done,
            {"tool_executed": True, "action": action.name},
        )

    def _call_to_action(self, calls: list[Any], text_response: str) -> Action:
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
        if calls:
            if len(calls) > 1:
                logger.debug("Multiple tool calls identified, only taking first.")
            tool_call = calls[0]
            try:
                params = (
                    json.loads(tool_call["parameters"])
                    if isinstance(tool_call["parameters"], str)
                    else tool_call["parameters"]
                )
                if not isinstance(params, dict):
                    logger.warning(f"{params} does not follow dict structure for action")
                else:
                    action = Action(name=tool_call["name"], kwargs=params)
            except json.JSONDecodeError as e:
                logger.warning(f"Failed to parse parameters as JSON: {e}")
        return action

    def close(self):
        pass

    def format_observation(self, observation: dict) -> dict:
        observation = observation or {}
        content = observation.get("obs_str", "")
        return {
            "role": observation.get("role", "user"),
            "content": content + "\n/no_think" if observation.get("role") == "user" else content,
        }

    @staticmethod
    def _to_dict(info: Any) -> dict:
        if hasattr(info, "model_dump"):
            return info.model_dump()
        if isinstance(info, dict):
            return info
        return {}


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> TauBenchEnv:
    tau_bench_config = getattr(args, "tau_bench_config", None)
    if tau_bench_config is None:
        raise RuntimeError("args.tau_bench_config is missing; generate_with_tau.generate must set it from TAU_CONFIGS")

    task_index = None
    if sample is not None and sample.prompt is not None:
        try:
            task_index = int(sample.prompt)
        except (ValueError, TypeError):
            pass

    max_turns = getattr(args, "max_turns", 30)
    if max_turns is None:
        max_turns = 30

    return TauBenchEnv(
        tau_config=tau_bench_config,
        task_index=task_index,
        max_turns=max_turns,
    )


def compute_process_reward(env: TauBenchEnv, base_reward: float) -> float:
    reward = 0.0
    if base_reward > 0:
        reward += 1.0
    if env.successful_tool_calls > 0:
        reward += 0.1 * env.successful_tool_calls
    if env.format_correct_calls > 0:
        reward += 0.05 * env.format_correct_calls
    reward = min(reward, 1.5)
    return reward


def _build_tools_section(tools_info: list[dict]) -> str:
    if not tools_info:
        return ""
    tools_json = json.dumps(tools_info, ensure_ascii=False)
    parts = [
        "",
        "",
        "# Tools",
        "",
        "You may call one or more functions to assist with the user query.",
        "",
        "You are provided with function signatures within <tools></tools> XML tags:",
        "<tools>",
        tools_json,
        "</tools>",
        "",
        "For each function call, return a json object with function name and arguments within",
        "<tool_call>",
        '{"name": <function-name>, "arguments": <args-json-object>}',
        "</tool_call>",
        "XML tags.",
    ]
    return "\n".join(parts)


def _messages_for_render(messages: list[dict]) -> list[dict]:
    out: list[dict] = []
    for msg in messages:
        out.append({"role": msg["role"], "content": msg.get("content", "")})
    return out


class TrainableTauBenchAgent:
    """Trainable tau-bench agent using vLLM render + /inference/v1/generate."""

    async def asolve(self, args: Any, sample: Sample, sampling_params) -> Sample:
        assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."

        state = GenerateState(args)
        base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"

        sample.metadata = sample.metadata or {}

        headers = None
        if getattr(args, "router_policy", None) == "consistent_hash":
            sample.session_id = sample.session_id or str(uuid.uuid4())
            headers = {"x-session-id": sample.session_id}

        try:
            env = build_env(sample=sample, args=args)
        except TypeError:
            env = build_env(sample, args)

        initial_obs = env.reset()
        wiki = initial_obs.get("wiki", "")
        short_wiki_chars = os.environ.get("TAU_SHORT_WIKI")
        if short_wiki_chars is not None:
            wiki = wiki[: int(short_wiki_chars)]
        tools_info = initial_obs.get("tools_info", [])
        if os.environ.get("TAU_SHORT_TOOLS") == "1":
            keep = {"find_user_id_by_email", "find_user_id_by_name_zip", "get_user_details", "get_order_details"}
            tools_info = [tool for tool in tools_info if tool.get("function", {}).get("name") in keep]
        tools_section = _build_tools_section(tools_info)

        messages: list[dict] = [
            {"role": "system", "content": wiki + tools_section + "\n/no_think"},
            {"role": "user", "content": initial_obs.get("obs_str", "")},
        ]

        response_tokens: list[int] = []
        sample.loss_mask = sample.loss_mask or []
        sample.rollout_log_probs = sample.rollout_log_probs or []
        sample.tokens = list(sample.tokens) if sample.tokens else []

        sampling_params = sampling_params.copy()
        sampling_params["repetition_penalty"] = 1.1
        sampling_params.setdefault("stop", ["</tool_call>"])
        inference_sampling_params = _build_inference_sampling_params(sampling_params)
        max_response_budget = sampling_params.get("max_new_tokens")

        def remaining_budget() -> int | None:
            return None if max_response_budget is None else max_response_budget - sample.response_length

        def _ensure_trainable_skeleton() -> None:
            """Megatron padding needs total_length >= 1 (F.pad uses prompt_length - 1)."""
            if not sample.tokens:
                eos_id = getattr(state.tokenizer, "eos_token_id", None)
                pad_id = getattr(state.tokenizer, "pad_token_id", None)
                token_id = eos_id if eos_id is not None else pad_id if pad_id is not None else 0
                sample.tokens = [int(token_id)]
            sample.loss_mask = sample.loss_mask or []
            sample.rollout_log_probs = sample.rollout_log_probs or []
            if len(sample.rollout_log_probs) < len(sample.loss_mask):
                sample.rollout_log_probs.extend([0.0] * (len(sample.loss_mask) - len(sample.rollout_log_probs)))
            sample.response_length = len(sample.loss_mask)

        def _mark_truncated(reward_base: float = 0.0, *, remove: bool = True) -> Sample:
            _ensure_trainable_skeleton()
            if remove:
                sample.remove_sample = True
            sample.reward = compute_process_reward(env, reward_base)
            sample.status = Sample.Status.TRUNCATED
            return sample

        async def safe_render() -> dict | None:
            try:
                render_messages = _messages_for_render(messages)
                payload = {"model": args.hf_checkpoint, "messages": render_messages}
                render_data = await post(
                    f"{base_url}/v1/chat/completions/render", payload, headers=headers, max_retries=3
                )
                return _mm_render_response_to_generate_body(render_data, args.hf_checkpoint)
            except Exception as exc:
                logger.warning("render failed, skipping task: %s", exc)
                return None

        def append_response_window(
            token_ids: list[int],
            loss_mask: list[int],
            log_probs: list[float] | None = None,
        ) -> None:
            if not token_ids:
                return
            if len(loss_mask) != len(token_ids):
                raise ValueError(f"loss_mask length {len(loss_mask)} != token_ids length {len(token_ids)}")
            sample.tokens.extend(token_ids)
            sample.loss_mask.extend(loss_mask)
            sample.rollout_log_probs.extend(log_probs if log_probs is not None else [0.0] * len(token_ids))
            sample.response_length += len(token_ids)

        def sampling_params_for_turn() -> dict | None:
            params = dict(inference_sampling_params)
            max_tokens = remaining_budget()
            if max_tokens is None:
                return params
            if max_tokens <= 0:
                return None
            params["max_tokens"] = max_tokens
            return params

        try:
            pending_obs_offset: int | None = None
            rendered_body = await safe_render()
            if rendered_body is None:
                return _mark_truncated()
            prompt_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
            if not sample.tokens:
                sample.tokens = list(prompt_ids)
            if args.rollout_max_context_len is not None:
                context_budget = max(0, args.rollout_max_context_len - len(sample.tokens))
                if max_response_budget is None:
                    max_response_budget = context_budget
                else:
                    max_response_budget = min(max_response_budget, context_budget)

            vllm_max_len = getattr(args, "vllm_max_model_len", 16384) or 16384
            if len(prompt_ids) >= vllm_max_len - 64:
                logger.info(f"prompt too long ({len(prompt_ids)} tokens >= {vllm_max_len - 64}), skipping task")
                return _mark_truncated()

            for turn_idx in range(args.max_turns):
                input_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))

                if pending_obs_offset is not None:
                    obs_tokens = input_ids[pending_obs_offset:]
                    remaining = remaining_budget()
                    if remaining is not None and len(obs_tokens) > remaining:
                        append_response_window(obs_tokens[: max(remaining, 0)], [0] * max(remaining, 0))
                        sample.status = Sample.Status.TRUNCATED
                        break
                    append_response_window(obs_tokens, [0] * len(obs_tokens))
                    pending_obs_offset = None

                current_sampling_params = sampling_params_for_turn()
                if current_sampling_params is None:
                    sample.status = Sample.Status.TRUNCATED
                    break

                body = dict(rendered_body)
                body["sampling_params"] = current_sampling_params
                output = await post(f"{base_url}/inference/v1/generate", body, headers=headers)
                choice = output["choices"][0]
                finish_reason = choice.get("finish_reason") or "stop"
                new_tokens, new_logprobs = _parse_choice_tokens_and_logprobs(choice)

                if not new_tokens:
                    if finish_reason in ("abort", "cancelled"):
                        sample.status = Sample.Status.ABORTED
                        break

                response_text = state.tokenizer.decode(new_tokens, skip_special_tokens=False) if new_tokens else ""
                train_tokens = list(new_tokens)
                train_logprobs = list(new_logprobs)
                train_loss_mask = [1] * len(train_tokens)

                stop = current_sampling_params.get("stop")
                if not stop:
                    stop = ["</tool_call>"]
                stop_strings = (stop,) if isinstance(stop, str) else tuple(stop) if stop else ()
                hit_stop_str = None
                hit_stop_pos = len(response_text)
                if stop_strings:
                    for ss in stop_strings:
                        pos = response_text.find(ss)
                        if pos != -1 and pos < hit_stop_pos:
                            hit_stop_str = ss
                            hit_stop_pos = pos
                if hit_stop_str is not None:
                    truncated_text = response_text[: hit_stop_pos + len(hit_stop_str)]
                    trunc_token_count = 0
                    for t in range(1, len(train_tokens) + 1):
                        partial = state.tokenizer.decode(train_tokens[:t], skip_special_tokens=False)
                        if partial >= truncated_text:
                            trunc_token_count = t
                            break
                    if trunc_token_count == 0:
                        trunc_token_count = len(train_tokens)
                    train_tokens = train_tokens[:trunc_token_count]
                    train_logprobs = train_logprobs[:trunc_token_count]
                    train_loss_mask = train_loss_mask[:trunc_token_count]
                    response_text = truncated_text
                    finish_reason = "stop"

                eos_token_id = getattr(state.tokenizer, "eos_token_id", None)
                append_stop_eos = (
                    stop
                    and eos_token_id is not None
                    and getattr(args, "append_eos_token_after_stop_str_in_multi_turn", True)
                )
                if append_stop_eos:
                    already_has_eos = bool(train_tokens and train_tokens[-1] == eos_token_id)
                    if stop_strings and response_text.endswith(stop_strings) and not already_has_eos:
                        if getattr(args, "use_rollout_routing_replay", False):
                            raise RuntimeError(
                                "Routing replay is not supported when appending an artificial EOS after a stop string, "
                                "because vLLM does not return routed experts for that extra token."
                            )
                        train_tokens.append(int(eos_token_id))
                        train_logprobs.append(0.0)
                        train_loss_mask.append(0)

                response_tokens.extend(new_tokens)
                append_response_window(train_tokens, train_loss_mask, train_logprobs)
                _maybe_apply_routed_experts(args, sample, choice)

                messages.append({"role": "assistant", "content": response_text})

                if finish_reason == "length":
                    sample.status = Sample.Status.TRUNCATED
                    break
                if finish_reason in ("abort", "cancelled"):
                    sample.status = Sample.Status.ABORTED
                    break

                observation, done, step_info = env.step(response_text)

                if done:
                    base_reward = observation.get("reward", 0.0)
                    sample.reward = compute_process_reward(env, base_reward)
                    sample.status = Sample.Status.COMPLETED
                    break

                next_user_message = env.format_observation(observation)
                messages.append(next_user_message)

                if turn_idx + 1 >= args.max_turns:
                    sample.reward = compute_process_reward(env, 0.0)
                    sample.status = Sample.Status.TRUNCATED
                    break

                pending_obs_offset = len(input_ids) + len(train_tokens)
                max_ctx = args.rollout_max_context_len or 8192
                if len(sample.tokens) >= max_ctx - 64:
                    logger.info(
                        f"[turn={turn_idx}] context overflow: {len(sample.tokens)} tokens >= {max_ctx - 64}, truncating"
                    )
                    sample.reward = compute_process_reward(env, 0.0)
                    sample.status = Sample.Status.TRUNCATED
                    break
                rendered_body = await safe_render()
                if rendered_body is None:
                    return _mark_truncated()
                rendered_ids = _coerce_flat_int_token_ids(rendered_body.get("token_ids"))
                is_prefix_stable = rendered_ids[:pending_obs_offset] == sample.tokens[:pending_obs_offset]
                sample.metadata["multiturn_render"] = {
                    "prefix_stable": is_prefix_stable,
                    "prefix_len": pending_obs_offset,
                    "sample_len": len(sample.tokens),
                    "rendered_len": len(rendered_ids),
                    "turn": turn_idx + 1,
                }
                if getattr(args, "strict_multiturn_render_token_match", False) and not is_prefix_stable:
                    raise RuntimeError(
                        "Full conversation render is not prefix-stable with the generated token stream: "
                        f"{sample.metadata['multiturn_render']}"
                    )

            sample.response = state.tokenizer.decode(response_tokens, skip_special_tokens=False)
            sample.response_length = len(sample.loss_mask)
            _ensure_trainable_skeleton()
            if sample.status == Sample.Status.PENDING:
                sample.status = Sample.Status.COMPLETED
            if sample.reward is None or sample.reward == 0.0:
                sample.reward = compute_process_reward(env, getattr(env, "total_reward", 0.0))
            return sample
        finally:
            try:
                env.close()
            except Exception:
                pass


def agent_factory() -> TrainableTauBenchAgent:
    return TrainableTauBenchAgent()
