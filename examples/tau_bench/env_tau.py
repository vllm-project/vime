from __future__ import annotations

import json
import logging
import os
from typing import Any

from tau_bench.agents.tool_calling_agent import RESPOND_ACTION_NAME
from tau_bench.envs import get_env
from tau_bench.types import Action, RunConfig

from .openai_tool_adapter import create_openai_adapter

from vime.utils.types import Sample

logger = logging.getLogger(__name__)


class TauBenchEnv:
    """
    Interaction environment wrapping tau-bench for vime multi-turn rollout.

    This env bridges the tau-bench toolkit with vime's multi-turn rollout
    architecture. It manages:
    - tau-bench environment initialization and task selection
    - Tool call parsing from LLM responses (via OpenAI adapter)
    - Action execution in the tau-bench environment
    - Observation formatting for the next turn
    """

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

    def reset(self):
        self.turn = 0
        self.total_reward = 0.0
        self.info = {}

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
        self.info = env_reset_res.info.model_dump() if hasattr(env_reset_res.info, "model_dump") else {}

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
            return {
                "obs_str": f"Failed to parse tool call. Please try again.",
                "role": "tool",
            }, is_final_turn, {"tool_executed": False, "parse_error": openai_result.get("error")}

        parsed = openai_result["parsed_result"]
        agent_content, calls = parsed["normal_text"], parsed["calls"]

        action = self._call_to_action(calls, agent_content)

        try:
            env_response = self.env.step(action)
        except Exception as e:
            logger.warning(f"Environment step failed: {e}")
            return {
                "obs_str": f"Environment error: {e}",
                "role": "tool",
            }, True, {"tool_executed": False, "env_error": str(e)}

        self.total_reward = env_response.reward
        self.info.update(env_response.info.model_dump() if hasattr(env_response.info, "model_dump") else {})

        if action.name != RESPOND_ACTION_NAME:
            obs_role = "tool"
            obs_content = env_response.observation
        else:
            obs_role = "user"
            obs_content = env_response.observation

        done = env_response.done or is_final_turn

        return {
            "obs_str": obs_content,
            "role": obs_role,
            "reward": env_response.reward,
        }, done, {"tool_executed": True, "action": action.name}

    def _call_to_action(self, calls: list[Any], text_response: str) -> Action:
        action = Action(name=RESPOND_ACTION_NAME, kwargs={"content": text_response})
        if calls:
            if len(calls) > 1:
                logger.debug("Multiple tool calls identified, only taking first.")
            tool_call = calls[0]
            try:
                params = json.loads(tool_call["parameters"]) if isinstance(tool_call["parameters"], str) else tool_call["parameters"]
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
        return {
            "role": observation.get("role", "user"),
            "content": observation.get("obs_str", ""),
        }


def build_env(sample: Sample | None = None, args: Any | None = None, **_: Any) -> TauBenchEnv:
    user_model = getattr(args, "tau_user_model", "openai/local-qwen3-4b")
    user_model_provider = getattr(args, "tau_user_model_provider", "openai")
    user_strategy = getattr(args, "tau_user_strategy", "llm")

    vllm_router_host = getattr(args, "vllm_router_host", "10.155.68.38")
    vllm_router_port = getattr(args, "vllm_router_port", 3250)
    vllm_model_name = getattr(args, "vllm_model_name", "/data/nfs_87/xky/models/Qwen3-4B")

    if user_model_provider == "openai" and "local" in user_model:
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
        os.environ["OPENAI_API_BASE"] = f"http://{vllm_router_host}:{vllm_router_port}/v1"
        user_model = vllm_model_name

    tau_config = RunConfig(
        env=getattr(args, "tau_env", "retail"),
        agent="tool-calling",
        user_model=user_model,
        user_model_provider=user_model_provider,
        task_split=getattr(args, "tau_task_split", "train"),
        user_strategy=user_strategy,
        model_provider="auto_router",
        model="qwen3-4b",
    )

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
        tau_config=tau_config,
        task_index=task_index,
        max_turns=max_turns,
    )
