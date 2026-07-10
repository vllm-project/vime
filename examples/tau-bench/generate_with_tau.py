"""Tau-bench multi-turn custom rollout for vime (vLLM render + generate)."""

from __future__ import annotations

import asyncio
import logging
import os
from typing import Any

from tau_bench.types import RunConfig
from trainable_agents import TrainableTauBenchAgent, agent_factory, patch_tau_user_retries

from vime.utils.types import Sample

logger = logging.getLogger(__name__)

_TAU_DEFAULT_MAX_TURNS = 10

_inflight_sem: asyncio.Semaphore | None = None

# Tau-bench user-simulator configuration (edit TAU_CONFIGS below).
# Agent rollout uses vLLM; only user_model / user_model_provider affect the user simulator here.
TAU_CONFIGS = {
    "env": "retail",  # Select between ["retail", "airline"]
    "agent_strategy": "tool-calling",  # Select between ["tool-calling", "act", "react", "few-shot"]
    # Default: local vLLM user sim (no external API). For Gemini API user sim, switch to:
    # "user_model": "gemini-2.5-flash-lite", "user_model_provider": "gemini",
    "user_model": "openai/local-qwen3-4b",
    "user_model_provider": "openai",
    "task_split": "train",  # Select between ["train", "test", "dev"] for retail
    "user_strategy": "llm",  # Select between ["llm", "react", "verify", "reflection"]
    "model_provider": "auto_router",  # Unused, required
    "model": "qwen3-4b",  # Unused, required
}
# Replace with your actual API key when user_model_provider is gemini.
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "NONE")
os.environ["GEMINI_API_KEY"] = GEMINI_API_KEY
tau_config = RunConfig(**TAU_CONFIGS)


def _get_inflight_sem() -> asyncio.Semaphore:
    global _inflight_sem
    if _inflight_sem is None:
        _inflight_sem = asyncio.Semaphore(int(os.environ.get("TAU_MAX_INFLIGHT", "8")))
    return _inflight_sem


patch_tau_user_retries()


def _ensure_tau_args(args: Any) -> None:
    if getattr(args, "max_turns", None) is None:
        env_max = os.environ.get("TAU_MAX_TURNS")
        args.max_turns = int(env_max) if env_max is not None else _TAU_DEFAULT_MAX_TURNS


def resolve_tau_config(args: Any) -> RunConfig:
    """Build RunConfig from TAU_CONFIGS, with optional local-vLLM user-sim routing."""
    user_model = tau_config.user_model
    user_model_provider = tau_config.user_model_provider

    if user_model_provider == "openai" and "local" in user_model:
        vllm_router_host = getattr(args, "vllm_router_ip", "127.0.0.1")
        vllm_router_port = getattr(args, "vllm_router_port", 3250)
        vllm_model_name = getattr(args, "vllm_model_name", getattr(args, "hf_checkpoint", ""))
        os.environ["OPENAI_API_KEY"] = os.environ.get("OPENAI_API_KEY", "dummy")
        os.environ["OPENAI_API_BASE"] = f"http://{vllm_router_host}:{vllm_router_port}/v1"
        user_model = vllm_model_name

    return RunConfig(
        env=tau_config.env,
        agent_strategy=tau_config.agent_strategy,
        user_model=user_model,
        user_model_provider=user_model_provider,
        task_split=tau_config.task_split,
        user_strategy=tau_config.user_strategy,
        model_provider=tau_config.model_provider,
        model=tau_config.model,
    )


async def batched_tau_bench_rm(args, samples, **kwargs) -> list[float] | float:
    if isinstance(samples, Sample):
        return samples.reward if samples.reward is not None else 0.0
    rewards = [s.reward if s.reward is not None else 0.0 for s in samples]
    max_r = max(rewards) if rewards else 1.0
    if max_r > 0:
        rewards = [r / max_r for r in rewards]
    return rewards


async def generate(args: Any, sample: Sample, sampling_params) -> Sample:
    assert not args.partial_rollout, "Partial rollout is not supported for tau-bench interactions."
    _ensure_tau_args(args)
    args.tau_bench_config = resolve_tau_config(args)

    task_index = sample.prompt
    logger.info(f"Starting agent-environment interaction for task {task_index}")

    async with _get_inflight_sem():
        agent: TrainableTauBenchAgent = agent_factory()
        result = await agent.asolve(args, sample, sampling_params)

    logger.info(f"Finished agent-environment interaction for task {task_index}")
    return result
