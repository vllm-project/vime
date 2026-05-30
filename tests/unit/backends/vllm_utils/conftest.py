"""Shared fixtures for vLLM backend unit tests."""

from __future__ import annotations

from types import SimpleNamespace

import pytest


@pytest.fixture
def vllm_args() -> SimpleNamespace:
    return SimpleNamespace(
        rollout_external=True,
        hf_checkpoint="/tmp/model",
        router_ip=None,
        router_port=None,
        vllm_weight_transfer_timeout_sec=900.0,
        num_gpus_per_node=8,
        rollout_num_gpus_per_engine=4,
        colocate=False,
        debug_rollout_only=False,
        actor_num_gpus_per_node=4,
        actor_num_nodes=1,
        use_critic=False,
        critic_num_gpus_per_node=0,
        critic_num_nodes=0,
    )


@pytest.fixture
def vllm_engine(vllm_args):
    from slime.backends.vllm_utils.vllm_engine import VLLMEngine

    engine = VLLMEngine(vllm_args, rank=0)
    engine.server_host = "127.0.0.1"
    engine.server_port = 8765
    return engine
