"""Unit tests for the multi-turn agent-loop scaffold (vime_plugins.rollout.multi_turn_agent).

The vLLM router (``post``) and ``GenerateState`` are mocked, so this validates the vime contract
bookkeeping (token accounting, loss masking, status, reward propagation) without a live server.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from contextlib import asynccontextmanager

import pytest

from vime.utils.types import Sample
from vime_plugins.rollout import multi_turn_agent as mod


class _FakeTokenizer:
    def encode(self, text, add_special_tokens=False):
        return [1] * len(text.split())

    def decode(self, token_ids, skip_special_tokens=True):
        return " ".join("tok" for _ in token_ids)

    def apply_chat_template(self, messages, add_generation_prompt=True, tokenize=True):
        return [0, 0, 0]


class _FakeState:
    def __init__(self, args):
        self.tokenizer = _FakeTokenizer()
        self.aborted = False

    @property
    def semaphore(self):
        @asynccontextmanager
        async def _cm():
            yield

        return _cm()


def _args(**kw):
    base = dict(
        vllm_router_ip="127.0.0.1",
        vllm_router_port=1234,
        hf_checkpoint="model",
        rollout_max_context_len=1000,
        eval_max_context_len=1000,
        router_policy=None,
        agent_step_path=None,
        agent_max_turns=8,
    )
    base.update(kw)
    return Namespace(**base)


@pytest.fixture(autouse=True)
def _patch_state(monkeypatch):
    monkeypatch.setattr(mod, "GenerateState", _FakeState)
    monkeypatch.setattr(mod, "_build_inference_sampling_params", lambda sp: sp)


def _post_returning(*token_id_lists):
    """Return an async ``post`` that yields one canned generation per call."""
    seq = iter(token_id_lists)

    async def _post(url, payload, headers=None):
        return {"choices": [{"token_ids": next(seq), "finish_reason": "stop"}]}

    return _post


def test_single_turn_default_all_trainable(monkeypatch):
    monkeypatch.setattr(mod, "post", _post_returning([5, 6, 7]))
    sample = Sample(prompt="hello world")

    out = asyncio.run(mod.generate(_args(), sample, {"max_new_tokens": 16}))

    assert out.response_length == 3
    assert out.loss_mask == [1, 1, 1]
    assert out.status == Sample.Status.COMPLETED
    # str prompt "hello world" -> encode() = 2 prompt tokens; + 3 response tokens
    assert len(out.tokens) == 5


def test_multi_turn_masks_observations(monkeypatch):
    monkeypatch.setattr(mod, "post", _post_returning([5, 6], [7, 8, 9]))

    async def agent_step(args, sample, assistant_text, turn_index):
        if turn_index == 0:
            return {"done": False, "observation": "obs tokens here", "reward": None}
        return {"done": True, "observation": None, "reward": 1.0}

    monkeypatch.setattr(mod, "_resolve_agent_step", lambda args: agent_step)
    sample = Sample(prompt="hi")

    out = asyncio.run(mod.generate(_args(), sample, {"max_new_tokens": 16}))

    # turn 0: 2 model tokens (mask 1), then observation "obs tokens here" -> 3 tokens (mask 0)
    # turn 1: 3 model tokens (mask 1)
    assert out.loss_mask == [1, 1, 0, 0, 0, 1, 1, 1]
    assert out.response_length == 8
    assert out.effective_response_length == 5  # only model tokens trained
    assert out.reward == 1.0
    assert out.status == Sample.Status.COMPLETED


def test_truncates_on_context_budget(monkeypatch):
    monkeypatch.setattr(mod, "post", _post_returning([1, 2, 3, 4, 5]))

    async def never_done(args, sample, assistant_text, turn_index):
        return {"done": False, "observation": "x y z", "reward": None}

    monkeypatch.setattr(mod, "_resolve_agent_step", lambda args: never_done)
    # tiny context budget so the loop truncates quickly
    sample = Sample(prompt="hi")
    out = asyncio.run(mod.generate(_args(rollout_max_context_len=9), sample, {"max_new_tokens": 16}))

    assert out.status == Sample.Status.TRUNCATED


def test_aborted(monkeypatch):
    monkeypatch.setattr(mod, "post", _post_returning([1]))

    class _Aborted(_FakeState):
        def __init__(self, args):
            super().__init__(args)
            self.aborted = True

    monkeypatch.setattr(mod, "GenerateState", _Aborted)
    out = asyncio.run(mod.generate(_args(), Sample(prompt="hi"), {"max_new_tokens": 4}))
    assert out.status == Sample.Status.ABORTED


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
