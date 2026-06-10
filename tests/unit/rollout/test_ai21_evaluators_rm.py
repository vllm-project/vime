"""Unit tests for the AI21 evaluators reward function (``vime_plugins.rm.ai21_evaluators``).

These cover the vime-specific glue (single vs. batched dispatch, reward shaping, exclusion on
non-success) without requiring the private ``ai21-evaluators`` deps: the request-building and the
actual evaluation call are monkeypatched.
"""

from __future__ import annotations

import asyncio
from argparse import Namespace
from types import SimpleNamespace

import pytest

from vime.utils.types import Sample
from vime_plugins.rm import ai21_evaluators as mod


def _fake_response(score, status="success"):
    # mimics AI21EvaluationResponse: .score and a str-enum-like .status with a .value
    return SimpleNamespace(score=score, status=SimpleNamespace(value=status))


@pytest.fixture(autouse=True)
def _stub_ai21(monkeypatch):
    """Stub out evaluator init and request building so no private deps are imported."""
    monkeypatch.setattr(mod, "_ensure_initialized", lambda: None)
    # request just carries the completion so _run_evaluation can key off it
    monkeypatch.setattr(mod, "_build_request", lambda args, sample: SimpleNamespace(completion=sample.response))


def _run_eval_returning(score, status="success"):
    async def _fake_run(request, timeout):
        return _fake_response(score, status)

    return _fake_run


def test_single_sample_returns_float(monkeypatch):
    monkeypatch.setattr(mod, "_run_evaluation", _run_eval_returning(1.0))
    args = Namespace(reward_key=None)
    sample = Sample(prompt="p", response="r", label="l", metadata={"reward_model": [{"evaluator_name": "x"}]})

    reward = asyncio.run(mod.ai21_reward(args, sample))

    assert reward == 1.0
    assert isinstance(reward, float)


def test_reward_key_returns_dict(monkeypatch):
    monkeypatch.setattr(mod, "_run_evaluation", _run_eval_returning(0.75))
    args = Namespace(reward_key="score")
    sample = Sample(prompt="p", response="r")

    reward = asyncio.run(mod.ai21_reward(args, sample))

    assert isinstance(reward, dict)
    assert reward["score"] == 0.75
    assert reward["status"] == "success"
    assert reward["do_exclude"] is False
    # the configured --reward-key must be selectable by Sample.get_reward_value
    sample.reward = reward
    assert sample.get_reward_value(args) == 0.75


def test_non_success_marks_exclusion(monkeypatch):
    monkeypatch.setattr(mod, "_run_evaluation", _run_eval_returning(0.0, status="timeout"))
    args = Namespace(reward_key="score")
    sample = Sample(prompt="p", response="r")

    reward = asyncio.run(mod.ai21_reward(args, sample))

    assert reward["do_exclude"] is True
    assert reward["status"] == "timeout"


def test_batched_returns_list(monkeypatch):
    monkeypatch.setattr(mod, "_run_evaluation", _run_eval_returning(1.0))
    args = Namespace(reward_key=None)
    samples = [Sample(prompt="p", response=f"r{i}") for i in range(3)]

    rewards = asyncio.run(mod.ai21_reward(args, samples))

    assert rewards == [1.0, 1.0, 1.0]


def test_none_score_coerced_to_zero(monkeypatch):
    monkeypatch.setattr(mod, "_run_evaluation", _run_eval_returning(None))
    args = Namespace(reward_key=None)
    sample = Sample(prompt="p", response="r")

    assert asyncio.run(mod.ai21_reward(args, sample)) == 0.0


def test_clean_thinking_trace(monkeypatch):
    monkeypatch.setenv("AI21_CLEAN_THINKING_TRACE", "true")
    args = Namespace()
    sample = Sample(response="reasoning...</think>final answer")

    assert mod._completion_from_sample(args, sample) == "final answer"


def test_build_request_missing_reward_model_raises(monkeypatch):
    # _build_request is stubbed by the fixture; test the real one here
    monkeypatch.undo()
    args = Namespace()
    sample = Sample(prompt="p", response="r", metadata={})

    with pytest.raises(ValueError, match="reward_model"):
        mod._build_request(args, sample)


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
