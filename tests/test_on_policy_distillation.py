"""CPU tests for the external OPD teacher's request and score contract."""

import asyncio
import base64
import io
from argparse import Namespace
from unittest.mock import AsyncMock

import numpy as np
import pytest
import torch

from vime.rollout import on_policy_distillation as opd
from vime.utils.types import Sample

NUM_GPUS = 0


def _args(**kwargs):
    return Namespace(
        **dict(rm_url="http://teacher/inference/v1/generate", rollout_temperature=0.7, reward_key=None) | kwargs
    )


def _encoded(scores):
    buffer = io.BytesIO()
    np.save(buffer, scores)
    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _legacy_reward():
    return {"prompt_logprobs": [None, {"20": {"logprob": -9.0}}, {"30": {"logprob": -2.0}}, {"40": {"logprob": -3.0}}]}


@pytest.mark.parametrize("mode", [None, "prompt-logprobs", "per-position"])
def test_scoring_request_preserves_context_and_temperature(monkeypatch, mode):
    args = _args(opd_teacher_model="teacher")
    if mode is not None:
        args.opd_teacher_scoring = mode
    post = AsyncMock(return_value={})
    monkeypatch.setattr(opd, "post", post)
    asyncio.run(opd.reward_func(args, Sample(tokens=[10, 20, 30, 40], response_length=2)))
    url, body = post.call_args.args
    assert url == args.rm_url
    assert body["token_ids"] == [10, 20, 30, 40]
    assert body["model"] == "teacher"
    params = body["sampling_params"]
    assert params["temperature"] == 0.7
    assert params["max_tokens"] == 1
    assert params["skip_special_tokens"] is False
    if mode == "per-position":
        assert "prompt_logprobs" not in params
        assert params["prompt_logprob_start"] == 1
        assert params["prompt_logprob_token_ids"] == [[30], [40]]
    else:
        assert params["prompt_logprobs"] == 1
        assert "prompt_logprob_token_ids" not in params


def test_first_response_token_is_scored_from_its_context(monkeypatch):
    post = AsyncMock(return_value={})
    monkeypatch.setattr(opd, "post", post)
    asyncio.run(opd.reward_func(_args(opd_teacher_scoring="per-position"), Sample(tokens=[10, 20], response_length=1)))
    params = post.call_args.args[1]["sampling_params"]
    assert params["prompt_logprob_start"] == 0
    assert params["prompt_logprob_token_ids"] == [[20]]
    assert "model" not in post.call_args.args[1]


@pytest.mark.parametrize("mode", ["prompt-logprobs", "per-position"])
def test_empty_response_skips_teacher_and_stores_empty_scores(monkeypatch, mode):
    args = _args(opd_teacher_scoring=mode, reward_key="unused")
    post = AsyncMock()
    monkeypatch.setattr(opd, "post", post)
    sample = Sample(tokens=[10, 20], response_length=0)
    sample.reward = asyncio.run(opd.reward_func(args, sample))
    assert opd.post_process_rewards(args, [sample]) == ([0.0], [0.0])
    assert sample.teacher_log_probs.shape == (0,)
    assert sample.teacher_log_probs.dtype == torch.float32
    post.assert_not_called()


@pytest.mark.parametrize("length", [-1, 4, 5])
def test_invalid_response_length_fails_before_request_or_score_assignment(monkeypatch, length):
    post = AsyncMock()
    monkeypatch.setattr(opd, "post", post)
    sample = Sample(tokens=[10, 20, 30, 40], response_length=length)
    with pytest.raises(ValueError, match="context token"):
        asyncio.run(opd.reward_func(_args(), sample))
    with pytest.raises(ValueError, match="context token"):
        opd.post_process_rewards(_args(), [sample])
    post.assert_not_called()
    assert sample.teacher_log_probs is None


@pytest.mark.parametrize("mode", ["prompt-logprobs", "per-position"])
def test_post_processing_preserves_response_scores_and_zero_task_reward(mode):
    reward = (
        _legacy_reward()
        if mode == "prompt-logprobs"
        else {"prompt_token_id_logprobs": _encoded(np.array([[-2.0], [-3.0]], dtype=np.float32))}
    )
    sample = Sample(tokens=[10, 20, 30, 40], response_length=2, reward={"teacher": reward})
    assert opd.post_process_rewards(_args(opd_teacher_scoring=mode, reward_key="teacher"), [sample]) == ([0.0], [0.0])
    torch.testing.assert_close(sample.teacher_log_probs, torch.tensor([-2.0, -3.0]))


@pytest.mark.parametrize(
    "reward",
    [
        _legacy_reward(),
        {"prompt_token_id_logprobs": None},
        {"prompt_token_id_logprobs": "not base64"},
        {"prompt_token_id_logprobs": base64.b64encode(b"not a numpy array").decode()},
    ],
)
def test_missing_or_malformed_candidate_scores_do_not_fall_back(reward):
    sample = Sample(tokens=[10, 20, 30, 40], response_length=2, reward=reward)
    with pytest.raises(ValueError, match="prompt_token_id_logprobs"):
        opd.post_process_rewards(_args(opd_teacher_scoring="per-position"), [sample])
    assert sample.teacher_log_probs is None


def test_candidate_array_read_error_has_teacher_context(monkeypatch):
    reward = {"prompt_token_id_logprobs": _encoded(np.array([[-2.0], [-3.0]], dtype=np.float32))}
    sample = Sample(tokens=[10, 20, 30, 40], response_length=2, reward=reward)
    read_error = OSError("array read failed")

    def fail_load(*args, **kwargs):
        raise read_error

    monkeypatch.setattr(opd.np, "load", fail_load)
    with pytest.raises(ValueError, match="teacher prompt_token_id_logprobs") as exc:
        opd.post_process_rewards(_args(opd_teacher_scoring="per-position"), [sample])
    assert exc.value.__cause__ is read_error
    assert sample.teacher_log_probs is None


@pytest.mark.parametrize(
    "scores",
    [
        np.array([-2.0, -3.0], dtype=np.float32),
        np.array([[-2.0, -3.0]], dtype=np.float32),
        np.array([[-2.0]], dtype=np.float32),
        np.array([[-2.0], [-3.0]], dtype=np.float64),
        np.array([[-2], [-3]], dtype=np.int32),
        np.array([[float("nan")], [-3.0]], dtype=np.float32),
        np.array([[float("-inf")], [-3.0]], dtype=np.float32),
        np.array([[1.0], [-3.0]], dtype=np.float32),
        np.array([[object()], [object()]], dtype=object),
    ],
)
def test_invalid_candidate_matrix_is_rejected(scores):
    sample = Sample(tokens=[10, 20, 30, 40], response_length=2, reward={"prompt_token_id_logprobs": _encoded(scores)})
    with pytest.raises(ValueError):
        opd.post_process_rewards(_args(opd_teacher_scoring="per-position"), [sample])
    assert sample.teacher_log_probs is None


def test_mixed_image_and_text_batch_uses_each_samples_requested_format():
    text = Sample(
        tokens=[10, 20, 30, 40],
        response_length=2,
        reward={"prompt_token_id_logprobs": _encoded(np.array([[-2.0], [-3.0]], dtype=np.float32))},
    )
    image = Sample(
        tokens=[10, 20, 30, 40], response_length=2, multimodal_inputs={"images": ["image"]}, reward=_legacy_reward()
    )
    empty = Sample(tokens=[10], response_length=0)
    assert opd.post_process_rewards(_args(opd_teacher_scoring="per-position"), [text, image, empty]) == (
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    )
    torch.testing.assert_close(text.teacher_log_probs, image.teacher_log_probs)
    assert empty.teacher_log_probs.numel() == 0


def test_one_bad_sample_does_not_partially_assign_batch_scores():
    valid = Sample(tokens=[10, 20, 30, 40], response_length=2, reward=_legacy_reward())
    invalid = Sample(tokens=[10, 20, 30, 40], response_length=2, reward={"prompt_logprobs": [None]})
    with pytest.raises(ValueError, match="align"):
        opd.post_process_rewards(_args(), [valid, invalid])
    assert valid.teacher_log_probs is None


def test_teacher_http_failure_is_propagated_without_legacy_retry(monkeypatch):
    post = AsyncMock(side_effect=RuntimeError("unsupported request"))
    monkeypatch.setattr(opd, "post", post)
    with pytest.raises(RuntimeError, match="unsupported request"):
        asyncio.run(
            opd.reward_func(_args(opd_teacher_scoring="per-position"), Sample(tokens=[10, 20], response_length=1))
        )
    post.assert_awaited_once()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
