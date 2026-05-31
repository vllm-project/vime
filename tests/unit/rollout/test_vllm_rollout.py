"""Unit tests for ``slime.rollout.vllm_rollout`` helpers and mocked async paths."""

from __future__ import annotations

import asyncio
import base64
import io
import logging
from argparse import Namespace
from contextlib import contextmanager
from unittest.mock import AsyncMock

import numpy as np
import pytest

from slime.rollout import vllm_rollout as mod
from slime.utils.eval_config import EvalDatasetConfig
from slime.utils.types import Sample


class _FakeTokenizer:
    def encode(self, prompt: str, add_special_tokens: bool = False) -> list[int]:
        assert add_special_tokens is False
        return [ord(c) % 100 for c in prompt[:3]] or [1, 2, 3]

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return "".join(chr(int(t)) for t in token_ids)


class _FakeProcessor:
    def __call__(self, text: str, **kwargs):
        return {
            "input_ids": [[10, 20, 30]],
            "pixel_values": [[1.0]],
        }


class _DummySemaphore:
    async def __aenter__(self):
        return None

    async def __aexit__(self, exc_type, exc, tb):
        return False


class _PatchedGenerateState:
    """Lightweight GenerateState for unit tests (no HF load)."""

    _instance = None

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.tokenizer = _FakeTokenizer()
        self.processor = None
        self.semaphore = _DummySemaphore()
        self.aborted = False
        self.remaining_batch_size = 0
        self.pendings: set = set()
        self.dp_counts = [0]
        self.dp_rank = 0
        self.group_sampling_seeds = None
        if getattr(args, "vllm_enable_deterministic_inference", False):
            self.group_sampling_seeds = [args.rollout_seed + i for i in range(args.n_samples_per_prompt)]

    @classmethod
    def clear_instances(cls) -> None:
        cls._instance = None

    @contextmanager
    def dp_rank_context(self):
        yield 0

    def reset(self) -> None:
        self.remaining_batch_size = 0
        self.pendings = set()
        self.aborted = False


def _rollout_args(**overrides) -> Namespace:
    base = dict(
        ci_test=False,
        hf_checkpoint="/tmp/model",
        vllm_router_ip="127.0.0.1",
        vllm_router_port=8000,
        partial_rollout=False,
        mask_offpolicy_in_partial_rollout=False,
        group_rm=False,
        custom_generate_function_path=None,
        vllm_speculative_config=None,
        router_policy=None,
        use_rollout_routing_replay=False,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=True,
        apply_chat_template=False,
        apply_chat_template_kwargs=None,
        eval_max_prompt_len=None,
        multimodal_keys=None,
        eval_reward_key=None,
        reward_key=None,
    )
    base.update(overrides)
    return Namespace(**base)


def _default_sampling_params(**overrides) -> dict:
    sp = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "max_new_tokens": 8,
        "stop": None,
        "stop_token_ids": None,
        "skip_special_tokens": True,
    }
    sp.update(overrides)
    return sp


def _generate_response(token_ids: list[int] | None = None) -> dict:
    tids = token_ids or [50, 51]
    return {
        "choices": [
            {
                "token_ids": tids,
                "finish_reason": "stop",
                "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": len(tids)},
    }


@pytest.fixture
def patch_generate_state(monkeypatch):
    monkeypatch.setattr(mod, "GenerateState", _PatchedGenerateState)
    _PatchedGenerateState.clear_instances()
    return mod


def _encode_routed(arr: np.ndarray) -> str:
    buf = io.BytesIO()
    np.save(buf, arr)
    return base64.b64encode(buf.getvalue()).decode("ascii")


@pytest.mark.unit
def test_coerce_flat_int_token_ids_nested_and_scalars():
    assert mod._coerce_flat_int_token_ids([1, [2, 3]]) == [1, 2, 3]
    assert mod._coerce_flat_int_token_ids(np.array([4, 5])) == [4, 5]
    assert mod._coerce_flat_int_token_ids(None) == []
    assert mod._coerce_flat_int_token_ids(7) == [7]


@pytest.mark.unit
def test_coerce_flat_int_token_ids_rejects_str():
    with pytest.raises(TypeError, match="must not be a str"):
        mod._coerce_flat_int_token_ids("hello")


@pytest.mark.unit
def test_prepare_prompt_ids_text_only():
    sample = Sample(prompt="abc")
    assert mod._prepare_prompt_ids(sample, _FakeTokenizer(), None) == [97, 98, 99]


@pytest.mark.unit
def test_prepare_prompt_ids_reuses_tokens_without_multimodal():
    sample = Sample(prompt="ignored", tokens=[9, 8, 7])
    assert mod._prepare_prompt_ids(sample, _FakeTokenizer(), None) == [9, 8, 7]


@pytest.mark.unit
def test_prepare_prompt_ids_multimodal_via_processor():
    sample = Sample(prompt="hi", multimodal_inputs={"images": ["img"]})
    ids = mod._prepare_prompt_ids(sample, _FakeTokenizer(), _FakeProcessor())
    assert ids == [10, 20, 30]
    assert sample.multimodal_train_inputs == {"pixel_values": [[1.0]]}


@pytest.mark.unit
def test_base_dataset_prompt_ids_ignores_sample_tokens():
    sample = Sample(prompt="abc", tokens=[99, 99, 99])
    assert mod._base_dataset_prompt_ids(sample, _FakeTokenizer(), None) == [97, 98, 99]


@pytest.mark.unit
def test_get_model_url_named_router_and_fallback():
    args = Namespace(
        vllm_router_ip="127.0.0.1",
        vllm_router_port=8000,
        vllm_model_routers={"ref": ("10.0.0.2", 9001)},
    )
    assert mod.get_model_url(args, "ref") == "http://10.0.0.2:9001/inference/v1/generate"
    assert mod.get_model_url(args, "missing") == "http://127.0.0.1:8000/inference/v1/generate"
    assert mod.get_model_url(args, "ref", "/v1/chat/completions/render") == (
        "http://10.0.0.2:9001/v1/chat/completions/render"
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("finish_reason", "expected_type"),
    [
        ("length", "length"),
        ("abort", "abort"),
        ("cancelled", "abort"),
        ("stop", "stop"),
    ],
)
def test_vllm_meta_from_generate_choice_finish_reason(finish_reason, expected_type):
    args = Namespace()
    meta = mod._vllm_meta_from_generate_choice(
        args,
        {"finish_reason": finish_reason},
        {"prompt_tokens": 3, "completion_tokens": 2},
    )
    assert meta["finish_reason"] == {"type": expected_type}
    assert meta["prompt_tokens"] == 3
    assert meta["completion_tokens"] == 2


@pytest.mark.unit
def test_vllm_meta_from_generate_choice_preserves_dict_finish_reason():
    args = Namespace()
    fr = {"type": "custom", "detail": "x"}
    meta = mod._vllm_meta_from_generate_choice(args, {"finish_reason": fr}, None)
    assert meta["finish_reason"] is fr


@pytest.mark.unit
def test_decode_vllm_routed_experts_roundtrip():
    arr = np.arange(24, dtype=np.int32).reshape(2, 3, 4)
    buf = io.BytesIO()
    np.save(buf, arr)
    encoded = base64.b64encode(buf.getvalue()).decode("ascii")
    decoded = mod._decode_vllm_routed_experts(encoded)
    np.testing.assert_array_equal(decoded, arr)


@pytest.mark.unit
def test_apply_vllm_routed_experts_requires_base64_choice_field_and_exact_rows():
    # sample.tokens includes prompt+gen; routed_experts rows must be len(tokens)-1.
    sample = Sample(tokens=[1, 2, 3, 4])
    args = Namespace(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1)
    routed = np.zeros((len(sample.tokens) - 1, 2, 1), dtype=np.int32)
    mod._apply_vllm_routed_experts(args, sample, {"routed_experts": _encode_routed(routed)})
    assert sample.rollout_routed_experts is not None
    assert sample.rollout_routed_experts.shape == (3, 2, 1)


@pytest.mark.unit
def test_apply_vllm_routed_experts_raises_on_row_mismatch():
    sample = Sample(tokens=[1, 2, 3, 4])
    args = Namespace(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1)
    routed = np.zeros((1, 2, 1), dtype=np.int32)  # should be 3 rows
    with pytest.raises(RuntimeError, match="rows"):
        mod._apply_vllm_routed_experts(args, sample, {"routed_experts": _encode_routed(routed)})


@pytest.mark.unit
def test_inference_generate_tokens_and_logprobs_full_content():
    choice = {
        "token_ids": [1, 2],
        "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
    }
    tids, lps = mod._inference_generate_tokens_and_logprobs(choice)
    assert tids == [1, 2]
    assert lps == pytest.approx([-0.1, -0.2])


@pytest.mark.unit
def test_inference_generate_tokens_and_logprobs_pads_partial_content():
    choice = {
        "token_ids": [1, 2, 3],
        "logprobs": {"content": [{"logprob": -0.5}]},
    }
    tids, lps = mod._inference_generate_tokens_and_logprobs(choice)
    assert tids == [1, 2, 3]
    assert lps == pytest.approx([-0.5, 0.0, 0.0])


@pytest.mark.unit
def test_inference_generate_tokens_and_logprobs_empty_when_not_int_list():
    tids, lps = mod._inference_generate_tokens_and_logprobs({"token_ids": ["a"]})
    assert tids == []
    assert lps == []


@pytest.mark.unit
@pytest.mark.parametrize(
    ("tokens", "logprobs", "expected_lps"),
    [
        ([1, 2], [-0.1, -0.2], [-0.1, -0.2]),
        ([1, 2, 3], [-0.1, -0.2, -0.3, -0.4], [-0.1, -0.2, -0.3]),
        ([1, 2, 3], [-0.1], [-0.1, 0.0, 0.0]),
        ([], [1.0], []),
    ],
)
def test_align_engine_tokens_and_logprobs(tokens, logprobs, expected_lps):
    out_toks, out_lps = mod._align_engine_tokens_and_logprobs(tokens, logprobs)
    assert out_toks == tokens
    assert out_lps == expected_lps


@pytest.mark.unit
def test_warn_if_sampling_filters_may_affect_logprobs(caplog):
    caplog.set_level(logging.WARNING, logger=mod.logger.name)
    mod._warn_if_sampling_filters_may_affect_logprobs({"top_p": 0.9, "top_k": -1})
    assert "processed_logprobs may make rollout logprobs differ" in caplog.text


@pytest.mark.unit
def test_warn_if_sampling_filters_may_affect_logprobs_skips_defaults(caplog):
    caplog.set_level(logging.WARNING, logger=mod.logger.name)
    mod._warn_if_sampling_filters_may_affect_logprobs({"top_p": 1.0, "top_k": -1})
    mod._warn_if_sampling_filters_may_affect_logprobs({"top_p": 1.0, "top_k": 0})
    assert caplog.text == ""


@pytest.mark.unit
def test_build_inference_sampling_params_maps_rollout_fields():
    sp = mod._build_inference_sampling_params(
        {
            "max_new_tokens": 16,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "stop": ["</s>"],
            "stop_token_ids": [2],
            "seed": 42,
            "skip_special_tokens": False,
        }
    )
    assert sp["max_tokens"] == 16
    assert sp["temperature"] == 0.7
    assert sp["top_p"] == 0.9
    assert sp["top_k"] == 40
    assert sp["stop"] == ["</s>"]
    assert sp["stop_token_ids"] == [2]
    assert sp["seed"] == 42
    assert sp["skip_special_tokens"] is False
    assert sp["logprobs"] == 1


@pytest.mark.unit
def test_build_inference_sampling_params_forwards_disabled_top_k():
    sp = mod._build_inference_sampling_params({"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
    assert sp["top_k"] == -1


@pytest.mark.unit
def test_mm_render_response_to_generate_body_flat_dict():
    body = mod._mm_render_response_to_generate_body(
        {"token_ids": [1, 2], "features": {"x": 1}},
        "model-a",
    )
    assert body["token_ids"] == [1, 2]
    assert body["model"] == "model-a"
    assert body["features"] == {"x": 1}


@pytest.mark.unit
def test_mm_render_response_to_generate_body_engine_prompts_list():
    body = mod._mm_render_response_to_generate_body(
        [
            {"messages": []},
            [{"prompt_token_ids": [5, 6], "multi_modal_data": {"img": 1}}],
        ],
        "model-b",
    )
    assert body == {
        "token_ids": [5, 6],
        "model": "model-b",
        "features": '{"img": 1}',
    }


@pytest.mark.unit
def test_mm_render_response_to_generate_body_invalid_shape():
    with pytest.raises(ValueError, match="unexpected JSON shape"):
        mod._mm_render_response_to_generate_body({"bad": True}, "m")


@pytest.mark.unit
def test_router_worker_urls_workers_endpoint(monkeypatch):
    monkeypatch.setattr(mod, "get", AsyncMock(return_value={"workers": [{"url": "http://w1"}, {"url": "http://w2"}]}))
    args = Namespace(vllm_router_ip="127.0.0.1", vllm_router_port=7000)
    urls = asyncio.run(mod._router_worker_urls(args))
    assert urls == ["http://w1", "http://w2"]


@pytest.mark.unit
def test_router_worker_urls_falls_back_to_list_workers(monkeypatch):
    async def fake_get(url: str):
        if url.endswith("/workers"):
            raise RuntimeError("not found")
        return {"urls": ["http://w3"]}

    monkeypatch.setattr(mod, "get", fake_get)
    args = Namespace(vllm_router_ip="127.0.0.1", vllm_router_port=7000)
    urls = asyncio.run(mod._router_worker_urls(args))
    assert urls == ["http://w3"]


@pytest.mark.unit
def test_resume_vllm_workers_posts_to_each_url(monkeypatch):
    post_mock = AsyncMock(return_value={"ok": True})
    monkeypatch.setattr(mod, "post", post_mock)
    asyncio.run(mod._resume_vllm_workers(["http://a/", "http://b"]))
    assert post_mock.await_count == 2
    endpoints = [call.args[0] for call in post_mock.await_args_list]
    assert endpoints == ["http://a/resume", "http://b/resume"]


@pytest.mark.unit
def test_resume_vllm_workers_noop_on_empty_urls():
    asyncio.run(mod._resume_vllm_workers([]))


@pytest.mark.unit
def test_prepare_prompt_ids_reuses_tokens_with_multimodal_train_inputs():
    sample = Sample(
        prompt="hi",
        tokens=[9, 8, 7],
        multimodal_inputs={"images": ["img"]},
        multimodal_train_inputs={"pixel_values": [[1.0]]},
    )
    assert mod._prepare_prompt_ids(sample, _FakeTokenizer(), _FakeProcessor()) == [9, 8, 7]


@pytest.mark.unit
def test_base_dataset_prompt_ids_multimodal_via_processor():
    sample = Sample(prompt="hi", multimodal_inputs={"images": ["img"]})
    assert mod._base_dataset_prompt_ids(sample, _FakeTokenizer(), _FakeProcessor()) == [10, 20, 30]


@pytest.mark.unit
def test_get_model_url_without_named_routers():
    args = Namespace(vllm_router_ip="10.0.0.3", vllm_router_port=9000)
    assert mod.get_model_url(args, "any") == "http://10.0.0.3:9000/inference/v1/generate"


@pytest.mark.unit
def test_vllm_meta_from_generate_choice_defaults_to_stop():
    meta = mod._vllm_meta_from_generate_choice(Namespace(), {}, None)
    assert meta == {"finish_reason": {"type": "stop"}}


@pytest.mark.unit
def test_apply_vllm_routed_experts_disabled_or_missing():
    sample = Sample(tokens=[1, 2, 3])
    mod._apply_vllm_routed_experts(Namespace(use_rollout_routing_replay=False), sample, {})
    assert sample.rollout_routed_experts is None
    with pytest.raises(RuntimeError, match="routing replay"):
        mod._apply_vllm_routed_experts(Namespace(use_rollout_routing_replay=True), sample, {})


@pytest.mark.unit
def test_apply_vllm_routed_experts_skips_bad_shape():
    arr = np.zeros((2, 2), dtype=np.int32)
    sample = Sample(tokens=[1, 2, 3])
    args = Namespace(use_rollout_routing_replay=True)
    with pytest.raises(RuntimeError, match="routing replay"):
        mod._apply_vllm_routed_experts(args, sample, {"routed_experts": _encode_routed(arr)})


@pytest.mark.unit
def test_apply_vllm_routed_experts_trims_when_too_many_rows():
    # New contract: require exact rows, no trimming.
    arr = np.zeros((9, 2, 1), dtype=np.int32)
    sample = Sample(tokens=[1, 2, 3])
    args = Namespace(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1)
    with pytest.raises(RuntimeError, match="rows"):
        mod._apply_vllm_routed_experts(args, sample, {"routed_experts": _encode_routed(arr)})


@pytest.mark.unit
def test_apply_vllm_routed_experts_raises_when_too_few_rows():
    arr = np.zeros((1, 2, 1), dtype=np.int32)
    sample = Sample(tokens=[1, 2, 3])
    args = Namespace(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1)
    with pytest.raises(RuntimeError, match="routing replay"):
        mod._apply_vllm_routed_experts(args, sample, {"routed_experts": _encode_routed(arr)})


@pytest.mark.unit
def test_apply_vllm_routed_experts_skips_layer_topk_mismatch():
    arr = np.zeros((2, 3, 4), dtype=np.int32)
    sample = Sample(tokens=[1, 2, 3])
    args = Namespace(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1)
    with pytest.raises(RuntimeError, match="routing replay"):
        mod._apply_vllm_routed_experts(args, sample, {"routed_experts": _encode_routed(arr)})


@pytest.mark.unit
def test_inference_generate_tokens_empty_list():
    assert mod._inference_generate_tokens_and_logprobs({"token_ids": []}) == ([], [])


@pytest.mark.unit
def test_inference_generate_tokens_without_logprobs_dict():
    tids, lps = mod._inference_generate_tokens_and_logprobs({"token_ids": [1, 2]})
    assert tids == [1, 2]
    assert lps == [0.0, 0.0]


@pytest.mark.unit
def test_build_inference_sampling_params_omits_zero_top_k():
    sp = mod._build_inference_sampling_params({"max_new_tokens": 4, "temperature": 0.0, "top_p": 1.0, "top_k": 0})
    assert "top_k" not in sp


@pytest.mark.unit
def test_mm_render_response_token_ids_alias_and_cache_salt():
    body = mod._mm_render_response_to_generate_body(
        [
            {},
            [{"token_ids": [7, 8], "features": {"f": 1}, "cache_salt": "salt-1"}],
        ],
        "model-c",
    )
    assert body["token_ids"] == [7, 8]
    assert body["features"] == {"f": 1}
    assert body["cache_salt"] == "salt-1"


@pytest.mark.unit
def test_mm_render_response_empty_engine_prompts_raises():
    with pytest.raises(ValueError, match="non-empty engine_prompts"):
        mod._mm_render_response_to_generate_body([{}, []], "m")


@pytest.mark.unit
def test_generate_text_path_updates_sample(patch_generate_state, monkeypatch):
    post_mock = AsyncMock(return_value=_generate_response([50, 51]))
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc")
    result = asyncio.run(
        mod.generate(
            _rollout_args(),
            sample,
            _default_sampling_params(max_new_tokens=8),
        )
    )

    assert result.tokens == [97, 98, 99, 50, 51]
    assert result.response_length == 2
    assert result.rollout_log_probs == pytest.approx([-0.1, -0.2])
    assert result.status == Sample.Status.COMPLETED
    body = post_mock.await_args_list[0].args[1]
    assert body["token_ids"] == [97, 98, 99]
    assert body["sampling_params"]["max_tokens"] == 8


@pytest.mark.unit
def test_generate_partial_continuation_uses_full_tokens(patch_generate_state, monkeypatch):
    post_mock = AsyncMock(return_value=_generate_response([60]))
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc", response="x", tokens=[97, 98, 99, 40, 41])
    asyncio.run(mod.generate(_rollout_args(), sample, _default_sampling_params(max_new_tokens=5)))

    body = post_mock.await_args_list[0].args[1]
    assert body["token_ids"] == [97, 98, 99, 40, 41]
    assert body["sampling_params"]["max_tokens"] == 3


@pytest.mark.unit
def test_generate_truncated_when_continuation_budget_zero(patch_generate_state, monkeypatch):
    post_mock = AsyncMock()
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc", response="done", tokens=[97, 98, 99, 40, 41])
    result = asyncio.run(mod.generate(_rollout_args(), sample, _default_sampling_params(max_new_tokens=2)))

    assert result.status == Sample.Status.TRUNCATED
    post_mock.assert_not_called()


@pytest.mark.unit
def test_generate_consistent_hash_header(patch_generate_state, monkeypatch):
    post_mock = AsyncMock(return_value=_generate_response())
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc", session_id="sess-42")
    asyncio.run(
        mod.generate(
            _rollout_args(router_policy="consistent_hash"),
            sample,
            _default_sampling_params(),
        )
    )

    headers = post_mock.await_args_list[0].kwargs.get("headers")
    assert headers == {"x-session-id": "sess-42"}


@pytest.mark.unit
def test_generate_multimodal_render_then_generate(patch_generate_state, monkeypatch):
    render_resp = {"token_ids": [11, 12]}
    gen_resp = _generate_response([13])

    async def fake_post(url, payload, headers=None, **kwargs):
        if url.endswith("/render"):
            return render_resp
        return gen_resp

    monkeypatch.setattr(mod, "post", fake_post)
    monkeypatch.setattr(mod, "encode_image_for_rollout_engine", lambda _img: "data:image/png;base64,xx")

    sample = Sample(index=0, prompt="look", multimodal_inputs={"images": ["img.png"]})
    result = asyncio.run(mod.generate(_rollout_args(), sample, _default_sampling_params()))

    assert result.response_length == 1
    assert result.tokens[-1] == 13


@pytest.mark.unit
def test_generate_applies_routed_experts(patch_generate_state, monkeypatch):
    # Fake tokenizer yields 3 prompt ids; +2 response => 5 tokens, 4 routing rows.
    routed_rows = np.concatenate(
        [
            np.ones((2, 2, 1), dtype=np.int32),
            np.full((2, 2, 1), 2, dtype=np.int32),
        ],
        axis=0,
    )

    post_mock = AsyncMock(
        return_value={
            "choices": [
                {
                    "token_ids": [50, 51],
                    "finish_reason": "stop",
                    "routed_experts": _encode_routed(routed_rows),
                    "logprobs": {"content": [{}, {}]},
                }
            ],
            "usage": {},
        }
    )
    monkeypatch.setattr(mod, "post", post_mock)

    # _FakeTokenizer encodes up to 3 chars => 3 prompt ids + 2 response = 5 tokens.
    sample = Sample(index=0, prompt="abc")
    asyncio.run(
        mod.generate(
            _rollout_args(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1),
            sample,
            _default_sampling_params(max_new_tokens=4),
        )
    )
    np.testing.assert_array_equal(sample.rollout_routed_experts, routed_rows)
    assert len(sample.tokens) == 5
    assert sample.rollout_routed_experts.shape[0] == len(sample.tokens) - 1


@pytest.mark.unit
def test_generate_and_rm_skips_completed_sample(patch_generate_state, monkeypatch):
    called = False

    async def fake_generate(args, sample, sampling_params):
        nonlocal called
        called = True
        return sample

    monkeypatch.setattr(mod, "generate", fake_generate)
    sample = Sample(
        index=0,
        prompt="p",
        response="done",
        response_length=1,
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )
    result = asyncio.run(mod.generate_and_rm(_rollout_args(), sample, _default_sampling_params()))
    assert result is sample
    assert called is False


@pytest.mark.unit
def test_generate_and_rm_aborted_marks_sample(patch_generate_state, monkeypatch):
    state = _PatchedGenerateState(_rollout_args())
    state.aborted = True
    monkeypatch.setattr(mod, "GenerateState", lambda args: state)

    sample = Sample(index=0, prompt="p")
    result = asyncio.run(mod.generate_and_rm(_rollout_args(), sample, _default_sampling_params()))
    assert result.status == Sample.Status.ABORTED


@pytest.mark.unit
def test_generate_r3_abort_without_routed_experts_does_not_raise(patch_generate_state, monkeypatch):
    post_mock = AsyncMock(
        return_value={
            "choices": [
                {
                    "token_ids": [],
                    "finish_reason": "abort",
                    "logprobs": {"content": []},
                }
            ],
            "usage": {"prompt_tokens": 3, "completion_tokens": 0},
        }
    )
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc")
    result = asyncio.run(
        mod.generate(
            _rollout_args(use_rollout_routing_replay=True),
            sample,
            _default_sampling_params(max_new_tokens=8),
        )
    )
    assert result.status == Sample.Status.ABORTED
    assert result.response_length == 0
    assert result.rollout_routed_experts is None


@pytest.mark.unit
def test_generate_and_rm_custom_generate_path(patch_generate_state, monkeypatch):
    async def custom_generate(args, sample, sampling_params, evaluation=False):
        sample.response = "custom"
        sample.response_length = 1
        sample.tokens = [1, 2, 3]
        sample.reward = 0.5
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(mod, "load_function", lambda _path: custom_generate)
    monkeypatch.setattr(mod, "async_rm", AsyncMock())

    sample = Sample(index=0, prompt="p")
    result = asyncio.run(
        mod.generate_and_rm(
            _rollout_args(custom_generate_function_path="fake.path"),
            sample,
            _default_sampling_params(),
            evaluation=True,
        )
    )
    assert result.response == "custom"
    assert result.reward == 0.5


@pytest.mark.unit
def test_generate_and_rm_group_assigns_session_ids(patch_generate_state, monkeypatch):
    async def fake_generate_and_rm(args, sample, sampling_params, evaluation=False):
        sample.response = "ok"
        sample.response_length = 1
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(mod, "generate_and_rm", fake_generate_and_rm)
    group = [Sample(index=0, prompt="a"), Sample(index=1, prompt="b")]
    result = asyncio.run(mod.generate_and_rm_group(_rollout_args(), group, _default_sampling_params()))
    assert all(s.session_id for s in result)
    assert result[0].session_id != result[1].session_id


@pytest.mark.unit
def test_eval_rollout_passk_requests_do_not_share_session_ids(patch_generate_state, monkeypatch):
    seen_session_ids: list[str | None] = []

    async def fake_generate_and_rm(args, sample, sampling_params, evaluation=False):
        seen_session_ids.append(sample.session_id)
        sample.response = "ok"
        sample.response_length = 1
        sample.reward = 0.0
        sample.status = Sample.Status.COMPLETED
        return sample

    monkeypatch.setattr(mod, "generate_and_rm", fake_generate_and_rm)
    monkeypatch.setattr(mod, "EVAL_PROMPT_DATASET", {})

    args = _rollout_args()
    dataset_cfg = EvalDatasetConfig(name="eval", path="/tmp/eval.jsonl", n_samples_per_eval_prompt=2)
    cache_key = dataset_cfg.cache_key + (args.hf_checkpoint, args.apply_chat_template)
    mod.EVAL_PROMPT_DATASET[cache_key] = type("DummyDataset", (), {"samples": [Sample(prompt="prompt")]})()

    result = asyncio.run(mod.eval_rollout_single_dataset(args, rollout_id=0, dataset_cfg=dataset_cfg))

    assert len(seen_session_ids) == 2
    assert None not in seen_session_ids
    assert len(set(seen_session_ids)) == 2
    assert result[dataset_cfg.name]["samples"][0].session_id != result[dataset_cfg.name]["samples"][1].session_id


@pytest.mark.unit
def test_abort_pauses_workers_and_resumes(patch_generate_state, monkeypatch):
    async def _run_abort():
        state = _PatchedGenerateState(_rollout_args(partial_rollout=False))

        async def done_group():
            return [Sample(index=0, prompt="p", response="x")]

        task = asyncio.create_task(done_group())
        await asyncio.sleep(0)
        state.pendings = {task}
        monkeypatch.setattr(mod, "GenerateState", lambda args: state)
        monkeypatch.setattr(mod, "_router_worker_urls", AsyncMock(return_value=["http://worker/"]))
        pause_mock = AsyncMock(return_value={"ok": True})
        resume_mock = AsyncMock(return_value={"ok": True})
        monkeypatch.setattr(mod, "post", pause_mock)
        monkeypatch.setattr(mod, "_resume_vllm_workers", resume_mock)

        aborted = await mod.abort(_rollout_args(), rollout_id=3)
        assert aborted == []
        pause_mock.assert_awaited()
        assert "pause?mode=abort" in pause_mock.await_args_list[0].args[0]
        resume_mock.assert_awaited_once()

    asyncio.run(_run_abort())
