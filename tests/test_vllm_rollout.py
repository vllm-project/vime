"""CPU unit tests for ``vime.rollout.vllm_rollout`` helpers and mocked async paths."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from argparse import Namespace
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import AsyncMock

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import numpy as np
import pytest

_unit_stubs.install_rollout_optional_stubs()

from vime.rollout import vllm_rollout as mod

NUM_GPUS = 0
from vime.utils.eval_config import EvalDatasetConfig
from vime.utils.types import Sample


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


def _generate_response(
    token_ids: list[int] | None = None,
    weight_version: str | None = None,
    request_spec_decode_stats: dict[str, int] | None = None,
    sampling_mask: list[list[int]] | None = None,
) -> dict:
    tids = token_ids or [50, 51]
    response = {
        "choices": [
            {
                "token_ids": tids,
                "finish_reason": "stop",
                "logprobs": {"content": [{"logprob": -(index + 1) / 10} for index in range(len(tids))]},
            }
        ],
        "usage": {"prompt_tokens": 3, "completion_tokens": len(tids)},
    }
    if weight_version is not None:
        response["weight_version"] = weight_version
    if request_spec_decode_stats is not None:
        response["request_spec_decode_stats"] = request_spec_decode_stats
    if sampling_mask is not None:
        response["choices"][0]["sampling_mask"] = sampling_mask
    return response


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
def test_prepare_prompt_ids_reuses_tokens_with_multimodal_train_inputs():
    sample = Sample(
        prompt="hi",
        tokens=[9, 8, 7],
        multimodal_inputs={"images": ["img"]},
        multimodal_train_inputs={"pixel_values": [[1.0]]},
    )
    assert mod._prepare_prompt_ids(sample, _FakeTokenizer(), _FakeProcessor()) == [9, 8, 7]


@pytest.mark.unit
def test_get_model_url_without_named_routers():
    args = Namespace(vllm_router_ip="10.0.0.3", vllm_router_port=9000)
    assert mod.get_model_url(args, "any") == "http://10.0.0.3:9000/inference/v1/generate"


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
    post_mock = AsyncMock(
        return_value=_generate_response(
            [50, 51],
            weight_version="step-7",
            sampling_mask=[[1, 50], [2, 3, 51]],
            request_spec_decode_stats={
                "num_accepted_draft_tokens": 6,
                "num_draft_tokens": 8,
                "num_spec_steps": 2,
            },
        )
    )
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc")
    result = asyncio.run(
        mod.generate(
            _rollout_args(vllm_speculative_config={"method": "mtp"}),
            sample,
            _default_sampling_params(max_new_tokens=8),
        )
    )

    assert result.tokens == [97, 98, 99, 50, 51]
    assert result.response_length == 2
    assert result.rollout_log_probs == pytest.approx([-0.1, -0.2])
    assert result.rollout_top_p_token_ids.tolist() == [1, 50, 2, 3, 51]
    assert result.rollout_top_p_token_offsets.tolist() == [0, 2, 5]
    assert result.weight_versions == ["step-7"]
    assert result.spec_info.spec_accept_token_num == 6
    assert result.spec_info.spec_draft_token_num == 8
    assert result.spec_info.spec_verify_ct == 2
    assert result.status == Sample.Status.COMPLETED
    body = post_mock.await_args_list[0].args[1]
    assert body["token_ids"] == [97, 98, 99]
    assert body["sampling_params"]["max_tokens"] == 8


@pytest.mark.unit
def test_generate_streaming_records_weight_version(patch_generate_state, monkeypatch):
    from vime.rollout import vllm_streaming_rollout as streaming

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            chunks = [
                {
                    "weight_version": "step-7",
                    "request_spec_decode_stats": {
                        "num_accepted_tokens": 6,
                        "num_draft_tokens": 8,
                        "num_verify_steps": 2,
                    },
                    "choices": [
                        {
                            "token_ids": [50],
                            "sampling_mask": [[1, 50]],
                            "finish_reason": None,
                            "logprobs": {"content": [{"logprob": -0.1}]},
                        }
                    ],
                },
                {
                    "weight_version": "step-7",
                    "choices": [
                        {
                            "token_ids": [51],
                            "sampling_mask": [[2, 3, 51]],
                            "finish_reason": "stop",
                            "logprobs": {"content": [{"logprob": -0.2}]},
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
            ]
            for chunk in chunks:
                yield f"data: {json.dumps(chunk)}"
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeStreamResponse()

    monkeypatch.setattr(streaming, "GenerateState", _PatchedGenerateState)
    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())

    result = asyncio.run(
        streaming.generate_streaming(
            _rollout_args(vllm_speculative_config={"method": "mtp"}),
            Sample(index=0, prompt="abc"),
            _default_sampling_params(max_new_tokens=8),
        )
    )

    assert result.tokens == [97, 98, 99, 50, 51]
    assert result.rollout_log_probs == pytest.approx([-0.1, -0.2])
    assert result.rollout_top_p_token_ids.tolist() == [1, 50, 2, 3, 51]
    assert result.rollout_top_p_token_offsets.tolist() == [0, 2, 5]
    assert result.weight_versions == ["step-7"]
    assert result.spec_info.spec_accept_token_num == 6
    assert result.spec_info.spec_draft_token_num == 8
    assert result.spec_info.spec_verify_ct == 2
    assert result.status == Sample.Status.COMPLETED


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
def test_generate_and_rm_rejects_batched_rm_length_mismatch_without_partial_assignment(
    patch_generate_state, monkeypatch
):
    from vime.rollout import rm_hub

    generated_samples: list[Sample] = []

    async def fake_generate(args, sample, sampling_params):
        sample.response = "a"
        sample.response_length = 1
        sample.tokens = [1]
        sample.reward = None
        sample.status = Sample.Status.COMPLETED

        sibling = Sample(index=1, prompt="p1", status=Sample.Status.COMPLETED)
        sibling.response = "b"
        sibling.response_length = 1
        sibling.tokens = [2]
        sibling.reward = None
        generated_samples[:] = [sample, sibling]
        return generated_samples

    async def short_batched_rm(args, samples, **kwargs):
        assert len(samples) == 2
        return [0.25]

    monkeypatch.setattr(mod, "generate", fake_generate)
    monkeypatch.setattr(rm_hub, "load_function", lambda _path: short_batched_rm)

    with pytest.raises(ValueError, match="returned 1 rewards for 2 samples"):
        asyncio.run(
            mod.generate_and_rm(
                _rollout_args(custom_rm_path="fake.rm"),
                Sample(index=0, prompt="p0"),
                _default_sampling_params(),
            )
        )

    assert [sample.reward for sample in generated_samples] == [None, None]


@pytest.mark.unit
@pytest.mark.parametrize(
    "invalid_rewards,type_name",
    [
        ({"first": 0.25, "second": 0.75}, "dict"),
        ("ab", "str"),
        (b"ab", "bytes"),
    ],
)
def test_generate_and_rm_rejects_deceptive_batched_rm_result_types_without_partial_assignment(
    patch_generate_state, monkeypatch, invalid_rewards, type_name
):
    from vime.rollout import rm_hub

    generated_samples: list[Sample] = []

    async def fake_generate(args, sample, sampling_params):
        sample.response = "a"
        sample.response_length = 1
        sample.tokens = [1]
        sample.reward = None
        sample.status = Sample.Status.COMPLETED

        sibling = Sample(index=1, prompt="p1", status=Sample.Status.COMPLETED)
        sibling.response = "b"
        sibling.response_length = 1
        sibling.tokens = [2]
        sibling.reward = None
        generated_samples[:] = [sample, sibling]
        return generated_samples

    async def invalid_batched_rm(args, samples, **kwargs):
        assert len(samples) == 2
        return invalid_rewards

    monkeypatch.setattr(mod, "generate", fake_generate)
    monkeypatch.setattr(rm_hub, "load_function", lambda _path: invalid_batched_rm)

    with pytest.raises(TypeError, match=f"returned {type_name} instead of an iterable of rewards"):
        asyncio.run(
            mod.generate_and_rm(
                _rollout_args(custom_rm_path="fake.rm"),
                Sample(index=0, prompt="p0"),
                _default_sampling_params(),
            )
        )

    assert [sample.reward for sample in generated_samples] == [None, None]


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
def test_generate_and_rm_group_rejects_batched_rm_length_mismatch_without_partial_assignment(
    patch_generate_state, monkeypatch
):
    from vime.rollout import rm_hub

    async def fake_generate(args, sample, sampling_params):
        sample.response = "ok"
        sample.response_length = 1
        sample.tokens = [sample.index or 0]
        sample.reward = None
        sample.status = Sample.Status.COMPLETED
        return sample

    async def long_batched_rm(args, samples, **kwargs):
        assert len(samples) == 2
        return [0.25, 0.75, 1.0]

    monkeypatch.setattr(mod, "generate", fake_generate)
    monkeypatch.setattr(rm_hub, "load_function", lambda _path: long_batched_rm)

    group = [Sample(index=0, prompt="p0"), Sample(index=1, prompt="p1")]
    with pytest.raises(ValueError, match="returned 3 rewards for 2 samples"):
        asyncio.run(
            mod.generate_and_rm_group(
                _rollout_args(group_rm=True, custom_rm_path="fake.rm"),
                group,
                _default_sampling_params(),
            )
        )

    assert [sample.reward for sample in group] == [None, None]


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
    cache_key = dataset_cfg.cache_key + (
        args.hf_checkpoint,
        args.apply_chat_template,
        None,
        None,
    )
    mod.EVAL_PROMPT_DATASET[cache_key] = type("DummyDataset", (), {"samples": [Sample(prompt="prompt")]})()

    result = asyncio.run(mod.eval_rollout_single_dataset(args, rollout_id=0, dataset_cfg=dataset_cfg))

    assert len(seen_session_ids) == 2
    assert None not in seen_session_ids
    assert len(set(seen_session_ids)) == 2
    assert result[dataset_cfg.name]["samples"][0].session_id != result[dataset_cfg.name]["samples"][1].session_id


@pytest.mark.unit
def test_abort_deletes_inflight_without_pause_resume(patch_generate_state, monkeypatch):
    from vime.backends.vllm_utils import server_control

    state = _PatchedGenerateState(_rollout_args())
    monkeypatch.setattr(mod, "GenerateState", lambda args: state)

    aborted = asyncio.Event()
    posted_paths: list[str] = []

    async def fake_get(url):
        return {"workers": [{"url": "http://w0:9000"}]}

    async def fake_post(url, payload, max_retries=60, headers=None):
        posted_paths.append(url)
        if url.endswith("/abort_requests"):
            aborted.set()
        return {}

    monkeypatch.setattr(mod, "get", fake_get)
    # abort() drives the delete-type sweep through the server_control helper.
    monkeypatch.setattr(server_control, "post", fake_post)

    sample = Sample(index=0, prompt="p")

    async def pending_group():
        # Delete-type abort makes the in-flight /generate return on its own.
        await aborted.wait()
        sample.status = Sample.Status.ABORTED
        return [sample]

    async def run_abort():
        state.pendings = {asyncio.create_task(pending_group())}
        return await asyncio.wait_for(mod.abort(_rollout_args(), rollout_id=0), timeout=5.0)

    aborted_samples = asyncio.run(run_abort())

    # Only /abort_requests is posted -- never /pause or /resume.
    assert posted_paths and all(u.endswith("/abort_requests") for u in posted_paths)
    assert state.pendings == set()
    # partial_rollout is off by default, so drained groups are discarded, not returned.
    assert aborted_samples == []


@pytest.mark.unit
def test_abort_collects_partial_samples_when_partial_rollout(patch_generate_state, monkeypatch):
    from vime.backends.vllm_utils import server_control

    args = _rollout_args(partial_rollout=True)
    state = _PatchedGenerateState(args)
    monkeypatch.setattr(mod, "GenerateState", lambda a: state)

    aborted = asyncio.Event()

    async def fake_get(url):
        return {"workers": [{"url": "http://w0:9000"}]}

    async def fake_post(url, payload, max_retries=60, headers=None):
        if url.endswith("/abort_requests"):
            aborted.set()
        return {}

    monkeypatch.setattr(mod, "get", fake_get)
    monkeypatch.setattr(server_control, "post", fake_post)

    sample = Sample(index=0, prompt="p")
    sample.response = "partial"

    async def pending_group():
        await aborted.wait()
        return [sample]

    async def run_abort():
        state.pendings = {asyncio.create_task(pending_group())}
        return await asyncio.wait_for(mod.abort(args, rollout_id=7), timeout=5.0)

    aborted_samples = asyncio.run(run_abort())

    assert aborted_samples == [[sample]]
    assert sample.metadata["start_rollout_id"] == 7


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
