"""CPU unit tests for ``vime.rollout.vllm_rollout`` helpers and mocked async paths."""

from __future__ import annotations

import asyncio
import base64
import io
import json
import sys
from argparse import Namespace
from contextlib import asynccontextmanager, contextmanager
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock

_tests_root = Path(__file__).resolve().parent
if str(_tests_root) not in sys.path:
    sys.path.insert(0, str(_tests_root))

import _unit_stubs
import httpx
import numpy as np
import pytest

_unit_stubs.install_rollout_optional_stubs()

from vime.rollout import vllm_rollout as mod

NUM_GPUS = 0
from vime.utils.async_utils import AsyncPacer
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


class _PatchedGenerateState:
    """Lightweight GenerateState for unit tests (no HF load)."""

    _instance = None

    def __init__(self, args: Namespace) -> None:
        self.args = args
        self.tokenizer = _FakeTokenizer()
        self.processor = None
        self.semaphore = asyncio.Semaphore(128)
        self.generation_pacer = AsyncPacer()
        self.aborted = False
        self.remaining_batch_size = 0
        self.pendings: set = set()
        self.cancellable_tasks: set = set()
        self.active_server_generations = 0
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
        self.cancellable_tasks = set()
        self.active_server_generations = 0
        self.aborted = False


def _rollout_args(**overrides) -> Namespace:
    base = dict(
        ci_test=False,
        hf_checkpoint="/tmp/model",
        vllm_router_ip="127.0.0.1",
        vllm_router_port=8000,
        rollout_external=False,
        rollout_health_check_timeout=600,
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
    request_metrics: dict[str, float] | None = None,
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
        response["choices"][0]["weight_versions"] = [{"version": weight_version, "start": 0, "end": len(tids)}]
    metrics = dict(request_metrics or {})
    if request_spec_decode_stats is not None:
        metrics["speculative_decoding"] = request_spec_decode_stats
    if sampling_mask is not None:
        response["choices"][0]["sampling_mask"] = sampling_mask
    if metrics:
        response["metrics"] = metrics
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
@pytest.mark.parametrize("finish", ["stop", "length", "abort", "cancelled"])
def test_geo3k_turn_preserves_sampling_metadata(monkeypatch, finish):
    from examples.geo3k_vlm_multi_turn import rollout as geo3k

    monkeypatch.setattr(geo3k, "GenerateState", _PatchedGenerateState)
    monkeypatch.setattr(
        geo3k,
        "post",
        AsyncMock(
            return_value={
                "choices": [
                    {
                        "finish_reason": finish,
                        "token_ids": [65],
                        "logprobs": {"content": [{"logprob": -0.1}]},
                        "sampling_mask": [[65, 66]],
                        "weight_versions": [{"version": "7", "start": 0, "end": 1}],
                    }
                ],
            }
        ),
    )
    sample = Sample(tokens=[1])
    rollout = geo3k._Geo3kRollout(
        _rollout_args(max_turns=2, rollout_top_p=0.9),
        sample,
        {"max_new_tokens": 8, "temperature": 0.8, "top_p": 0.9},
    )
    turn = asyncio.run(rollout._generate_turn(rollout.inference_params))
    rollout._append_generated(turn)
    assert sample.rollout_top_p_token_ids.tolist() == [65, 66]
    assert sample.rollout_top_p_token_offsets.tolist() == [0, 2]
    assert sample.weight_versions == ["7"]
    assert sample.tokens == [1, 65]
    assert sample.loss_mask == [1]
    assert sample.rollout_log_probs == [-0.1]
    assert turn.finish == ("abort" if finish == "cancelled" else finish)


@pytest.mark.unit
@pytest.mark.parametrize(
    "context_limit,max_new_tokens,finish,existing,max_turns,done,expected_budget,expected_tail,expected_status",
    [
        (20, 4, "stop", False, 1, False, 18, [65, 66, 67], "COMPLETED"),
        (None, 6, "stop", False, 1, False, 4, [65, 66, 67], "COMPLETED"),
        (None, 5, "stop", False, 1, False, 3, [65, 66, 67], "TRUNCATED"),
        (None, 4, "stop", False, 1, False, 2, [65, 66, 67], "TRUNCATED"),
        (None, 3, "stop", False, 1, True, 1, [65], "TRUNCATED"),
        (None, 6, "stop", False, 1, True, 4, [65], "COMPLETED"),
        (None, None, "stop", True, 1, False, None, [64, 65, 66, 67], "COMPLETED"),
        (8, 3, "cancelled", True, 1, False, 5, [64, 65], "ABORTED"),
        (None, 6, "length", False, 1, False, 4, [65], "TRUNCATED"),
        (None, 6, "stop", False, 0, False, 4, [], "PENDING"),
        (None, -1, "stop", False, 1, False, -3, [], "TRUNCATED"),
        (-1, 6, "stop", False, 1, False, -3, [], "TRUNCATED"),
    ],
)
def test_geo3k_source_lifecycle(
    monkeypatch,
    context_limit,
    max_new_tokens,
    finish,
    existing,
    max_turns,
    done,
    expected_budget,
    expected_tail,
    expected_status,
):
    from examples.geo3k_vlm_multi_turn import rollout as geo3k

    monkeypatch.setattr(geo3k, "GenerateState", _PatchedGenerateState)
    requests = []

    async def post(url, body, **kwargs):
        requests.append(body)
        if url.endswith("/render"):
            return {"token_ids": [1, 2]}
        assert body["token_ids"] == [1, 2] + ([64] if existing else [])
        if expected_budget is not None:
            assert body["sampling_params"]["max_tokens"] == expected_budget
        return {
            "choices": [{"finish_reason": finish, "token_ids": [65], "logprobs": {"content": [{"logprob": -0.1}]}}]
        }

    monkeypatch.setattr(geo3k, "post", post)
    monkeypatch.setattr(geo3k, "_observation_token_ids", lambda *args, **kwargs: [66, 67])
    steps = []

    def step(text):
        steps.append(text)
        return "feedback", done, {}

    env = SimpleNamespace(step=step, format_observation=lambda text: {"role": "user", "content": text})
    sample = Sample(
        prompt="question",
        tokens=[1, 2, 64] if existing else [],
        response="old" if existing else "",
        response_length=1 if existing else 0,
        loss_mask=[1] if existing else None,
        rollout_log_probs=[-0.2] if existing else None,
        status=Sample.Status.ABORTED if existing else Sample.Status.PENDING,
    )
    rollout = geo3k._Geo3kRollout(
        _rollout_args(max_turns=max_turns, rollout_max_context_len=context_limit),
        sample,
        {"max_new_tokens": max_new_tokens, "stop": "A", "temperature": 0.0, "top_p": 1.0},
    )
    rollout.state.tokenizer.eos_token_id = 99
    rollout.state.tokenizer.convert_tokens_to_ids = lambda token: -1
    asyncio.run(rollout._initialize_prompt())
    assert rollout._remaining_budget == expected_budget
    asyncio.run(rollout._run_turns(env))
    result = rollout._finalize()
    assert result.tokens == [1, 2] + expected_tail
    assert result.response == "".join(chr(token) for token in expected_tail)
    assert result.response_length == len(expected_tail)
    assert result.status == getattr(Sample.Status, expected_status)
    assert result.loss_mask == [0 if token in [66, 67] else 1 for token in expected_tail]
    assert result.rollout_log_probs == [
        -0.2 if token == 64 else -0.1 if token == 65 else 0.0 for token in expected_tail
    ]
    generated = max_turns > 0 and (expected_budget is None or expected_budget > 0)
    assert len(requests) == 1 + int(generated)
    advanced = generated and finish == "stop" and (expected_budget is None or expected_budget > 1)
    assert steps == (["A"] if advanced else [])


@pytest.mark.unit
def test_build_inference_sampling_params_maps_rollout_fields():
    sp = mod._build_inference_sampling_params(
        {
            "max_new_tokens": 16,
            "min_new_tokens": 4,
            "repetition_penalty": 1.2,
            "temperature": 0.7,
            "top_p": 0.9,
            "top_k": 40,
            "stop": ["</s>"],
            "stop_token_ids": [2],
            "seed": 42,
            "skip_special_tokens": False,
            "spaces_between_special_tokens": False,
        }
    )
    assert sp["max_tokens"] == 16
    assert sp["min_tokens"] == 4
    assert sp["repetition_penalty"] == 1.2
    assert sp["temperature"] == 0.7
    assert sp["top_p"] == 0.9
    assert sp["top_k"] == 40
    assert sp["stop"] == ["</s>"]
    assert sp["stop_token_ids"] == [2]
    assert sp["seed"] == 42
    assert sp["skip_special_tokens"] is False
    assert sp["spaces_between_special_tokens"] is False
    assert sp["logprobs"] == 1
    assert sp["detokenize"] is True


@pytest.mark.unit
def test_build_inference_sampling_params_forwards_disabled_top_k():
    sp = mod._build_inference_sampling_params({"max_new_tokens": 8, "temperature": 0.0, "top_p": 1.0, "top_k": -1})
    assert sp["top_k"] == -1
    assert sp["detokenize"] is False


@pytest.mark.unit
def test_inference_generate_tokens_and_logprobs_aligns_partial_content():
    token_ids, log_probs = mod._inference_generate_tokens_and_logprobs(
        {
            "token_ids": [11, 12, 13],
            "logprobs": {"content": [{"logprob": -0.1}, {"logprob": -0.2}]},
        }
    )
    assert token_ids == [11, 12]
    assert log_probs == [-0.1, -0.2]


@pytest.mark.unit
def test_inference_generate_tokens_and_logprobs_rejects_invalid_token_ids():
    assert mod._inference_generate_tokens_and_logprobs({"token_ids": [1, "2"]}) == ([], [])


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
    with pytest.raises(ValueError, match="expected a dict with token_ids"):
        mod._mm_render_response_to_generate_body(
            [
                {"messages": []},
                [{"prompt_token_ids": [5, 6], "multi_modal_data": {"img": 1}}],
            ],
            "model-b",
        )


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
        {"token_ids": [7, 8], "features": {"f": 1}, "cache_salt": "salt-1"},
        "model-c",
    )
    assert body["token_ids"] == [7, 8]
    assert body["features"] == {"f": 1}
    assert body["cache_salt"] == "salt-1"


@pytest.mark.unit
def test_mm_render_response_empty_engine_prompts_raises():
    with pytest.raises(ValueError, match="expected a dict with token_ids"):
        mod._mm_render_response_to_generate_body([{}, []], "m")


@pytest.mark.unit
@pytest.mark.parametrize("legacy_scalar", [False, True])
def test_generate_text_path_updates_sample(patch_generate_state, monkeypatch, legacy_scalar):
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
            request_metrics={
                "queue_time_ms": 100,
                "time_to_first_token_ms": 200,
                "generation_time_ms": 300,
                "tokens_per_second": 20,
                "remote_kv_wait_time_ms": 50,
                "kv_transfer_worker_time_ms": 50,
            },
        )
    )
    if legacy_scalar:
        post_mock.return_value["choices"][0].pop("weight_versions")
        post_mock.return_value["weight_version"] = "step-7"
    monkeypatch.setattr(mod, "post", post_mock)

    sample = Sample(index=0, prompt="abc")
    result = asyncio.run(
        mod.generate(
            _rollout_args(vllm_speculative_config={"method": "mtp"}),
            sample,
            _default_sampling_params(max_new_tokens=8, min_new_tokens=2, repetition_penalty=1.2),
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
    generate_span = next(
        event for event in result.trace["events"] if event["type"] == "span_end" and event["name"] == "vllm_generate"
    )
    assert generate_span["attrs"] == {
        "pd_decode_remote_kv_wait_duration": pytest.approx(0.05),
        "pd_transfer_worker_duration": pytest.approx(0.05),
        "prompt_tokens": 3,
        "completion_tokens": 2,
        "cached_tokens": 0,
        "finish_reason": "stop",
        "queue_time": pytest.approx(0.1),
        "e2e_latency": pytest.approx(0.6),
        "decode_throughput": pytest.approx(20),
    }
    assert not any(
        event["type"] == "span_end" and event["name"] == "vllm_pd_decode_transfer" for event in result.trace["events"]
    )
    body = post_mock.await_args_list[0].args[1]
    assert body["token_ids"] == [97, 98, 99]
    assert body["sampling_params"]["max_tokens"] == 8
    assert body["sampling_params"]["min_tokens"] == 2
    assert body["sampling_params"]["repetition_penalty"] == 1.2


@pytest.mark.unit
@pytest.mark.parametrize(
    ("versions", "expected_staleness"),
    [(["3", "7"], 7), (["7", "3"], 7), (["3", "11"], None), (["step-7"], None), ([], None), (None, None)],
)
def test_native_weight_versions_preserve_staleness(versions, expected_staleness):
    from vime.utils.staleness import sample_staleness

    output = _generate_response([50, 51])
    output["weight_version"] = "ignored-legacy-field"
    output["choices"][0]["weight_versions"] = [
        {"version": version, "start": index, "end": index + 1} for index, version in enumerate(versions or [])
    ] + [{"version": "unused", "start": 2, "end": 2}]
    if versions is None:
        output["choices"][0]["weight_versions"] = None
    sample = Sample()
    sample._apply_meta_info(_rollout_args(), mod._inference_generate_meta_info(output), new_token_count=2)

    assert sample.weight_versions == (versions or [])
    assert sample_staleness(sample, 10) == expected_staleness
    sample._apply_meta_info(_rollout_args(), {"weight_versions": ["9"]}, new_token_count=0)
    assert sample.weight_versions == (versions or [])


@pytest.mark.unit
def test_native_weight_versions_accumulate_across_partial_calls():
    sample = Sample()
    for version, finish_reason in [("3", "abort"), ("7", "stop")]:
        output = _generate_response([50], weight_version=version)
        output["choices"][0]["finish_reason"] = finish_reason
        sample._apply_meta_info(_rollout_args(), mod._inference_generate_meta_info(output), new_token_count=1)
    assert sample.weight_versions == ["3", "7"]
    assert sample.status == Sample.Status.COMPLETED


@pytest.mark.unit
def test_custom_generator_scalar_weight_version_remains_supported():
    sample = Sample()
    sample._apply_meta_info(
        _rollout_args(), {"weight_version": "3", "finish_reason": {"type": "stop"}}, new_token_count=1
    )
    assert sample.weight_versions == ["3"]


@pytest.mark.unit
@pytest.mark.parametrize("terminal_only", [False, True])
@pytest.mark.parametrize("with_encoder", [False, True])
@pytest.mark.parametrize(("mixed_versions", "legacy_scalar"), [(False, False), (True, False), (False, True)])
def test_generate_streaming_records_weight_version(
    patch_generate_state, monkeypatch, terminal_only, with_encoder, mixed_versions, legacy_scalar
):
    from vime.rollout import vllm_streaming_rollout as streaming

    ec_params = {"caller-media": {"peer_host": "encoder", "peer_port": 9000, "metadata": {}}}
    versions = (
        [{"version": "3", "start": 0, "end": 1}, {"version": "7", "start": 1, "end": 2}]
        if mixed_versions
        else [{"version": "step-7", "start": 0, "end": 2}]
    )

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
                    "request_id": "stream-7",
                    "metrics": {
                        "speculative_decoding": {
                            "num_accepted_tokens": 6,
                            "num_draft_tokens": 8,
                            "num_verify_steps": 2,
                        }
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
                    "choices": [
                        {
                            "token_ids": [51],
                            "sampling_mask": [[2, 3, 51]],
                            "finish_reason": "stop",
                            "weight_versions": versions,
                            "logprobs": {"content": [{"logprob": -0.2}]},
                        }
                    ],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                },
                {
                    "request_id": "stream-7",
                    "choices": [],
                    "usage": {"prompt_tokens": 3, "completion_tokens": 2},
                    "metrics": {"queue_time_ms": 100},
                },
            ]
            if terminal_only:
                chunks[1]["choices"][0]["finish_reason"] = None
                chunks[1]["choices"][0].pop("weight_versions")
                chunks.insert(
                    2, {"choices": [{"token_ids": [], "finish_reason": "stop", "weight_versions": versions}]}
                )
            for chunk in chunks:
                chunk["weight_version"] = "step-7" if legacy_scalar else "ignored-legacy-field"
                for choice in chunk.get("choices", []):
                    if legacy_scalar:
                        choice.pop("weight_versions", None)
                    else:
                        choice.setdefault("weight_versions", None)
                yield f"data: {json.dumps(chunk)}"
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, *args, **kwargs):
            if with_encoder:
                assert kwargs["json"]["ec_transfer_params"] == ec_params
            else:
                assert "ec_transfer_params" not in kwargs["json"]
            return FakeStreamResponse()

    monkeypatch.setattr(streaming, "GenerateState", _PatchedGenerateState)
    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())
    if with_encoder:
        monkeypatch.setattr(streaming, "build_multimodal_messages", lambda *_args: [{"role": "user", "content": []}])
        monkeypatch.setattr(streaming, "prime_encoder", AsyncMock(return_value=ec_params))
        monkeypatch.setattr(streaming.http_utils, "post", AsyncMock(return_value={"token_ids": [97, 98, 99]}))

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
    assert result.weight_versions == (["3", "7"] if mixed_versions else ["step-7"])
    assert result.spec_info.spec_accept_token_num == 6
    assert result.spec_info.spec_draft_token_num == 8
    assert result.spec_info.spec_verify_ct == 2
    assert result.status == Sample.Status.COMPLETED

    event = next(
        event
        for event in result.trace["events"]
        if event["type"] == "span_end" and event["name"] == "vllm_inference_generate_stream"
    )
    assert event["attrs"]["vllm_request_id"] == "stream-7"
    assert event["attrs"]["finish_reason"] == "stop"
    assert event["attrs"]["completion_tokens"] == 2
    assert event["attrs"]["queue_time"] == pytest.approx(0.1)


@pytest.mark.unit
@pytest.mark.parametrize("stream_interval", [1, 20, 64])
def test_generate_streaming_preserves_metadata_across_stream_intervals(
    patch_generate_state, monkeypatch, stream_interval
):
    from vime.rollout import vllm_streaming_rollout as streaming

    response_tokens = list(range(11, 76))

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            for start in range(0, len(response_tokens), stream_interval):
                tokens = response_tokens[start : start + stream_interval]
                chunk = _generate_response(tokens, sampling_mask=[[token + 1000] for token in tokens])
                chunk["choices"][0]["logprobs"]["content"] = [{"logprob": -float(token)} for token in tokens]
                chunk["choices"][0]["finish_reason"] = None
                yield f"data: {json.dumps(chunk)}"
            yield 'data: {"choices": [{"token_ids": [], "finish_reason": "stop"}]}'
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeStreamResponse()

    monkeypatch.setattr(streaming, "GenerateState", _PatchedGenerateState)
    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())
    result = asyncio.run(
        streaming.generate_streaming(
            _rollout_args(), Sample(prompt="abc"), _default_sampling_params(max_new_tokens=len(response_tokens))
        )
    )
    assert result.status == Sample.Status.COMPLETED
    assert result.tokens == [97, 98, 99, *response_tokens]
    assert result.response_length == len(response_tokens)
    assert result.rollout_log_probs == [-float(token) for token in response_tokens]
    assert result.rollout_top_p_token_ids.tolist() == [token + 1000 for token in response_tokens]
    assert result.rollout_top_p_token_offsets.tolist() == list(range(len(response_tokens) + 1))


@pytest.mark.unit
def test_generate_streaming_handles_encoded_terminal_metadata(patch_generate_state, monkeypatch):
    from vime.rollout import vllm_streaming_rollout as streaming

    args = _rollout_args(use_rollout_routing_replay=True, num_layers=1, moe_router_topk=1)
    state = _PatchedGenerateState(args)
    state.tokenizer.decode = lambda token_ids, skip_special_tokens: "".join(
        {11: "λ", 12: "" if skip_special_tokens else "<|eot_id|>"}[token_id] for token_id in token_ids
    )
    monkeypatch.setattr(streaming, "GenerateState", lambda _args: state)
    monkeypatch.setattr(streaming, "_prepare_prompt_ids", lambda *_args: [1, 2])
    routed_experts = np.asarray([[[101]], [[102]], [[103]]], dtype=np.int32)
    encoded = io.BytesIO()
    np.save(encoded, routed_experts, allow_pickle=False)

    async def lines():
        for token_id in (11, 12):
            chunk = _generate_response([token_id], sampling_mask=[[token_id + 1000]])
            chunk["choices"][0]["logprobs"]["content"][0]["logprob"] = -(token_id - 10) / 10
            chunk["choices"][0]["finish_reason"] = None
            yield "data: " + json.dumps(chunk)
        yield "data: " + json.dumps(
            {
                "choices": [
                    {
                        "token_ids": [],
                        "finish_reason": "length",
                        "routed_experts": base64.b64encode(encoded.getvalue()).decode("ascii"),
                    }
                ]
            }
        )

    @asynccontextmanager
    async def stream(*_args, **_kwargs):
        yield SimpleNamespace(raise_for_status=lambda: None, aiter_lines=lines)

    monkeypatch.setattr(streaming.http_utils, "_http_client", SimpleNamespace(stream=stream))
    result = asyncio.run(
        streaming.generate_streaming(
            args, Sample(prompt="hello"), _default_sampling_params(max_new_tokens=2, skip_special_tokens=False)
        )
    )
    assert result.status == Sample.Status.TRUNCATED
    assert result.response == "λ<|eot_id|>"
    assert result.tokens == [1, 2, 11, 12]
    assert result.response_length == 2
    assert result.rollout_log_probs == [-0.1, -0.2]
    assert result.rollout_top_p_token_ids.tolist() == [1011, 1012]
    assert result.rollout_top_p_token_offsets.tolist() == [0, 1, 2]
    assert result.rollout_routed_experts.tolist() == routed_experts.tolist()


@pytest.mark.unit
def test_server_abort_keeps_last_observed_prefix(patch_generate_state, monkeypatch):
    from vime.rollout import vllm_streaming_rollout as streaming

    args = _rollout_args()
    state = _PatchedGenerateState(args)
    monkeypatch.setattr(streaming, "GenerateState", lambda _args: state)

    async def lines():
        state.aborted = True
        chunk = _generate_response([120], weight_version="step-7")
        chunk["choices"][0]["finish_reason"] = None
        yield "data: " + json.dumps(chunk)
        raise AssertionError("server-aborted stream should stop after the next observed chunk")

    @asynccontextmanager
    async def stream(*_args, **_kwargs):
        yield SimpleNamespace(raise_for_status=lambda: None, aiter_lines=lines)

    monkeypatch.setattr(streaming.http_utils, "_http_client", SimpleNamespace(stream=stream))
    result = asyncio.run(streaming.generate_streaming(args, Sample(prompt="abc"), _default_sampling_params()))
    assert result.status == Sample.Status.ABORTED
    assert (result.tokens, result.rollout_log_probs, result.response) == ([97, 98, 99, 120], [-0.1], "x")
    assert result.response_length == 1
    assert result.weight_versions == ["step-7"]
    assert not state.cancellable_tasks


@pytest.mark.unit
def test_generate_streaming_stops_at_partial_token_budget(patch_generate_state, monkeypatch):
    from vime.rollout import vllm_streaming_rollout as streaming

    monkeypatch.setattr(streaming, "GenerateState", _PatchedGenerateState)
    monkeypatch.setattr(streaming.http_utils, "_http_client", None)
    sample = Sample(index=0, prompt="abc", tokens=[97, 98, 99, 0], response_length=1)
    sample.status = Sample.Status.ABORTED
    result = asyncio.run(
        streaming.generate_streaming(_rollout_args(), sample, _default_sampling_params(max_new_tokens=1))
    )
    assert result.status == Sample.Status.TRUNCATED
    assert result.tokens == [97, 98, 99, 0]


@pytest.mark.unit
@pytest.mark.parametrize("logprob_count", [0, 1, 2])
def test_generate_streaming_rejects_unexpected_eof(patch_generate_state, monkeypatch, logprob_count):
    from vime.rollout import vllm_streaming_rollout as streaming

    class FakeStreamResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            chunk = _generate_response([50])
            chunk["choices"][0]["finish_reason"] = None
            chunk["choices"][0]["logprobs"]["content"] = [{"logprob": -0.1}] * logprob_count
            yield "data: " + json.dumps(chunk)
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeStreamResponse()

    monkeypatch.setattr(streaming, "GenerateState", _PatchedGenerateState)
    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())

    error, message = (
        (RuntimeError, "without a terminal finish_reason")
        if logprob_count == 1
        else (ValueError, "token IDs and logprob pairs must have matching lengths")
    )
    with pytest.raises(error, match=message):
        asyncio.run(
            streaming.generate_streaming(
                _rollout_args(),
                Sample(index=0, prompt="abc"),
                _default_sampling_params(max_new_tokens=8),
            )
        )


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
@pytest.mark.parametrize(("with_encoder", "encoder_failure"), [(False, False), (True, False), (True, True)])
@pytest.mark.parametrize("modality", ["image_url", "audio_url", "video_url"])
def test_generate_multimodal_render_then_generate(
    patch_generate_state, monkeypatch, with_encoder, encoder_failure, modality
):
    features = {
        "kwargs_data": {"media": ["encoded-original-media"]},
        "mm_hashes": {"media": ["caller-media"]},
        "mm_placeholders": {"media": [{"offset": 0, "length": 1}]},
    }
    render_resp = {"token_ids": [11, 12], "features": features}
    gen_resp = _generate_response([13])
    item = {"type": modality, modality: {"url": "https://example.com/media"}, "uuid": "caller-media"}
    ec_params = {"caller-media": {"peer_host": "encoder", "peer_port": 9000, "metadata": {}}}
    args = _rollout_args(
        vllm_model_encoder_endpoints={"default": ("/tmp/model", ["http://encoder"])} if with_encoder else {}
    )

    async def fake_post(url, payload, headers=None, **kwargs):
        if url.startswith("http://encoder"):
            assert payload["messages"][0]["content"] == [item]
            if encoder_failure:
                raise mod.httpx.ConnectError("encoder unavailable")
            return {"ec_transfer_params": ec_params}
        if url.endswith("/render"):
            assert payload["messages"][0]["content"] == [item]
            return render_resp
        if with_encoder and not encoder_failure:
            assert payload["ec_transfer_params"] == ec_params
        else:
            assert "ec_transfer_params" not in payload
        assert payload["features"]["kwargs_data"] == features["kwargs_data"]
        assert payload["features"]["mm_hashes"] == features["mm_hashes"]
        return gen_resp

    monkeypatch.setattr(mod, "post", fake_post)
    monkeypatch.setattr(mod, "build_multimodal_messages", lambda *_args: [{"role": "user", "content": [item]}])

    sample = Sample(index=0, prompt="look", multimodal_inputs={"images": ["img.png"]})
    result = asyncio.run(mod.generate(args, sample, _default_sampling_params()))

    assert result.response_length == 1
    assert result.tokens[-1] == 13


@pytest.mark.unit
def test_build_multimodal_messages_supports_audio_and_video():
    messages = mod.build_multimodal_messages(
        "describe",
        {"audio": ["https://example.com/audio.wav"], "videos": ["https://example.com/video.mp4"]},
    )
    assert [item["type"] for item in messages[0]["content"]] == ["text", "audio_url", "video_url"]


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
                    "logprobs": {"content": [{"logprob": 0.0}, {"logprob": 0.0}]},
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
def test_generate_continuation_appends_only_new_routed_experts(patch_generate_state, monkeypatch):
    prefix = np.array([[[1], [2]]], dtype=np.int32)
    suffix = np.array([[[3], [4]]], dtype=np.int32)
    sample = Sample(
        index=0,
        prompt="x",
        tokens=[9, 3],
        response_length=1,
        rollout_log_probs=[-0.2],
        status=Sample.Status.ABORTED,
    )
    sample.rollout_routed_experts = prefix.copy()
    post_mock = AsyncMock(
        return_value={
            "choices": [
                {
                    "token_ids": [4],
                    "finish_reason": "stop",
                    "routed_experts": _encode_routed(suffix),
                    "logprobs": {"content": [{"logprob": -0.5}]},
                }
            ],
            "usage": {},
        }
    )
    monkeypatch.setattr(mod, "post", post_mock)

    result = asyncio.run(
        mod.generate(
            _rollout_args(use_rollout_routing_replay=True, num_layers=2, moe_router_topk=1),
            sample,
            _default_sampling_params(),
        )
    )

    assert post_mock.await_args.args[1]["sampling_params"]["routed_experts_prompt_start"] == 1
    np.testing.assert_array_equal(result.materialize_rollout_routed_experts(), np.concatenate((prefix, suffix)))


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
@pytest.mark.parametrize("rollout_external", [False, True])
def test_abort_resumes_only_after_pending_groups_finish(patch_generate_state, monkeypatch, rollout_external):
    state = _PatchedGenerateState(_rollout_args())
    state.active_server_generations = 1
    monkeypatch.setattr(mod, "GenerateState", lambda args: state)

    aborted = asyncio.Event()
    aborted_urls: list[str] = []

    async def fake_get(url):
        return {"workers": [{"url": "http://w0:9000"}]}

    async def abort_servers(urls, *, paused):
        if paused:
            aborted_urls.extend(urls)
            aborted.set()
        else:
            assert not state.pendings

    monkeypatch.setattr(mod, "get", fake_get)
    prune_mock = AsyncMock(return_value=[{"url": "http://w0:9000"}])
    monkeypatch.setattr(mod, "get_live_router_workers", prune_mock)
    monkeypatch.setattr(mod, "set_servers_paused", abort_servers)

    sample = Sample(index=0, prompt="p")

    async def pending_group():
        # Delete-type abort makes the in-flight /generate return on its own.
        await aborted.wait()
        sample.status = Sample.Status.ABORTED
        return [sample]

    async def run_abort():
        state.pendings = {asyncio.create_task(pending_group())}
        return await asyncio.wait_for(
            mod.abort(_rollout_args(rollout_external=rollout_external), rollout_id=0), timeout=5.0
        )

    aborted_samples = asyncio.run(run_abort())

    assert aborted_urls == ["http://w0:9000"]
    assert prune_mock.await_count == (0 if rollout_external else 1)
    assert state.pendings == set()
    # partial_rollout is off by default, so drained groups are discarded, not returned.
    assert aborted_samples == []


@pytest.mark.unit
@pytest.mark.parametrize("aborted", [False, True])
def test_late_paused_request_is_aborted_only_during_rollout_abort(aborted):
    state = SimpleNamespace(active_server_generations=0, aborted=aborted)
    sample = Sample(prompt="p")

    async def rejected():
        response = httpx.Response(
            503,
            json={"error": {"message": "Generation is paused. Please retry after it resumes."}},
            request=httpx.Request("POST", "http://engine/inference/v1/generate"),
        )
        response.raise_for_status()

    async def exercise():
        if aborted:
            assert await mod._run_server_abort_generate(state, sample, rejected()) is sample
            assert sample.status == Sample.Status.ABORTED
        else:
            with pytest.raises(httpx.HTTPStatusError):
                await mod._run_server_abort_generate(state, sample, rejected())
        assert state.active_server_generations == 0

    asyncio.run(exercise())


@pytest.mark.unit
@pytest.mark.parametrize("rollout_external", [False, True])
def test_abort_collects_partial_samples_when_partial_rollout(patch_generate_state, monkeypatch, rollout_external):
    args = _rollout_args(partial_rollout=True, rollout_external=rollout_external)
    state = _PatchedGenerateState(args)
    state.active_server_generations = 1
    monkeypatch.setattr(mod, "GenerateState", lambda a: state)

    aborted = asyncio.Event()

    async def fake_get(url):
        return {"workers": [{"url": "http://w0:9000"}]}

    async def abort_servers(urls, *, paused):
        if paused:
            aborted.set()
        else:
            assert not state.pendings

    monkeypatch.setattr(mod, "get", fake_get)
    prune_mock = AsyncMock(return_value=[{"url": "http://w0:9000"}])
    monkeypatch.setattr(mod, "get_live_router_workers", prune_mock)
    monkeypatch.setattr(mod, "set_servers_paused", abort_servers)

    sample = Sample(index=0, prompt="p")
    sample.response = "partial"
    sample.response_length = 1

    async def pending_group():
        await aborted.wait()
        sample.status = Sample.Status.ABORTED
        return [sample]

    async def run_abort():
        state.pendings = {asyncio.create_task(pending_group())}
        return await asyncio.wait_for(mod.abort(args, rollout_id=7), timeout=5.0)

    aborted_samples = asyncio.run(run_abort())

    assert aborted_samples == [[sample]]
    assert prune_mock.await_count == (0 if rollout_external else 1)
    assert sample.metadata["start_rollout_id"] == 7


@pytest.mark.unit
def test_abort_cancels_request_without_server_abort(patch_generate_state, monkeypatch):
    args = _rollout_args(partial_rollout=True)
    state = _PatchedGenerateState(args)
    monkeypatch.setattr(mod, "GenerateState", lambda a: state)
    get_mock = AsyncMock()
    monkeypatch.setattr(mod, "get", get_mock)

    sample = Sample(index=0, prompt="p")
    sample.response = "partial"
    sample.response_length = 1

    async def request():
        await asyncio.Future()

    async def run():
        generate_task = asyncio.create_task(mod._run_request_abortable_generate(state, sample, request()))
        await asyncio.sleep(0)

        async def group():
            return [await generate_task]

        state.pendings = {asyncio.create_task(group())}
        return await asyncio.wait_for(mod.abort(args, rollout_id=9), timeout=5.0)

    assert asyncio.run(run()) == [[sample]]
    assert sample.status == Sample.Status.ABORTED
    assert sample.metadata["start_rollout_id"] == 9
    get_mock.assert_not_awaited()


@pytest.mark.unit
@pytest.mark.parametrize("unrelated_cancel", [False, True])
def test_stream_cancellation_closes_http_and_preserves_prefix(patch_generate_state, monkeypatch, unrelated_cancel):
    from vime.rollout import vllm_streaming_rollout as streaming

    assert streaming.generate_streaming.abort_mode == "request"
    args = _rollout_args()
    state = _PatchedGenerateState(args)
    monkeypatch.setattr(mod, "GenerateState", lambda args: state)
    monkeypatch.setattr(streaming, "GenerateState", lambda args: state)
    monkeypatch.setattr(mod, "load_function", lambda path: streaming.generate_streaming)
    prefix_seen = asyncio.Event()
    request_closed = asyncio.Event()
    server_started = asyncio.Event()
    server_released = asyncio.Event()

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            request_closed.set()
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            chunk = _generate_response([120], weight_version="step-7", sampling_mask=[[12, 120]])
            chunk["choices"][0]["finish_reason"] = None
            yield f"data: {json.dumps(chunk)}"
            prefix_seen.set()
            await asyncio.Future()

    class FakeClient:
        def stream(self, *args, **kwargs):
            return FakeResponse()

    async def server_generate(args, sample, sampling_params):
        server_started.set()
        await server_released.wait()
        sample.status = Sample.Status.ABORTED
        return sample

    async def abort_servers(urls, *, paused):
        assert urls == ["http://worker:9000"]
        server_released.set()

    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())
    monkeypatch.setattr(mod, "generate", server_generate)
    get_mock = AsyncMock(return_value={"workers": [{"url": "http://worker:9000"}]})
    monkeypatch.setattr(mod, "get", get_mock)
    prune_mock = AsyncMock(return_value=[{"url": "http://worker:9000"}])
    monkeypatch.setattr(mod, "get_live_router_workers", prune_mock)
    monkeypatch.setattr(mod, "set_servers_paused", abort_servers)
    sample = Sample(prompt="abc", generate_function_path="streaming")

    async def exercise():
        request_task = asyncio.create_task(mod.generate_and_rm(args, sample, _default_sampling_params()))
        await prefix_seen.wait()
        if unrelated_cancel:
            state.aborted = True
            request_task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await request_task
            get_mock.assert_not_awaited()
            prune_mock.assert_not_awaited()
        else:
            server_task = asyncio.create_task(mod.generate_and_rm(args, Sample(prompt="abc"), {}))
            await server_started.wait()
            await mod.abort(args, rollout_id=7)
            result, server_result = await asyncio.gather(request_task, server_task)
            assert result is sample
            assert result.status == server_result.status == Sample.Status.ABORTED
            get_mock.assert_not_awaited()
            prune_mock.assert_awaited_once()

    async def run():
        await asyncio.wait_for(exercise(), timeout=5)

    asyncio.run(run())
    assert request_closed.is_set()
    assert sample.tokens == [97, 98, 99, 120]
    assert sample.response == "x"
    assert sample.response_length == 1
    assert sample.rollout_log_probs == [-0.1]
    assert sample.rollout_top_p_token_ids.tolist() == [12, 120]
    assert sample.rollout_top_p_token_offsets.tolist() == [0, 2]
    assert sample.weight_versions == ["step-7"]
    assert not state.cancellable_tasks
    assert state.active_server_generations == 0


@pytest.mark.unit
def test_partial_abort_resumes_only_aborted_siblings(patch_generate_state, monkeypatch):
    from vime.rollout import vllm_streaming_rollout as streaming

    args = _rollout_args(
        partial_rollout=True,
        mask_offpolicy_in_partial_rollout=True,
        custom_generate_function_path="streaming",
    )
    state = _PatchedGenerateState(args)
    monkeypatch.setattr(mod, "GenerateState", lambda args: state)
    monkeypatch.setattr(streaming, "GenerateState", lambda args: state)
    monkeypatch.setattr(mod, "load_function", lambda path: streaming.generate_streaming)
    partial = Sample(
        prompt="abc",
        tokens=[97, 98, 99, 120],
        response="x",
        response_length=1,
        rollout_log_probs=[-0.1],
        weight_versions=["step-3"],
        loss_mask=[1],
        status=Sample.Status.ABORTED,
    )
    terminal = Sample(
        prompt="abc",
        tokens=[97, 98, 99, 122],
        response="z",
        response_length=1,
        rollout_log_probs=[-0.3],
        reward=1.0,
        status=Sample.Status.COMPLETED,
    )
    empty = Sample(status=Sample.Status.ABORTED)
    mixed_group = [terminal, partial]
    payloads = []

    class FakeResponse:
        async def __aenter__(self):
            return self

        async def __aexit__(self, exc_type, exc, traceback):
            return False

        def raise_for_status(self):
            return None

        async def aiter_lines(self):
            yield f"data: {json.dumps(_generate_response([121], weight_version='step-7'))}"
            yield "data: [DONE]"

    class FakeClient:
        def stream(self, method, url, *, json, headers):
            payloads.append(json)
            return FakeResponse()

    monkeypatch.setattr(streaming.http_utils, "_http_client", FakeClient())
    reward = AsyncMock(return_value=2.0)
    monkeypatch.setattr(mod, "async_rm", reward)

    async def exercise():
        state.pendings = {
            asyncio.create_task(asyncio.sleep(0, result=group)) for group in (mixed_group, [terminal], [empty])
        }
        buffered = await mod.abort(args, rollout_id=7)
        assert buffered == [mixed_group]
        assert not state.pendings
        state.aborted = False
        return await mod.generate_and_rm_group(args, buffered[0], _default_sampling_params(max_new_tokens=8))

    async def run():
        return await asyncio.wait_for(exercise(), timeout=5)

    assert asyncio.run(run()) == mixed_group
    assert len(payloads) == 1
    assert payloads[0]["token_ids"] == [97, 98, 99, 120]
    assert payloads[0]["sampling_params"]["max_tokens"] == 7
    assert partial.tokens == [97, 98, 99, 120, 121]
    assert partial.response == "xy"
    assert partial.response_length == 2
    assert partial.rollout_log_probs == [-0.1, -0.1]
    assert partial.weight_versions == ["step-3", "step-7"]
    assert partial.loss_mask == [0, 1]
    assert partial.status == terminal.status == Sample.Status.COMPLETED
    assert partial.reward == 2.0
    assert terminal.tokens == [97, 98, 99, 122]
    assert terminal.response == "z"
    assert terminal.reward == 1.0
    assert partial.metadata["start_rollout_id"] == terminal.metadata["start_rollout_id"] == 7
    reward.assert_awaited_once_with(args, partial)


@pytest.mark.unit
@pytest.mark.parametrize("custom", [False, True])
def test_generation_pacing_preserves_concurrency_cancellation_and_reuse(monkeypatch, custom):
    async def exercise():
        args = _rollout_args(group_rm=True, custom_generate_function_path="test.generate" if custom else None)
        state = _PatchedGenerateState(args)
        state.semaphore = asyncio.Semaphore(128)
        started = []
        first_batch, all_started, finish = asyncio.Event(), asyncio.Event(), asyncio.Event()
        timers = []
        loop = asyncio.get_running_loop()
        call_later = loop.call_later

        def schedule(delay, callback, *args, **kwargs):
            if delay == 0.001:
                timers.append(callback)
            else:
                return call_later(delay, callback, *args, **kwargs)

        async def generate(_args, sample, params, evaluation=False):
            assert params == {"temperature": 0.5}
            assert evaluation == custom
            started.append(sample.index)
            if len(started) == 64:
                first_batch.set()
            if len(started) == 127:
                all_started.set()
            await finish.wait()
            sample.status = Sample.Status.COMPLETED
            return sample

        monkeypatch.setattr(loop, "call_later", schedule)
        monkeypatch.setattr(mod, "GenerateState", lambda _args: state)
        monkeypatch.setattr(mod, "generate", generate)
        monkeypatch.setattr(mod, "load_function", lambda _path: generate)

        def start(index):
            return asyncio.create_task(
                mod.generate_and_rm(args, Sample(index=index), {"temperature": 0.5}, evaluation=custom)
            )

        tasks = [start(index) for index in range(128)]
        await asyncio.wait_for(first_batch.wait(), 1)
        assert started == list(range(64))
        assert len(timers) == 1
        assert not any(task.done() for task in tasks)
        tasks[70].cancel()
        with pytest.raises(asyncio.CancelledError):
            await tasks[70]
        timers.pop()()
        await asyncio.wait_for(all_started.wait(), 1)
        assert started == [index for index in range(128) if index != 70]
        assert state.active_server_generations == 127
        finish.set()
        await asyncio.gather(*(task for index, task in enumerate(tasks) if index != 70))
        assert (await start(128)).status == Sample.Status.COMPLETED
        assert not state.generation_pacer.pending

    asyncio.run(exercise())


@pytest.mark.unit
def test_abort_while_waiting_for_generation_pacer_does_not_start_request(monkeypatch):
    async def exercise():
        args = _rollout_args(group_rm=True)
        state = _PatchedGenerateState(args)
        queued, release = asyncio.Event(), asyncio.Event()
        wait = state.generation_pacer.wait

        async def wait_for_release():
            queued.set()
            await release.wait()
            await wait()

        async def unexpected_generate(*args, **kwargs):
            raise AssertionError("An aborted queued sample must not start generation")

        monkeypatch.setattr(state.generation_pacer, "wait", wait_for_release)
        monkeypatch.setattr(mod, "GenerateState", lambda _args: state)
        monkeypatch.setattr(mod, "generate", unexpected_generate)
        task = asyncio.create_task(mod.generate_and_rm(args, Sample(), {}))
        await queued.wait()
        state.aborted = True
        release.set()
        assert (await task).status == Sample.Status.ABORTED
        assert state.active_server_generations == 0

    asyncio.run(exercise())


@pytest.mark.unit
def test_multi_agent_generate_response_preserves_request_metadata(monkeypatch):
    from examples.multi_agent import agent_system

    class CallableTokenizer(_FakeTokenizer):
        def __call__(self, prompt, add_special_tokens=False):
            return {"input_ids": self.encode(prompt, add_special_tokens=add_special_tokens)}

    args = _rollout_args(
        tokenizer=CallableTokenizer(),
        sampling_params=_default_sampling_params(),
        rollout_max_context_len=32,
        sample=Sample(),
        results_dict={"solver": []},
        vllm_speculative_config={"method": "mtp"},
    )
    post_mock = AsyncMock(
        return_value=_generate_response(
            weight_version="step-7",
            request_spec_decode_stats={"num_accepted_tokens": 6, "num_draft_tokens": 8, "num_verify_steps": 2},
        )
    )
    monkeypatch.setattr(agent_system, "post", post_mock)

    asyncio.run(agent_system.generate_response(args, "abc", "solver"))

    post_mock.assert_awaited_once()
    assert len(args.results_dict["solver"]) == 1
    sample = args.results_dict["solver"][0]
    assert sample.weight_versions == ["step-7"]
    assert sample.spec_info.spec_accept_token_num == 6
    assert sample.spec_info.spec_draft_token_num == 8
    assert sample.spec_info.spec_verify_ct == 2


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
