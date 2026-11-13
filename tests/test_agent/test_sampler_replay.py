"""Agent adapters preserve the sampler's top-p, SC and R3 snapshots."""

import asyncio
import base64
import copy
import io
import sys
from dataclasses import asdict, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from aiohttp import web
from aiohttp.test_utils import TestServer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tests.agent_e2e_helpers import audit_token_records
from tests.test_agent._fakes import FakeTokenizer

from vime.agent.adapters.common import Session, call_vllm_generate
from vime.agent.adapters.openai import OpenAIAdapter
from vime.agent.trajectory import TrajectoryManager
from vime.observability.rollout_metrics import _compute_prefix_cache_metrics, _compute_spec_metrics
from vime.observability.trace_utils import bind_trace
from vime.utils.types import Sample

NUM_GPUS = 0


def capture(top_p, sc, r3, missing=None):
    async def run():
        args = SimpleNamespace(
            rollout_top_p=top_p,
            rollout_temperature=1.0,
            use_score_centering=sc,
            score_centering_top_k=2,
            use_rollout_routing_replay=r3,
            num_layers=2,
            moe_router_topk=2,
        )

        async def reply(request):
            payload = await request.json()
            assert payload["token_ids"] == [51, 52, 53]
            params = payload["sampling_params"]
            assert "return_sampling_mask" not in params
            assert "return_sampling_mask_logprobs" not in params
            assert params["top_p"] == top_p and params.get("top_k", -1) == -1
            assert (params.get("routed_experts_prompt_start") is not None) == r3
            ids = [101, 201, 102, 202]
            logps = np.log([0.7, 0.3, 0.6, 0.4]).astype(np.float32)
            choice = {
                "token_ids": [101, 102],
                "logprobs": {
                    "content": [
                        {"token_id": 101, "logprob": float(logps[0])},
                        {"token_id": 102, "logprob": float(logps[2])},
                    ]
                },
                "finish_reason": "stop",
            }
            if top_p < 1:
                choice["sampling_mask"] = [ids[:2], ids[2:]]
                if sc:
                    choice["sampling_mask_logprobs"] = [logps[:2].tolist(), logps[2:].tolist()]
            elif sc:
                assert params["logprobs"] == 3
                for row, entry in enumerate(choice["logprobs"]["content"]):
                    entry["top_logprobs"] = [
                        {"token_id": token_id, "logprob": float(logprob)}
                        for token_id, logprob in zip(
                            ids[row * 2 : row * 2 + 2], logps[row * 2 : row * 2 + 2], strict=True
                        )
                    ]
            if r3:
                buffer = io.BytesIO()
                np.save(buffer, (np.arange(16, dtype=np.int32) % 4).reshape(4, 2, 2), allow_pickle=False)
                choice["routed_experts"] = base64.b64encode(buffer.getvalue()).decode()
            if missing == "nucleus":
                del choice["sampling_mask"]
            elif missing == "probabilities":
                del choice["sampling_mask_logprobs"]
            elif missing == "routes":
                del choice["routed_experts"]
            return web.json_response({"choices": [choice], "usage": {"completion_tokens": 2}})

        app = web.Application()
        app.router.add_post("/inference/v1/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(), vllm_url=str(server.make_url("")).rstrip("/"), rollout_args=args
            )
            return await call_vllm_generate(
                [51, 52, 53], Session(sampling_defaults={"top_p": top_p, "temperature": 1.0}), {}, adapter=adapter
            )

    return asyncio.run(run())


@pytest.mark.parametrize(
    "top_p,sc,r3",
    [(0.95, False, False), (0.95, True, False), (1.0, True, False), (0.95, True, True), (1.0, True, True)],
)
@pytest.mark.parametrize("truncate", [False, True])
def test_replay_survives_exact_prefix_extensions_and_context_limit(top_p, sc, r3, truncate):
    first = replace(
        capture(top_p, sc, r3),
        spec_info=Sample.SpecInfo(1, 3, 1, 2),
        prefix_cache_info=Sample.PrefixCacheInfo(1, 3),
    )
    replay = copy.deepcopy(first.replay)
    prompt = first.prompt_ids + first.output_ids + [9]
    if r3:
        replay["rollout_routed_experts"] = torch.arange(28, dtype=torch.int32).reshape(7, 2, 2) % 4
    second = replace(first, prompt_ids=prompt, replay=replay)
    manager = TrajectoryManager(fork_threshold_tokens=1024)
    user = {"role": "user", "content": "repair"}
    assistant = {"role": "assistant", "content": "read"}
    manager.record_turn("s", turn=first, prompt_messages=[user], response_message=assistant)
    manager.record_turn(
        "s",
        turn=second,
        prompt_messages=[user, assistant, {"role": "tool", "content": "files"}],
        response_message={"role": "assistant", "content": "fixed"},
    )
    samples = manager.get_trajectory(
        "s", base_sample=Sample(index=4, group_index=2), reward=1, max_sample_tokens=7 if truncate else 0
    )
    assert len(samples) == (2 if r3 else 1)
    assert {(s.index, s.rollout_id, s.group_index, s.reward) for s in samples} == {(4, 4, 2, 1)}
    assert sum(s.spec_info.spec_accept_token_num for s in samples) == 2
    assert sum(s.spec_info.spec_draft_token_num for s in samples) == 6
    assert sum(s.spec_info.completion_token_num for s in samples) == 4
    assert sum(s.prefix_cache_info.total_prompt_tokens for s in samples) == 6
    if not truncate:
        report = audit_token_records(samples, [asdict(first), asdict(second)])
        assert set(report["replay_metadata_fields_verified"]) == set(replay)
    else:
        sample = samples[-1]
        assert sample.response_length == (1 if r3 else 4) and sample.tokens == prompt + [101]
        assert sample.loss_mask == ([1] if r3 else [1, 1, 0, 1])
        if top_p < 1:
            assert sample.rollout_top_p_token_offsets.tolist() == ([0, 2] if r3 else [0, 2, 4, 4, 6])
            assert sample.rollout_top_p_token_ids.tolist() == ([101, 201] if r3 else [101, 201, 102, 202, 101, 201])
            if sc:
                assert len(sample.rollout_top_p_log_probs) == (2 if r3 else 6)
        else:
            assert sample.rollout_topk_token_ids.tolist() == (
                [[101, 201]] if r3 else [[101, 201], [102, 202], [0, 1], [101, 201]]
            )
        if r3:
            assert sample.rollout_routed_experts.shape == (6, 2, 2)


@pytest.mark.parametrize("missing", ["nucleus", "probabilities", "routes"])
def test_missing_requested_replay_metadata_fails_closed(missing):
    with pytest.raises(ValueError):
        capture(0.95, True, True, missing)


def test_merged_ragged_distributions_follow_tokens_and_shared_branches_train_once():
    first = replace(
        capture(0.95, True, False),
        spec_info=Sample.SpecInfo(1, 3, 1, 2),
        prefix_cache_info=Sample.PrefixCacheInfo(1, 3),
    )
    second = replace(
        first,
        prompt_ids=first.prompt_ids + first.output_ids + [9, 8, 7],
        output_ids=[103, 104, 105],
        output_log_probs=[-0.1, -0.2, -0.3],
        spec_info=Sample.SpecInfo(2, 6, 2, 3),
        prefix_cache_info=Sample.PrefixCacheInfo(4, 8),
        replay={
            "rollout_top_p_token_ids": torch.tensor([103, 104, 204, 105], dtype=torch.int32),
            "rollout_top_p_token_offsets": torch.tensor([0, 1, 3, 4], dtype=torch.int32),
            "rollout_top_p_log_probs": torch.tensor([-0.1, -0.2, -1.0, -0.3]),
        },
    )
    third = replace(
        second,
        prompt_ids=first.prompt_ids + first.output_ids + [6, 5],
        prefix_cache_info=Sample.PrefixCacheInfo(4, 7),
    )
    user, assistant = {"role": "user", "content": "repair"}, {"role": "assistant", "content": "inspect"}
    manager = TrajectoryManager(fork_threshold_tokens=0)
    manager.record_turn("s", turn=first, prompt_messages=[user], response_message=assistant)
    for turn, branch in [(second, "left"), (third, "right")]:
        manager.record_turn(
            "s",
            turn=turn,
            prompt_messages=[user, assistant, {"role": "tool", "content": branch}],
            response_message={"role": "assistant", "content": "fixed"},
        )
    samples = manager.get_trajectory("s", base_sample=Sample(index=4, group_index=2), reward=1)
    assert len(samples) == 2
    assert samples[0].loss_mask == [1, 1, 0, 0, 0, 1, 1, 1]
    assert samples[1].loss_mask == [0, 0, 0, 0, 1, 1, 1]
    assert samples[1].rollout_log_probs[:2] == first.output_log_probs
    assert samples[0].rollout_top_p_token_offsets.tolist() == [0, 2, 4, 4, 4, 4, 5, 7, 8]
    assert samples[0].rollout_top_p_token_ids.tolist() == [101, 201, 102, 202, 103, 104, 204, 105]
    assert [s.spec_info for s in samples] == [Sample.SpecInfo(3, 9, 3, 5), Sample.SpecInfo(2, 6, 2, 3)]
    assert _compute_spec_metrics(
        SimpleNamespace(vllm_speculative_config={"method": "eagle"}), samples
    ) == pytest.approx({"spec_accept_rate": 1 / 3, "spec_accept_length": 19 / 12})
    assert _compute_prefix_cache_metrics(samples)["prefix_cache_hit_rate"] == pytest.approx(9 / 18)
    # A serialized worker handoff must retain the counters as well as tokens.
    assert Sample.from_dict(samples[0].to_dict()).spec_info == samples[0].spec_info
    turns = [asdict(t) for t in (first, second, third)]
    report = audit_token_records(samples, turns)
    assert report["sampled_tokens"] == 8
    samples[0].rollout_top_p_log_probs[-1] -= 0.5
    with pytest.raises(ValueError, match="sampler distribution must include the sampled token"):
        audit_token_records(samples, turns)


def _partial_response(meta):
    pairs = meta["output_token_logprobs"]
    choice = {
        "token_ids": [pair[1] for pair in pairs],
        "logprobs": {"content": [{"token_id": token, "logprob": logprob} for logprob, token in pairs]},
        "finish_reason": meta["finish_reason"]["type"],
    }
    if "top_p_token_ids" in meta:
        offsets = meta["top_p_token_offsets"]
        choice["sampling_mask"] = [
            meta["top_p_token_ids"][start:end] for start, end in zip(offsets, offsets[1:], strict=False)
        ]
        if "top_p_log_probs" in meta:
            choice["sampling_mask_logprobs"] = [
                meta["top_p_log_probs"][start:end] for start, end in zip(offsets, offsets[1:], strict=False)
            ]
    if "output_topk_token_ids" in meta:
        width = meta["output_topk_shape"][1]
        for index, entry in enumerate(choice["logprobs"]["content"]):
            entry["top_logprobs"] = [
                {"token_id": token, "logprob": logprob}
                for token, logprob in zip(
                    meta["output_topk_token_ids"][index * width : (index + 1) * width],
                    meta["output_topk_log_probs"][index * width : (index + 1) * width],
                    strict=True,
                )
            ]
    if "routed_experts" in meta:
        buffer = io.BytesIO()
        np.save(buffer, np.asarray(meta["routed_experts"], dtype=np.int32).reshape(-1, 2, 2))
        choice["routed_experts"] = base64.b64encode(buffer.getvalue()).decode()
    return {
        "choices": [choice],
        "weight_version": meta.get("weight_version"),
        "usage": {
            "completion_tokens": meta.get("completion_tokens", len(pairs)),
            "prompt_tokens": meta.get("prompt_tokens", 0),
            "prompt_tokens_details": {"cached_tokens": meta.get("cached_tokens", 0)},
        },
        "metrics": {
            "speculative_decoding": {
                "num_accepted_draft_tokens": meta.get("spec_num_correct_drafts", 0),
                "num_draft_tokens": meta.get("spec_num_proposed_drafts", 0),
                "num_spec_steps": meta.get("spec_verify_ct", 0),
            }
        },
    }


@pytest.mark.parametrize("top_p,r3", [(0.95, False), (1.0, False), (0.95, True)])
def test_partial_turn_continues_exact_tokens_and_preserves_each_policy(top_p, r3):
    async def run():
        args = SimpleNamespace(
            partial_rollout=True,
            vllm_speculative_config={"method": "eagle"},
            rollout_top_p=top_p,
            rollout_temperature=1.0,
            use_score_centering=True,
            score_centering_top_k=2,
            use_rollout_routing_replay=r3,
            num_layers=2,
            moe_router_topk=2,
        )
        requests = []
        resume = asyncio.Event()
        waiting = asyncio.Event()

        async def reply(request):
            payload = await request.json()
            requests.append(payload)
            index = len(requests) - 1
            assert request.headers["x-session-id"] == "kept-sandbox"
            assert payload["token_ids"] == [51, 52, 53] + [101] * (index == 2)
            assert payload["sampling_params"]["max_tokens"] == (3 if index < 2 else 2)
            assert len({r["request_id"] for r in requests}) == len(requests)
            if index == 2:
                waiting.set()
                await resume.wait()
            tokens = [] if index == 0 else [100 + index]
            meta = {
                "finish_reason": {"type": "stop" if index == 2 else "abort"},
                "completion_tokens": len(tokens),
                "output_token_logprobs": [[float(np.log(0.7)), t] for t in tokens],
                "weight_version": str(index),
                "spec_num_correct_drafts": int(index == 1),
                "spec_num_proposed_drafts": 3 if tokens else 0,
                "spec_verify_ct": int(bool(tokens)),
                "prompt_tokens": len(payload["token_ids"]),
                "cached_tokens": index,
            }
            if tokens:
                ids, logps = [tokens[0], 200 + index], np.log([0.7, 0.3])
                if top_p < 1:
                    meta.update(
                        top_p_token_ids=np.asarray(ids, dtype="<i4").tolist(),
                        top_p_token_offsets=np.asarray([0, 2], dtype="<i4").tolist(),
                        top_p_log_probs=np.asarray(logps, dtype="<f4").tolist(),
                    )
                else:
                    meta.update(
                        output_topk_token_ids=np.asarray(ids, dtype="<i4").tolist(),
                        output_topk_log_probs=np.asarray(logps, dtype="<f4").tolist(),
                        output_topk_shape=[1, 2],
                    )
                if r3:
                    assert payload["sampling_params"].get("routed_experts_prompt_start", 0) == (3 if index == 2 else 0)
                    meta["routed_experts"] = np.asarray([index] * (4 if index == 2 else 12), dtype="<i4").tolist()
            return web.json_response(_partial_response(meta))

        app = web.Application()
        app.router.add_post("/inference/v1/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(tokenizer=FakeTokenizer(), vllm_url=str(server.make_url("")), rollout_args=args)
            task = asyncio.create_task(
                call_vllm_generate(
                    [51, 52, 53],
                    Session(sampling_defaults={"top_p": top_p, "max_new_tokens": 3}),
                    {},
                    adapter=adapter,
                    session_id="kept-sandbox",
                )
            )
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                assert not task.done(), "The CLI must not receive an unfinished model turn"
            finally:
                resume.set()
            turn = await asyncio.wait_for(task, 5)
        assert turn.output_ids == [101, 102] and turn.finish_reason == "stop"
        assert turn.output_log_probs == pytest.approx([np.log(0.7)] * 2)
        assert turn.weight_versions == ["1", "2"]
        assert turn.spec_info == Sample.SpecInfo(1, 6, 2, 2)
        assert turn.prefix_cache_info == Sample.PrefixCacheInfo(3, 10)
        if top_p < 1:
            assert turn.replay["rollout_top_p_token_ids"].tolist() == [101, 201, 102, 202]
            assert turn.replay["rollout_top_p_token_offsets"].tolist() == [0, 2, 4]
        else:
            assert turn.replay["rollout_topk_token_ids"].tolist() == [[101, 201], [102, 202]]
        if r3:
            assert turn.replay["rollout_routed_experts"].tolist() == [[[1, 1], [1, 1]]] * 3 + [[[2, 2], [2, 2]]]
        manager = TrajectoryManager()
        manager.record_turn(
            "s",
            turn=turn,
            prompt_messages=[{"role": "user", "content": "fix"}],
            response_message={"role": "assistant", "content": "fixed"},
        )
        samples = manager.get_trajectory("s", base_sample=Sample(index=0), reward=1)
        assert samples[0].weight_versions == ["1", "2"]
        assert samples[0].loss_mask == [1, 1]
        assert _compute_spec_metrics(args, samples) == pytest.approx(
            {"spec_accept_rate": 1 / 6, "spec_accept_length": 1}
        )
        assert _compute_prefix_cache_metrics(samples)["prefix_cache_hit_rate"] == pytest.approx(3 / 10)
        audit_token_records(samples, [asdict(turn)])

    asyncio.run(run())


@pytest.mark.parametrize(
    "partial,budget,expected_calls,finish", [(True, 2, 2, "stop"), (True, 1, 1, "length"), (False, 2, 1, "abort")]
)
def test_partial_turn_budget_and_opt_in(partial, budget, expected_calls, finish):
    async def run():
        calls = []

        async def reply(request):
            assert request.headers["x-session-id"] == "agent-trace-test"
            calls.append(await request.json())
            return web.json_response(
                _partial_response(
                    {
                        "finish_reason": {"type": "abort" if len(calls) == 1 else "stop"},
                        "completion_tokens": 1,
                        "output_token_logprobs": [[-0.5, 101]],
                    }
                )
            )

        app = web.Application()
        app.router.add_post("/inference/v1/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                vllm_url=str(server.make_url("")),
                rollout_args=SimpleNamespace(partial_rollout=partial),
            )
            sample = Sample(index=0)
            turn = await call_vllm_generate(
                [51],
                Session(sampling_defaults={"max_new_tokens": budget}, trace=bind_trace(sample)),
                {},
                adapter=adapter,
                session_id="agent-trace-test",
            )
        assert len(calls) == expected_calls and turn.finish_reason == finish
        assert len(turn.output_ids) == expected_calls
        events = sample.trace["events"]
        starts = [event for event in events if event["type"] == "span_start"]
        ends = [event for event in events if event["type"] == "span_end"]
        assert len(starts) == len(ends) == expected_calls
        assert [event["attrs"]["continuation_tokens"] for event in starts] == list(range(expected_calls))
        assert sum(event["attrs"]["completion_tokens"] for event in ends) == expected_calls
        assert all(event["attrs"]["http_status"] == 200 for event in ends)

    asyncio.run(run())


@pytest.mark.parametrize("error", ["http", "missing_replay"])
def test_partial_turn_does_not_retry_errors_or_train_incomplete_replay(error):
    async def run():
        calls = []

        async def reply(request):
            calls.append(await request.json())
            if len(calls) == 2 and error == "http":
                return web.Response(status=503, text="model unavailable")
            meta = {
                "finish_reason": {"type": "abort" if len(calls) == 1 else "stop"},
                "output_token_logprobs": [[float(np.log(0.7)), 101]],
            }
            if len(calls) == 1:
                meta.update(
                    top_p_token_ids=np.asarray([101, 201], dtype="<i4").tolist(),
                    top_p_token_offsets=np.asarray([0, 2], dtype="<i4").tolist(),
                )
            return web.json_response(_partial_response(meta))

        app = web.Application()
        app.router.add_post("/inference/v1/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                vllm_url=str(server.make_url("")),
                rollout_args=SimpleNamespace(partial_rollout=True),
            )
            with pytest.raises(RuntimeError if error == "http" else ValueError):
                await call_vllm_generate([51], Session(sampling_defaults={"top_p": 0.95}), {}, adapter=adapter)
        assert len(calls) == 2

    asyncio.run(run())


def test_partial_turn_cancel_aborts_current_continuation():
    async def run():
        calls, aborted = [], []
        waiting, release, cancelled = asyncio.Event(), asyncio.Event(), asyncio.Event()

        async def reply(request):
            calls.append(await request.json())
            if len(calls) == 2:
                waiting.set()
                try:
                    await release.wait()
                except asyncio.CancelledError:
                    aborted.append(calls[-1]["request_id"])
                    cancelled.set()
                    raise
            return web.json_response(
                _partial_response(
                    {
                        "finish_reason": {"type": "abort"},
                        "output_token_logprobs": [[-0.5, 101]],
                    }
                )
            )

        app = web.Application()
        app.router.add_post("/inference/v1/generate", reply)
        async with TestServer(app) as server:
            adapter = OpenAIAdapter(
                tokenizer=FakeTokenizer(),
                vllm_url=str(server.make_url("")),
                rollout_args=SimpleNamespace(partial_rollout=True),
            )
            task = asyncio.create_task(call_vllm_generate([51], Session(), {}, adapter=adapter))
            try:
                await asyncio.wait_for(waiting.wait(), 5)
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
                await asyncio.wait_for(cancelled.wait(), 5)
                assert aborted == [calls[1]["request_id"]]
            finally:
                release.set()
                task.cancel()

    asyncio.run(run())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
