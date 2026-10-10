"""Sampler heads survive the real rollout-manager, DP and microbatch boundaries."""

import asyncio
import json
import sys
import types
from types import SimpleNamespace

import _cp_dist_helpers  # noqa: F401
import numpy as np
import pytest
import torch
from test_score_centering import args, meta

from vime.observability.rollout_data_utils import tensorize_rollout_data_for_training
from vime.utils.async_utils import AsyncPacer
from vime.utils.types import Sample

NUM_GPUS = 0


def test_score_centering_reads_native_tito_integer_token_ids():
    from vllm.entrypoints.scale_out.token_in_token_out.serving import ServingTokens
    from vllm.logprobs import Logprob

    from vime.rollout.vllm_rollout import _score_centering_metadata

    serving = object.__new__(ServingTokens)
    logprobs = serving._create_tokens_logprobs(
        [42],
        [{42: Logprob(-3.0, 3), 4: Logprob(-0.1, 1), 7: Logprob(-0.2, 2)}],
        num_output_top_logprobs=3,
    )
    metadata, sampled = _score_centering_metadata({"logprobs": logprobs.model_dump()}, [42], top_p=1.0, top_k=2)
    ids, probabilities = metadata["score_centering_topk"]
    np.testing.assert_array_equal(ids, [[4, 7]])
    np.testing.assert_allclose(probabilities, [[-0.1, -0.2]])
    assert sampled is None


@pytest.mark.parametrize("finish_reason", ["abort", "length"])
@pytest.mark.parametrize("return_token_ids", [False, True])
@pytest.mark.parametrize("terminal_only", [False, True])
def test_tito_stream_preserves_empty_finish_reason(finish_reason, return_token_ids, terminal_only):
    serving_module = pytest.importorskip("vllm.entrypoints.scale_out.token_in_token_out.serving")
    outputs = pytest.importorskip("vllm.outputs")
    from vllm.sampling_params import SamplingParams

    serving = object.__new__(serving_module.ServingTokens)
    serving.enable_log_outputs = False
    serving.enable_prompt_tokens_details = False
    serving.enable_per_request_metrics = False
    serving.request_logger = None
    request = SimpleNamespace(
        output_mode="tokens",
        sampling_params=SamplingParams(),
        stream_options=None,
        return_token_ids=return_token_ids,
        _response_mm_placeholders=None,
        kv_transfer_params=None,
    )

    async def results():
        chunks = [([], finish_reason)] if terminal_only else [([42], None), ([], finish_reason)]
        for token_ids, reason in chunks:
            completion = outputs.CompletionOutput(
                index=0, text="", token_ids=token_ids, cumulative_logprob=None, logprobs=None, finish_reason=reason
            )
            yield outputs.RequestOutput("probe", None, [1, 2, 3], None, [completion], reason is not None)

    async def collect():
        return [
            json.loads(chunk[6:])
            async for chunk in serving.serve_tokens_stream_generator(
                request, results(), "probe", "model", SimpleNamespace()
            )
            if chunk.startswith("data: {")
        ]

    chunks = asyncio.run(collect())
    assert chunks[-1]["choices"][0]["finish_reason"] == finish_reason
    assert chunks[-1]["choices"][0]["token_ids"] == []


@pytest.mark.parametrize("aggregate", [False, True])
def test_sampling_mask_logprobs_survive_output_coalescing(aggregate):
    outputs = pytest.importorskip("vllm.outputs")

    def output(token, support, logprobs):
        completion = outputs.CompletionOutput(
            index=0,
            text="",
            token_ids=[token],
            cumulative_logprob=None,
            logprobs=None,
            sampling_mask=outputs.SamplingMask([support], [logprobs]),
        )
        return outputs.RequestOutput("audit", None, None, None, [completion], False)

    first = output(2, [1, 2], [-1.2, -0.4])
    first.add(output(3, [3, 4], [-0.3, -1.4]), aggregate=aggregate)
    mask = first.outputs[0].sampling_mask
    assert mask.token_ids == ([[1, 2], [3, 4]] if aggregate else [[3, 4]])
    assert mask.logprobs == ([[-1.2, -0.4], [-0.3, -1.4]] if aggregate else [[-0.3, -1.4]])


@pytest.fixture(autouse=True)
def no_gpu_server_imports(monkeypatch):
    deployment = types.ModuleType("vime.backends.vllm_utils.deployment")
    deployment.start_rollout_servers = lambda *args: None
    monkeypatch.setitem(sys.modules, deployment.__name__, deployment)
    if "vllm_router" not in sys.modules:
        monkeypatch.setitem(sys.modules, "vllm_router", SimpleNamespace(__version__="0.3.0"))


def test_terminal_spec_metrics_survive_output_coalescing():
    outputs = pytest.importorskip("vllm.outputs")
    metrics = outputs.RequestSpecDecodeMetrics.new(2)
    metrics.observe(num_draft_tokens=2, num_accepted=1)
    completions = [
        outputs.CompletionOutput(index=0, text="", token_ids=[token], cumulative_logprob=None, logprobs=None)
        for token in (2, 3)
    ]
    completions[1].spec_decode_metrics = metrics
    first = outputs.RequestOutput("audit", None, None, None, [completions[0]], False)
    first.add(outputs.RequestOutput("audit", None, None, None, [completions[1]], True), aggregate=True)
    assert first.outputs[0].spec_decode_metrics is metrics


def manager(**overrides):
    from vime.data.batch_builder import BatchBuilder

    result = BatchBuilder(
        args(custom_reward_post_process_path=None, custom_convert_samples_to_train_data_path=None, **overrides)
    )
    result._post_process_rewards = lambda samples: ([1.0] * len(samples), [1.0] * len(samples))
    return result


def samples():
    result = []
    for i in range(2):
        sample = Sample(index=i, tokens=[9])
        sample.append_response_tokens(args(), tokens=[3], log_probs=[-0.5], meta_info=meta())
        result.append(sample)
    return result


def test_topk_training_transport_and_microbatch_order(monkeypatch):
    packed = types.ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = object
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: None
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    from vime.backends.megatron_utils.data import DataIterator

    data = samples()
    data[1].rollout_topk_token_ids[0] = [5, 6, 7]
    batch = manager().convert(data)
    tensorize_rollout_data_for_training(batch)
    assert batch["rollout_topk_token_ids"][0].dtype == torch.int32
    assert batch["rollout_topk_log_probs"][0].dtype == torch.float32
    iterator = DataIterator(batch, micro_batch_indices=[[1], [0]])
    keys = ["rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"]
    assert iterator.get_next(keys)["rollout_topk_token_ids"][0].tolist() == [[5, 6, 7]]
    assert iterator.get_next(keys)["rollout_topk_token_ids"][0].tolist() == [[3, 1, 4]]


@pytest.mark.parametrize("field", ["rollout_topk_token_ids", "rollout_topk_log_probs", "rollout_log_probs"])
def test_missing_sampler_metadata_rejected_by_manager(field):
    data = samples()
    setattr(data[1], field, None)
    with pytest.raises(ValueError, match="Score centering"):
        manager().convert(data)


def test_generate_requests_sampler_topk(monkeypatch):
    from vime.rollout import vllm_rollout as rollout

    a = args(
        hf_checkpoint="model",
        vllm_router_ip="localhost",
        vllm_router_port=1234,
        use_rollout_routing_replay=False,
        ci_test=False,
    )
    monkeypatch.setattr(
        rollout,
        "GenerateState",
        lambda _: SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "x"), processor=None),
    )
    monkeypatch.setattr(rollout, "_prepare_prompt_ids", lambda *_: [9])
    captured = []

    async def post(url, payload, **kwargs):
        captured.append((payload, kwargs))
        return {
            "choices": [
                {
                    "token_ids": [3],
                    "finish_reason": "stop",
                    "logprobs": {
                        "content": [
                            {
                                "logprob": -0.5,
                                "top_logprobs": [
                                    {"token_id": 9, "logprob": -4.0},
                                    {"token_id": 4, "logprob": -3.0},
                                    {"token_id": 1, "logprob": -2.0},
                                    {"token_id": 3, "logprob": -0.5},
                                ],
                            }
                        ]
                    },
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(rollout, "post", post)
    sample = asyncio.run(rollout.generate(a, Sample(prompt="test"), {"max_new_tokens": 8}))
    payload, kwargs = captured[0]
    assert payload["sampling_params"]["logprobs"] == 4
    assert kwargs == {"headers": None}
    assert sample.rollout_topk_token_ids.tolist() == [[3, 1, 4]]


def test_streaming_score_centering_rejected():
    from vime.rollout.vllm_streaming_rollout import generate_streaming

    with pytest.raises(ValueError, match="streaming"):
        asyncio.run(generate_streaming(args(), Sample(), {}))


@pytest.mark.parametrize("returned_rows", [1, 2])
def test_r3_resume_preserves_routes_and_score_centering_heads(monkeypatch, returned_rows):
    import base64
    import io

    from vime.rollout import vllm_rollout as rollout

    configured = args(
        hf_checkpoint="model",
        vllm_router_ip="localhost",
        vllm_router_port=1234,
        use_rollout_routing_replay=True,
        ci_test=False,
        num_layers=2,
        moe_router_topk=2,
    )
    sample = samples()[0]
    sample.status = Sample.Status.ABORTED
    prefix = torch.tensor([[[1, 2], [3, 4]]], dtype=torch.int32)
    sample.rollout_routed_experts = prefix.clone()
    tokenizer = SimpleNamespace(decode=lambda *_args, **_kwargs: "x")
    monkeypatch.setattr(rollout, "GenerateState", lambda _: SimpleNamespace(tokenizer=tokenizer, processor=None))
    monkeypatch.setattr(rollout, "_prepare_prompt_ids", lambda sample, *_: sample.tokens)

    async def post(url, payload, **kwargs):
        assert payload["sampling_params"]["routed_experts_prompt_start"] == 1
        assert payload["token_ids"] == [9, 3]
        capture = io.BytesIO()
        np.save(capture, np.array([[[5, 6], [7, 8]]] * returned_rows, dtype=np.int32))
        head_ids, head_logprobs = meta()["score_centering_topk"]
        return {
            "choices": [
                {
                    "token_ids": [3],
                    "finish_reason": "stop",
                    "routed_experts": base64.b64encode(capture.getvalue()).decode(),
                    "logprobs": {
                        "content": [
                            {
                                "logprob": -0.5,
                                "top_logprobs": [
                                    {"token_id": token_id, "logprob": float(logprob)}
                                    for token_id, logprob in zip(head_ids[0], head_logprobs[0], strict=True)
                                ],
                            }
                        ]
                    },
                }
            ]
        }

    monkeypatch.setattr(rollout, "post", post)
    if returned_rows == 2:
        with pytest.raises(ValueError, match="element count"):
            asyncio.run(rollout.generate(configured, sample, {"max_new_tokens": 8}))
        assert torch.equal(sample.materialize_rollout_routed_experts(), prefix)
    else:
        result = asyncio.run(rollout.generate(configured, sample, {"max_new_tokens": 8}))
        assert result.tokens == [9, 3, 3]
        assert torch.equal(result.materialize_rollout_routed_experts()[:1], prefix)
        assert result.materialize_rollout_routed_experts()[1:].flatten().tolist() == [5, 6, 7, 8]
        assert result.rollout_topk_token_ids.tolist() == [[3, 1, 4]] * 2


@pytest.mark.parametrize("transport", ["object-store", "nixl"])
def test_dp_transport_keeps_heads_aligned(monkeypatch, transport):
    from vime.data import batch_builder as rollout

    mgr = manager(rollout_data_transport=transport, global_batch_size=2)
    mgr.train_parallel_config = {"dp_size": 2}
    monkeypatch.setattr(rollout, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    captured = []

    def put(data, **kwargs):
        captured.append(kwargs)
        return data

    monkeypatch.setattr(rollout.ray, "put", put)
    data = samples()
    data[1].rollout_topk_token_ids[0] = [5, 6, 7]
    refs = mgr.split_by_dp(mgr.convert(data))
    assert refs[0].inner["rollout_topk_token_ids"][0].tolist() == [[5, 6, 7]]
    assert refs[1].inner["rollout_topk_token_ids"][0].tolist() == [[3, 1, 4]]
    assert captured == ([{"_tensor_transport": "nixl"}] * 2 if transport == "nixl" else [{}, {}])


def test_evaluation_preserves_training_score_centering(monkeypatch):
    from contextlib import nullcontext

    from vime.rollout import vllm_rollout as rollout

    a = args(partial_rollout=False, group_rm=True, custom_generate_function_path=None)
    state = SimpleNamespace(
        semaphore=asyncio.Semaphore(1),
        generation_pacer=AsyncPacer(),
        aborted=False,
        active_server_generations=0,
        dp_rank_context=lambda: nullcontext(),
    )
    flags = []
    state_args = []

    def get_state(received):
        state_args.append(received)
        return state

    async def generate(received, sample, params):
        flags.append(received.use_score_centering)
        return sample

    async def hooks(received, sample, **kwargs):
        return sample

    monkeypatch.setattr(rollout, "GenerateState", get_state)
    monkeypatch.setattr(rollout, "generate", generate)
    monkeypatch.setattr(rollout, "apply_rollout_sample_hooks", hooks)

    async def run():
        await rollout.generate_and_rm(a, Sample(), {"temperature": 0}, evaluation=True)
        await rollout.generate_and_rm(a, Sample(), {"temperature": 0.8})

    asyncio.run(run())
    assert flags == [False, True]
    assert all(value is a for value in state_args)
    assert a.use_score_centering


@pytest.mark.parametrize("disk", [False, True])
def test_training_metrics_ignore_sampler_head_payloads(monkeypatch, tmp_path, disk):
    from megatron.core import mpu

    from vime.observability import train_metric_utils as metrics

    for name, value in {
        "get_tensor_model_parallel_rank": 0,
        "is_pipeline_last_stage": True,
        "get_context_parallel_world_size": 1,
        "get_data_parallel_world_size": 1,
    }.items():
        monkeypatch.setattr(mpu, name, lambda *a, _value=value, **kw: _value, raising=False)
    reported = []
    monkeypatch.setattr(metrics, "gather_log_data", lambda name, args, rollout_id, data: reported.append(data))
    batch = manager().convert(samples())
    tensorize_rollout_data_for_training(batch)
    batch.update(total_lengths=[2, 2], global_batch_sizes=[2])
    if disk:
        from straw import SharedFilesystemStore
        from straw.tensor import publish_tensors

        with SharedFilesystemStore(tmp_path, "metrics", codecs=("tensor.v1",)) as store:
            for key in ("rollout_topk_token_ids", "rollout_topk_log_probs"):
                batch[key] = list(
                    publish_tensors(store, {str(i): x for i, x in enumerate(batch[key])}, submission_id=key)
                )
    metrics.log_rollout_data(
        0, args(ci_test=False, log_multi_turn=False, log_passrate=False, log_correct_samples=False), batch
    )
    assert "rollout_topk_token_ids" not in reported[0]
    assert "rollout_topk_log_probs" not in reported[0]
    assert "rollout_log_probs" in reported[0]


@pytest.mark.parametrize("transport", ["object-store", "nixl", "straw"])
def test_exact_top_p_transport_and_microbatch(monkeypatch, tmp_path, transport):
    import numpy as np
    from test_score_centering import top_p_meta

    from vime.data import batch_builder as rollout

    packed = types.ModuleType("megatron.core.packed_seq_params")
    packed.PackedSeqParams = object
    training = types.ModuleType("megatron.training")
    training.get_args = lambda: None
    monkeypatch.setitem(sys.modules, "megatron.core.packed_seq_params", packed)
    monkeypatch.setitem(sys.modules, "megatron.training", training)
    from vime.backends.megatron_utils.data import DataIterator
    from vime.data.tensor import TensorRef
    from vime.data.transport import pack_rollout_payload, seal_rollout_store

    mgr = manager(
        rollout_top_p=0.9, rollout_data_transport=transport, rollout_data_dir=str(tmp_path), global_batch_size=2
    )
    mgr.rollout_id = 0
    mgr.train_parallel_config = {"dp_size": 2}
    monkeypatch.setattr(rollout, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    monkeypatch.setattr(rollout.ray, "put", lambda data, **kwargs: data)
    samples = []
    for i in range(2):
        sample = Sample(index=i, tokens=[9])
        sample.append_response_tokens(
            mgr.args, tokens=[4, 2], log_probs=[float(np.log(0.7)), 0.0], meta_info=top_p_meta()
        )
        if i == 1:
            sample.append_response_tokens(mgr.args, tokens=[8], trainable=False)
        samples.append(sample)
    if transport == "straw":
        samples = pack_rollout_payload(samples, mgr.args, 0).load()
    batch = mgr.convert(samples)
    refs = mgr.split_by_dp(batch)
    shards = [ref.inner.load() if transport == "straw" else ref.inner for ref in refs]
    offsets = shards[0]["rollout_top_p_token_offsets"][0]
    assert (offsets.load() if isinstance(offsets, TensorRef) else offsets).tolist() == [0, 2, 3, 3]
    for shard in shards:
        tensorize_rollout_data_for_training(shard)
        iterator = DataIterator(shard, micro_batch_indices=[[0]])
        data = iterator.get_next(["rollout_top_p_log_probs", "rollout_top_p_token_ids", "rollout_top_p_token_offsets"])
        logps = data["rollout_top_p_log_probs"][0]
        if transport == "straw":
            assert isinstance(logps, TensorRef)
            logps = logps.load()
        assert logps.dtype == torch.float32
        torch.testing.assert_close(logps.exp(), torch.tensor([0.3, 0.7, 1.0]))
        ids = data["rollout_top_p_token_ids"][0]
        assert (ids.load() if isinstance(ids, TensorRef) else ids).tolist() == [1, 4, 2]
    if transport == "straw":
        seal_rollout_store(mgr.args)


def test_generate_requests_complete_top_p_probabilities(monkeypatch):
    from vime.rollout import vllm_rollout as rollout

    a = args(
        hf_checkpoint="model",
        rollout_top_p=0.9,
        vllm_router_ip="localhost",
        vllm_router_port=1234,
        use_rollout_routing_replay=False,
        ci_test=False,
    )
    monkeypatch.setattr(
        rollout,
        "GenerateState",
        lambda _: SimpleNamespace(tokenizer=SimpleNamespace(decode=lambda *_args, **_kwargs: "x"), processor=None),
    )
    monkeypatch.setattr(rollout, "_prepare_prompt_ids", lambda *_: [9])

    async def post(url, payload, **kwargs):
        assert payload["sampling_params"]["logprobs"] == 1
        assert kwargs == {"headers": None}
        return {
            "choices": [
                {
                    "token_ids": [4, 2],
                    "finish_reason": "stop",
                    "sampling_mask": [[1, 4], [2]],
                    "sampling_mask_logprobs": [[float(np.log(0.3)), float(np.log(0.7))], [0.0]],
                    "logprobs": {
                        "content": [
                            {
                                "logprob": float(np.log(0.7)),
                                "top_logprobs": [{"token_id": 4, "logprob": float(np.log(0.7))}],
                            },
                            {
                                "logprob": -2.0,
                                "top_logprobs": [{"token_id": 2, "logprob": -2.0}],
                            },
                        ]
                    },
                }
            ],
            "usage": {},
        }

    monkeypatch.setattr(rollout, "post", post)
    sample = asyncio.run(rollout.generate(a, Sample(prompt="test"), {"max_new_tokens": 8}))
    np.testing.assert_allclose(np.exp(sample.rollout_top_p_log_probs), [0.3, 0.7, 1.0], rtol=1e-6)
    assert sample.rollout_top_p_token_ids.tolist() == [1, 4, 2]


def test_straw_heads_survive_buffer_and_debug_dump_lifetimes(tmp_path):
    from vime.data.tensor import TensorRef
    from vime.data.transport import pack_rollout_payload, seal_rollout_store
    from vime.observability.rollout_data_utils import load_debug_rollout_data, save_debug_rollout_data
    from vime.utils.score_centering import validate_sampler_topk

    a = args(rollout_data_transport="straw", rollout_data_dir=str(tmp_path))
    sample = pack_rollout_payload(samples()[0], a, 1).load()
    assert isinstance(sample.rollout_topk_token_ids, TensorRef)
    sample = pack_rollout_payload(sample, a, 2).load()
    path = str(tmp_path / "debug.pt")
    save_debug_rollout_data(path, [sample], rollout_id=2, evaluation=False)
    seal_rollout_store(a)
    validate_sampler_topk(sample, 3)
    for pack in tmp_path.rglob("*.pack"):
        pack.unlink()
    restored = load_debug_rollout_data(path, rollout_id=2)[0]
    validate_sampler_topk(restored, 3)
    restored.append_response_tokens(a, tokens=[3], log_probs=[-0.5], meta_info=meta())
    assert restored.rollout_topk_token_ids.tolist() == [[3, 1, 4]] * 2


@pytest.mark.parametrize("loss_mask,expected", [(None, 1.5), ([1, 0], 2.0), ([0, 0], None)])
def test_rollout_metrics_read_only_top_p_offsets_from_straw(tmp_path, monkeypatch, loss_mask, expected):
    import numpy as np
    from test_score_centering import top_p_meta

    from vime.data.tensor import TensorRef
    from vime.data.transport import pack_rollout_payload, seal_rollout_store
    from vime.observability import rollout_metrics

    a = args(
        rollout_top_p=0.95,
        rollout_data_transport="straw",
        rollout_data_dir=str(tmp_path),
        rollout_queue_online_gc=True,
        rollout_num_gpus=0,
        log_reward_category=None,
        reward_key=None,
        custom_rollout_log_function_path=None,
        load_debug_rollout_data=None,
        wandb_always_use_train_step=False,
    )
    sample = Sample(tokens=[9], reward=1.0, response="answer")
    sample.append_response_tokens(a, tokens=[4, 2], log_probs=[float(np.log(0.7)), 0.0], meta_info=top_p_meta())
    sample.loss_mask = loss_mask
    sample.status = Sample.Status.COMPLETED
    reported = []
    monkeypatch.setattr(rollout_metrics.logging_utils, "log", lambda args, metrics, **kw: reported.append(metrics))
    rollout_metrics.log_rollout_data(0, a, [sample], None, 1.0)

    restored = pack_rollout_payload([sample], a, 0).load()
    seal_rollout_store(a)
    fields = ("rollout_top_p_token_ids", "rollout_top_p_token_offsets", "rollout_top_p_log_probs")
    refs = {key: getattr(restored[0], key) for key in fields}
    assert all(isinstance(ref, TensorRef) for ref in refs.values())
    load = TensorRef.load
    reads = []

    def load_offsets(ref, **kwargs):
        assert ref.kind == "rollout_top_p_token_offsets", "Logging must not load the full sampler payload"
        reads.append(ref)
        return load(ref, **kwargs)

    monkeypatch.setattr(TensorRef, "load", load_offsets)
    rollout_metrics.log_rollout_data(0, a, restored, None, 1.0)

    assert reported[0] == reported[1]
    key = "rollout/top_p_kept_vocab_per_token"
    if expected is None:
        assert key not in reported[1]
    else:
        assert reported[1][key] == pytest.approx(expected)
    assert reads == [refs["rollout_top_p_token_offsets"]]
    assert all(getattr(restored[0], key) is ref for key, ref in refs.items())


@pytest.mark.parametrize("score_centering", [False, True])
@pytest.mark.parametrize("append", ["model", "tool", "terminal"])
def test_top_p_resume_from_straw(tmp_path, monkeypatch, score_centering, append):
    import numpy as np
    from test_score_centering import top_p_meta

    from vime.data.tensor import TensorRef
    from vime.data.transport import pack_rollout_payload, seal_rollout_store

    a = args(
        rollout_top_p=0.95,
        use_score_centering=score_centering,
        rollout_data_transport="straw",
        rollout_data_dir=str(tmp_path),
        rollout_queue_online_gc=True,
    )
    sample = Sample(tokens=[9])
    sample.append_response_tokens(a, tokens=[4, 2], log_probs=[float(np.log(0.7)), 0.0], meta_info=top_p_meta())
    sample.status = Sample.Status.ABORTED
    original = pack_rollout_payload(sample, a, 0)
    restored = original.load()
    assert isinstance(restored.rollout_top_p_token_ids, TensorRef)
    kwargs = {
        "model": dict(tokens=[4, 2], log_probs=[float(np.log(0.7)), 0.0], meta_info=top_p_meta()),
        "tool": dict(tokens=[8], trainable=False),
        "terminal": dict(tokens=[], meta_info={"finish_reason": {"type": "stop"}}),
    }[append]
    sample.append_response_tokens(a, **kwargs)
    load = TensorRef.load
    prefix_ids, prefix_logps = restored.rollout_top_p_token_ids, restored.rollout_top_p_log_probs

    def load_changed_field(ref, **kwargs):
        if append != "model":
            assert ref.kind == "rollout_top_p_token_offsets", "Unchanged supports must remain shared"
        return load(ref, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(TensorRef, "load", load_changed_field)
        restored.append_response_tokens(a, **kwargs)
    if append != "model":
        assert restored.rollout_top_p_token_ids is prefix_ids
        assert restored.rollout_top_p_log_probs is prefix_logps
    republished = pack_rollout_payload(restored, a, 1).load()
    seal_rollout_store(a)
    assert republished.tokens == sample.tokens
    assert republished.loss_mask == sample.loss_mask
    for key in ("rollout_top_p_token_ids", "rollout_top_p_token_offsets", "rollout_top_p_log_probs"):
        expected, actual = getattr(sample, key), getattr(republished, key)
        if expected is not None:
            torch.testing.assert_close(actual.load(), torch.as_tensor(expected))
    # Continuing one reader must leave the persisted prefix usable by another.
    assert original.load().rollout_top_p_token_offsets.load().tolist() == [0, 2, 3]


@pytest.mark.parametrize("rank", [0, 1])
@pytest.mark.parametrize("allgather", [False, True])
def test_top_p_mask_reads_only_local_cp_support(tmp_path, monkeypatch, rank, allgather):
    from megatron.core import mpu
    from straw import SharedFilesystemStore
    from straw.tensor import TensorRef, publish_tensors

    from vime.backends.megatron_utils.loss import _build_topp_keep_mask

    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 2)
    monkeypatch.setattr(mpu, "get_context_parallel_rank", lambda: rank)
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_rank", lambda: 0, raising=False)
    with SharedFilesystemStore(tmp_path, "cp-mask", codecs=("tensor.v1",)) as store:
        ids, offsets = publish_tensors(
            store,
            {"ids": torch.arange(6, dtype=torch.int32), "offsets": torch.arange(7, dtype=torch.int32)},
            submission_id="support",
        )
    getitem = TensorRef.__getitem__
    reads = []

    def read_support(ref, rows):
        assert ref is ids
        reads.append((rows.start, rows.stop))
        return getitem(ref, rows)

    monkeypatch.setattr(TensorRef, "__getitem__", read_support)
    mask = _build_topp_keep_mask(4, 8, torch.device("cpu"), [ids], [offsets], [8], [6], allgather)
    positions = (
        list(range(rank * 4, rank * 4 + 4)) if allgather else [2 * rank, 2 * rank + 1, 6 - 2 * rank, 7 - 2 * rank]
    )
    expected = torch.ones(4, 8, dtype=torch.bool)
    for row, position in enumerate(positions):
        if 1 <= position <= 6:
            expected[row] = False
            expected[row, position - 1] = True
    torch.testing.assert_close(mask, expected)
    expected_reads = (
        ([(0, 3)] if rank == 0 else [(3, 6)]) if allgather else ([(0, 1), (5, 6)] if rank == 0 else [(1, 3), (3, 5)])
    )
    assert reads == expected_reads


def test_top_p_validation_checks_unvalidated_support_in_chunks(tmp_path, monkeypatch):
    from vime.data.tensor import TensorRef
    from vime.data.transport import pack_rollout_payload, seal_rollout_store
    from vime.utils.score_centering import validate_sampler_top_p

    count = 4097
    a = args(rollout_top_p=0.95, rollout_data_transport="straw", rollout_data_dir=str(tmp_path))
    sample = Sample(
        tokens=[9] + [4] * count,
        response_length=count,
        rollout_log_probs=[0.0] * count,
        rollout_top_p_token_ids=torch.full((count,), 4, dtype=torch.int32),
        rollout_top_p_token_offsets=torch.arange(count + 1, dtype=torch.int32),
        rollout_top_p_log_probs=torch.zeros(count),
        # Interrupted captures remain unvalidated until their continuation is
        # ready for training; publishing a completed sample validates it first.
        status=Sample.Status.ABORTED,
    )
    restored = pack_rollout_payload(sample, a, 0).load()
    seal_rollout_store(a)
    fields = (restored.rollout_top_p_token_ids, restored.rollout_top_p_token_offsets, restored.rollout_top_p_log_probs)
    assert all(not ref.validated for ref in fields)
    load, getitem = TensorRef.load, TensorRef.__getitem__
    reads = []

    def load_offsets(ref, **kwargs):
        assert ref.kind == "rollout_top_p_token_offsets"
        return load(ref, **kwargs)

    def read_chunk(ref, rows):
        reads.append((ref.kind, rows.start, rows.stop))
        return getitem(ref, rows)

    monkeypatch.setattr(TensorRef, "load", load_offsets)
    monkeypatch.setattr(TensorRef, "__getitem__", read_chunk)
    validate_sampler_top_p(*fields, count, tokens=[4] * count, sampled_logps=[0.0] * count)
    assert reads == [
        (key, start, stop)
        for start, stop in ((0, 4096), (4096, count))
        for key in ("rollout_top_p_token_ids", "rollout_top_p_log_probs")
    ]
    # Sharing an unvalidated capture must not skip the sampled-token check.
    with pytest.raises(ValueError, match="sampled token"):
        validate_sampler_top_p(*fields, count, tokens=[4] * (count - 1) + [5], sampled_logps=[0.0] * count)


def test_top_p_published_captures_need_only_metadata_checks(tmp_path, monkeypatch):
    from test_score_centering import args

    from vime.data.tensor import TensorRef
    from vime.data.transport import pack_rollout_payload, seal_rollout_store
    from vime.utils.score_centering import validate_sampler_top_p

    count = 4097
    a = args(rollout_top_p=0.95, rollout_data_transport="straw", rollout_data_dir=str(tmp_path))
    sample = Sample(
        tokens=[9] + [4] * count,
        response_length=count,
        rollout_log_probs=[0.0] * count,
        rollout_top_p_token_ids=torch.full((count,), 4, dtype=torch.int32),
        rollout_top_p_token_offsets=torch.arange(count + 1, dtype=torch.int32),
        rollout_top_p_log_probs=torch.zeros(count),
        status=Sample.Status.COMPLETED,
    )
    restored = pack_rollout_payload(sample, a, 0).load()
    seal_rollout_store(a)
    fields = (restored.rollout_top_p_token_ids, restored.rollout_top_p_token_offsets, restored.rollout_top_p_log_probs)
    assert all(ref.validated for ref in fields)

    # Round-end republication must not read or revalidate immutable payloads.
    def no_payload_read(*args, **kwargs):
        raise AssertionError("validated top-p payload was reread during republication")

    with monkeypatch.context() as guarded:
        for method in ("load", "__getitem__", "validate"):
            guarded.setattr(TensorRef, method, no_payload_read)
        validate_sampler_top_p(*fields, count)
        republished = pack_rollout_payload({"buffer": [restored]}, a, 1)
        assert republished.manifest is not None

    with monkeypatch.context() as guarded:
        for method in ("load", "__getitem__", "validate"):
            guarded.setattr(TensorRef, method, no_payload_read)
        validate_sampler_top_p(*fields, count, tokens=[4] * count, sampled_logps=[0.0] * count)
        with pytest.raises(ValueError, match="align"):
            validate_sampler_top_p(*fields, count, tokens=[4], sampled_logps=[0.0])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
