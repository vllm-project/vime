"""CPU checks of deferred decoding against dense, causal Ouro equations."""

import math

import pytest
import torch

pytest.importorskip("vllm_rlt")
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from vime_plugins.ouro.deferred_decode import DeferredOuroRunner, generate_deferred

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def make_model(kv_heads=2, loops=4):
    torch.manual_seed(19)
    return OuroForCausalLM(
        OuroConfig(
            vocab_size=31,
            hidden_size=32,
            intermediate_size=48,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=kv_heads,
            head_dim=8,
            max_position_embeddings=64,
            total_ut_steps=loops,
        )
    ).eval()


@torch.inference_mode()
def dense_reference(model, tokens):
    """All four causal planes, without recurrent_prepared or a paged cache."""
    hidden = model.model.embed_tokens(torch.tensor(tokens))
    cos, sin = model.model.rotary_emb(hidden, torch.arange(len(tokens)))
    future = torch.ones(len(tokens), len(tokens), dtype=torch.bool).triu(1)
    levels, planes = [], []
    for _ in range(4):
        layers = []
        for layer in model.model.layers:
            value = layer.input_layernorm(hidden)
            q, k, v = [
                projection(value).view(len(tokens), -1, model.config.head_dim)
                for projection in (layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj)
            ]

            def rotate(x):
                first, second = x.chunk(2, -1)
                return x * cos + torch.cat((-second, first), -1) * sin

            q, k = rotate(q), rotate(k)
            layers.append((k.clone(), v.clone()))
            groups = model.config.num_attention_heads // model.config.num_key_value_heads
            keys, values = k.repeat_interleave(groups, 1), v.repeat_interleave(groups, 1)
            scores = torch.einsum("thd,shd->hts", q, keys) / math.sqrt(model.config.head_dim)
            scores.masked_fill_(future, -torch.inf)
            attention = torch.einsum("hts,shd->thd", scores.softmax(-1), values).reshape(len(tokens), -1)
            hidden = hidden + layer.input_layernorm_2(layer.self_attn.o_proj(attention))
            hidden = hidden + layer.post_attention_layernorm_2(layer.mlp(layer.post_attention_layernorm(hidden)))
        hidden = model.model.norm(hidden)
        levels.append(hidden.clone())
        planes.append(layers)
    return levels, planes


@pytest.mark.parametrize("interval", [1, 2, 4, 8])
@pytest.mark.parametrize("kv_heads", [1, 2, 4])
@pytest.mark.parametrize("defer", [False, True])
@torch.inference_mode()
def test_readouts_and_all_committed_kv_match_dense(interval, kv_heads, defer):
    model = make_model(kv_heads)
    prompts = [[3, 5, 7], [11, 4, 9, 6, 8]]
    prompt_lengths = [len(prompt) for prompt in prompts]
    with DeferredOuroRunner(model, [32, 32], interval=interval, backend="torch", defer=defer) as runner:
        hidden = runner.prefill(prompts)
        for index, prompt in enumerate(prompts):
            expected, _ = dense_reference(model, prompt)
            torch.testing.assert_close(hidden[index], expected[3][-1], atol=2e-5, rtol=2e-5)
        for step in range(9):
            tokens = [3 + step, 17 + step]
            hidden, depths = runner.step([0, 1], tokens)
            for index, token in enumerate(tokens):
                prompts[index].append(token)
                expected, planes = dense_reference(model, prompts[index])
                torch.testing.assert_close(hidden[index], expected[depths[index] - 1][-1], atol=2e-5, rtol=2e-5)
                for depth in range(4):
                    for layer in range(model.config.num_hidden_layers):
                        actual = runner.cache.read(layer, runner.requests[index].request_id, depth)
                        committed = (
                            prompt_lengths[index] + (step + 1) // interval * interval
                            if defer and depth >= 2
                            else len(prompts[index])
                        )
                        for cached, reference in zip(actual, planes[depth][layer], strict=True):
                            assert len(cached) == committed
                            torch.testing.assert_close(cached, reference[: len(cached)], atol=2e-5, rtol=2e-5)
        assert depths == ([4, 4] if 9 % interval == 0 else [2, 2])
    assert runner.cache.num_free_blocks == runner.cache.num_blocks
    assert all(not request.pending_hidden and not request.live for request in runner.requests)


@torch.inference_mode()
def test_staggered_blocks_and_request_release():
    model = make_model()
    with DeferredOuroRunner(model, [32, 32], interval=4, backend="torch") as runner:
        runner.prefill([[3, 5], [7, 9, 11]])
        runner.step([0], [4])
        runner.step([0], [6])
        runner.step([0, 1], [8, 12])
        hidden, depths = runner.step([0, 1], [10, 14])
        assert depths == [4, 2]
        expected, _ = dense_reference(model, [3, 5, 4, 6, 8, 10])
        torch.testing.assert_close(hidden[0], expected[3][-1], atol=2e-5, rtol=2e-5)
        runner.release([0])
        runner.step([1], [16])
        hidden, depths = runner.step([1], [18])
        assert depths == [4]
        expected, _ = dense_reference(model, [7, 9, 11, 12, 14, 16, 18])
        torch.testing.assert_close(hidden[0], expected[3][-1], atol=2e-5, rtol=2e-5)


@torch.inference_mode()
def test_stop_tokens_processed_scores_and_batch_shrink(monkeypatch):
    model = make_model()
    samples = iter([2, 7, 8, 0])
    monkeypatch.setattr(torch, "multinomial", lambda *args, **kwargs: torch.tensor([next(samples)]))
    outputs = generate_deferred(model, [[3, 5], [4, 6]], [42, 43], max_tokens=8, backend="torch")
    assert [output.token_ids for output in outputs] == [[2], [7, 8, 0]]
    assert [output.finish_reason for output in outputs] == ["stop", "stop"]
    assert [output.readout_depths for output in outputs] == [[4], [4, 2, 2]]
    for prompt, output in zip([[3, 5], [4, 6]], outputs, strict=True):
        for index, (token, depth) in enumerate(zip(output.token_ids, output.readout_depths, strict=True)):
            levels, _ = dense_reference(model, prompt + output.token_ids[:index])
            expected = model.coda(levels[depth - 1][-1]).float().log_softmax(-1)[token]
            assert output.logprobs[index] == pytest.approx(float(expected), abs=2e-5)


@torch.inference_mode()
def test_length_limit_and_repeatable_rng():
    model = make_model()
    args = dict(prompts=[[3, 5], [4, 6]], seeds=[42, 43], max_tokens=7, stop_token_ids=(), backend="torch")
    first = generate_deferred(model, **args)
    second = generate_deferred(model, **args)
    assert first == second
    assert all(len(output.token_ids) == 7 and output.finish_reason == "length" for output in first)
    assert all(output.readout_depths == [4, 2, 2, 2, 4, 2, 2] for output in first)


def test_invalid_configuration_and_step_are_rejected():
    with pytest.raises(ValueError, match="four-loop"):
        DeferredOuroRunner(make_model(loops=3), [16], backend="torch")
    with pytest.raises(ValueError, match="interval"):
        DeferredOuroRunner(make_model(), [16], interval=3, backend="torch")
    with DeferredOuroRunner(make_model(), [3], backend="torch") as runner:
        with pytest.raises(ValueError, match="after prefill"):
            runner.step([0], [1])
        runner.prefill([[3, 4, 5]])
        with pytest.raises(ValueError, match="capacity"):
            runner.step([0], [1])
