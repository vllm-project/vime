"""Tiny CPU GRPO signals, packed replay and complete physical publications."""

import copy
from argparse import Namespace
from uuid import uuid4

import pytest
import torch
import torch.nn.functional as F
from safetensors.torch import save_file

pytest.importorskip("megatron")
pytest.importorskip("vllm_rlt")
from megatron.core.packed_seq_params import PackedSeqParams
from megatron.core.transformer.transformer_config import TransformerConfig
from vllm_rlt import LLM, CacheConfig
from vllm_rlt.models.huginn import HuginnConfig, HuginnForCausalLM
from vllm_rlt.models.nanbeige import NanbeigeConfig, NanbeigeForCausalLM

from vime.backends.megatron_utils.megatron_to_hf import _convert_to_hf_core
from vime.backends.vllm_rlt_utils.engine import NativeEngine
from vime.utils.ppo_utils import compute_policy_loss, get_grpo_returns
from vime.utils.reward_normalization import normalize_rewards
from vime.utils.types import Sample
from vime_plugins.huginn.model import HuginnMegatronModel
from vime_plugins.nanbeige.model import NanbeigeMegatronModel


def model_pair(family, recompute, role="actor"):
    if family == "nanbeige":
        native = NanbeigeForCausalLM(
            NanbeigeConfig(
                vocab_size=11,
                hidden_size=8,
                intermediate_size=16,
                num_hidden_layers=1,
                num_attention_heads=2,
                num_key_value_heads=1,
                head_dim=8,
                num_loops=2,
                max_position_embeddings=32,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
            )
        )
        config = TransformerConfig(
            num_layers=1, hidden_size=8, num_attention_heads=2, kv_channels=8, ffn_hidden_size=16
        )
        actor = NanbeigeMegatronModel(config, copy.deepcopy(native), recompute=recompute, role=role)
    else:
        native = HuginnForCausalLM(
            HuginnConfig(
                n_embd=8,
                n_heads=2,
                n_layers=3,
                n_layers_in_prelude=1,
                n_layers_in_recurrent_block=1,
                n_layers_in_coda=1,
                intermediate_size=16,
                mean_recurrence=2,
                block_size=32,
                vocab_size=11,
                padded_vocab_size=11,
                bos_token_id=None,
                eos_token_id=None,
                pad_token_id=None,
            )
        )
        config = TransformerConfig(num_layers=3, hidden_size=8, num_attention_heads=2, ffn_hidden_size=16)
        actor = HuginnMegatronModel(
            config, copy.deepcopy(native), recompute=recompute, model_revision="tiny-cpu", role=role
        )
    return native, actor


def cpu_engine(native, family):
    engine = NativeEngine()
    engine.args = Namespace(
        seed=42,
        rlt_model_family=family,
        rlt_depth=2,
        rlt_model_revision="tiny-cpu",
        rlt_engine_revision="test",
        rollout_temperature=0.8,
        rollout_max_response_len=2,
    )
    engine.llm = LLM(copy.deepcopy(native), cache_config=CacheConfig(num_blocks=16))
    engine.epoch = uuid4().hex
    engine.paused, engine.ready, engine.committed_digest = True, False, None
    return engine


def publish(actor, engine, family, path, version):
    path.mkdir()
    name = "huginn" if family == "huginn_raven" else family
    state = dict(
        item
        for key, parameter in actor.named_parameters()
        for item in _convert_to_hf_core(None, name, key, parameter.detach())
    )
    if family == "huginn_raven":
        state["freqs_cis"] = actor.freqs_cis
    save_file(state, str(path / "model.safetensors"))
    engine.pause_generation()
    engine.flush_cache()
    assert engine.update_weights_from_disk(str(path), str(version)) == version
    engine.continue_generation()


def samples(start=0):
    return [
        Sample(tokens=list(prompt), group_index=start // 2 + group, index=start + group * 2 + sibling)
        for group, prompt in enumerate(([1, 2], [3]))
        for sibling in range(2)
    ]


def selected_scores(actor, batch):
    # Training may sort by length; keep each completion and its trace together.
    ordered = sorted(batch, key=lambda sample: len(sample.tokens))
    sequences = [torch.tensor(sample.tokens[:-1]) for sample in ordered]
    lengths = torch.tensor(
        [0, *torch.tensor([len(sequence) for sequence in sequences]).cumsum(0).tolist()], dtype=torch.int32
    )
    packed = PackedSeqParams(cu_seqlens_q=lengths, cu_seqlens_kv=lengths, qkv_format="thd")
    logits = actor(
        torch.cat(sequences)[None],
        packed_seq_params=packed,
        recurrent_inputs=[sample.recurrent_trace for sample in ordered],
    )
    result = {}
    for i, sample in enumerate(ordered):
        start = int(lengths[i]) + len(sample.tokens) - sample.response_length - 1
        log_probs = F.log_softmax(logits[0, start : int(lengths[i + 1])] / 0.8, dim=-1)
        tokens = torch.tensor(sample.tokens[-sample.response_length :])
        result[sample.index] = log_probs.gather(1, tokens[:, None])[:, 0]
    return result


@pytest.mark.parametrize("family", ["nanbeige", "huginn_raven"])
@pytest.mark.parametrize("recompute", [False, True])
def test_grouped_signal_packed_replay_publication_and_cpu_restore(family, recompute, tmp_path, record_property):
    torch.manual_seed(42)
    native, actor = model_pair(family, recompute)
    engine = cpu_engine(native, family)
    publish(actor, engine, family, tmp_path / "initial", 1)
    batch = engine.generate(samples(), 0)
    assert len({sample.recurrent_trace.seed for sample in batch}) == 4
    for sample in batch:
        assert sample.recurrent_trace.policy_version == 1
        assert sample.recurrent_trace.decode_depths == [2, 2]
        if family == "huginn_raven":
            assert sample.recurrent_trace.latent_seed == sample.recurrent_trace.seed
    current = selected_scores(actor, batch)
    record_property("scope", "tiny FP32 CPU; no Ray, official model, CUDA graph or GPU E2E")
    record_property("family", family)
    record_property("recompute", recompute)
    record_property(
        "max_selected_logprob_error",
        max(
            (current[sample.index].detach() - torch.tensor(sample.rollout_log_probs)).abs().max().item()
            for sample in batch
        ),
    )
    for sample in batch:
        torch.testing.assert_close(current[sample.index], torch.tensor(sample.rollout_log_probs), atol=3e-6, rtol=3e-5)
    args = Namespace(
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
    )
    advantages = normalize_rewards(args, [0.0, 1.0, 7.0, 7.0])
    assert advantages[0] == -advantages[1] and advantages[1] > 0
    assert advantages[2:] == [0.0, 0.0]
    assert normalize_rewards(args, [10.0, 11.0, -3.0, -3.0]) == advantages
    returns = get_grpo_returns(torch.tensor(advantages), [torch.zeros(sample.response_length) for sample in batch])
    losses = [
        compute_policy_loss(torch.tensor(sample.rollout_log_probs) - current[sample.index], advantage, 0.2, 0.2)[
            0
        ].mean()
        for sample, advantage in zip(batch, returns, strict=True)
    ]
    loss = torch.stack(losses).mean()
    assert torch.isfinite(loss)
    loss.backward()
    assert all(parameter.grad is not None and torch.isfinite(parameter.grad).all() for parameter in actor.parameters())
    assert sum(parameter.grad.square().sum() for parameter in actor.parameters()) > 0
    record_property(
        "mixed_reward_grad_norm", sum(parameter.grad.square().sum() for parameter in actor.parameters()).sqrt().item()
    )
    record_property("mixed_reward_loss", loss.item())
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
    before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    optimizer.step()
    assert any(not torch.equal(before[name], value) for name, value in actor.named_parameters())
    publish(actor, engine, family, tmp_path / "updated", 2)
    checkpoint = tmp_path / "actor.pt"
    torch.save({"actor": actor.state_dict(), "optimizer": optimizer.state_dict(), "next_index": 4}, checkpoint)
    restored = torch.load(checkpoint, weights_only=True)
    _, resumed_actor = model_pair(family, recompute)
    resumed_actor.load_state_dict(restored["actor"])
    resumed_optimizer = torch.optim.SGD(resumed_actor.parameters(), lr=0.01)
    resumed_optimizer.load_state_dict(restored["optimizer"])
    resumed_engine = cpu_engine(native, family)
    publish(resumed_actor, resumed_engine, family, tmp_path / "restored", 2)
    assert resumed_engine.committed_digest == engine.committed_digest
    record_property("publication_digest", engine.committed_digest)
    record_property("cpu_actor_optimizer_restore", "exact")
    continuous = engine.generate(samples(4), 1)
    resumed = resumed_engine.generate(samples(restored["next_index"]), 1)
    for a, b in zip(continuous, resumed, strict=True):
        assert a.tokens == b.tokens and a.rollout_log_probs == b.rollout_log_probs
        assert a.recurrent_trace.seed == b.recurrent_trace.seed
        assert a.recurrent_trace.latent_seed == b.recurrent_trace.latent_seed
    # With no entropy/KL term, uniform group rewards must leave policy weights unchanged.
    optimizer.zero_grad(set_to_none=True)
    scores = selected_scores(actor, continuous)
    zero = normalize_rewards(args, [1.0] * 4)
    assert zero == [0.0] * 4
    zero_loss = torch.stack(
        [
            compute_policy_loss(
                torch.tensor(sample.rollout_log_probs) - scores[sample.index],
                torch.full_like(scores[sample.index], advantage),
                0.2,
                0.2,
            )[0].mean()
            for sample, advantage in zip(continuous, zero, strict=True)
        ]
    ).mean()
    zero_loss.backward()
    assert zero_loss.item() == 0 and all(torch.count_nonzero(p.grad) == 0 for p in actor.parameters())
    record_property("equal_reward_loss", zero_loss.item())
    record_property("equal_reward_grad_norm", 0.0)
    before = {name: value.detach().clone() for name, value in actor.named_parameters()}
    optimizer.step()
    assert all(torch.equal(before[name], value) for name, value in actor.named_parameters())
    engine.close()
    resumed_engine.close()
