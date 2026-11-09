"""Tiny CPU three-update pipelines; complete Torch state and fresh engine replay."""

import copy
from time import perf_counter

import pytest
import torch

pytest.importorskip("megatron")
pytest.importorskip("vllm_rlt")
from megatron.core import mpu
from megatron.core.packed_seq_params import PackedSeqParams

from tests.plugins.test_looped_dppo import loss_args
from tests.plugins.test_looped_grpo import cpu_engine, model_pair, publish, samples
from vime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from vime.backends.megatron_utils.loss import (
    get_log_probs_and_entropy,
    get_values,
    policy_loss_function,
    value_loss_function,
)
from vime.utils.ppo_utils import get_grpo_returns, vanilla_gae
from vime.utils.reward_normalization import normalize_rewards


def update(actor, critic, optimizer, critic_optimizer, engine, family, algorithm, path, step):
    publish(actor, engine, family, path / f"before-{step}", step + 1)
    rollout = engine.generate(samples(step * 4), step)
    tokens = [torch.tensor(sample.tokens) for sample in rollout]
    lengths = [len(sequence) for sequence in tokens]
    boundaries = torch.tensor([0, *torch.tensor(lengths).cumsum(0).tolist()], dtype=torch.int32)
    packed = PackedSeqParams(cu_seqlens_q=boundaries, cu_seqlens_kv=boundaries, qkv_format="thd")
    inputs = dict(
        input_ids=torch.cat(tokens)[None],
        packed_seq_params=packed,
        recurrent_inputs=[s.recurrent_trace for s in rollout],
    )
    args = loss_args(algorithm)
    batch = dict(
        unconcat_tokens=tokens,
        total_lengths=lengths,
        response_lengths=[2] * 4,
        rollout_log_probs=[torch.tensor(s.rollout_log_probs) for s in rollout],
        loss_masks=[torch.ones(2) for _ in rollout],
        rollout_mask_sums=[torch.tensor(8.0)] * 4,
    )
    logits = actor(**inputs)
    _, frozen = get_log_probs_and_entropy(
        logits.detach(),
        args=args,
        with_full_distribution=algorithm == "flow-dppo",
        **{key: batch[key] for key in ("unconcat_tokens", "total_lengths", "response_lengths")},
    )
    batch.update(frozen)
    error = max(
        (a - b).abs().max().item() for a, b in zip(batch["log_probs"], batch["rollout_log_probs"], strict=True)
    )
    assert error < 3e-6
    if critic is None:
        rewards = normalize_rewards(args, [0.0, 1.0, 7.0, 7.0])
        batch["advantages"] = get_grpo_returns(torch.tensor(rewards), batch["rollout_log_probs"])
    else:
        values = critic(**inputs)
        _, old_values = get_values(
            values.detach(),
            args=args,
            **{key: batch[key] for key in ("unconcat_tokens", "total_lengths", "response_lengths")},
        )
        batch.update(old_values)
        rewards = torch.zeros(4, 2)
        rewards[:, -1] = torch.tensor([0.0, 1.0, 7.0, 7.0])
        advantages, returns = vanilla_gae(rewards, torch.stack([v.flatten() for v in batch["values"]]), 1.0, 0.95)
        batch.update(advantages=list(advantages.detach()), returns=list(returns.detach()))
    reducer = get_sum_of_sample_mean(lengths, [2] * 4, batch["loss_masks"], batch["rollout_mask_sums"])
    for epoch in range(2 if algorithm == "flow-dppo" else 1):
        optimizer.zero_grad(set_to_none=True)
        loss, _ = policy_loss_function(args, batch, actor(**inputs) if epoch else logits, reducer)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in actor.parameters())
        optimizer.step()
        if critic is not None:
            critic_optimizer.zero_grad(set_to_none=True)
            value_loss, _ = value_loss_function(args, batch, critic(**inputs) if epoch else values, reducer)
            value_loss.backward()
            assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in critic.parameters())
            critic_optimizer.step()
    publish(actor, engine, family, path / f"after-{step}", step + 2)
    return {
        "tokens": [s.tokens for s in rollout],
        "scores": [s.rollout_log_probs for s in rollout],
        "seeds": [s.recurrent_trace.seed for s in rollout],
        "digest": engine.committed_digest,
        "error": error,
        "loss": loss.item(),
    }


@pytest.mark.parametrize("family", ["nanbeige"])
@pytest.mark.parametrize("algorithm", ["ppo", "grpo", "dppo", "flow-dppo"])
def test_three_updates_and_fresh_resume(family, algorithm, tmp_path, monkeypatch, record_property):
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    torch.manual_seed(42)
    native, actor = model_pair(family, True)
    critic = None if algorithm == "grpo" else model_pair(family, True, role="critic")[1]
    optimizer = torch.optim.SGD(actor.parameters(), lr=0.01)
    critic_optimizer = None if critic is None else torch.optim.SGD(critic.parameters(), lr=0.01)
    engine = cpu_engine(native, family)
    started = perf_counter()
    for step in range(2):
        update(actor, critic, optimizer, critic_optimizer, engine, family, algorithm, tmp_path, step)
    checkpoint = {
        "actor": copy.deepcopy(actor.state_dict()),
        "optimizer": optimizer.state_dict(),
        "rng": torch.get_rng_state(),
        "next_step": 2,
    }
    if critic is not None:
        checkpoint.update(critic=copy.deepcopy(critic.state_dict()), critic_optimizer=critic_optimizer.state_dict())
    torch.save(checkpoint, tmp_path / "checkpoint.pt")
    expected = update(actor, critic, optimizer, critic_optimizer, engine, family, algorithm, tmp_path, 2)
    elapsed = perf_counter() - started
    saved = torch.load(tmp_path / "checkpoint.pt", weights_only=True)
    _, restored = model_pair(family, True)
    restored.load_state_dict(saved["actor"])
    restored_optimizer = torch.optim.SGD(restored.parameters(), lr=0.01)
    restored_optimizer.load_state_dict(saved["optimizer"])
    restored_critic = None if critic is None else model_pair(family, True, role="critic")[1]
    restored_critic_optimizer = None if critic is None else torch.optim.SGD(restored_critic.parameters(), lr=0.01)
    if critic is not None:
        restored_critic.load_state_dict(saved["critic"])
        restored_critic_optimizer.load_state_dict(saved["critic_optimizer"])
    resumed_engine = cpu_engine(native, family)
    resumed_path = tmp_path / "resume"
    resumed_path.mkdir()
    torch.set_rng_state(saved["rng"])
    actual = update(
        restored,
        restored_critic,
        restored_optimizer,
        restored_critic_optimizer,
        resumed_engine,
        family,
        algorithm,
        resumed_path,
        saved["next_step"],
    )
    assert actual == expected
    for name, value in actor.state_dict().items():
        assert torch.equal(value, restored.state_dict()[name]), name
    if critic is not None:
        for name, value in critic.state_dict().items():
            assert torch.equal(value, restored_critic.state_dict()[name]), name
        assert restored_critic_optimizer.state_dict() == critic_optimizer.state_dict()
    assert restored_optimizer.state_dict() == optimizer.state_dict()
    assert torch.equal(torch.get_rng_state(), saved["rng"])
    record_property(
        "scope", "tiny FP32 CPU, synthetic rewards, Torch checkpoint; no Ray/MCore DCP or official GPU weights"
    )
    record_property("family", family)
    record_property("algorithm", algorithm)
    record_property("continuous_updates", 3)
    record_property("continuous_seconds", elapsed)
    record_property("max_selected_logprob_error", expected["error"])
    record_property("fresh_resume", "exact actor/critic/tokens/scores/SGD/RNG/publication digest")
    engine.close()
    resumed_engine.close()
