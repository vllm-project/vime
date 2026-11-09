"""Analytical categorical KL and both outward/corrective gradient directions."""

from argparse import Namespace

import pytest
import torch

from vime_plugins.flow_dppo.loss import categorical_flow_dppo


def test_full_vocabulary_kl_and_asymmetric_gradient_gate():
    # Same distribution and actions; only the advantage sign changes direction.
    probabilities = torch.tensor([[0.9, 0.1]] * 4)
    logits = probabilities.log().requires_grad_()
    old = torch.full((4, 2), 0.5).log().requires_grad_()
    actions = torch.tensor([0, 1, 1, 0])
    advantages = torch.tensor([1.0, -1.0, 1.0, -1.0])
    loss, blocked, divergence = categorical_flow_dppo(logits, old, actions, advantages, 0.05)
    expected = 0.5 * torch.log(torch.tensor(0.5 / 0.9, dtype=torch.float64)) + 0.5 * torch.log(
        torch.tensor(0.5 / 0.1, dtype=torch.float64)
    )
    torch.testing.assert_close(divergence.double(), expected.expand(4), atol=1e-7, rtol=1e-6)
    assert blocked.tolist() == [1, 1, 0, 0]
    loss.sum().backward()
    assert old.grad is None
    assert torch.equal(logits.grad[:2], torch.zeros(2, 2))
    reference = logits.detach().clone().requires_grad_()
    ratio = (reference.log_softmax(-1) - old.detach()).gather(1, actions[:, None])[:, 0].exp()
    (-ratio[2:] * advantages[2:]).sum().backward()
    torch.testing.assert_close(logits.grad[2:], reference.grad[2:])


@pytest.mark.parametrize("budget", [0.05, 1.0])
def test_identical_policy_has_zero_kl_and_live_gradient(budget):
    old = torch.tensor([[0.25, 0.75]]).log()
    logits = old.clone().requires_grad_()
    loss, blocked, divergence = categorical_flow_dppo(logits, old, torch.tensor([1]), torch.tensor([1.0]), budget)
    assert blocked.item() == 0 and divergence.item() == 0
    loss.sum().backward()
    torch.testing.assert_close(logits.grad, torch.tensor([[0.25, -0.25]]))


@pytest.mark.parametrize("budget", [None, 0.05])
def test_packed_frozen_distribution_and_training_loss(monkeypatch, budget):
    from megatron.core import mpu

    from vime.backends.megatron_utils.data import DataIterator
    from vime.backends.megatron_utils.loss import get_log_probs_and_entropy, policy_loss_function

    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    args = Namespace(
        rollout_temperature=0.5,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        entropy_coef=0.0,
        use_rollout_logprobs=True,
        advantage_estimator="ppo",
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=False,
        use_kl_loss=False,
        custom_pg_loss_reducer_function_path=None,
        rollout_top_p=1,
        flow_dppo_divergence_budget=budget,
    )
    batch = {
        "unconcat_tokens": [torch.tensor([3, 3, 0]), torch.tensor([3, 3, 1, 2])],
        "total_lengths": [3, 4],
        "response_lengths": [1, 2],
        "advantages": [torch.tensor([1.0]), torch.tensor([-1.0, -1.0])],
    }
    probabilities = torch.full((7, 4), 0.25)
    probabilities[1] = torch.tensor([0.1, 0.2, 0.3, 0.4])
    probabilities[4] = torch.tensor([0.4, 0.2, 0.1, 0.3])
    probabilities[5] = torch.tensor([0.2, 0.1, 0.3, 0.4])
    old_logits = (probabilities.log() * 0.5).unsqueeze(0).requires_grad_()
    capture_args = {key: batch[key] for key in ("unconcat_tokens", "total_lengths", "response_lengths")}
    _, frozen = get_log_probs_and_entropy(old_logits, args=args, with_full_distribution=True, **capture_args)
    assert all(not tensor.requires_grad and tensor.device.type == "cpu" for tensor in frozen["old_policy_log_probs"])
    torch.testing.assert_close(frozen["old_policy_log_probs"][0], probabilities[1:2].log())
    torch.testing.assert_close(frozen["old_policy_log_probs"][1], probabilities[4:6].log())
    iterator = DataIterator(frozen, [[1], [0]])
    torch.testing.assert_close(
        iterator.get_next(["old_policy_log_probs"])["old_policy_log_probs"][0], probabilities[4:6].log()
    )
    _, ordinary = get_log_probs_and_entropy(old_logits, args=args, **capture_args)
    assert "old_policy_log_probs" not in ordinary
    batch.update(frozen)
    batch["rollout_log_probs"] = [value.detach() for value in frozen["log_probs"]]

    probabilities[1] = torch.tensor([0.9, 0.05, 0.03, 0.02])
    probabilities[4:6] = torch.tensor([0.05, 0.1, 0.75, 0.1])
    logits = (probabilities.log() * 0.5).unsqueeze(0).requires_grad_()

    def reduce_samples(values):
        return (values[0] + values[1:].mean()) / 2

    loss, metrics = policy_loss_function(args, batch, logits, reduce_samples)
    loss.backward()
    torch.testing.assert_close(loss, torch.tensor(0.225 if budget is None else -3.75), atol=1e-6, rtol=1e-6)
    assert logits.grad[0, 1].count_nonzero() == 0 and logits.grad[0, 4].count_nonzero() == 0
    assert logits.grad[0, 5].abs().sum() > 0 and old_logits.grad is None
    assert ("categorical_kl" in metrics) == (budget is not None)
    if budget is not None:
        assert metrics["categorical_kl"] > budget
