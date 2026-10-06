"""Three distinct policies, clipped behavior correction, and masked gradients."""

from argparse import Namespace

import pytest
import torch

pytest.importorskip("megatron")
from megatron.core import mpu

from vime.backends.megatron_utils.cp_utils import get_sum_of_sample_mean
from vime.backends.megatron_utils.loss import policy_loss_function


def loss_args(algorithm):
    return Namespace(
        rollout_temperature=0.8,
        log_probs_chunk_size=-1,
        allgather_cp=False,
        entropy_coef=0.0,
        use_rollout_logprobs=algorithm != "dppo",
        advantage_estimator="grpo" if algorithm == "grpo" else "ppo",
        use_opsm=False,
        eps_clip=0.2,
        eps_clip_high=0.2,
        eps_clip_c=None,
        get_mismatch_metrics=False,
        use_tis=algorithm == "dppo",
        tis_clip_low=0.5,
        tis_clip=2.0,
        custom_tis_function_path=None,
        calculate_per_token_loss=False,
        use_kl_loss=False,
        custom_pg_loss_reducer_function_path=None,
        rollout_top_p=1,
        value_clip=0.2,
        rewards_normalization=True,
        grpo_std_normalization=True,
        n_samples_per_prompt=2,
        rollout_batch_size=2,
        flow_dppo_divergence_budget=0.01 if algorithm == "flow-dppo" else None,
    )


def test_decoupled_loss_matches_three_policy_reference(monkeypatch):
    monkeypatch.setattr(mpu, "get_tensor_model_parallel_group", lambda: None)
    monkeypatch.setattr(mpu, "get_context_parallel_world_size", lambda: 1)
    args = loss_args("dppo")
    masks = [torch.tensor([1.0]), torch.tensor([1.0, 0.0, 1.0])]
    batch = {
        "unconcat_tokens": [torch.tensor([1, 1, 0]), torch.tensor([1, 1, 0, 1, 0])],
        "total_lengths": [3, 5],
        "response_lengths": [1, 3],
        "advantages": [torch.tensor([1.0]), torch.tensor([-1.0, 1.0, -1.0])],
        "log_probs": [torch.tensor([0.3]).log(), torch.tensor([0.8, 0.2, 0.8]).log()],
        "rollout_log_probs": [torch.tensor([0.1]).log(), torch.tensor([0.9, 0.9, 0.1]).log()],
        "loss_masks": masks,
        "rollout_mask_sums": [torch.tensor(1.0), torch.tensor(2.0)],
    }
    probabilities = torch.tensor([[0.5, 0.5]] * 8)
    probabilities[1] = torch.tensor([0.32, 0.68])
    probabilities[4:7] = torch.tensor([[0.75, 0.25], [0.25, 0.75], [0.4, 0.6]])
    logits = (probabilities.log() * 0.8)[None].requires_grad_()
    reducer = get_sum_of_sample_mean([3, 5], [1, 3], masks, batch["rollout_mask_sums"], False)
    loss, metrics = policy_loss_function(args, batch, logits, reducer)
    reference = logits.detach().clone().requires_grad_()
    selected = reference[0, [1, 4, 5, 6]].div(0.8).log_softmax(-1).gather(1, torch.tensor([[0], [0], [1], [0]]))[:, 0]
    proximal = torch.cat(batch["log_probs"])
    behavior = torch.cat(batch["rollout_log_probs"])
    advantages = torch.cat(batch["advantages"])
    ratio = (selected - proximal).exp()
    correction = (proximal - behavior).exp().clamp(0.5, 2.0)
    expected = reducer(torch.maximum(-ratio * advantages, -ratio.clamp(0.8, 1.2) * advantages) * correction)
    torch.testing.assert_close(loss, expected)
    loss.backward()
    expected.backward()
    torch.testing.assert_close(logits.grad, reference.grad)
    assert logits.grad[0, 5].count_nonzero() == 0
    assert logits.grad[0, 1].abs().sum() > 0
    assert metrics["tis_clipfrac"] > 0
    # Collapsing the proximal anchor into behavior changes the algorithm.
    bypass = reducer(
        torch.maximum(
            -(selected - behavior).exp() * advantages, -(selected - behavior).exp().clamp(0.8, 1.2) * advantages
        )
    )
    assert not torch.isclose(loss, bypass)
