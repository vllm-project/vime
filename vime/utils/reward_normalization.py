"""Shared reward normalization for rollout backends."""

from argparse import Namespace

import torch


def normalize_rewards(args: Namespace, raw_rewards: list[float]) -> list[float]:
    if (
        args.advantage_estimator in ["grpo", "gspo", "cispo", "reinforce_plus_plus_baseline"]
        and args.rewards_normalization
    ):
        rewards = torch.tensor(raw_rewards, dtype=torch.float)
        if rewards.shape[-1] == args.n_samples_per_prompt * args.rollout_batch_size:
            rewards = rewards.reshape(-1, args.n_samples_per_prompt)
        else:
            rewards = rewards.view(-1, rewards.shape[-1])
        rewards = rewards - rewards.mean(dim=-1, keepdim=True)
        if args.advantage_estimator in ["grpo", "gspo", "cispo"] and args.grpo_std_normalization:
            rewards = rewards / (rewards.std(dim=-1, keepdim=True) + 1e-6)
        return rewards.flatten().tolist()
    return raw_rewards
