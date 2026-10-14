"""MemAgent multi-turn custom convert — unroll turns into independent training samples."""

from __future__ import annotations

import torch


def _sample_has_rollout_log_probs(sample) -> bool:
    meta = sample.train_metadata
    if meta and "turns" in meta:
        return any(t.get("rollout_log_probs") is not None for t in meta["turns"])
    return sample.rollout_log_probs is not None


def custom_convert(args, samples):
    raw_rewards = [s.get_reward_value(args) for s in samples]
    rewards_tensor = torch.tensor(raw_rewards, dtype=torch.float)

    if getattr(args, "advantage_estimator", None) in ["grpo", "gspo", "reinforce_plus_plus_baseline"] and getattr(
        args, "rewards_normalization", False
    ):
        n = getattr(args, "n_samples_per_prompt", 1)
        if rewards_tensor.shape[-1] == n * getattr(args, "rollout_batch_size", 1):
            rewards_tensor = rewards_tensor.reshape(-1, n)
        else:
            rewards_tensor = rewards_tensor.view(-1, rewards_tensor.shape[-1])

        mean = rewards_tensor.mean(dim=-1, keepdim=True)
        rewards_tensor = rewards_tensor - mean

        if getattr(args, "advantage_estimator", None) in ["grpo", "gspo"] and getattr(
            args, "grpo_std_normalization", False
        ):
            std = rewards_tensor.std(dim=-1, keepdim=True)
            rewards_tensor = rewards_tensor / (std + 1e-6)

    normalized_rewards = rewards_tensor.flatten().tolist()

    parent_ids = [sample.rollout_id if sample.rollout_id is not None else sample.index for sample in samples]
    used_ids = {parent_id for parent_id in parent_ids if parent_id is not None}
    next_id = 0
    for position, parent_id in enumerate(parent_ids):
        if parent_id is None:
            while next_id in used_ids:
                next_id += 1
            parent_ids[position] = next_id
            used_ids.add(next_id)

    rollout_ids = []
    tokens_list = []
    response_lengths = []
    loss_masks = []
    rewards = []
    raw_reward_list = []
    truncated_list = []
    sample_indices = []
    rollout_log_probs_list = []
    has_rollout_log_probs = any(
        _sample_has_rollout_log_probs(s) for s in samples if s.status not in (s.Status.FAILED, s.Status.ABORTED)
    )

    for i, sample in enumerate(samples):
        if sample.status in (sample.Status.FAILED, sample.Status.ABORTED):
            continue

        meta = sample.train_metadata
        if meta is None or "turns" not in meta:
            if not sample.tokens:
                continue
            tokens_list.append(sample.tokens)
            response_lengths.append(sample.response_length)
            lm = sample.loss_mask if sample.loss_mask is not None else [1] * sample.response_length
            if sample.remove_sample:
                lm = [0] * sample.response_length
            loss_masks.append(lm)
            rewards.append(normalized_rewards[i])
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(1 if sample.status == sample.Status.TRUNCATED else 0)
            sample_indices.append(sample.index)
            rollout_ids.append(parent_ids[i])
            if has_rollout_log_probs:
                lp = sample.rollout_log_probs
                if lp is None:
                    lp = [0.0] * sample.response_length
                rollout_log_probs_list.append(lp)
            continue

        turns = meta["turns"]
        if not turns:
            continue
        norm_reward = normalized_rewards[i] / len(turns)
        is_truncated = 1 if sample.status == sample.Status.TRUNCATED else 0

        for turn in turns:
            tokens_list.append(turn["tokens"])
            response_lengths.append(turn["response_length"])
            lm = list(turn["loss_mask"])
            if sample.remove_sample:
                lm = [0] * turn["response_length"]
            loss_masks.append(lm)
            rewards.append(norm_reward)
            raw_reward_list.append(raw_rewards[i])
            truncated_list.append(is_truncated)
            sample_indices.append(sample.index)
            rollout_ids.append(parent_ids[i])
            if has_rollout_log_probs:
                lp = turn.get("rollout_log_probs")
                if lp is None:
                    lp = [0.0] * turn["response_length"]
                rollout_log_probs_list.append(lp)

    train_data = {
        "tokens": tokens_list,
        "response_lengths": response_lengths,
        "loss_masks": loss_masks,
        "rewards": rewards,
        "raw_reward": raw_reward_list,
        "truncated": truncated_list,
        "sample_indices": sample_indices,
        "rollout_ids": rollout_ids,
    }
    if has_rollout_log_probs:
        train_data["rollout_log_probs"] = rollout_log_probs_list

    return train_data
