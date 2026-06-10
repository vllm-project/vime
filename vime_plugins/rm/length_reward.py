"""Length-reward shaping (Kimi 1.5 style), ported from ai21-verl.

Re-homes ``verl/ai21/trainer/algos.py::compute_length_reward`` as a vime reward
post-processor. Within each prompt group, shorter responses get up to +0.5 and longer
ones down to -0.5 (reversely linear between the group's min and max length), scaled by
a coefficient and added to the raw reward *before* GRPO group normalization — the same
ordering as ai21-verl, where the shaping term was added to ``token_level_rewards``
before advantage computation.

Wire it in with::

    --custom-reward-post-process-path vime_plugins.rm.length_reward.length_reward_post_process

Config is read from ``args`` if present (settable via ``--custom-config-path`` YAML),
else the matching env var, else a default:

    length_reward_coeff             AI21_LENGTH_REWARD_COEFF              1.0
    length_reward_min_length        AI21_LENGTH_REWARD_MIN_LENGTH         0
    length_reward_mode              AI21_LENGTH_REWARD_MODE               "penalty_only"

Modes (``LengthRewardMode``):
    penalty_only       only the negative half is applied — short-correct samples never
                       receive a positive bonus.
    bonus_and_penalty  full +/-0.5 shaping for correct samples (raw reward == 1),
                       negative half only for incorrect ones.

Note on the ai21-verl ``valid_len-1`` fix (#106): verl read the "is this sample correct"
signal from the padded ``token_level_rewards`` tensor and originally indexed the wrong
(padding) token. vime rewards are per-sample scalars, so the correctness check here uses
the raw reward directly and that bug class cannot occur.

After shaping, this function replicates vime's default normalization (group mean-center,
plus std-normalization for grpo/gspo when ``--grpo-std-normalization`` is on), since a
custom post-processor *replaces* the default one. Groups are identified by
``sample.group_index`` when set, else by consecutive chunks of ``n_samples_per_prompt``.
"""

import os
from collections import defaultdict
from enum import Enum

import torch

from vime.utils.types import Sample
from vime_plugins.utils.config_dump import maybe_dump_resolved_config

__all__ = ["LengthRewardMode", "compute_length_rewards", "length_reward_post_process"]


class LengthRewardMode(str, Enum):
    PENALTY_ONLY = "penalty_only"
    BONUS_AND_PENALTY = "bonus_and_penalty"


def _cfg(args, attr, env, default):
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


def _flatten(samples) -> list[Sample]:
    while samples and isinstance(samples[0], list):
        samples = [s for sub in samples for s in sub]
    return samples


def _group_ids(args, samples: list[Sample]) -> list:
    if all(s.group_index is not None for s in samples):
        return [s.group_index for s in samples]
    n = getattr(args, "n_samples_per_prompt", None) or 1
    return [i // n for i in range(len(samples))]


def compute_length_rewards(
    raw_rewards: list[float],
    response_lengths: list[int],
    group_ids: list,
    coeff: float = 1.0,
    min_length_for_reward: int = 0,
    mode: LengthRewardMode = LengthRewardMode.PENALTY_ONLY,
) -> list[float]:
    """Per-sample length shaping term (Kimi 1.5, https://arxiv.org/pdf/2501.12599).

    Lengths are clamped to ``min_length_for_reward`` so all responses shorter than it
    map to the same shaping value; a group with equal min and max length gets 0.
    """
    mode = LengthRewardMode(mode)
    clamped = [max(rl, min_length_for_reward) for rl in response_lengths]

    group_min: dict = {}
    group_max: dict = {}
    for gid, length in zip(group_ids, clamped, strict=True):
        group_min[gid] = min(length, group_min.get(gid, length))
        group_max[gid] = max(length, group_max.get(gid, length))

    shaping = []
    for reward, gid, length in zip(raw_rewards, group_ids, clamped, strict=True):
        if group_max[gid] == group_min[gid]:
            shaping.append(0.0)
            continue
        # map lengths between min and max to +0.5 .. -0.5 (reversely linear)
        lambda_factor = 0.5 - (length - group_min[gid]) / (group_max[gid] - group_min[gid])
        if mode == LengthRewardMode.PENALTY_ONLY:
            value = min(0.0, lambda_factor)
        else:
            value = lambda_factor if reward == 1 else min(0.0, lambda_factor)
        shaping.append(value * coeff)
    return shaping


def _normalize_like_default(args, shaped: list[float], group_ids: list) -> list[float]:
    """Replicate ``RolloutManager._post_process_rewards``'s default normalization."""
    if getattr(args, "advantage_estimator", None) not in (
        "grpo",
        "gspo",
        "reinforce_plus_plus_baseline",
    ) or not getattr(args, "rewards_normalization", False):
        return shaped

    by_group: dict = defaultdict(list)
    for i, gid in enumerate(group_ids):
        by_group[gid].append(i)

    out = list(shaped)
    use_std = args.advantage_estimator in ("grpo", "gspo") and getattr(args, "grpo_std_normalization", False)
    for indices in by_group.values():
        values = torch.tensor([shaped[i] for i in indices], dtype=torch.float)
        values = values - values.mean()
        if use_std:
            values = values / (values.std() + 1e-6)
        for i, v in zip(indices, values.tolist(), strict=True):
            out[i] = v
    return out


def length_reward_post_process(args, samples) -> tuple[list[float], list[float]]:
    """``--custom-reward-post-process-path`` entrypoint: ``(args, samples) -> (raw, rewards)``."""
    maybe_dump_resolved_config(args)
    samples = _flatten(samples)
    raw_rewards = [sample.get_reward_value(args) for sample in samples]
    group_ids = _group_ids(args, samples)

    shaping = compute_length_rewards(
        raw_rewards,
        [sample.response_length for sample in samples],
        group_ids,
        coeff=float(_cfg(args, "length_reward_coeff", "AI21_LENGTH_REWARD_COEFF", 1.0)),
        min_length_for_reward=int(_cfg(args, "length_reward_min_length", "AI21_LENGTH_REWARD_MIN_LENGTH", 0)),
        mode=LengthRewardMode(_cfg(args, "length_reward_mode", "AI21_LENGTH_REWARD_MODE", "penalty_only")),
    )
    shaped = [r + s for r, s in zip(raw_rewards, shaping, strict=True)]
    return raw_rewards, _normalize_like_default(args, shaped, group_ids)
