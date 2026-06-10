"""Curriculum dynamic-sampling filter with snoozing, ported from ai21-verl.

This is a stateful drop-in superset of vime's built-in
``vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std``:

1. Drops zero-variance groups (all responses got the same reward → no learning signal),
   exactly like ``check_reward_nonzero_std``.
2. Additionally, when a zero-variance group is also "easy" (mean reward >= a threshold, i.e.
   consistently solved), it **snoozes** that prompt id: the next N times the same prompt is
   encountered it is dropped, so training focuses on harder prompts for a while.

This re-homes ai21-verl's ``filter_groups`` + ``snooze_easy_examples`` (verl/ai21/trainer/
trainer_utils.py) + ``SnoozingDataset`` (verl/ai21/datasets/snoozing.py). In vime the dynamic
filter runs per group right after reward (see ``vime/rollout/vllm_rollout.py``); a dropped group
is transparently replaced by re-generating other prompts, which is the curriculum effect.

Wire it in with::

    --dynamic-sampling-filter-path vime_plugins.filters.snoozing.snoozing_filter

Config is read from ``args`` if present, else the matching env var, else a default:

    snooze_num_times              AI21_SNOOZE_NUM_TIMES               0     (0 disables snoozing;
                                                                            filter then == check_reward_nonzero_std)
    snooze_mean_score_threshold   AI21_SNOOZE_MEAN_SCORE_THRESHOLD    1.0   (>= this mean == "easy")
    snooze_id_key                 AI21_SNOOZE_ID_KEY                  "id"  (metadata key for the stable prompt id)

Note: snooze state is per-process (the rollout driver). Counts decrement per encounter, matching
the original ``SnoozingDataset`` semantics.
"""

import os
from collections import defaultdict

import torch

from vime.rollout.filter_hub.base_types import DynamicFilterOutput
from vime.utils.types import Sample

__all__ = ["snoozing_filter", "reset_snooze_state"]

# prompt_id -> number of remaining times to skip this prompt
_snooze_counts: dict[str, int] = defaultdict(int)


def reset_snooze_state() -> None:
    """Clear the snooze registry (used by tests)."""
    _snooze_counts.clear()


def _cfg(args, attr, env, default):
    value = getattr(args, attr, None)
    if value is not None:
        return value
    if env in os.environ:
        return os.environ[env]
    return default


def _flatten_group(group: list) -> list[Sample]:
    # The standard rollout path passes list[Sample]; compact/fanout passes list[list[Sample]].
    if group and isinstance(group[0], list):
        return [s for sub in group for s in sub]
    return group


def _prompt_id(args, sample: Sample) -> str:
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    id_key = str(_cfg(args, "snooze_id_key", "AI21_SNOOZE_ID_KEY", "id"))
    pid = metadata.get(id_key, sample.group_index)
    return str(pid)


def snoozing_filter(args, samples, **kwargs) -> DynamicFilterOutput:
    samples = _flatten_group(samples)
    pid = _prompt_id(args, samples[0])

    # 1) Currently snoozed → drop and decrement.
    if _snooze_counts.get(pid, 0) > 0:
        _snooze_counts[pid] -= 1
        if _snooze_counts[pid] <= 0:
            _snooze_counts.pop(pid, None)
        return DynamicFilterOutput(keep=False, reason="snoozed")

    rewards = [sample.get_reward_value(args) for sample in samples]
    std = torch.tensor(rewards, dtype=torch.float64).std()
    keep = bool(std > 1e-6) or len(rewards) == 1
    if keep:
        return DynamicFilterOutput(keep=True)

    # 2) Zero-variance group: snooze it if it's "easy" (consistently solved).
    num_times = int(_cfg(args, "snooze_num_times", "AI21_SNOOZE_NUM_TIMES", 0))
    if num_times > 0:
        threshold = float(_cfg(args, "snooze_mean_score_threshold", "AI21_SNOOZE_MEAN_SCORE_THRESHOLD", 1.0))
        mean_reward = sum(rewards) / len(rewards)
        if mean_reward >= threshold:
            _snooze_counts[pid] += num_times

    return DynamicFilterOutput(keep=False, reason=f"zero_std_{round(rewards[0], 1)}")
