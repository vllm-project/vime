"""AI21 rollout observability metrics, ported from ai21-verl ``AI21GRPOTrainer``.

Re-homes the trainer's metric layer — ``compute_prefilter_data_metrics``,
``_compute_constant_score_metrics``, ``compute_extra_data_metrics`` (the reward-side
parts), and the snooze-adjusted ("unsnoozed") reward — that ``AI21GRPOTrainer.fit()``
computed inline (verl/ai21/trainer/trainer_utils.py + ai21_grpo_ray_trainer.py).

In verl this lived in one process that saw the whole batch. vime splits rollout from
training across Ray actors, so this is wired through the two rollout-side seams that run
back-to-back inside the same ``RolloutManager.generate`` call:

    --rollout-all-samples-process-path  vime_plugins.metrics.rollout_metrics.capture_prefilter_metrics
    --custom-rollout-log-function-path  vime_plugins.metrics.rollout_metrics.ai21_rollout_log

``capture_prefilter_metrics`` receives **all** generated groups — including the ones the
dynamic filter dropped — which is exactly ai21-verl's "unfiltered rollouts batch", and
stashes the per-rollout prefilter metrics. ``ai21_rollout_log`` then merges them with
kept-batch reward min/max and the snooze-adjusted reward, logs everything under
``rollout/step``, and returns ``False`` so vime's default rollout logging still runs
unchanged (this layer is purely additive).

Metric names map ai21-verl's ``critic/score/*`` (verl logged reward stats under the
critic namespace even without a critic) into vime's ``rollout/`` namespace so they share
the ``rollout/step`` x-axis:

    ai21-verl                                   vime
    critic/score/{mean,max,min}/prefilter       rollout/score/{mean,max,min}/prefilter
    critic/score/mean/prefilter/unsnoozed       rollout/score/mean/prefilter/unsnoozed
    critic/score/frac_prompts_const_score_in_X  rollout/score/frac_prompts_const_score_in_X/prefilter
    response_length/*/prefilter                 rollout/response_length/*/prefilter
    train/step_newly_snoozed_prompts            rollout/snooze/step_newly_snoozed_prompts
    train/step_snooze_skips                     rollout/snooze/step_skips

Per-dataset breakdowns (``per-dataset/<source>/...``) are emitted when samples carry a
data-source tag in ``metadata`` (key configurable, default ``data_source``); absent that
key the per-dataset block is silently skipped.

Not ported here (train-side, no additive seam): ``critic/advantages/*`` and
``critic/returns/*`` min/max + ``all_zero_frac`` need the advantage/return tensors that
only exist on the Megatron actor (``vime/backends/megatron_utils/data.py`` logs their
means). The rollout-vs-actor log-prob divergence (``training/rollout_actor_logprobs_*``)
is already covered natively by vime's mismatch (``mis_*``) metrics.
"""

import numpy as np

from vime.utils import logging_utils
from vime.utils.metric_utils import compute_rollout_step
from vime.utils.misc import group_by
from vime.utils.types import Sample
from vime_plugins.utils.common import cfg as _cfg
from vime_plugins.utils.common import flatten_samples as _flatten

__all__ = [
    "capture_prefilter_metrics",
    "ai21_rollout_log",
    "compute_prefilter_metrics",
    "constant_score_bucket_metrics",
    "reset_pending",
]

# Single-slot stash: capture_prefilter_metrics writes it, ai21_rollout_log reads+clears it.
# Safe because both run back-to-back inside one synchronous RolloutManager.generate() call,
# so at most one rollout's metrics are ever pending.
_PENDING: dict = {}


def reset_pending() -> None:
    """Clear stashed prefilter metrics (used by tests)."""
    _PENDING.clear()


def _is_aborted(sample: Sample) -> bool:
    return sample.status == Sample.Status.ABORTED or sample.response_length == 0


def _data_source(args, sample: Sample) -> str:
    key = str(_cfg(args, "metrics_data_source_key", "AI21_METRICS_DATA_SOURCE_KEY", "data_source"))
    metadata = sample.metadata if isinstance(sample.metadata, dict) else {}
    return str(metadata.get(key, "unknown"))


def constant_score_bucket_metrics(args, samples: list[Sample], n_buckets: int = 4) -> dict[str, float]:
    """Fraction of prompt groups where every rollout got the same non-zero score, bucketed.

    Replicates ai21-verl ``_compute_constant_score_metrics``: with ``n_buckets=4`` the
    buckets are [0,0.25), [0.25,0.5), [0.5,0.75), [0.75,1]. Denominator is all groups.
    """
    groups = group_by(samples, lambda s: s.group_index)
    n_prompts = len(groups)
    if n_prompts == 0:
        return {}

    in_bucket = [0 for _ in range(n_buckets)]
    for group in groups.values():
        scores = [s.get_reward_value(args) for s in group]
        if len(set(scores)) == 1 and scores[0]:
            bucket_idx = max(min(int(scores[0] * n_buckets), n_buckets - 1), 0)
            in_bucket[bucket_idx] += 1

    metrics = {}
    for bucket_idx, count in enumerate(in_bucket):
        s_range = f"{bucket_idx / n_buckets:.2f}-{(bucket_idx + 1) / n_buckets:.2f}"
        metrics[f"score/frac_prompts_const_score_in_{s_range}/prefilter"] = count / n_prompts
    return metrics


def compute_prefilter_metrics(args, all_samples) -> dict[str, float]:
    """Reward/length stats over the full (pre-filter) batch — vime port of
    ``compute_prefilter_data_metrics``. ``all_samples`` is ``list[list[Sample]]``."""
    samples = _flatten(all_samples)
    if not samples:
        return {}

    scores = np.array([s.get_reward_value(args) for s in samples], dtype=np.float64)
    lengths = np.array([s.response_length for s in samples], dtype=np.float64)
    aborted = np.array([_is_aborted(s) for s in samples], dtype=bool)
    non_aborted = ~aborted

    max_len = getattr(args, "rollout_max_response_len", None)

    metrics: dict[str, float] = {
        "score/mean/prefilter": float(np.mean(scores)),
        "score/max/prefilter": float(np.max(scores)),
        "score/min/prefilter": float(np.min(scores)),
        "response_length/mean/prefilter": float(np.mean(lengths)),
        "response_length/max/prefilter": float(np.max(lengths)),
        "response_length/min/prefilter": float(np.min(lengths)),
        "response/aborted_ratio/prefilter": float(np.mean(aborted)),
    }
    if max_len:
        metrics["response_length/clip_ratio/prefilter"] = float(np.mean(lengths >= max_len))

    if non_aborted.any():
        na_lengths = lengths[non_aborted]
        metrics["score/mean_non_aborted/prefilter"] = float(np.mean(scores[non_aborted]))
        metrics["response_length_non_aborted/mean/prefilter"] = float(np.mean(na_lengths))

    metrics.update(constant_score_bucket_metrics(args, samples))
    return metrics


def _snooze_adjusted(prefilter: dict, num_prompts_unfiltered: int, num_snooze_skips: int) -> dict[str, float]:
    """ai21-verl ``compute_snooze_adjusted_metrics``: the mean reward we'd have seen if the
    snoozed (assumed-solved, reward 1.0) prompts had not been skipped this step."""
    mean_key = "score/mean/prefilter"
    if mean_key not in prefilter or num_snooze_skips <= 0 or num_prompts_unfiltered <= 0:
        return {}
    mean_reward = float(prefilter[mean_key])
    snoozed_reward = 1.0
    adjusted = (mean_reward * num_prompts_unfiltered + snoozed_reward * num_snooze_skips) / (
        num_prompts_unfiltered + num_snooze_skips
    )
    return {"score/mean/prefilter/unsnoozed": adjusted}


def _per_dataset_metrics(args, samples: list[Sample], suffix: str) -> dict[str, float]:
    by_source = group_by(samples, lambda s: _data_source(args, s))
    if len(by_source) <= 1 and "unknown" in by_source:
        return {}
    out: dict[str, float] = {}
    for source, group in by_source.items():
        scores = np.array([s.get_reward_value(args) for s in group], dtype=np.float64)
        lengths = np.array([s.response_length for s in group], dtype=np.float64)
        prefix = f"per-dataset/{source}"
        out[f"{prefix}/score/mean{suffix}"] = float(np.mean(scores))
        out[f"{prefix}/response_length/mean{suffix}"] = float(np.mean(lengths))
        out[f"{prefix}/num_samples{suffix}"] = float(len(group))
    return out


def capture_prefilter_metrics(args, all_samples, data_source) -> None:
    """``--rollout-all-samples-process-path`` entrypoint: ``(args, all_samples, data_source)``.

    Computes prefilter metrics over every generated group (including dropped ones) and
    stashes them keyed by ``rollout_id`` for ``ai21_rollout_log`` to pick up. Never raises
    into the rollout loop — observability must not break training.
    """
    try:
        metrics = compute_prefilter_metrics(args, all_samples)
        flat = _flatten(all_samples)
        metrics["num_prompts_unfiltered"] = float(len(group_by(flat, lambda s: s.group_index)))
        metrics.update(_per_dataset_metrics(args, flat, suffix="/prefilter"))
        _PENDING["current"] = metrics
    except Exception as exc:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning("capture_prefilter_metrics failed: %s", exc)


def ai21_rollout_log(rollout_id, args, samples, rollout_extra_metrics, rollout_time) -> bool:
    """``--custom-rollout-log-function-path`` entrypoint.

    Logs AI21 prefilter / snooze-adjusted / kept-batch reward metrics under ``rollout/step``
    and returns ``False`` so vime's default rollout logging is left intact (additive).
    """
    try:
        log_dict: dict[str, float] = {}

        prefilter = _PENDING.pop("current", None)
        if prefilter is not None:
            num_prompts_unfiltered = int(prefilter.pop("num_prompts_unfiltered", 0))

            snooze_skips = 0
            newly_snoozed = 0
            try:
                from vime_plugins.filters.snoozing import pop_snooze_step_stats

                stats = pop_snooze_step_stats()
                snooze_skips = stats["snooze_skips"]
                newly_snoozed = stats["newly_snoozed"]
            except Exception:
                # Fall back to the drop counter the MetricGatherer already collected.
                snooze_skips = int((rollout_extra_metrics or {}).get("rollout/dynamic_filter/drop_snoozed", 0))

            prefilter.update(_snooze_adjusted(prefilter, num_prompts_unfiltered, snooze_skips))
            log_dict["rollout/snooze/step_skips"] = float(snooze_skips)
            log_dict["rollout/snooze/step_newly_snoozed_prompts"] = float(newly_snoozed)
            for key, val in prefilter.items():
                log_dict[key if key.startswith("per-dataset/") else f"rollout/{key}"] = val

        # Kept-batch reward min/max (vime logs the mean as rollout/raw_reward, not min/max).
        flat = _flatten(samples)
        if flat:
            scores = np.array([s.get_reward_value(args) for s in flat], dtype=np.float64)
            log_dict["rollout/score/max"] = float(np.max(scores))
            log_dict["rollout/score/min"] = float(np.min(scores))
            log_dict.update(_per_dataset_metrics(args, flat, suffix=""))

        if log_dict:
            log_dict["rollout/step"] = compute_rollout_step(args, rollout_id)
            logging_utils.log(args, log_dict, step_key="rollout/step")
    except Exception as exc:  # pragma: no cover - defensive
        import logging

        logging.getLogger(__name__).warning("ai21_rollout_log failed: %s", exc)

    # Always additive: let vime's default rollout logging run too.
    return False
