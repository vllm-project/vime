"""Unit tests for the curriculum snoozing dynamic-sampling filter."""

from __future__ import annotations

from argparse import Namespace

import pytest

from vime.utils.types import Sample
from vime_plugins.filters import snoozing as mod


@pytest.fixture(autouse=True)
def _clean_state():
    mod.reset_snooze_state()
    yield
    mod.reset_snooze_state()


def _group(rewards, pid="p1"):
    return [Sample(reward=r, metadata={"id": pid}, group_index=0) for r in rewards]


def _args(**kw):
    base = dict(reward_key=None, snooze_num_times=0, snooze_mean_score_threshold=1.0, snooze_id_key="id")
    base.update(kw)
    return Namespace(**base)


def test_keeps_nonzero_variance_group():
    out = mod.snoozing_filter(_args(), _group([0.0, 1.0]))
    assert out.keep is True


def test_drops_zero_variance_group():
    out = mod.snoozing_filter(_args(), _group([1.0, 1.0]))
    assert out.keep is False
    assert out.reason.startswith("zero_std_")


def test_single_sample_group_kept():
    out = mod.snoozing_filter(_args(), _group([1.0]))
    assert out.keep is True


def test_easy_group_is_snoozed_for_n_encounters():
    args = _args(snooze_num_times=2, snooze_mean_score_threshold=1.0)
    pid = "easy"

    # First encounter: all-correct, zero variance, mean >= threshold -> dropped + snoozed for 2.
    out = mod.snoozing_filter(args, _group([1.0, 1.0], pid=pid))
    assert out.keep is False and out.reason.startswith("zero_std_")

    # Next 2 encounters: dropped as snoozed (even if now it has signal).
    for _ in range(2):
        out = mod.snoozing_filter(args, _group([0.0, 1.0], pid=pid))
        assert out.keep is False and out.reason == "snoozed"

    # Snooze exhausted -> a group with signal is kept again.
    out = mod.snoozing_filter(args, _group([0.0, 1.0], pid=pid))
    assert out.keep is True


def test_hard_zero_variance_group_not_snoozed():
    # all-wrong (mean 0 < threshold) is dropped but NOT snoozed.
    args = _args(snooze_num_times=3, snooze_mean_score_threshold=1.0)
    pid = "hard"
    out = mod.snoozing_filter(args, _group([0.0, 0.0], pid=pid))
    assert out.keep is False
    # next encounter is evaluated normally (not snoozed)
    out = mod.snoozing_filter(args, _group([0.0, 1.0], pid=pid))
    assert out.keep is True


def test_snoozing_disabled_by_default_behaves_like_nonzero_std():
    # num_times=0 -> never snoozes; identical drop behaviour to check_reward_nonzero_std.
    args = _args(snooze_num_times=0)
    pid = "x"
    mod.snoozing_filter(args, _group([1.0, 1.0], pid=pid))  # dropped, no snooze registered
    out = mod.snoozing_filter(args, _group([0.0, 1.0], pid=pid))
    assert out.keep is True


def test_reward_key_dict_rewards():
    args = _args(reward_key="score")
    group = [Sample(reward={"score": v}, metadata={"id": "p"}) for v in (0.0, 1.0)]
    out = mod.snoozing_filter(args, group)
    assert out.keep is True


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
