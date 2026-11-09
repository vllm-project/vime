import json
from dataclasses import asdict

import pytest

from vime.rollout.transfer_queue_contract import GroupLedger, Phase, TrajectoryChild, TrajectoryGroup, group_from_dict


def _child(sample_id, versions=("v3",)):
    return TrajectoryChild(
        sample_id=sample_id,
        rollout_id=8,
        weight_versions=versions,
        token_count=6,
        response_length=2,
        loss_mask_length=2,
        reward=0.5,
        rollout_log_probs_length=2,
        top_p_token_offsets=(0, 2, 3),
        top_p_token_count=3,
    )


def _group(**changes):
    values = dict(
        schema_version=1,
        job_id="job-a",
        restart_epoch=2,
        group_id="prompt-draw-17",
        attempt_id="attempt-0",
        expected_children=2,
        completed_children=2,
        sampling_config_digest="sha256-config",
        rollout_policy_version="v3",
        children=(_child(0), _child(1)),
        payload_ref="job-a/group-17",
        byte_count=1024,
        ready=True,
    )
    values.update(changes)
    return TrajectoryGroup(**values)


def test_complete_group_preserves_child_identity_and_exact_policy_version():
    group = _group()
    group.require_consumer_version("v3")
    assert [child.sample_id for child in group.children] == [0, 1]
    assert [child.rollout_id for child in group.children] == [8, 8]
    with pytest.raises(ValueError, match="does not match"):
        group.require_consumer_version("v2")
    with pytest.raises(ValueError, match="does not match"):
        group.require_consumer_version("v4")


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        ({"completed_children": 1}, "incomplete"),
        ({"children": (_child(0), _child(0))}, "duplicate"),
        ({"children": (_child(0), _child(1, ()))}, "policy version"),
        ({"children": (_child(0), _child(1, ("v3", "v4")))}, "policy version"),
        ({"payload_ref": ""}, "needs payload"),
        ({"schema_version": 2}, "schema"),
    ],
)
def test_incomplete_or_incompatible_group_never_becomes_ready(changes, message):
    with pytest.raises(ValueError, match=message):
        _group(**changes)


def test_ragged_top_p_metadata_must_match_response():
    with pytest.raises(ValueError, match="top-p"):
        TrajectoryChild(1, 8, ("v3",), 6, 2, 2, 0.5, top_p_token_offsets=(0, 2), top_p_token_count=2)


def test_lease_generation_prevents_late_owner_from_committing():
    ledger = GroupLedger(_group())
    old = ledger.acquire("worker-a", "token-a", now=0, ttl=5)
    ledger.prepare(old, "batch-1", now=4)
    new = ledger.acquire("worker-b", "token-b", now=5, ttl=10)
    assert new.generation == old.generation + 1
    with pytest.raises(ValueError, match="stale"):
        ledger.start_training(old, now=6)
    ledger.prepare(new, "batch-2", now=6)
    ledger.start_training(new, now=7)
    ledger.finish_training(new, now=7.5)
    ledger.commit(new, "checkpoint-7", now=8)
    assert (ledger.phase, ledger.consumer_batch_id, ledger.commit_id) == (Phase.COMMITTED, "batch-2", "checkpoint-7")


def test_training_timeout_is_unknown_and_never_automatically_requeued():
    ledger = GroupLedger(_group())
    lease = ledger.acquire("worker-a", "token-a", now=0, ttl=5)
    ledger.prepare(lease, "batch-1", now=1)
    ledger.start_training(lease, now=2)
    ledger.expire(now=5)
    assert ledger.phase == Phase.UNKNOWN
    with pytest.raises(ValueError, match="unavailable"):
        ledger.acquire("worker-b", "token-b", now=6, ttl=5)
    with pytest.raises(ValueError, match="stale"):
        ledger.commit(lease, "late-commit", now=6)


def test_fetch_and_prepare_do_not_commit_a_group():
    ledger = GroupLedger(_group())
    lease = ledger.acquire("worker-a", "token-a", now=0, ttl=5)
    ledger.prepare(lease, "batch-1", now=1)
    with pytest.raises(ValueError, match="not finished"):
        ledger.commit(lease, "early", now=2)
    ledger.release_before_training(lease, now=2)
    assert ledger.phase == Phase.READY


def test_training_start_is_not_a_commit_boundary():
    ledger = GroupLedger(_group())
    lease = ledger.acquire("worker-a", "token-a", now=0, ttl=5)
    ledger.prepare(lease, "batch-1", now=1)
    ledger.start_training(lease, now=2)
    with pytest.raises(ValueError, match="not finished"):
        ledger.commit(lease, "early", now=3)
    ledger.finish_training(lease, now=4)
    ledger.expire(now=5)
    assert ledger.phase == Phase.UNKNOWN


@pytest.mark.parametrize("field,value", [("now", True), ("now", float("nan")), ("ttl", float("inf")), ("ttl", True)])
def test_lease_clock_rejects_non_finite_and_bool(field, value):
    ledger = GroupLedger(_group())
    values = dict(now=0.0, ttl=5.0)
    values[field] = value
    with pytest.raises(ValueError, match="finite"):
        ledger.acquire("owner", "token", **values)


def test_heartbeat_preserves_fencing_and_clock_is_monotonic():
    ledger = GroupLedger(_group())
    lease = ledger.acquire("owner", "token", now=0, ttl=5)
    renewed = ledger.renew(lease, now=4, ttl=5)
    assert renewed.generation == lease.generation
    with pytest.raises(ValueError, match="stale"):
        ledger.prepare(lease, "batch", now=4)
    ledger.prepare(renewed, "batch", now=4)
    with pytest.raises(ValueError, match="backwards"):
        ledger.start_training(renewed, now=3)


def test_nested_json_roundtrip_and_unknown_fields():
    original = _group()
    encoded = json.loads(json.dumps(asdict(original)))
    assert group_from_dict(encoded) == original
    encoded["children"][0]["weight_versions"] = "v3"
    with pytest.raises(ValueError, match="array"):
        group_from_dict(encoded)
    encoded = json.loads(json.dumps(asdict(original)))
    encoded["unexpected"] = 1
    with pytest.raises(ValueError, match="fields"):
        group_from_dict(encoded)


@pytest.mark.parametrize("offset", [True, -1, 0.5])
def test_top_p_offsets_are_integer_coordinates(offset):
    with pytest.raises(ValueError, match="top_p"):
        TrajectoryChild(1, 8, ("v3",), 6, 2, 2, 0.5, top_p_token_offsets=(0, offset, 3), top_p_token_count=3)


@pytest.mark.parametrize("reward", [True, "1", float("nan"), float("inf"), 10**1000])
def test_reward_validation_rejects_invalid_numeric_values(reward):
    with pytest.raises(ValueError, match="reward"):
        TrajectoryChild(1, 8, ("v3",), 6, 2, 2, reward)
