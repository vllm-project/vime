from dataclasses import replace

import pytest

from vime.rollout.transfer_queue_adapter import decode_group, encode_group, to_tensor_dict
from vime.rollout.transfer_queue_coordinator import QueueCoordinator
from vime.utils.types import Sample

pytestmark = pytest.mark.unit
NUM_GPUS = 0


def group(round_id=0, group_id="g0", policy="v3"):
    samples = [
        Sample(
            index=i,
            group_index=5,
            rollout_id=8 + i,
            tokens=[1, 2, 3 + i],
            response="answer",
            response_length=1,
            reward=float(i),
            loss_mask=[1],
            weight_versions=[policy],
            rollout_log_probs=[-0.5],
            status=Sample.Status.COMPLETED,
            rollout_top_p_token_ids=[2, 3],
            rollout_top_p_token_offsets=[0, 2],
        )
        for i in range(2)
    ]
    return encode_group(
        samples,
        job_id="job",
        restart_epoch=1,
        group_id=group_id,
        attempt_id="0",
        expected_children=2,
        policy_version=policy,
        sampling_digest="digest",
        reward_key=None,
        collection_round=round_id,
    )


def test_codec_preserves_every_training_field_and_ragged_top_p():
    encoded = group()
    td = to_tensor_dict(encoded)
    decoded = decode_group(td, encoded)
    assert [s.reward for s in decoded] == [0.0, 1.0]
    assert [s.tokens for s in decoded] == [[1, 2, 3], [1, 2, 4]]
    assert [s.rollout_id for s in decoded] == [8, 9]
    assert decoded[0].weight_versions == ["v3"]
    assert decoded[0].rollout_log_probs == [-0.5]
    assert decoded[0].loss_mask == [1]
    assert decoded[0].rollout_top_p_token_ids == [2, 3]
    assert decoded[0].rollout_top_p_token_offsets == [0, 2]
    assert sum(t.numel() * t.element_size() for t in td.values()) == encoded.manifest.byte_count
    td["payload"][0, 0] = 0
    with pytest.raises(ValueError, match="manifest"):
        decode_group(td, encoded)


def test_duplicate_capacity_and_train_ack_boundary():
    encoded = group()
    c = QueueCoordinator(1, 6, encoded.working_bytes, 60)
    assert c.reserve(encoded)
    assert not c.reserve(encoded)
    with pytest.raises(ValueError, match="capacity"):
        c.reserve(group(group_id="g1"))
    c.mark_readable(encoded.manifest.payload_ref)
    c.seal((encoded.manifest.payload_ref,), "full-plan", "v3")
    with pytest.raises(ValueError, match="training"):
        c.finish_training(0, "full-plan")
    c.start_training()
    c.heartbeat()
    c.finish_training(0, "full-plan")
    with pytest.raises(RuntimeError, match="quiescent"):
        c.snapshot()
    c.reclaimed(encoded.manifest.payload_ref)
    c.publication_confirmed()
    assert c.snapshot() == {"schema_version": 1, "committed_round": 0}
    assert c.used == [0, 0, 0]
    with pytest.raises(ValueError, match="late"):
        c.reserve(encoded)
    c.reserve(group(round_id=1))


def test_optimizer_unknown_cannot_be_requeued_or_checkpointed():
    encoded = group()
    c = QueueCoordinator(2, 100, 100000, 60)
    c.reserve(encoded)
    c.mark_readable(encoded.manifest.payload_ref)
    c.seal((encoded.manifest.payload_ref,), "batch", "v3")
    c.start_training()
    c.fail()
    assert c.entries[encoded.manifest.payload_ref].ledger.phase.value == "unknown"
    with pytest.raises(ValueError):
        c.finish_training(0, "batch")
    with pytest.raises(RuntimeError):
        c.publication_confirmed()
    with pytest.raises(RuntimeError):
        c.snapshot()


def test_checkpoint_restores_only_committed_round():
    c = QueueCoordinator(2, 100, 100000, 60)
    c.restore({"schema_version": 1, "committed_round": 3})
    c.reserve(group(round_id=4))
    with pytest.raises(ValueError):
        c.reserve(group(round_id=3, group_id="late"))


def test_declared_byte_count_and_native_ready_boundary():
    encoded = group()
    changed = replace(encoded, manifest=replace(encoded.manifest, byte_count=encoded.manifest.byte_count + 1))
    with pytest.raises(ValueError, match="bytes"):
        to_tensor_dict(changed)
    c = QueueCoordinator(1, 100, encoded.working_bytes, 60)
    c.reserve(encoded)
    with pytest.raises(ValueError, match="read and validated"):
        c.seal((encoded.manifest.payload_ref,), "batch", "v3")


def test_working_set_budget_includes_codec_and_prepared_copies():
    encoded = group()
    assert encoded.working_bytes > 5 * encoded.manifest.byte_count
    with pytest.raises(ValueError, match="capacity"):
        QueueCoordinator(1, 100, encoded.working_bytes - 1, 60).reserve(encoded)


def test_default_rollout_id_and_mask_remain_nullable():
    decoded = decode_group(to_tensor_dict(group()), group())
    for sample in decoded:
        sample.rollout_id = None
        sample.loss_mask = None
        sample.session_id = "original-session"
        sample.spec_info.spec_accept_token_num = 3
        sample.prefix_cache_info.cached_tokens = 2
        sample.non_generation_time = 0.25
    encoded = encode_group(
        decoded,
        job_id="job",
        restart_epoch=1,
        group_id="g",
        attempt_id="0",
        expected_children=2,
        policy_version="v3",
        sampling_digest="digest",
        reward_key=None,
    )
    recovered = decode_group(to_tensor_dict(encoded), encoded)
    assert [sample.to_dict() for sample in recovered] == [sample.to_dict() for sample in decoded]
    assert [child.rollout_id for child in encoded.manifest.children] == [sample.index for sample in decoded]


@pytest.mark.parametrize(
    "snapshot", [{"schema_version": True, "committed_round": 0}, {"schema_version": 1, "committed_round": True}]
)
def test_checkpoint_rejects_boolean_schema_and_cursor(snapshot):
    with pytest.raises(ValueError, match="recovery"):
        QueueCoordinator(1, 100, 100000, 60).restore(snapshot)
