from __future__ import annotations

import pytest
import torch

from vime.backends.speculative_training.feature_schema import DraftFeatureSample, VersionedFeatureQueue


def _sample(version: str = "7", sample_id: str = "sample-0", rows: int = 6) -> DraftFeatureSample:
    return DraftFeatureSample(
        input_ids=torch.arange(rows),
        loss_mask=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32)[:rows],
        position_ids=torch.arange(4, 4 + rows),
        aux_hidden_states=torch.randn(rows, 12),
        final_hidden_states=torch.randn(rows, 4),
        rollout_id=3,
        target_weight_version=version,
        original_sample_id=sample_id,
        prompt_length=2,
        response_length=rows - 2,
        window_start=4,
        window_end=4 + rows,
        aux_layer_ids=(2, 8, 13),
    )


@pytest.mark.unit
def test_feature_round_trip_normalizes_cpu_dtypes():
    sample = _sample()
    sample.algorithm = "dspark"
    payload = sample.to_payload()
    restored = DraftFeatureSample.from_payload(payload)

    assert restored.input_ids.dtype == torch.long
    assert restored.loss_mask.dtype == torch.float32
    assert restored.aux_hidden_states.dtype == torch.bfloat16
    assert restored.final_hidden_states.dtype == torch.bfloat16
    assert restored.target_weight_version == "7"
    assert restored.algorithm == "dspark"


@pytest.mark.unit
def test_feature_rejects_non_contiguous_positions():
    sample = _sample()
    sample.position_ids[-1] += 1

    with pytest.raises(ValueError, match="contiguous"):
        sample.validate(strict=True)


@pytest.mark.unit
def test_versioned_queue_never_accepts_mismatched_head_version():
    queue = VersionedFeatureQueue(max_samples=4)

    assert queue.add([_sample("1")], expected_version="2") == 0
    assert len(queue) == 0
    assert queue.rejected_version_mismatch == 1


@pytest.mark.unit
def test_versioned_queue_evicts_oldest_sample():
    queue = VersionedFeatureQueue(max_samples=2)
    queue.add([_sample("1", "a"), _sample("1", "b"), _sample("1", "c")])

    assert len(queue) == 2
    assert [sample.original_sample_id for sample in queue.take("1", 2)] == ["b", "c"]


@pytest.mark.unit
def test_consumed_queue_records_do_not_evict_newer_samples():
    queue = VersionedFeatureQueue(max_samples=2)
    queue.add([_sample("1", "a"), _sample("1", "b")])
    assert queue.take("1", 1)[0].original_sample_id == "a"

    queue.add([_sample("1", "c"), _sample("1", "d")])

    assert queue.count("1") == 2
    assert [sample.original_sample_id for sample in queue.take("1", 2)] == ["c", "d"]


@pytest.mark.unit
def test_repeated_training_batches_rotate_across_queued_samples():
    queue = VersionedFeatureQueue(max_samples=4)
    queue.add([_sample("1", "a"), _sample("1", "b"), _sample("1", "c")])

    assert [sample.original_sample_id for sample in queue.take("1", 2, repeat=True)] == ["a", "b"]
    assert [sample.original_sample_id for sample in queue.take("1", 2, repeat=True)] == ["c", "a"]
    assert queue.count("1") == 3


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
