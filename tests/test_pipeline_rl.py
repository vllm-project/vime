"""CPU checks for PipelineRL weight-sync cache policy."""

import pytest

from vime.utils.weight_sync import should_flush_cache

NUM_GPUS = 0


@pytest.mark.parametrize(
    "interval,expected",
    [(-1, [1]), (-100, [1]), (0, [1]), (1, [1, 2, 3, 4, 5, 6, 7]), (2, [1, 3, 5, 7]), (3, [1, 4, 7])],
)
def test_flush_schedule_counts_training_updates_and_handles_restored_versions(interval, expected):
    assert [version for version in range(1, 8) if should_flush_cache(interval, version)] == expected
    # Restoring a nonzero serving version keeps the same phase.
    assert [version for version in range(5, 8) if should_flush_cache(interval, version)] == [
        version for version in expected if version >= 5
    ]


@pytest.mark.parametrize("interval", [-1, 0, 1, 2, 3])
def test_first_publication_after_resume_flushes_health_check_cache(interval):
    assert should_flush_cache(interval, weight_version=6, initial_weight_version=5)
    assert should_flush_cache(interval, weight_version=7, initial_weight_version=5) == should_flush_cache(interval, 7)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
