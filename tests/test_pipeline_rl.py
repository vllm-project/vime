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


def test_engine_reset_uses_consistent_collective_order_across_workers(monkeypatch):
    from concurrent.futures import ThreadPoolExecutor
    from threading import Barrier
    from types import SimpleNamespace

    import _unit_stubs

    _unit_stubs.install_vllm_cli_stubs()
    from vime.backends.vllm_utils import vllm_engine as engine_module

    collective = Barrier(2)
    engines = []
    calls = [[], []]
    for index in range(2):
        engine = engine_module.VLLMEngine.__new__(engine_module.VLLMEngine)
        engine.args = SimpleNamespace(rollout_health_check_timeout=2)

        def request(route, payload, timeout, index=index):
            assert route == "shutdown_weight_transfer_engine"
            assert payload == {}
            collective.wait(timeout=timeout)
            calls[index].append(route)

        monkeypatch.setattr(engine, "_make_request", request)
        engines.append(engine)
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda engine: engine.reset_weights_update_groups(), engines))
    assert calls == [["shutdown_weight_transfer_engine"], ["shutdown_weight_transfer_engine"]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
