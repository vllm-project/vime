"""Exercise the scheduler contract without launching vLLM or hardware."""

from types import SimpleNamespace

import pytest

from vime.backends.vllm_utils import vllm_engine

NUM_GPUS = 0


@pytest.mark.parametrize("r3", [False, True])
@pytest.mark.parametrize("node_rank", [0, 1])
def test_abort_then_r3_reset_fully_pauses_without_losing_abort(monkeypatch, r3, node_rank):
    engine = object.__new__(vllm_engine.VLLMEngine)
    engine.args = SimpleNamespace(use_rollout_routing_replay=r3)
    engine.node_rank = node_rank
    engine.server_host, engine.server_port = "localhost", 8000
    state = "running"
    events = []

    def post(url, *, params=None, **kwargs):
        nonlocal state
        endpoint = url.rsplit("/", 1)[-1]
        events.append((endpoint, params))
        if endpoint == "pause":
            state = params["mode"]
        elif endpoint == "reset_prefix_cache":
            assert not r3 or state == "keep", "AuxOutput reset requires PAUSED_ALL"
            assert params == {"reset_running_requests": "true", "reset_external": "true"}
        elif endpoint == "resume":
            state = "running"
        return SimpleNamespace(raise_for_status=lambda: None, status_code=200, json=lambda: {"success": True})

    monkeypatch.setattr(vllm_engine.requests, "post", post)
    engine.pause_generation("abort")
    engine.flush_cache()
    engine.continue_generation()
    if node_rank:
        assert events == []
    else:
        expected = [("pause", {"mode": "abort", "clear_cache": "false"})]
        if r3:
            expected.append(("pause", {"mode": "keep", "clear_cache": "false"}))
        expected.extend(
            [
                ("reset_prefix_cache", {"reset_running_requests": "true", "reset_external": "true"}),
                ("resume", None),
            ]
        )
        assert events == expected
        assert state == "running"
