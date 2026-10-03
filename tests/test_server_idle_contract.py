"""Contract tests for the rollout server-idle completion check.

``abort_inflight_requests`` is best-effort by design, so the abort sweeps in
``vime/rollout/vllm_rollout.py`` returning does not by itself prove that the
engines have stopped working on requests. These tests pin the contract of the
check that answers that question. Ordering is driven through a controlled
transport rather than sleeps, so the results are deterministic.

The important direction is the strict one: only an explicit zero count may be
read as idle. A failed request, a malformed body, a missing ``num_requests`` or a
half-finished probe must all come back as UNKNOWN rather than idle, and the
``requests`` sample list (which the endpoint truncates by ``inflight_limit``)
must never be used to decide idleness.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import time

import httpx
import pytest

from vime.backends.vllm_utils import server_control
from vime.backends.vllm_utils.server_control import (
    DrainReport,
    EngineDrainState,
    abort_inflight_requests,
    query_engine_inflight,
    verify_server_drain,
)

URL_A = "http://engine-a:8000"
URL_B = "http://engine-b:8000"


def _load_body(counts: dict[str, int], *, samples: int = 0, ranks: int = 1) -> dict:
    """Build a ``/load?include_inflight=true`` body in the engine's real shape.

    Captured from a live engine (see ``REAL_IDLE_BODY``): ``inflight`` is a list
    with one entry per data-parallel rank, and the counts live in that rank's
    ``queues`` list. Reading ``num_requests`` off the rank entry itself finds
    nothing, which is the shape this contract silently got wrong before it was
    run against hardware.
    """
    return {
        "server_load": {},
        "inflight": [
            {
                "data_parallel_rank": rank,
                "queues": [
                    {"name": name, "num_requests": count, "requests": [{"id": f"r{i}"} for i in range(samples)]}
                    for name, count in counts.items()
                ],
            }
            for rank in range(ranks)
        ],
    }


# Verbatim from a live engine: GET /load?include_inflight=true&inflight_limit=1
# on vllm/vime:latest, idle. Kept as a literal so a schema change upstream fails
# here instead of silently degrading every probe to UNKNOWN.
REAL_IDLE_BODY = {
    "server_load": 0,
    "inflight": [
        {
            "data_parallel_rank": 0,
            "queues": [
                {"name": "running", "num_requests": 0, "requests": []},
                {"name": "waiting", "num_requests": 0, "requests": []},
                {"name": "skipped_waiting", "num_requests": 0, "requests": []},
            ],
        }
    ],
}


def _transport(handler) -> httpx.MockTransport:
    return httpx.MockTransport(handler)


def test_all_engines_idle_is_drained() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load_body({"waiting": 0, "running": 0}))

    report = asyncio.run(verify_server_drain([URL_A, URL_B], deadline_s=5.0, transport=_transport(handler)))
    assert report.drained
    assert report.busy == ()
    assert report.unknown_urls == ()


def test_real_engine_body_is_read_as_idle() -> None:
    """The verbatim idle body from a live engine must read as drained.

    Regression for the shape this contract originally got wrong: the counts sit
    one level deeper than the rank entry, so a shallow read returns UNKNOWN for
    every probe and the check can never confirm anything on real hardware.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=REAL_IDLE_BODY)

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert report.drained
    assert report.engines[0].num_requests == 0
    assert report.unknown_urls == ()


def test_counts_are_summed_across_data_parallel_ranks() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load_body({"running": 2, "waiting": 1}, ranks=3))

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert report.engines[0].num_requests == 9


def test_busy_engine_prevents_a_drained_verdict() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "engine-b":
            return httpx.Response(200, json=_load_body({"running": 3}))
        return httpx.Response(200, json=_load_body({"running": 0}))

    report = asyncio.run(verify_server_drain([URL_A, URL_B], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert [state.url for state in report.busy] == [URL_B]
    assert f"{URL_B}=3" in report.describe()


def test_truncated_sample_list_with_nonzero_count_is_not_idle() -> None:
    """The endpoint caps ``requests``; an empty list does not mean empty queues."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=_load_body({"running": 2}, samples=0))

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert report.engines[0].num_requests == 2


@pytest.mark.parametrize(
    "payload",
    [
        {"server_load": {}},
        {"inflight": "nope"},
        {"inflight": []},
        {"inflight": [{"data_parallel_rank": 0}]},
        {"inflight": [{"data_parallel_rank": 0, "queues": []}]},
        {"inflight": [{"data_parallel_rank": 0, "queues": "nope"}]},
        {"inflight": [{"data_parallel_rank": 0, "queues": [{"name": "running"}]}]},
        # The pre-hardware assumption: counts read off the rank entry itself.
        {"inflight": [{"name": "running", "num_requests": 0}]},
        {"inflight": [{"data_parallel_rank": 0, "queues": [{"name": "running", "num_requests": "3"}]}]},
        {"inflight": [{"data_parallel_rank": 0, "queues": [{"name": "running", "num_requests": -1}]}]},
        {"inflight": [{"data_parallel_rank": 0, "queues": [{"name": "running", "num_requests": True}]}]},
        "not an object",
    ],
)
def test_unreadable_counts_are_unknown_never_idle(payload: object) -> None:
    """Any body the check cannot read must not be mistaken for an idle engine."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json=payload)

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert report.engines[0].num_requests is None
    assert report.engines[0].detail


@pytest.mark.parametrize("status", [404, 500])
def test_http_errors_are_unknown_not_idle(status: int) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, json={})

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert report.unknown_urls == (URL_A,)


def test_unparseable_body_is_unknown_not_idle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"not json")

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert report.engines[0].num_requests is None


def test_an_unreachable_engine_is_unknown_not_idle() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("connection refused", request=request)

    report = asyncio.run(verify_server_drain([URL_A], deadline_s=5.0, transport=_transport(handler)))
    assert not report.drained
    assert report.unknown_urls == (URL_A,)


def test_deadline_bounds_the_whole_probe() -> None:
    """A hung engine consumes the shared budget instead of extending it."""
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "engine-b":
            await release.wait()
            return httpx.Response(200, json=_load_body({"running": 0}))
        return httpx.Response(200, json=_load_body({"running": 0}))

    async def run() -> tuple[object, float]:
        started = time.monotonic()
        try:
            report = await verify_server_drain([URL_A, URL_B], deadline_s=0.5, transport=_transport(handler))
            return report, time.monotonic() - started
        finally:
            release.set()

    report, elapsed = asyncio.run(run())
    assert elapsed < 2.0, f"probe overran its deadline: {elapsed:.3f}s"
    assert not report.drained
    assert URL_B in report.unknown_urls


def test_engines_are_probed_in_parallel() -> None:
    """Neither engine may finish before the other has started."""
    both_started = asyncio.Event()
    arrived: set[str] = set()

    async def handler(request: httpx.Request) -> httpx.Response:
        arrived.add(request.url.host)
        if len(arrived) == 2:
            both_started.set()
        await asyncio.wait_for(both_started.wait(), timeout=2.0)
        return httpx.Response(200, json=_load_body({"running": 0}))

    report = asyncio.run(verify_server_drain([URL_A, URL_B], deadline_s=5.0, transport=_transport(handler)))
    assert report.drained, report.describe()


def test_no_engines_is_not_a_drained_verdict() -> None:
    report = asyncio.run(verify_server_drain([], deadline_s=1.0))
    assert not report.drained


def test_probe_asks_for_the_inflight_diagnostics() -> None:
    seen: list[httpx.URL] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url)
        return httpx.Response(200, json=_load_body({"running": 0}))

    state = asyncio.run(query_engine_inflight(URL_A, timeout_s=5.0, transport=_transport(handler)))
    assert state.num_requests == 0
    assert seen[0].path == "/load"
    assert seen[0].params["include_inflight"] == "true"


def test_abort_reports_only_the_engines_it_could_not_reach(monkeypatch: pytest.MonkeyPatch) -> None:
    reached: list[str] = []

    async def fake_post(url: str, payload: dict, max_retries: int = 3, headers: dict | None = None) -> dict:
        reached.append(url)
        if url.startswith(URL_B):
            raise httpx.ConnectError("connection refused")
        return {}

    monkeypatch.setattr(server_control, "post", fake_post)
    failures = asyncio.run(abort_inflight_requests([URL_A, URL_B]))

    assert set(failures) == {URL_B}
    assert all(url.endswith("/abort_requests") for url in reached)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))


# --------------------------------------------------------------------------
# Controlled reproduction of the timing gap, against the real abort() flow.
#
# abort() cancels local tasks, sweeps the engines once, and then re-sweeps only
# while the local pending set is non-empty. Nothing else observes the server, so
# "local pending is empty" was the only completion signal. These tests drive the
# real abort() with a controlled router and engine to show what that means when
# a request is still queued server-side.
# --------------------------------------------------------------------------


class _FakeState:
    """Minimal GenerateState stand-in; no HF, no network."""

    def __init__(self, args: object) -> None:
        self.args = args
        self.aborted = False
        self.pendings: set = set()
        self.cancellable_tasks: set = set()
        self.active_server_generations = 1


def _abort_args(**overrides: object) -> argparse.Namespace:
    base = {
        "vllm_router_ip": "127.0.0.1",
        "vllm_router_port": 8000,
        "partial_rollout": False,
    }
    base.update(overrides)
    return argparse.Namespace(**base)


def test_abort_stops_sweeping_once_local_pending_is_empty(monkeypatch: pytest.MonkeyPatch) -> None:
    """The reproduction: local pending empty, server still reporting work.

    abort() must not be read as "the engines are idle". It sweeps once, sees no
    local pendings to wait for, and returns - so a request queued server-side
    after that sweep is never looked for again.
    """
    from vime.rollout import vllm_rollout

    sweeps: list[list[str]] = []

    async def fake_abort_inflight(urls: list[str]) -> dict[str, str]:
        sweeps.append(list(urls))
        return {}

    async def fake_get(url: str) -> dict:
        return {"workers": [{"url": URL_A}]}

    async def fake_verify(urls: list[str], *, deadline_s: float):
        return DrainReport((EngineDrainState(URL_A, 1),), elapsed_s=0.01, deadline_s=deadline_s)

    monkeypatch.setattr(vllm_rollout, "GenerateState", _FakeState)
    monkeypatch.setattr(vllm_rollout, "get", fake_get)
    monkeypatch.setattr(vllm_rollout, "abort_inflight_requests", fake_abort_inflight)
    monkeypatch.setattr(vllm_rollout, "verify_server_drain", fake_verify)

    aborted = asyncio.run(vllm_rollout.abort(_abort_args(), rollout_id=0))

    assert aborted == []
    assert len(sweeps) == 1, f"expected only the initial sweep, got {len(sweeps)}"
    assert sweeps[0] == [URL_A]


def test_abort_warns_when_the_engines_are_not_confirmed_idle(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """The gap must be visible in the log instead of being reported as success."""
    from vime.rollout import vllm_rollout

    async def fake_abort_inflight(urls: list[str]) -> dict[str, str]:
        return {}

    async def fake_get(url: str) -> dict:
        return {"workers": [{"url": URL_A}]}

    async def fake_verify(urls: list[str], *, deadline_s: float):
        return DrainReport(
            (EngineDrainState(URL_A, 2), EngineDrainState(URL_B, None, "timeout after 5.000s")),
            elapsed_s=5.0,
            deadline_s=deadline_s,
        )

    monkeypatch.setattr(vllm_rollout, "GenerateState", _FakeState)
    monkeypatch.setattr(vllm_rollout, "get", fake_get)
    monkeypatch.setattr(vllm_rollout, "abort_inflight_requests", fake_abort_inflight)
    monkeypatch.setattr(vllm_rollout, "verify_server_drain", fake_verify)

    with caplog.at_level(logging.WARNING):
        asyncio.run(vllm_rollout.abort(_abort_args(), rollout_id=0))

    warnings = [record.getMessage() for record in caplog.records if record.levelno == logging.WARNING]
    assert any("not confirmed idle" in message for message in warnings), warnings
    joined = " ".join(warnings)
    assert f"{URL_A}=2" in joined
    assert f"{URL_B}=unknown" in joined


def test_abort_skips_the_probe_when_the_server_was_never_used(monkeypatch: pytest.MonkeyPatch) -> None:
    """No server-side generations means there is nothing to verify."""
    from vime.rollout import vllm_rollout

    calls: list[str] = []

    class _IdleState(_FakeState):
        def __init__(self, args: object) -> None:
            super().__init__(args)
            self.active_server_generations = 0

    async def fake_verify(urls: list[str], *, deadline_s: float):
        calls.append("verify")
        return DrainReport((), elapsed_s=0.0, deadline_s=deadline_s)

    monkeypatch.setattr(vllm_rollout, "GenerateState", _IdleState)
    monkeypatch.setattr(vllm_rollout, "verify_server_drain", fake_verify)

    asyncio.run(vllm_rollout.abort(_abort_args(), rollout_id=0))
    assert calls == []


def test_cancelling_the_caller_leaves_no_probe_running() -> None:
    """A cancelled drain check must not leave probes polling the engines.

    ``asyncio.wait`` does not cancel the tasks it was given, so the cleanup has
    to happen on the way out of the coroutine rather than after it: otherwise a
    caller that is cancelled mid-check leaves one background probe per engine
    running to completion.
    """
    release = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        await release.wait()
        return httpx.Response(200, json=_load_body({"running": 0}))

    async def run() -> list[asyncio.Task]:
        probe = asyncio.create_task(
            verify_server_drain([URL_A, URL_B], deadline_s=30.0, transport=_transport(handler))
        )
        # Let both probes reach their await before cancelling the caller.
        for _ in range(4):
            await asyncio.sleep(0)
        probe.cancel()
        with pytest.raises(asyncio.CancelledError):
            await probe
        for _ in range(4):
            await asyncio.sleep(0)
        leftover = [task for task in asyncio.all_tasks() if task is not asyncio.current_task() and not task.done()]
        release.set()
        return leftover

    assert asyncio.run(run()) == []


@pytest.mark.parametrize("deadline", [True, 0, -1, float("nan"), float("inf")])
def test_drain_rejects_invalid_deadlines(deadline):
    with pytest.raises(ValueError, match="deadline"):
        asyncio.run(verify_server_drain([URL_A], deadline_s=deadline))
