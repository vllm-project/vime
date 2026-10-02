"""Control-plane helpers for the rollout engines.

Two related jobs live here:

* aborting in-flight requests, which is inherently best-effort, and
* verifying that every rollout engine has actually gone idle, which callers need
  before they may treat a control-plane window as complete.
"""

from __future__ import annotations

import asyncio
import logging
import math
import time
from dataclasses import dataclass

import httpx

from vime.utils.http_utils import post

logger = logging.getLogger(__name__)

# Only the per-queue counts matter for idleness, so ask for as few request
# samples as the endpoint allows.
_INFLIGHT_SAMPLE_LIMIT = 1

# Upper bound on a single probe request. The caller's deadline still wins; this
# only stops one unresponsive engine from consuming the whole budget.
_PROBE_REQUEST_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class EngineDrainState:
    """What one engine reported about its in-flight queues.

    ``num_requests is None`` means UNKNOWN, never idle: a failed request, a
    malformed body or a missing count all land here.
    """

    url: str
    num_requests: int | None
    detail: str = ""

    @property
    def idle(self) -> bool:
        return self.num_requests == 0


@dataclass(frozen=True)
class DrainReport:
    engines: tuple[EngineDrainState, ...]
    elapsed_s: float
    deadline_s: float

    @property
    def drained(self) -> bool:
        """True only when every probed engine answered and reported zero."""
        return bool(self.engines) and all(state.idle for state in self.engines)

    @property
    def unknown_urls(self) -> tuple[str, ...]:
        return tuple(state.url for state in self.engines if state.num_requests is None)

    @property
    def busy(self) -> tuple[EngineDrainState, ...]:
        return tuple(state for state in self.engines if state.num_requests not in (0, None))

    def describe(self) -> str:
        parts = [
            f"{state.url}={'unknown' if state.num_requests is None else state.num_requests}"
            + (f"({state.detail})" if state.detail else "")
            for state in self.engines
        ]
        return (
            f"drained={self.drained} elapsed={self.elapsed_s:.3f}s "
            f"deadline={self.deadline_s:.3f}s engines=[{', '.join(parts)}]"
        )


async def abort_inflight_requests(urls: list[str]) -> dict[str, str]:
    """Abort all in-flight requests on each worker (one best-effort sweep).

    Posts to ``/abort_requests`` with an empty body; failures are logged, not
    raised. Idempotent, so the caller may re-issue it to converge.

    Returns the per-URL failure reason for the URLs that could not be reached,
    so a caller that needs stronger semantics can tell "aborted" from "cannot
    tell" instead of assuming success.
    """

    async def _abort_one(url: str) -> tuple[str, str]:
        try:
            await post(f"{url.rstrip('/')}/abort_requests", {}, max_retries=3)
        except httpx.HTTPError as e:
            logger.warning(f"Failed to abort requests on {url}: {e}")
            return url, str(e)
        return url, ""

    return {url: reason for url, reason in await asyncio.gather(*(_abort_one(u) for u in urls)) if reason}


def _discard_outcome(task: asyncio.Task[EngineDrainState]) -> None:
    """Mark a probe's result as consumed so cancellation cannot warn at GC."""
    if not task.cancelled():
        task.exception()


def _count_from_payload(payload: object) -> tuple[int | None, str]:
    """Extract the total in-flight count from a ``/load`` response body.

    The engine reports one entry per data-parallel rank, and the counts live in
    that rank's queue list::

        {"inflight": [{"data_parallel_rank": 0,
                       "queues": [{"name": "running", "num_requests": 3, ...}]}]}

    Only ``num_requests`` is read. The ``requests`` sample list is capped by
    ``inflight_limit`` and is legitimately empty while the count is not, so it
    must never be used to decide idleness.
    """
    if not isinstance(payload, dict):
        return None, "response is not an object"
    ranks = payload.get("inflight")
    if not isinstance(ranks, list) or not ranks:
        return None, "response has no inflight ranks"
    total = 0
    for rank in ranks:
        if not isinstance(rank, dict):
            return None, "inflight rank entry is not an object"
        queues = rank.get("queues")
        if not isinstance(queues, list) or not queues:
            return None, "inflight rank without queues"
        for queue in queues:
            if not isinstance(queue, dict) or "num_requests" not in queue:
                return None, "queue entry without num_requests"
            count = queue["num_requests"]
            if type(count) is not int or count < 0:
                return None, "num_requests is not a non-negative integer"
            total += count
    return total, ""


async def query_engine_inflight(
    url: str,
    *,
    timeout_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> EngineDrainState:
    """Ask one engine for its in-flight queue counts within ``timeout_s``."""
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError("probe timeout must be finite and positive")
    endpoint = f"{url.rstrip('/')}/load"
    params = {"include_inflight": "true", "inflight_limit": _INFLIGHT_SAMPLE_LIMIT}
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(timeout_s), transport=transport) as client:
            response = await client.get(endpoint, params=params)
            response.raise_for_status()
            payload = response.json()
    except httpx.TimeoutException:
        return EngineDrainState(url, None, f"timeout after {timeout_s:.3f}s")
    except httpx.HTTPError as e:
        return EngineDrainState(url, None, f"http error: {e}")
    except ValueError as e:
        return EngineDrainState(url, None, f"invalid json: {e}")

    count, detail = _count_from_payload(payload)
    return EngineDrainState(url, count, detail)


async def verify_server_drain(
    urls: list[str],
    *,
    deadline_s: float,
    transport: httpx.AsyncBaseTransport | None = None,
) -> DrainReport:
    """Check whether every engine has gone idle, inside a single deadline.

    All engines are probed in parallel and ``deadline_s`` bounds the whole
    operation, not each request: a slow engine consumes the shared budget rather
    than extending it. Anything other than an explicit zero count is reported as
    UNKNOWN/FAILED, so an unreachable or unrecognised engine can never be
    mistaken for an idle one.
    """
    if isinstance(deadline_s, bool) or not math.isfinite(deadline_s) or deadline_s <= 0:
        raise ValueError("drain deadline must be finite and positive")
    started = time.monotonic()
    if not urls:
        return DrainReport((), time.monotonic() - started, deadline_s)

    per_request_timeout = min(_PROBE_REQUEST_TIMEOUT_S, max(deadline_s, 0.0))
    tasks = [
        asyncio.create_task(query_engine_inflight(url, timeout_s=per_request_timeout, transport=transport))
        for url in urls
    ]
    done: set[asyncio.Task[EngineDrainState]] = set()
    try:
        done, _ = await asyncio.wait(tasks, timeout=deadline_s)
    finally:
        # Runs on every exit path, including the caller being cancelled: an
        # unfinished probe must never be left polling an engine in the
        # background. The callback consumes each outcome so a cancelled probe
        # cannot surface later as "exception was never retrieved".
        for task in tasks:
            if not task.done():
                task.cancel()
            task.add_done_callback(_discard_outcome)

    states = [task.result() for task in done]
    probed = {state.url for state in states}
    states.extend(
        EngineDrainState(url, None, "deadline exceeded before a result arrived") for url in urls if url not in probed
    )
    return DrainReport(tuple(states), time.monotonic() - started, deadline_s)
