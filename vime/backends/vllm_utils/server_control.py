import asyncio
import logging
from typing import Any

from vime.utils.http_utils import get, post

logger = logging.getLogger(__name__)

ABORT_RETRY_INTERVAL_SECONDS = 3
DEFAULT_ABORT_TIMEOUT_SECONDS = 180.0
DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS = 10.0


def num_requests_from_load(load: Any) -> int:
    if isinstance(load, list):
        return sum(num_requests_from_load(item) for item in load)
    if not isinstance(load, dict):
        return 0
    if "loads" in load:
        return num_requests_from_load(load["loads"])

    core_requests = load.get("server_load", 0)
    if not isinstance(core_requests, int) or isinstance(core_requests, bool):
        core_requests = 0
    detailed_counts = [core_requests]
    for key in ("inflight", "queues"):
        if key in load:
            detailed_counts.append(num_requests_from_load(load[key]))
    for key in ("num_requests", "num_reqs"):
        value = load.get(key)
        if isinstance(value, int) and not isinstance(value, bool):
            detailed_counts.append(value)
    return max(detailed_counts, default=0)


def non_idle_request_details(load: Any) -> list[dict[str, Any]]:
    if not isinstance(load, dict):
        return []

    details: list[dict[str, Any]] = []
    inflight = load.get("inflight", [])
    if not isinstance(inflight, list):
        return details
    for rank_load in inflight:
        if not isinstance(rank_load, dict):
            continue
        dp_rank = rank_load.get("data_parallel_rank")
        queues = rank_load.get("queues", [])
        if not isinstance(queues, list):
            continue
        for queue in queues:
            if not isinstance(queue, dict):
                continue
            requests = queue.get("requests") if isinstance(queue.get("requests"), list) else []
            count = queue.get("num_requests", len(requests))
            if isinstance(count, int) and count > 0:
                details.append(
                    {
                        "data_parallel_rank": dp_rank,
                        "queue": queue.get("name", "unknown"),
                        "num_requests": count,
                        "requests": requests,
                    }
                )
    return details


async def _abort_server_once(url: str, request_timeout: float) -> None:
    await asyncio.wait_for(
        post(f"{url.rstrip('/')}/abort_requests", {}, max_retries=1, timeout=request_timeout),
        timeout=request_timeout,
    )


async def _get_server_load(url: str, request_timeout: float) -> Any:
    return await asyncio.wait_for(
        get(
            f"{url.rstrip('/')}/load?include_inflight=true&inflight_limit=100",
            timeout=request_timeout,
        ),
        timeout=request_timeout,
    )


async def abort_server_until_idle(
    url: str,
    retry_interval: float = ABORT_RETRY_INTERVAL_SECONDS,
    *,
    timeout: float = DEFAULT_ABORT_TIMEOUT_SECONDS,
    request_timeout: float = DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS,
) -> None:
    if timeout <= 0 or request_timeout <= 0:
        raise ValueError("abort and control-request timeouts must be positive")

    loop = asyncio.get_running_loop()
    deadline = loop.time() + timeout
    attempt = 1
    last_error: Exception | None = None
    last_num_requests: int | None = None
    last_load: Any = None
    while True:
        remaining = deadline - loop.time()
        if remaining <= 0:
            detail = (
                f"last observed request count={last_num_requests}"
                if last_num_requests is not None
                else f"last error={last_error!r}"
            )
            if last_load is not None:
                detail += f", non-idle queues={non_idle_request_details(last_load)!r}"
            raise TimeoutError(f"Timed out draining vLLM server {url} after {timeout:.1f}s ({detail})")

        per_request_timeout = min(request_timeout, remaining)
        try:
            await _abort_server_once(url, per_request_timeout)
        except Exception as error:
            last_error = error
            logger.warning(f"Failed to abort vLLM server at {url}: {error}")

        try:
            remaining = deadline - loop.time()
            if remaining <= 0:
                continue
            load = await _get_server_load(url, min(request_timeout, remaining))
            num_requests = num_requests_from_load(load)
        except Exception as error:
            last_error = error
            logger.warning(f"Failed to get vLLM server load from {url}: {error}")
        else:
            last_load = load
            last_num_requests = num_requests
            if num_requests <= 0:
                return
            logger.info(
                "vLLM server %s still has %d requests after abort attempt %d; "
                "non-idle queues=%r; retrying in %s seconds.",
                url,
                num_requests,
                attempt,
                non_idle_request_details(load),
                retry_interval,
            )

        remaining = deadline - loop.time()
        if remaining > 0:
            await asyncio.sleep(min(retry_interval, remaining))
        attempt += 1


async def abort_servers_until_idle(
    urls: list[str],
    *,
    timeout: float = DEFAULT_ABORT_TIMEOUT_SECONDS,
    request_timeout: float = DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS,
) -> None:
    results = await asyncio.gather(
        *(abort_server_until_idle(url, timeout=timeout, request_timeout=request_timeout) for url in urls),
        return_exceptions=True,
    )
    failures = [(url, result) for url, result in zip(urls, results, strict=True) if isinstance(result, BaseException)]
    if failures:
        detail = "; ".join(f"{url}: {error}" for url, error in failures)
        raise RuntimeError(f"Failed to abort all vLLM servers to a confirmed idle state: {detail}") from failures[0][1]
