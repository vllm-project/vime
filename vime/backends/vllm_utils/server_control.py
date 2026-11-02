import asyncio
import logging
import re
import sys
from typing import Any
from urllib.parse import quote

import httpx
import requests

from vime.utils.http_utils import post

logger = logging.getLogger(__name__)

DEFAULT_ABORT_TIMEOUT_SECONDS = 180.0
DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS = 10.0


def register_worker(router_url: str, payload: dict[str, Any]) -> None:
    """Register a worker with the router, keeping a matching registration it already holds.

    An external engine outlives the run that registered it, and a run that ends
    uncleanly leaves its registrations behind. A router whose POST /workers is
    create-only answers the next run's POST with 409 WORKER_ALREADY_EXISTS; the
    registration it holds is kept when it matches this payload.
    """
    response = requests.post(f"{router_url}/workers", json=payload)
    if response.status_code == 409 and _holds_registration(router_url, payload, response):
        logger.info(f"Router {router_url} already has worker {payload['url']} registered; keeping it.")
        return
    response.raise_for_status()


def _holds_registration(router_url: str, payload: dict[str, Any], conflict: requests.Response) -> bool:
    try:
        code = conflict.json().get("code")
    except ValueError:
        return False
    if code != "WORKER_ALREADY_EXISTS":
        return False
    response = requests.get(f"{router_url}/workers", timeout=DEFAULT_CONTROL_REQUEST_TIMEOUT_SECONDS)
    response.raise_for_status()
    workers = [worker for worker in response.json()["workers"] if _worker_base_url(worker["url"]) == payload["url"]]
    return bool(workers) and all(
        all(worker.get(key) == value for key, value in payload.items() if key != "url") for worker in workers
    )


def _worker_base_url(worker_url: str) -> str:
    """Resolve native router @usize identities without rewriting malformed suffixes."""
    parts = worker_url.split("@")
    if len(parts) == 2 and re.fullmatch(r"\+?[0-9]+", parts[1]):
        rank = parts[1].lstrip("+0") or "0"
        max_rank = 2 * sys.maxsize + 1
        if len(rank) <= len(str(max_rank)) and int(rank) <= max_rank:
            return parts[0]
    return worker_url


def unregister_worker(router_url: str, worker_url: str, *, timeout: float) -> None:
    """Remove a worker even when its engine actor can no longer answer RPCs."""
    with requests.Session() as client:
        client.trust_env = False
        response = client.get(f"{router_url}/workers", timeout=timeout)
        response.raise_for_status()
        for worker in response.json()["workers"]:
            if worker["url"] == worker_url or _worker_base_url(worker["url"]) == worker_url:
                response = client.delete(f"{router_url}/workers/{quote(worker['url'], safe='')}", timeout=timeout)
                # Another health check may already have removed this worker.
                if response.status_code != 404:
                    response.raise_for_status()


async def get_live_router_workers(router_url: str, *, timeout: float) -> list[dict]:
    """Prune failed workers before rollout drains requests through the router.

    This precedes the manager's end-of-rollout check: the rollout function itself
    must finish abort/drain before it can return to that manager. Probe HTTP
    directly so cleanup never waits for a dead engine actor to answer an RPC.
    """
    async with httpx.AsyncClient(timeout=timeout, trust_env=False) as client:
        response = await client.get(f"{router_url}/workers")
        response.raise_for_status()

        async def check(worker):
            try:
                response = await client.get(f"{_worker_base_url(worker['url'])}/health_generate")
                response.raise_for_status()
            except httpx.HTTPError as error:
                # A registered URL is not proof of a live server. Remove it
                # before abort/drain can wait on requests the server cannot finish.
                logger.warning("Removing failed rollout worker %s: %s", worker["url"], error)
                response = await client.delete(f"{router_url}/workers/{quote(worker['url'], safe='')}")
                if response.status_code != 404:
                    response.raise_for_status()
                return None
            return worker

        # Probe concurrently so multiple failed servers do not multiply the wait.
        workers = await asyncio.gather(*(check(worker) for worker in response.json()["workers"]))
        return [worker for worker in workers if worker is not None]


async def set_servers_paused(urls: list[str], *, paused: bool, timeout: float = DEFAULT_ABORT_TIMEOUT_SECONDS) -> None:
    if timeout <= 0:
        raise ValueError("control timeout must be positive")
    urls = list(dict.fromkeys(_worker_base_url(url).rstrip("/") for url in urls))
    endpoint = "pause?mode=abort&clear_cache=false" if paused else "resume"
    results = await asyncio.gather(
        *(
            asyncio.wait_for(
                post(f"{url}/{endpoint}", {}, max_retries=1, timeout=timeout),
                timeout=timeout,
            )
            for url in urls
        ),
        return_exceptions=True,
    )
    failures = [(url, result) for url, result in zip(urls, results, strict=True) if isinstance(result, BaseException)]
    if failures:
        detail = "; ".join(f"{url}: {error}" for url, error in failures)
        raise RuntimeError(f"Failed to {endpoint.split('?')[0]} vLLM servers: {detail}") from failures[0][1]
