"""Control-plane helper for aborting in-flight requests on vLLM workers."""

import asyncio
import logging

from vime.utils.http_utils import post

logger = logging.getLogger(__name__)


async def abort_inflight_requests(urls: list[str]) -> None:
    """Abort all in-flight requests on each worker (one best-effort sweep).

    Posts to ``/abort_requests`` with an empty body; failures are logged, not
    raised. Idempotent, so the caller may re-issue it to converge.
    """

    async def _abort_one(url: str) -> None:
        try:
            await post(f"{url.rstrip('/')}/abort_requests", {}, max_retries=3)
        except Exception as e:
            logger.warning(f"Failed to abort requests on {url}: {e}")

    await asyncio.gather(*(_abort_one(url) for url in urls))
