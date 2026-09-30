import asyncio
import json

import pytest

from vime.backends.vllm_utils import server_control
from vime.utils import http_utils

NUM_GPUS = 0


@pytest.mark.unit
def test_num_requests_includes_all_dp_queues():
    load = {
        "server_load": 3,
        "inflight": [
            {
                "data_parallel_rank": 0,
                "queues": [{"name": "running", "num_requests": 2, "requests": ["a", "b"]}],
            },
            {
                "data_parallel_rank": 1,
                "queues": [{"name": "waiting", "num_requests": 3, "requests": ["c", "d", "e"]}],
            },
        ],
    }

    assert server_control.num_requests_from_load(load) == 5
    assert server_control.non_idle_request_details(load) == [
        {
            "data_parallel_rank": 0,
            "queue": "running",
            "num_requests": 2,
            "requests": ["a", "b"],
        },
        {
            "data_parallel_rank": 1,
            "queue": "waiting",
            "num_requests": 3,
            "requests": ["c", "d", "e"],
        },
    ]


@pytest.mark.unit
def test_abort_retries_load_failure_until_idle(monkeypatch):
    calls = 0

    async def abort_once(url, request_timeout):
        return None

    async def get_load(url, request_timeout):
        nonlocal calls
        calls += 1
        if calls == 1:
            raise OSError("temporary control-plane failure")
        return {"server_load": 0, "inflight": []}

    monkeypatch.setattr(server_control, "_abort_server_once", abort_once)
    monkeypatch.setattr(server_control, "_get_server_load", get_load)

    asyncio.run(server_control.abort_server_until_idle("http://engine", retry_interval=0, timeout=1))
    assert calls == 2


@pytest.mark.unit
def test_abort_servers_surfaces_partial_failure(monkeypatch):
    async def abort(url, **kwargs):
        if url.endswith("bad"):
            raise TimeoutError("not idle")

    monkeypatch.setattr(server_control, "abort_server_until_idle", abort)
    with pytest.raises(RuntimeError, match="bad.*not idle"):
        asyncio.run(server_control.abort_servers_until_idle(["http://good", "http://bad"]))


@pytest.mark.unit
def test_http_post_forwards_timeout_and_closes_response():
    class Response:
        text = ""
        closed = False

        def raise_for_status(self):
            return None

        async def aread(self):
            return json.dumps({"ok": True}).encode()

        async def aclose(self):
            self.closed = True

    response = Response()

    class Client:
        async def post(self, url, **kwargs):
            assert kwargs["timeout"] == 2.5
            return response

    assert asyncio.run(http_utils._post(Client(), "http://engine", {}, max_retries=1, timeout=2.5)) == {"ok": True}
    assert response.closed


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
