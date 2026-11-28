import asyncio
import json
import sys
from types import SimpleNamespace
from urllib.parse import quote

import httpx
import pytest
import requests

from vime.backends.vllm_utils import server_control
from vime.utils import http_utils

NUM_GPUS = 0


def _router_response(status_code, payload):
    response = requests.Response()
    response.status_code = status_code
    response._content = json.dumps(payload).encode()
    response.url = "http://router/workers"
    return response


def _fake_router(monkeypatch, *, post_status, post_body, workers):
    calls = []

    def fake_post(url, json):
        calls.append(("POST", url, json))
        return _router_response(post_status, post_body)

    def fake_get(url, timeout):
        calls.append(("GET", url))
        return _router_response(200, {"workers": workers})

    monkeypatch.setattr(server_control.requests, "post", fake_post)
    monkeypatch.setattr(server_control.requests, "get", fake_get)
    return calls


EXISTS = {"error": "Worker already exists at URL 'http://e:1'", "code": "WORKER_ALREADY_EXISTS"}


@pytest.mark.unit
@pytest.mark.parametrize("status", [200, 202])
def test_register_worker_posts_the_payload(monkeypatch, status):
    calls = _fake_router(monkeypatch, post_status=status, post_body={"worker_id": "w1"}, workers=[])

    server_control.register_worker("http://router", {"url": "http://e:1", "worker_type": "regular"})

    assert calls == [("POST", "http://router/workers", {"url": "http://e:1", "worker_type": "regular"})]


@pytest.mark.unit
def test_register_worker_keeps_the_registration_an_earlier_run_left(monkeypatch):
    # A router whose POST /workers is create-only (409 WORKER_ALREADY_EXISTS)
    # still holds an external engine registered by the previous run.
    payload = {"url": "http://e:1", "worker_type": "prefill", "bootstrap_port": 8998}
    calls = _fake_router(
        monkeypatch,
        post_status=409,
        post_body=EXISTS,
        workers=[{"id": "w1", "url": "http://e:1", "worker_type": "prefill", "bootstrap_port": 8998, "priority": 50}],
    )

    server_control.register_worker("http://router", payload)

    assert [call[0] for call in calls] == ["POST", "GET"]


@pytest.mark.unit
@pytest.mark.parametrize(
    "existing",
    [
        {"id": "w1", "url": "http://e:1", "worker_type": "decode"},
        {"id": "w1", "url": "http://e:1", "worker_type": "prefill", "bootstrap_port": 9000},
        {"id": "w1", "url": "http://e:1", "worker_type": "prefill"},
    ],
)
def test_register_worker_refuses_a_registration_that_differs(monkeypatch, existing):
    _fake_router(monkeypatch, post_status=409, post_body=EXISTS, workers=[existing])

    with pytest.raises(requests.HTTPError):
        server_control.register_worker(
            "http://router", {"url": "http://e:1", "worker_type": "prefill", "bootstrap_port": 8998}
        )


@pytest.mark.unit
@pytest.mark.parametrize(
    "status, body",
    [(409, {"error": "in progress", "code": "WORKER_CREATE_IN_PROGRESS"}), (409, {}), (500, EXISTS)],
)
def test_register_worker_raises_any_other_refusal(monkeypatch, status, body):
    _fake_router(
        monkeypatch,
        post_status=status,
        post_body=body,
        workers=[{"id": "w1", "url": "http://e:1", "worker_type": "regular"}],
    )

    with pytest.raises(requests.HTTPError):
        server_control.register_worker("http://router", {"url": "http://e:1", "worker_type": "regular"})


@pytest.mark.parametrize("conflicting_rank", [None, 1])
def test_register_worker_checks_every_native_dp_slot(monkeypatch, conflicting_rank):
    workers = [{"url": f"http://e:1@{rank}", "worker_type": "prefill", "bootstrap_port": 8998} for rank in range(2)]
    if conflicting_rank is not None:
        workers[conflicting_rank]["bootstrap_port"] = 9000
    _fake_router(monkeypatch, post_status=409, post_body=EXISTS, workers=workers)
    payload = {"url": "http://e:1", "worker_type": "prefill", "bootstrap_port": 8998}
    if conflicting_rank is None:
        server_control.register_worker("http://router", payload)
    else:
        with pytest.raises(requests.HTTPError):
            server_control.register_worker("http://router", payload)


@pytest.mark.parametrize("worker_type", ["prefill", "decode"])
@pytest.mark.parametrize("suffixes", [("",), ("@0", "@1"), ("@+1", "@01"), (f"@{2 * sys.maxsize + 1}",)])
def test_unregister_worker_removes_only_matching_router_slots(monkeypatch, worker_type, suffixes):
    base_url = f"http://{worker_type}:8000"
    matched_urls = [base_url + suffix for suffix in suffixes]
    unrelated_urls = [
        "http://other:8000@0",
        base_url + "-other@0",
        *[
            base_url + suffix
            for suffix in ("@", "@abc", "@-1", "@1.0", "@1@0", "@ 1", "@１", "@1_0", f"@{2 * sys.maxsize + 2}")
        ],
    ]
    workers = [
        {"id": str(index), "url": url, "worker_type": worker_type}
        for index, url in enumerate(matched_urls + unrelated_urls)
    ]
    calls = []

    def send(client, request, **kwargs):
        assert not client.trust_env
        assert kwargs["timeout"] == 0.1
        calls.append((request.method, request.url))
        response = server_control.requests.Response()
        response.url = request.url
        response.status_code = 200
        if request.method == "GET":
            response._content = json.dumps({"workers": workers}).encode()
        elif request.url == f"http://router/workers/{quote(matched_urls[0], safe='')}":
            response.status_code = 404
        return response

    monkeypatch.setattr(server_control.requests.Session, "send", send)
    server_control.unregister_worker("http://router", base_url, timeout=0.1)

    assert calls == [("GET", "http://router/workers")] + [
        ("DELETE", f"http://router/workers/{quote(url, safe='')}") for url in matched_urls
    ]


@pytest.mark.parametrize("failed_type", ["prefill", "decode"])
def test_dp_worker_health_prunes_slots_and_drains_base_servers(monkeypatch, failed_type):
    bases = {"prefill": "http://prefill:8000", "decode": "http://decode:9000"}
    workers = [
        {"id": f"{worker_type}-{rank}", "url": f"{base}@{rank}", "worker_type": worker_type}
        for worker_type, base in bases.items()
        for rank in range(2)
    ] + [{"id": "plain", "url": "http://plain:7000", "worker_type": "regular"}]
    calls = []
    rpc_calls = []

    async def respond(request):
        calls.append((request.method, str(request.url)))
        if request.url.path == "/workers":
            return httpx.Response(200, json={"workers": workers})
        if request.method == "DELETE":
            return httpx.Response(404 if str(request.url).endswith("%400") else 200)
        assert str(request.url) in {f"{base}/health_generate" for base in [*bases.values(), "http://plain:7000"]}
        return httpx.Response(503 if str(request.url) == f"{bases[failed_type]}/health_generate" else 200)

    async def post(url, payload, **kwargs):
        rpc_calls.append(("POST", url))
        assert payload == {}

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(server_control.httpx, "AsyncClient", lambda **kwargs: client)
    monkeypatch.setattr(server_control, "post", post)

    async def prune_and_drain():
        live = await server_control.get_live_router_workers("http://router", timeout=0.1)
        await server_control.set_servers_paused([worker["url"] for worker in live], paused=True)
        return live

    live = asyncio.run(prune_and_drain())
    assert live == [worker for worker in workers if worker["worker_type"] != failed_type]
    assert sorted(url for method, url in calls if method == "DELETE") == sorted(
        f"http://router/workers/{quote(worker['url'], safe='')}"
        for worker in workers
        if worker["worker_type"] == failed_type
    )
    live_bases = [base for worker_type, base in bases.items() if worker_type != failed_type] + ["http://plain:7000"]
    assert sorted(rpc_calls) == sorted([("POST", f"{base}/pause?mode=abort&clear_cache=false") for base in live_bases])


def test_abort_dp_worker_list_drains_each_base_once_without_health_pruning(monkeypatch):
    bases = ["http://prefill:8000", "http://decode:9000", "http://plain:7000"]
    urls = [f"{base}@{rank}" for base in bases[:2] for rank in range(2)] + [bases[2], bases[2]]
    calls = []

    async def post(url, payload, **kwargs):
        calls.append(("POST", url))

    monkeypatch.setattr(server_control, "post", post)
    asyncio.run(server_control.set_servers_paused(urls, paused=True))
    assert sorted(calls) == sorted([("POST", f"{base}/pause?mode=abort&clear_cache=false") for base in bases])


@pytest.mark.parametrize("suffix", ["", "@0", "@+1", "@01"])
def test_abort_dp_worker_uses_ipv6_base_endpoint(monkeypatch, suffix):
    calls = []
    base_url = "http://[::1]:8000"

    async def post(url, payload, **kwargs):
        calls.append(url)

    monkeypatch.setattr(server_control, "post", post)
    asyncio.run(server_control.set_servers_paused([base_url + suffix], paused=True))
    assert calls == [f"{base_url}/pause?mode=abort&clear_cache=false"]


@pytest.mark.parametrize("shutdown_fails", [False, True])
def test_colocated_engine_retirement_waits_for_gpu_release(monkeypatch, shutdown_fails):
    from vime.backends.vllm_utils import engine_group

    engine = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: "shutdown"))
    group = engine_group.ServerGroup(
        args=SimpleNamespace(num_gpus_per_node=1),
        pg=None,
        all_engines=[engine],
        num_gpus_per_engine=1,
        num_new_engines=0,
        worker_type="encoder",
        needs_offload=True,
    )
    events = []

    def get(refs, *, timeout):
        assert refs == ["shutdown"] and timeout == 0.1
        events.append("wait for shutdown")
        if shutdown_fails:
            raise TimeoutError("server still holds GPUs")

    monkeypatch.setattr(engine_group.ray, "get", get)
    monkeypatch.setattr(engine_group.ray, "kill", lambda actor, **kw: events.append("kill actor"))
    if shutdown_fails:
        with pytest.raises(TimeoutError, match="server still holds GPUs"):
            group.retire_engine(0, timeout=0.1)
        assert group.all_engines == [engine]
    else:
        group.retire_engine(0, timeout=0.1)
        assert group.all_engines == [None]
    assert events == ["wait for shutdown", "kill actor"]


@pytest.mark.parametrize("failure", ["connection", "timeout", "status"])
def test_rollout_boundary_prunes_failed_router_workers(monkeypatch, failure):
    calls = []

    async def respond(request):
        calls.append((request.method, str(request.url)))
        if request.url.path == "/workers":
            return httpx.Response(
                200, json={"workers": [{"id": "good", "url": "http://good"}, {"id": "bad", "url": "http://bad"}]}
            )
        if request.method == "DELETE":
            return httpx.Response(200)
        if request.url.host == "bad":
            if failure == "connection":
                raise httpx.ConnectError("server stopped", request=request)
            if failure == "timeout":
                raise httpx.ReadTimeout("server stuck", request=request)
            return httpx.Response(503)
        return httpx.Response(200)

    client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    monkeypatch.setattr(server_control.httpx, "AsyncClient", lambda **kwargs: client)
    workers = asyncio.run(server_control.get_live_router_workers("http://router", timeout=0.1))
    assert workers == [{"id": "good", "url": "http://good"}]
    assert ("DELETE", "http://router/workers/http%3A%2F%2Fbad") in calls
    assert ("DELETE", "http://router/workers/http%3A%2F%2Fgood") not in calls


@pytest.mark.parametrize("actor_failure", ["dead", "wedged"])
def test_boundary_health_check_bypasses_grace_and_cleans_dead_actor(monkeypatch, actor_failure):
    from vime.backends.vllm_utils import engine_group
    from vime.ray.serving import ServingCluster
    from vime.utils import health_monitor

    calls = []
    dead = SimpleNamespace(
        health_generate=SimpleNamespace(remote=lambda **kw: "dead"),
        shutdown=SimpleNamespace(remote=lambda: "shutdown"),
    )
    alive = SimpleNamespace(health_generate=SimpleNamespace(remote=lambda **kw: "alive"))
    group = engine_group.ServerGroup(
        args=SimpleNamespace(num_gpus_per_node=1),
        pg=None,
        num_gpus_per_engine=1,
        num_new_engines=0,
        all_engines=[dead, alive],
        worker_type="regular",
        engine_urls={0: "http://dead", 1: "http://alive"},
        router_ip="router",
        router_port=8000,
    )
    args = SimpleNamespace(
        rollout_health_check_interval=600, rollout_health_check_first_wait=600, rollout_health_check_timeout=0.1
    )
    monitor = health_monitor.RolloutHealthMonitor(group, args)
    serving = object.__new__(ServingCluster.__ray_metadata__.modified_class)
    serving.args = args
    serving._health_monitors = [monitor]
    serving.servers = {"model": group}

    def wait(refs, *, num_returns, timeout):
        assert timeout == 0.1
        if refs == ["shutdown"]:
            return [], refs
        assert refs == ["dead", "alive"] and num_returns == 2
        return (["alive"], ["dead"]) if actor_failure == "wedged" else (refs, [])

    def get(ref, *, timeout):
        if ref in {"dead", "shutdown"}:
            raise TimeoutError(actor_failure)
        return True

    def unregister(router_url, worker_url, *, timeout):
        calls.append(("unregister", router_url, worker_url))

    monkeypatch.setattr(health_monitor.ray, "wait", wait)
    monkeypatch.setattr(health_monitor.ray, "get", get)
    monkeypatch.setattr(health_monitor.ray, "kill", lambda engine, **kw: calls.append(("kill", engine)))
    monkeypatch.setattr(engine_group, "unregister_worker", unregister)
    monitor.start()
    try:
        snapshot = serving.finish_rollout()
        assert calls == [("unregister", "http://router:8000", "http://dead"), ("kill", dead)]
        assert snapshot["model"].all_engines == [None, alive]
    finally:
        monitor.stop()


@pytest.mark.unit
@pytest.mark.parametrize("paused", [False, True])
def test_server_control_waits_for_native_completion(monkeypatch, paused):
    completed = []
    endpoint = "pause?mode=abort&clear_cache=false" if paused else "resume"

    async def post(url, payload, **kwargs):
        assert url == f"http://engine/{endpoint}"
        assert payload == {} and kwargs == {"max_retries": 1, "timeout": 1}
        await asyncio.sleep(0)
        completed.append(url)

    monkeypatch.setattr(server_control, "post", post)
    asyncio.run(server_control.set_servers_paused(["http://engine"], paused=paused, timeout=1))
    assert completed == [f"http://engine/{endpoint}"]


@pytest.mark.unit
def test_server_control_surfaces_partial_failure(monkeypatch):
    completed = []

    async def post(url, payload, **kwargs):
        if url.startswith("http://bad/"):
            raise OSError("engine unreachable")
        await asyncio.sleep(0)
        completed.append(url)

    monkeypatch.setattr(server_control, "post", post)
    with pytest.raises(RuntimeError, match="bad.*engine unreachable"):
        asyncio.run(server_control.set_servers_paused(["http://good", "http://bad"], paused=True))
    assert completed == ["http://good/pause?mode=abort&clear_cache=false"]


@pytest.mark.unit
def test_server_control_timeout_is_not_success(monkeypatch):
    async def post(url, payload, **kwargs):
        await asyncio.Event().wait()

    monkeypatch.setattr(server_control, "post", post)
    with pytest.raises(RuntimeError, match="Failed to pause"):
        asyncio.run(server_control.set_servers_paused(["http://engine"], paused=True, timeout=0.01))


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


@pytest.mark.unit
def test_paused_generation_is_not_retried(monkeypatch):
    calls = []

    async def respond(request):
        calls.append(request)
        return httpx.Response(503, json={"error": {"message": "Generation is paused. Please retry after it resumes."}})

    async def exercise():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            with pytest.raises(httpx.HTTPStatusError) as caught:
                await http_utils._post(client, "http://engine/inference/v1/generate", {})
            assert http_utils.is_generation_paused_error(caught.value)

    asyncio.run(exercise())
    assert len(calls) == 1


@pytest.mark.unit
@pytest.mark.parametrize("concurrency", [2, 257])
def test_http_transport_preserves_streams_timeouts_and_connection_budget(concurrency):
    import httpx

    async def check():
        active = peak = accepted = 0
        tasks = set()

        async def serve(reader, writer):
            nonlocal active, peak, accepted
            task = asyncio.current_task()
            tasks.add(task)
            active += 1
            accepted += 1
            peak = max(peak, active)
            try:
                while True:
                    headers = await reader.readuntil(b"\r\n\r\n")
                    length = next(
                        (
                            int(line.split(b":", 1)[1])
                            for line in headers.split(b"\r\n")
                            if line.lower().startswith(b"content-length:")
                        ),
                        0,
                    )
                    body = await reader.readexactly(length)
                    if b"/slow " in headers:
                        await asyncio.sleep(0.1)
                    writer.write(b"HTTP/1.1 200 OK\r\nContent-Length: " + str(len(body)).encode() + b"\r\n\r\n" + body)
                    await writer.drain()
            except (asyncio.IncompleteReadError, ConnectionError):
                pass
            finally:
                active -= 1
                writer.close()
                await writer.wait_closed()
                tasks.discard(task)

        server = await asyncio.start_server(serve, "127.0.0.1", 0, backlog=2 * concurrency)
        url = f"http://127.0.0.1:{server.sockets[0].getsockname()[1]}"
        try:
            async with httpx.AsyncClient(transport=http_utils._ShardedHTTPTransport(concurrency)) as client:

                async def request(index):
                    body = str(index).encode()
                    async with client.stream("POST", url, content=body) as response:
                        assert await response.aread() == body

                for _ in range(2):
                    await asyncio.gather(*(request(index) for index in range(concurrency + 3)))
                assert peak <= concurrency
                assert accepted <= concurrency  # The second wave reused live sockets.
                with pytest.raises(httpx.ReadTimeout):
                    await client.post(url + "/slow", content=b"x", timeout=0.01)
                assert (await client.post(url, content=b"after-timeout")).content == b"after-timeout"
        finally:
            server.close()
            await server.wait_closed()
            await asyncio.gather(*tasks, return_exceptions=True)
        assert active == 0

    asyncio.run(check())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
