import asyncio
import json
from types import SimpleNamespace

import httpx
import pytest

from vime.backends.vllm_utils import server_control
from vime.utils import http_utils

NUM_GPUS = 0


@pytest.mark.parametrize(
    "operation,endpoint,payload",
    [
        ("init", "init_weight_transfer_engine", {"init_info": {}}),
        ("shutdown", "shutdown_weight_transfer_engine", {}),
        ("start", "start_weight_update", {}),
        ("start_draft", "start_draft_weight_update", {}),
        ("update", "update_weights", {"update_info": {}}),
        ("finish", "finish_weight_update", {"weight_version": "1"}),
    ],
)
@pytest.mark.parametrize("fail", [False, True])
def test_weight_operation_routes_record_success_and_failure(monkeypatch, operation, endpoint, payload, fail):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from prometheus_client import CollectorRegistry

    pytest.importorskip("vllm")
    from vllm.entrypoints.rl.online import api_router
    from vllm.entrypoints.rl.online import metrics as metrics_module

    registry = CollectorRegistry()
    metrics = metrics_module.WeightOperationMetrics(registry)
    monkeypatch.setattr(api_router, "weight_operation_metrics", lambda: metrics)
    timestamps = iter([1.0, 3.5])
    monkeypatch.setattr(metrics_module, "perf_counter", lambda: next(timestamps))
    labels = {"operation": operation}
    in_flight = "vllm:rl_weight_update_operations_in_flight"
    duration = "vllm:rl_weight_update_operation_duration_seconds"

    async def execute(*args, **kwargs):
        assert registry.get_sample_value(in_flight, labels) == 1
        if fail:
            raise RuntimeError("weight operation failed")

    app = FastAPI()
    app.include_router(api_router.router)
    app.state.engine_client = SimpleNamespace(**{endpoint: execute})
    with TestClient(app) as client:
        if fail:
            with pytest.raises(RuntimeError, match="weight operation failed"):
                client.post(f"/{endpoint}", json=payload)
        else:
            assert client.post(f"/{endpoint}", json=payload).status_code == 200

    assert registry.get_sample_value(in_flight, labels) == 0
    assert registry.get_sample_value(f"{duration}_count", labels) == 1
    assert registry.get_sample_value(f"{duration}_sum", labels) == 2.5


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
def test_abort_uses_supported_load_sections_and_waits_for_pd_transfers(monkeypatch):
    abort_calls = 0
    load_calls = 0

    async def post(url, payload, **kwargs):
        nonlocal abort_calls
        assert url == "http://engine/abort_requests"
        assert payload == {}
        abort_calls += 1

    async def get(url, **kwargs):
        nonlocal load_calls
        assert url == "http://engine/load?include_inflight=true&inflight_limit=100"
        load_calls += 1
        return {
            "server_load": 0,
            "inflight": [
                {
                    "data_parallel_rank": 0,
                    "has_pending_work": load_calls == 1,
                    "queues": [{"name": "kv_holding_waiting", "num_requests": int(load_calls == 1)}],
                }
            ],
        }

    monkeypatch.setattr(server_control, "post", post)
    monkeypatch.setattr(server_control, "get", get)
    asyncio.run(server_control.abort_server_until_idle("http://engine", retry_interval=0, timeout=0.1))
    assert abort_calls == load_calls == 2


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
@pytest.mark.parametrize("pending_work", [False, True])
def test_num_requests_includes_connector_activity(pending_work):
    load = {"server_load": 0, "inflight": [{"queues": [], "has_pending_work": pending_work}]}
    assert server_control.num_requests_from_load(load) == int(pending_work)


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


@pytest.mark.parametrize("cleanup_fails", [False, True])
def test_retirement_fences_cached_processes_before_releasing_slot(monkeypatch, cleanup_fails):
    from vime.backends.vllm_utils import engine_group

    calls = []
    actor = SimpleNamespace(shutdown=SimpleNamespace(remote=lambda: "shutdown"))
    group = engine_group.ServerGroup(
        args=SimpleNamespace(num_gpus_per_node=1),
        pg=None,
        num_gpus_per_engine=1,
        num_new_engines=0,
        all_engines=[actor],
        worker_type="encoder",
        engine_process_refs={0: "cached-identity"},
    )
    identity = {"node_id": "owner-node", "process": {"pid": 123}}

    def get(ref, **kwargs):
        if ref == "cached-identity":
            return identity
        assert ref == "fence"
        calls.append("wait-for-release")
        if cleanup_fails:
            raise TimeoutError("server still owns device")

    def options(**kwargs):
        assert kwargs["scheduling_strategy"].node_id == "owner-node"
        return SimpleNamespace(remote=lambda snapshot: "fence" if snapshot == identity["process"] else None)

    monkeypatch.setattr(engine_group.ray, "get", get)
    monkeypatch.setattr(engine_group.ray, "wait", lambda *a, **kw: ([], ["shutdown"]))
    monkeypatch.setattr(engine_group.ray, "kill", lambda *a, **kw: calls.append("kill-actor"))
    monkeypatch.setattr(engine_group, "_fence_server_processes", SimpleNamespace(options=options))
    if cleanup_fails:
        with pytest.raises(TimeoutError, match="still owns device"):
            group.retire_engine(0, timeout=0.1)
        assert group.all_engines == [actor] and group.engine_process_refs
    else:
        group.retire_engine(0, timeout=0.1)
        assert group.all_engines == [None] and not group.engine_process_refs
    assert calls == ["kill-actor", "wait-for-release"]
