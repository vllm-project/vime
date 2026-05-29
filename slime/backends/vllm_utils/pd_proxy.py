"""Prefill/Decode (PD) disaggregation proxy for the vLLM rollout backend.

vLLM does PD via the NIXL KV connector: a *prefill* engine computes the prompt
KV and a *decode* engine pulls it over a NIXL side channel, then generates. The
two engines coordinate through ``kv_transfer_params`` carried on the
``/inference/v1/generate`` request/response (vLLM's ``GenerateRequest`` /
``GenerateResponse`` both expose this field).

The stock ``vllm_router`` only implements SGLang-style PD (simultaneous dispatch
+ bootstrap room), so this module supplies a vLLM-native proxy instead. It is a
drop-in for the rollout flow: it exposes the same worker-registration surface
the engines already use against the router (``POST/GET/DELETE /workers``) plus
``POST /inference/v1/generate`` (the relay), ``GET /health``, and broadcast
control routes. Engines register themselves with ``worker_type`` ("prefill" /
"decode"); the proxy round-robins across the registered pools and drives the
two-step handshake:

    1. POST to a *prefill* worker with ``kv_transfer_params={"do_remote_decode": True}``
       and ``max_tokens=1``. It returns ``kv_transfer_params`` (remote handles).
    2. POST the original request to a *decode* worker with
       ``kv_transfer_params={"do_remote_prefill": True, **prefill_params}``. It
       pulls the KV via NIXL and generates the full response.

Engines are launched with the NIXL ``--kv-transfer-config`` by
``vllm_engine._build_vllm_cmd_and_env`` (role-driven by ``worker_type``).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import logging
import threading

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

GENERATE_PATH = "/inference/v1/generate"


class PDProxy:
    def __init__(self, request_timeout: int = 14400):
        self._workers: dict[str, str] = {}  # url -> worker_type
        self._lock = threading.Lock()
        self._pf_rr = None
        self._dc_rr = None
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._session: aiohttp.ClientSession | None = None

    # ---- worker registry (mirrors the vllm-router /workers surface) --------
    def _rebuild_rr(self):
        pf = [u for u, t in self._workers.items() if t == "prefill"]
        dc = [u for u, t in self._workers.items() if t == "decode"]
        self._pf_rr = itertools.cycle(pf) if pf else None
        self._dc_rr = itertools.cycle(dc) if dc else None

    async def add_worker(self, request: web.Request) -> web.Response:
        body = await request.json()
        url = body["url"].rstrip("/")
        wtype = body.get("worker_type", "regular")
        with self._lock:
            self._workers[url] = wtype
            self._rebuild_rr()
        logger.info("PD proxy: registered %s worker %s (total=%d)", wtype, url, len(self._workers))
        return web.json_response({"ok": True})

    async def list_workers(self, request: web.Request) -> web.Response:
        with self._lock:
            workers = [{"url": u, "worker_type": t} for u, t in self._workers.items()]
        return web.json_response({"workers": workers, "urls": [w["url"] for w in workers]})

    async def remove_worker(self, request: web.Request) -> web.Response:
        from urllib.parse import unquote

        url = unquote(request.match_info["url"]).rstrip("/")
        with self._lock:
            self._workers.pop(url, None)
            self._rebuild_rr()
        return web.json_response({"ok": True})

    # ---- generation relay --------------------------------------------------
    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def handle_generate(self, request: web.Request) -> web.Response:
        await self._ensure_session()
        with self._lock:
            prefill = next(self._pf_rr) if self._pf_rr else None
            decode = next(self._dc_rr) if self._dc_rr else None
        if prefill is None or decode is None:
            return web.json_response(
                {"error": f"PD proxy missing workers (prefill={prefill}, decode={decode})"}, status=503
            )
        body = await request.json()

        # Step 1: prefill (1 token, mark for remote decode).
        pf_body = copy.deepcopy(body)
        pf_body.setdefault("sampling_params", {})["max_tokens"] = 1
        pf_body["kv_transfer_params"] = {"do_remote_decode": True}
        pf_body["stream"] = False
        try:
            async with self._session.post(f"{prefill}{GENERATE_PATH}", json=pf_body) as r:
                pf_json = await r.json()
                if r.status != 200:
                    return web.json_response(pf_json, status=r.status)
        except Exception as e:  # noqa: BLE001
            logger.exception("PD proxy: prefill request to %s failed", prefill)
            return web.json_response({"error": f"prefill failed: {e}"}, status=502)

        ktp = pf_json.get("kv_transfer_params")
        if not ktp:
            return web.json_response(
                {"error": "prefill returned no kv_transfer_params; NIXL handshake not engaged"}, status=502
            )

        # Step 2: decode (full generation, consume remote prefill KV).
        dc_body = copy.deepcopy(body)
        dc_body["kv_transfer_params"] = {"do_remote_prefill": True, **ktp}
        try:
            async with self._session.post(f"{decode}{GENERATE_PATH}", json=dc_body) as r:
                dc_json = await r.json()
                return web.json_response(dc_json, status=r.status)
        except Exception as e:  # noqa: BLE001
            logger.exception("PD proxy: decode request to %s failed", decode)
            return web.json_response({"error": f"decode failed: {e}"}, status=502)

    async def handle_health(self, request: web.Request) -> web.Response:
        return web.Response(text="OK")

    async def handle_broadcast(self, request: web.Request) -> web.Response:
        await self._ensure_session()
        body = await request.read()
        path = request.path
        ctype = request.headers.get("Content-Type", "application/json")
        with self._lock:
            urls = list(self._workers)

        async def _one(url):
            try:
                async with self._session.post(f"{url}{path}", data=body, headers={"Content-Type": ctype}) as r:
                    return await r.read()
            except Exception as e:  # noqa: BLE001
                logger.warning("PD proxy: broadcast %s to %s failed: %s", path, url, e)
                return None

        await asyncio.gather(*[_one(u) for u in urls])
        return web.json_response({"ok": True})

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=512 * 1024 * 1024)
        app.router.add_post(GENERATE_PATH, self.handle_generate)
        app.router.add_get("/health", self.handle_health)
        app.router.add_post("/workers", self.add_worker)
        app.router.add_get("/workers", self.list_workers)
        app.router.add_delete("/workers/{url}", self.remove_worker)
        for ctl in ("/abort_requests", "/resume", "/pause"):
            app.router.add_post(ctl, self.handle_broadcast)
        return app


def run_pd_proxy(host: str, port: int, request_timeout: int = 14400) -> None:
    """Entrypoint for a spawned process (parallel to ``http_utils.run_router``)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s [pd_proxy] %(message)s")
    proxy = PDProxy(request_timeout=request_timeout)
    logger.info("PD proxy listening on %s:%s (workers register dynamically)", host, port)
    web.run_app(proxy.build_app(), host=host, port=port, access_log=None)


def main():
    ap = argparse.ArgumentParser(description="vLLM NIXL prefill/decode proxy")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--request-timeout", type=int, default=14400)
    args = ap.parse_args()
    run_pd_proxy(args.host, args.port, args.request_timeout)


if __name__ == "__main__":
    main()
