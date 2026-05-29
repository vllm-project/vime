"""Prefill/Decode (PD) disaggregation proxy for the vLLM rollout backend.

vLLM does PD via the NIXL KV connector: a *prefill* engine computes the prompt
KV and a *decode* engine pulls it over a NIXL side channel, then generates. The
two engines coordinate through ``kv_transfer_params`` carried on the
``/inference/v1/generate`` request/response (vLLM's ``GenerateRequest`` /
``GenerateResponse`` both expose this field).

The stock ``vllm_router`` only implements SGLang-style PD (simultaneous dispatch
+ bootstrap room), so this module supplies a vLLM-native proxy instead. It speaks
the same surface slime's rollout already targets — ``POST /inference/v1/generate``,
``GET /health``, worker listing, ``POST /abort_requests`` — and drives the
two-step handshake:

    1. POST the request to a *prefill* worker with
       ``kv_transfer_params={"do_remote_decode": True}`` and ``max_tokens=1``.
       The prefill worker computes+stores the prompt KV and returns
       ``kv_transfer_params`` describing the remote handles.
    2. POST the original request to a *decode* worker with
       ``kv_transfer_params={"do_remote_prefill": True, **prefill_params}``. The
       decode worker pulls the KV via NIXL and generates the full response.

The engines themselves are launched with the NIXL ``--kv-transfer-config`` by
``vllm_engine._build_vllm_cmd_and_env`` (role-driven by ``worker_type``).
"""

from __future__ import annotations

import argparse
import asyncio
import copy
import itertools
import logging

import aiohttp
from aiohttp import web

logger = logging.getLogger(__name__)

GENERATE_PATH = "/inference/v1/generate"


class PDProxy:
    def __init__(self, prefill_urls: list[str], decode_urls: list[str], request_timeout: int = 14400):
        assert prefill_urls and decode_urls, "PD proxy needs >=1 prefill and >=1 decode URL"
        self.prefill_urls = [u.rstrip("/") for u in prefill_urls]
        self.decode_urls = [u.rstrip("/") for u in decode_urls]
        self._pf_rr = itertools.cycle(self.prefill_urls)
        self._dc_rr = itertools.cycle(self.decode_urls)
        self._timeout = aiohttp.ClientTimeout(total=request_timeout)
        self._session: aiohttp.ClientSession | None = None

    async def _ensure_session(self):
        if self._session is None or self._session.closed:
            self._session = aiohttp.ClientSession(timeout=self._timeout)

    async def handle_generate(self, request: web.Request) -> web.Response:
        await self._ensure_session()
        body = await request.json()
        prefill = next(self._pf_rr)
        decode = next(self._dc_rr)

        # --- Step 1: prefill (1 token, mark for remote decode) ----------------
        pf_body = copy.deepcopy(body)
        sp = pf_body.setdefault("sampling_params", {})
        sp["max_tokens"] = 1
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
                {"error": "prefill returned no kv_transfer_params; NIXL handshake not engaged"},
                status=502,
            )

        # --- Step 2: decode (full generation, consume remote prefill KV) ------
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

    async def handle_list_workers(self, request: web.Request) -> web.Response:
        # Mirrors the router's worker-listing surface used by
        # slime.rollout.vllm_rollout._router_worker_urls (urls / workers keys).
        urls = self.prefill_urls + self.decode_urls
        return web.json_response({"urls": urls, "workers": [{"url": u} for u in urls]})

    async def handle_broadcast(self, request: web.Request) -> web.Response:
        # Fan a control request (e.g. /abort_requests, /resume) out to every worker.
        await self._ensure_session()
        body = await request.read()
        path = request.path
        ctype = request.headers.get("Content-Type", "application/json")

        async def _one(url):
            try:
                async with self._session.post(f"{url}{path}", data=body, headers={"Content-Type": ctype}) as r:
                    return await r.read()
            except Exception as e:  # noqa: BLE001
                logger.warning("PD proxy: broadcast %s to %s failed: %s", path, url, e)
                return None

        await asyncio.gather(*[_one(u) for u in self.prefill_urls + self.decode_urls])
        return web.json_response({"ok": True})

    def build_app(self) -> web.Application:
        app = web.Application(client_max_size=512 * 1024 * 1024)
        app.router.add_post(GENERATE_PATH, self.handle_generate)
        app.router.add_get("/health", self.handle_health)
        app.router.add_get("/list_workers", self.handle_list_workers)
        app.router.add_get("/workers", self.handle_list_workers)
        for ctl in ("/abort_requests", "/resume", "/pause"):
            app.router.add_post(ctl, self.handle_broadcast)
        return app


def main():
    ap = argparse.ArgumentParser(description="vLLM NIXL prefill/decode proxy")
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, required=True)
    ap.add_argument("--prefill-url", action="append", default=[], help="prefill worker base URL (repeatable)")
    ap.add_argument("--decode-url", action="append", default=[], help="decode worker base URL (repeatable)")
    ap.add_argument("--request-timeout", type=int, default=14400)
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    proxy = PDProxy(args.prefill_url, args.decode_url, request_timeout=args.request_timeout)
    logger.info("PD proxy on %s:%s | prefill=%s decode=%s", args.host, args.port, proxy.prefill_urls, proxy.decode_urls)
    web.run_app(proxy.build_app(), host=args.host, port=args.port, access_log=None)


if __name__ == "__main__":
    main()
