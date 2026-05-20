from __future__ import annotations

import ipaddress
import logging
import multiprocessing
import os
import time

import requests
from urllib.parse import quote

from slime.ray.ray_actor import RayActor
from slime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)

_spawn_ctx = multiprocessing.get_context("spawn")

# vLLM sleep/wake only supports these tags (SGLang also uses ``cuda_graph``, which must be dropped).
_VLLM_WAKE_TAGS = frozenset({"weights", "kv_cache"})


def _normalize_vllm_wake_tags(tags: list[str] | None) -> list[str] | None:
    if not tags:
        return tags
    normalized = [t for t in tags if t in _VLLM_WAKE_TAGS]
    dropped = set(tags) - set(normalized)
    if dropped:
        logger.debug("vLLM wake_up: dropped tags not supported by vLLM: %s", sorted(dropped))
    return normalized or None


def get_base_gpu_id(args, rank):
    """First local GPU index on this node for rollout engine *rank* (colocate vs actor[/critic]-offset layout)."""
    num_gpus = min(args.num_gpus_per_node, args.rollout_num_gpus_per_engine)
    if args.colocate:
        start_index = (rank * num_gpus) % args.num_gpus_per_node
    else:
        num_actor_gpus = 0 if args.debug_rollout_only else args.actor_num_gpus_per_node * args.actor_num_nodes
        start_index = (num_actor_gpus + rank * num_gpus) % args.num_gpus_per_node
        if args.use_critic:
            num_critic_gpus = args.critic_num_gpus_per_node * args.critic_num_nodes
            start_index = (num_actor_gpus + num_critic_gpus + rank * num_gpus) % args.num_gpus_per_node
    return start_index


def _to_local_gpu_id(physical_gpu_id: int) -> int:
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES")
    if not cvd:
        return physical_gpu_id
    visible = [int(x) for x in cvd.split(",") if x.strip() != ""]
    if physical_gpu_id in visible:
        return visible.index(physical_gpu_id)
    if 0 <= physical_gpu_id < len(visible):
        return physical_gpu_id
    raise RuntimeError(
        f"GPU id {physical_gpu_id} is not valid under CUDA_VISIBLE_DEVICES={cvd}. "
        f"Expected one of {visible} (physical) or 0..{len(visible)-1} (local)."
    )


def _format_v6_uri(addr: str | None) -> str | None:
    if not addr or addr.startswith("["):
        return addr
    try:
        if ipaddress.ip_address(addr).version == 6:
            return f"[{addr}]"
    except ValueError:
        pass
    return addr


def _exec_vllm_cmd(cmd: list[str], env: dict[str, str]) -> None:
    """Entry point for multiprocessing child process."""
    os.execvpe(cmd[0], cmd, env)


def launch_server_process(
    *,
    bind_host: str,
    server_port: int,
    args,
    rank: int,
    visible_devices: str,
    model_path: str,
) -> multiprocessing.Process:
    """Spawn ``vllm serve`` (OpenAI API server) in a subprocess.

    Contrasts with SGLang's launcher, which starts the HTTP server in-process from ``ServerArgs``.
    """
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env["CUDA_VISIBLE_DEVICES"] = visible_devices
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")

    host_for_subprocess = bind_host.strip("[]")
    model = getattr(args, "vllm_model", None) or model_path
    tp = args.rollout_num_gpus_per_engine
    seed = getattr(args, "seed", 1234) + rank

    cmd = [
        "vllm",
        "serve",
        str(model),
        "--tensor-parallel-size",
        str(tp),
        "--port",
        str(server_port),
        "--host",
        host_for_subprocess,
        "--seed",
        str(seed),
        "--trust-remote-code",
        "--gpu-memory-utilization",
        str(getattr(args, "vllm_gpu_memory_utilization", 0.4)),
    ]
    if getattr(args, "vllm_weight_sync_mode", "auto") == "native":
        cmd += ["--weight-transfer-config", '{"backend":"nccl"}']
    if getattr(args, "offload_rollout", False) or getattr(args, "vllm_enable_sleep_mode", False):
        cmd += ["--enable-sleep-mode"]
    if getattr(args, "vllm_enforce_eager", False):
        cmd += ["--enforce-eager"]
    if getattr(args, "fp16", False):
        cmd += ["--dtype", "float16"]
    if getattr(args, "vllm_kv_cache_memory_bytes", None) is not None:
        cmd += ["--kv-cache-memory-bytes", str(args.vllm_kv_cache_memory_bytes)]
    if args.rollout_max_context_len is not None:
        cmd += ["--max-model-len", str(args.rollout_max_context_len)]
    if getattr(args, "use_rollout_routing_replay", False):
        cmd += ["--enable-return-routed-experts"]

    logger.info("Launching vLLM server: %s", " ".join(cmd))

    p = _spawn_ctx.Process(target=_exec_vllm_cmd, args=(cmd, env))
    p.start()
    return p


def _wait_server_healthy(base_url: str, process: multiprocessing.Process | None, timeout_s: float = 300.0) -> None:
    """Wait until the vLLM server responds on ``GET /health`` (SGLang stacks typically use ``GET /health_generate``)."""
    start = time.time()
    while True:
        try:
            response = requests.get(f"{base_url}/health", timeout=3)
            if response.status_code == 200:
                return
        except requests.RequestException:
            pass

        if process is not None and not process.is_alive():
            raise RuntimeError(f"vLLM server exited unexpectedly with code {process.exitcode}")
        if time.time() - start > timeout_s:
            raise TimeoutError(f"Timeout waiting for vLLM server healthy: {base_url}")
        time.sleep(2)


class VLLMEngine(RayActor):
    """Ray actor for vLLM OpenAI HTTP rollout (connect or spawn local ``vllm serve``)."""

    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        model_path: str | None = None,
        sglang_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.model_path = model_path or args.hf_checkpoint
        # Uniform Ray ``start_engines`` kwargs; unused when launching vLLM over HTTP.
        self.sglang_overrides = sglang_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self.process: multiprocessing.Process | None = None
        self._weight_version: str | None = None
        self._sync_mode = getattr(args, "vllm_weight_sync_mode", "auto")
        self._warned_sync_fallback = False
        self._pending_reload_version: str | None = None
        self._is_local_server = not args.rollout_external
        self._native_weight_update_ready = False
        # Slime runs one vLLM HTTP process per logical engine; multi-node worker rank is not used.
        self.node_rank = 0

    def _http_base(self) -> str:
        return f"http://{self.server_host}:{self.server_port}"

    def init(
        self,
        dist_init_addr,
        port,
        nccl_port,
        host=None,
        disaggregation_bootstrap_port=None,
        router_ip=None,
        router_port=None,
    ):
        del dist_init_addr, nccl_port, disaggregation_bootstrap_port

        self.router_ip = router_ip if router_ip is not None else self.args.sglang_router_ip
        self.router_port = router_port if router_port is not None else self.args.sglang_router_port

        host = host or get_host_info()[1]
        self.server_host = _format_v6_uri(host)
        self.server_port = port

        if self.worker_type != "regular":
            logger.warning(
                "vLLMEngine: worker_type=%s is not used by current vLLM deployment (treated as regular).",
                self.worker_type,
            )

        if self.args.rollout_external:
            self._init_external()
        else:
            self._init_normal()

        if self.node_rank == 0 and self.router_ip and self.router_port:
            self._register_worker_with_router()

    def _register_worker_with_router(self) -> None:
        worker_url = self._http_base()
        payload = {"url": worker_url, "worker_type": self.worker_type}
        response = requests.post(
            f"http://{self.router_ip}:{self.router_port}/workers",
            json=payload,
            timeout=30,
        )
        response.raise_for_status()

    def _deregister_worker_from_router(self) -> None:
        if self.node_rank != 0 or not self.router_ip or not self.router_port:
            return
        worker_url = self._http_base()
        try:
            all_workers = requests.get(
                f"http://{self.router_ip}:{self.router_port}/workers", timeout=30
            ).json()["workers"]
            for worker in all_workers:
                if worker["url"] == worker_url:
                    response = requests.delete(
                        f"http://{self.router_ip}:{self.router_port}/workers/{quote(worker_url, safe='')}",
                        timeout=30,
                    )
                    response.raise_for_status()
                    return
            logger.warning("Worker %s not found in vllm-router during shutdown.", worker_url)
        except Exception as e:
            logger.warning("Failed to list/remove worker on vllm-router: %s", e)

    def _init_external(self) -> None:
        logger.info("Use external vLLM engine (rank=%s) at %s:%s", self.rank, self.server_host, self.server_port)
        base = self._http_base()
        _wait_server_healthy(base, process=None)
        self._wait_external_config_ready()

    def _wait_external_config_ready(self) -> None:
        """External engine: best-effort ``GET /server_info`` TP check (non-fatal)."""
        try:
            # SGLang external mode uses ``/get_server_info``; vLLM exposes ``/server_info``.
            actual = requests.get(f"{self._http_base()}/server_info", params={"config_format": "json"}, timeout=30)
            actual.raise_for_status()
            body = actual.json()
        except requests.RequestException as e:
            logger.warning("External vLLM: could not GET /server_info (non-fatal): %s", e)
            return

        expect_tp = self.args.rollout_num_gpus_per_engine
        parallel_cfg = body.get("vllm_config", {}).get("parallel_config", {})
        actual_tp = parallel_cfg.get("tensor_parallel_size")
        if actual_tp is not None and actual_tp != expect_tp:
            logger.warning(
                "External vLLM server_info TP mismatch: expect=%s actual=%s (weak check)",
                expect_tp,
                actual_tp,
            )

    def _init_normal(self) -> None:
        logger.info("Launch vLLM OpenAI api_server at: %s:%s", self.server_host, self.server_port)
        num_gpus = min(self.args.num_gpus_per_node, self.args.rollout_num_gpus_per_engine)
        base = self.base_gpu_id if self.base_gpu_id is not None else get_base_gpu_id(self.args, self.rank)
        base = _to_local_gpu_id(base)
        visible_devices = ",".join(str(base + i) for i in range(num_gpus))

        bind_host = self.server_host
        self.process = launch_server_process(
            bind_host=bind_host,
            server_port=self.server_port,
            args=self.args,
            rank=self.rank,
            visible_devices=visible_devices,
            model_path=self.model_path,
        )
        _wait_server_healthy(self._http_base(), process=self.process)

    def _restart_local_server(self) -> None:
        if not self._is_local_server:
            logger.warning("Skip vLLM reload for external server mode.")
            return
        if self.process and self.process.is_alive():
            self.process.terminate()
            try:
                self.process.join(timeout=15)
            except Exception:
                pass
            if self.process.is_alive():
                self.process.kill()
                self.process.join(timeout=10)
        self._init_normal()

    def _post_json(self, endpoint: str, payload: dict, timeout: float) -> requests.Response:
        url = f"{self._http_base()}/{endpoint.lstrip('/')}"
        return requests.post(url, json=payload, timeout=timeout)

    def _post_vllm_update_weights_http(self, update_info: dict) -> dict:
        """POST ``/update_weights`` with ``{"update_info": ...}`` (vLLM RLHF control plane).

        Same contract as upstream ``examples/online_serving/new_weight_syncing/rlhf_http_nccl.py``:
        no ``start_weight_update`` / ``finish_weight_update`` wrapper.
        """
        timeout_s = float(
            os.environ.get(
                "SLIME_VLLM_WEIGHT_TRANSFER_UPDATE_TIMEOUT_SEC",
                os.environ.get("SLIME_VLLM_WEIGHT_TRANSFER_HTTP_TIMEOUT_SEC", "900"),
            )
        )
        response = self._post_json("update_weights", {"update_info": update_info}, timeout=timeout_s)
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"ok": True, "raw": response.text}

    def _run_vllm_weight_update(self, update_info: dict, *, is_checkpoint_format: bool = False):
        """Backward-compatible alias for non-NCCL ``update_info`` shapes (e.g. tensor/IPC path)."""
        del is_checkpoint_format
        return self._post_vllm_update_weights_http(update_info)

    def health_generate(self, timeout: float = 5.0) -> bool:
        """Return True if ``GET /health`` succeeds (SGLang uses ``GET /health_generate`` for the same role)."""
        if self.node_rank != 0:
            return True
        response = requests.get(f"{self._http_base()}/health", timeout=timeout)
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        serialized_named_tensors: list[str],
        load_format: str | None = None,
        flush_cache: bool = False,
        weight_version: str | None = None,
    ):
        """
        Post tensor metadata via ``/update_weights`` when native weight transfer is ready; otherwise record
        ``weight_version`` and return a reload placeholder (no HTTP update on the fallback path).
        Contrasts with SGLang, which posts to ``update_weights_from_tensor`` with a different payload shape.
        """
        del load_format
        if self.node_rank != 0:
            return

        if weight_version is not None:
            self._weight_version = str(weight_version)
            self._pending_reload_version = self._weight_version
        if flush_cache:
            self.flush_cache()

        update_info = {
            "serialized_named_tensors": serialized_named_tensors,
            "format": "serialized_named_tensors",
            "weight_version": self._weight_version,
        }
        if self._native_weight_update_ready:
            try:
                return self._run_vllm_weight_update(update_info, is_checkpoint_format=False)
            except Exception as e:
                if self._sync_mode == "native":
                    raise RuntimeError(f"Native vLLM tensor weight update failed: {e}") from e
                logger.warning("Native vLLM tensor weight update failed, fallback: %s", e)
                self._native_weight_update_ready = False

        self._pending_reload_version = self._weight_version
        if not self._warned_sync_fallback:
            logger.warning(
                "vLLM tensor weight update fallback to reload-on-continue "
                "(init_weight_transfer_engine not ready or update failed)."
            )
            self._warned_sync_fallback = True
        if self._sync_mode == "native":
            raise RuntimeError("Native mode requested but weight transfer is not ready.")
        return {"ok": True, "mode": "reload", "weight_version": self._weight_version}

    def flush_cache(self):
        """Clear prefix cache via ``POST /reset_prefix_cache`` (SGLang uses ``GET /flush_cache``)."""
        if self.node_rank != 0:
            return
        reset_running = bool(getattr(self.args, "vllm_reset_prefix_cache_reset_running", False))
        reset_external = bool(getattr(self.args, "vllm_reset_prefix_cache_reset_external", False))
        params = {"reset_running_requests": reset_running, "reset_external": reset_external}
        for _ in range(60):
            try:
                response = requests.post(f"{self._http_base()}/reset_prefix_cache", params=params, timeout=60)
                if response.status_code == 200:
                    return
            except requests.ConnectionError:
                raise
            except Exception as e:
                logger.info("Error resetting vLLM prefix cache: %s", e)
                time.sleep(1)
                continue
        raise TimeoutError("Timeout while resetting vLLM prefix cache (reset_prefix_cache).")

    def get_url(self):
        """Worker HTTP base URL, or ``None`` when ``node_rank != 0``."""
        if self.node_rank != 0:
            return None
        return self._http_base()

    def shutdown(self):
        logger.info("Shutdown vLLM engine %s:%s...", self.server_host, self.server_port)
        self._deregister_worker_from_router()
        if self.args.rollout_external:
            return

        if self.process is None or not self.process.is_alive():
            return
        pid = self.process.pid
        try:
            from vllm.utils.system_utils import kill_process_tree

            kill_process_tree(pid)
        except Exception as e:
            logger.warning("vLLM kill_process_tree failed (%s); terminate root only.", e)
            if self.process.is_alive():
                self.process.terminate()
                try:
                    self.process.join(timeout=15)
                except Exception:
                    pass
                if self.process.is_alive():
                    self.process.kill()
        try:
            self.process.join(timeout=30)
        except Exception:
            pass
        self.process = None

    def get_weight_version(self):
        """
        Prefer ``_weight_version`` if weight sync already set it; else try ``GET /v1/models`` for a stable id string.

        SGLang exposes ``GET /get_weight_version``; vLLM has no name-equivalent route, so semantics differ from that endpoint.
        """
        if self.node_rank != 0:
            return
        if self._weight_version is not None:
            return self._weight_version
        try:
            r = requests.get(f"{self._http_base()}/v1/models", timeout=10)
            r.raise_for_status()
            data = r.json().get("data") or []
            if data and isinstance(data[0], dict) and "id" in data[0]:
                return str(data[0]["id"])
        except requests.RequestException as e:
            logger.info("get_weight_version: /v1/models failed (%s)", e)
        return None

    def release_memory_occupation(self):
        """
        ``POST /sleep`` when sleep mode is enabled (SGLang: ``POST /release_memory_occupation``); otherwise a no-op placeholder dict.
        """
        self.flush_cache()
        if not getattr(self.args, "vllm_enable_sleep_mode", False):
            return {"ok": True, "sleep_mode": False, "note": "vLLM sleep mode disabled; no /sleep call."}
        # vLLM ``POST /sleep`` reads ``level`` from query params, not JSON body
        # (``vllm.entrypoints.serve.sleep.api_router.sleep``).
        level = int(getattr(self.args, "vllm_sleep_level", 1))
        response = requests.post(
            f"{self._http_base()}/sleep",
            params={"level": level},
            timeout=30,
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"ok": True, "raw": response.text}

    def resume_memory_occupation(self, tags: list[str] | None = None):
        """``POST /wake_up`` when sleep mode is on (SGLang: ``POST /resume_memory_occupation``); else a small placeholder dict."""
        if not getattr(self.args, "vllm_enable_sleep_mode", False):
            return {"ok": True, "sleep_mode": False}
        tags = _normalize_vllm_wake_tags(tags)
        # vLLM ``POST /wake_up`` uses ``query_params.getlist("tags")``, not JSON.
        # Omit params when ``tags`` is empty so the server wakes all tags (see api_router.wake_up).
        wake_params: list[tuple[str, str]] | None = (
            [("tags", t) for t in tags] if tags else None
        )
        response = requests.post(
            f"{self._http_base()}/wake_up",
            params=wake_params,
            timeout=30,
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"ok": True, "raw": response.text}

    def check_weights(self, action: str):
        """No vLLM ``weights_checker`` route; return a placeholder (SGLang posts to ``/weights_checker``)."""
        del action
        return {"ok": True, "supported": False, "note": "vLLM has no weights_checker endpoint."}

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        """
        Call ``POST /init_weight_transfer_engine`` with an ``init_info`` block (SGLang: ``/init_weights_update_group``).

        ``group_name`` / ``backend`` are accepted for a uniform caller signature but are not sent to vLLM.
        If ``vllm_weight_sync_mode`` is not ``native``, the HTTP call is skipped (SGLang still posts to its endpoint).
        """
        del group_name, backend
        if self._sync_mode != "native":
            return {"ok": True, "mode": self._sync_mode, "skipped": True}

        payload = {
            "init_info": {
                "master_address": master_address,
                "master_port": master_port,
                "rank_offset": rank_offset,
                "world_size": world_size,
            }
        }
        init_timeout_s = float(os.environ.get("SLIME_VLLM_WEIGHT_TRANSFER_HTTP_TIMEOUT_SEC", "900"))
        last_error = None
        for attempt in range(1, 4):
            try:
                response = self._post_json("init_weight_transfer_engine", payload, timeout=init_timeout_s)
                response.raise_for_status()
                self._native_weight_update_ready = True
                try:
                    return response.json()
                except Exception:
                    return {"ok": True, "raw": response.text}
            except Exception as e:
                last_error = e
                self._native_weight_update_ready = False
                if attempt < 3:
                    logger.warning("init_weight_transfer_engine attempt %s/3 failed: %s", attempt, e)
                    time.sleep(2 * attempt)
        if self._sync_mode == "native":
            raise RuntimeError(f"vLLM init_weight_transfer_engine failed: {last_error}") from last_error
        logger.warning("vLLM native weight transfer init failed, fallback: %s", last_error)
        return {"ok": False, "error": str(last_error)}

    def destroy_weights_update_group(self, group_name):
        """No vLLM destroy call; return ``None`` (SGLang may ``POST /destroy_weights_update_group`` and swallow errors)."""
        del group_name
        return None

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        flush_cache=False,
        weight_version: str | None = None,
        packed: bool = True,
    ):
        """NCCL/native path posts ``/update_weights`` (SGLang: ``POST /update_weights_from_distributed``)."""
        del group_name
        if weight_version is not None:
            self._weight_version = str(weight_version)
        if flush_cache:
            self.flush_cache()
        dtype_names = [str(d).replace("torch.", "") for d in dtypes]
        if self._native_weight_update_ready:
            # Payload matches vLLM NCCL weight transfer (see upstream rlhf_http_nccl example).
            update_info = {
                "names": names,
                "dtype_names": dtype_names,
                "shapes": [list(s) for s in shapes],
                "packed": bool(packed),
            }
            try:
                return self._post_vllm_update_weights_http(update_info)
            except Exception as e:
                if self._sync_mode == "native":
                    raise RuntimeError(f"Native vLLM weight update failed: {e}") from e
                logger.warning("Native vLLM weight update failed, fallback: %s", e)
                self._native_weight_update_ready = False

        self._pending_reload_version = self._weight_version
        if self._sync_mode == "native":
            raise RuntimeError("Native mode requested but weight transfer is not ready.")
        if not self._warned_sync_fallback and self._sync_mode in ("auto", "reload"):
            logger.warning("vLLM weight sync mode=%s: reload-on-continue path may apply.", self._sync_mode)
            self._warned_sync_fallback = True
        return {"ok": True, "mode": self._sync_mode, "weight_version": self._weight_version}

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        """``POST /collective_rpc`` with ``reload_weights`` and ``weights_path`` (SGLang uses a dedicated disk API)."""
        if self.node_rank != 0:
            return
        del load_format
        response = requests.post(
            f"{self._http_base()}/collective_rpc",
            json={
                "method": "reload_weights",
                "kwargs": {"weights_path": model_path, "is_checkpoint_format": True},
            },
            timeout=600,
        )
        response.raise_for_status()
        try:
            return response.json()
        except Exception:
            return {"ok": True, "raw": response.text}

    def pause_generation(self):
        """``POST /pause`` with mode query (SGLang: ``POST /pause_generation``); returns the ``requests.Response``."""
        if self.node_rank != 0:
            return None
        mode = getattr(self.args, "vllm_pause_mode", "keep")
        response = requests.post(
            f"{self._http_base()}/pause",
            params={"mode": mode, "clear_cache": "false"},
            json={},
            timeout=120,
        )
        response.raise_for_status()
        return response

    def continue_generation(self):
        """
        ``POST /resume`` (SGLang: ``POST /continue_generation``); may restart the local child process after a pending reload.
        """
        if self.node_rank != 0:
            return None
        response = requests.post(f"{self._http_base()}/resume", json={}, timeout=120)
        response.raise_for_status()
        if self._pending_reload_version is not None:
            logger.info("Reload vLLM server after weight update, version=%s", self._pending_reload_version)
            self._restart_local_server()
            self._pending_reload_version = None
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        """No vLLM HTTP hook (SGLang: ``POST /post_process_weights``); return a noop placeholder dict."""
        del restore_weights_before_load, post_process_quantization
        return {"ok": True, "noop": True, "note": "vLLM post_process is internal to load; no HTTP API."}

    def start_profile(
        self,
        output_dir: str | None = None,
        start_step: int | None = None,
        num_steps: int | None = None,
        activities: list[str] | None = None,
        profile_by_stage: bool = False,
        with_stack: bool | None = None,
        record_shapes: bool | None = None,
    ):
        """``POST /start_profile`` with an empty JSON body; kwargs are not forwarded and may be ignored by the server."""
        if self.node_rank != 0:
            return None
        if any(
            x is not None and x is not False
            for x in (
                output_dir,
                start_step,
                num_steps,
                activities,
                profile_by_stage,
                with_stack,
                record_shapes,
            )
        ):
            logger.warning("vLLM start_profile: extra kwargs may be ignored by server; see vLLM profiling docs.")
        response = requests.post(f"{self._http_base()}/start_profile", json={}, timeout=30)
        response.raise_for_status()
        return response

    def stop_profile(self):
        """POST ``/stop_profile`` to stop an active server-side profile."""
        if self.node_rank != 0:
            return None
        response = requests.post(f"{self._http_base()}/stop_profile", json={}, timeout=30)
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return
        logger.info("Simulating crash on vLLM engine %s:%s...", self.server_host, self.server_port)
        self.shutdown()
