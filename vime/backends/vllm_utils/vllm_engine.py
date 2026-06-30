import argparse
import base64
import dataclasses
import ipaddress
import logging
import multiprocessing
import os
import time
from typing import Any
from urllib.parse import quote

import cloudpickle
import requests
from vllm.utils.system_utils import kill_process_tree

from vime.backends.vllm_utils.external import get_server_info
from vime.ray.ray_actor import RayActor
from vime.utils.http_utils import get_host_info

logger = logging.getLogger(__name__)

_VLLM_WAKE_TAGS = frozenset({"weights", "kv_cache"})


def get_base_gpu_id(args, rank):
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


def launch_server_process(server_args_dict: dict) -> multiprocessing.Process:
    env = _build_subprocess_env(server_args_dict)
    kwargs = {k: v for k, v in server_args_dict.items() if not k.startswith("_")}
    logger.info("Launching vLLM server: %s", kwargs)

    multiprocessing.set_start_method("spawn", force=True)
    p = multiprocessing.Process(target=_run_vllm_server, args=(kwargs, env))
    p.start()

    if server_args_dict.get("node_rank", 0) != 0:
        return p

    _wait_server_healthy(
        base_url=f"http://{(server_args_dict['host'] or '127.0.0.1').strip('[]')}:{server_args_dict['port']}",
        is_process_alive=lambda: p.is_alive(),
    )

    return p


def _build_subprocess_env(server_args_dict: dict[str, Any]) -> dict[str, str]:
    args = server_args_dict["_args"]
    env = os.environ.copy()
    env.pop("PYTORCH_CUDA_ALLOC_CONF", None)
    env.setdefault("NCCL_CUMEM_ENABLE", "0")
    env["CUDA_VISIBLE_DEVICES"] = server_args_dict["_visible_devices"]
    env.setdefault("VLLM_SERVER_DEV_MODE", "1")
    if getattr(args, "vllm_enable_deterministic_inference", False):
        env["VLLM_BATCH_INVARIANT"] = "1"
    if getattr(args, "colocate", False):
        import vime

        vime_root = os.path.dirname(os.path.dirname(os.path.abspath(vime.__file__)))
        existing_pp = env.get("PYTHONPATH", "")
        if vime_root not in {p for p in existing_pp.split(os.pathsep) if p}:
            env["PYTHONPATH"] = os.pathsep.join(filter(None, [vime_root, existing_pp]))
        env.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")

    worker_type = server_args_dict.get("_worker_type", "regular")
    if worker_type in ("prefill", "decode") and server_args_dict.get("node_rank", 0) == 0:
        host_for_subprocess = (server_args_dict.get("host") or "127.0.0.1").strip("[]")
        env["VLLM_NIXL_SIDE_CHANNEL_HOST"] = host_for_subprocess
        env["VLLM_NIXL_SIDE_CHANNEL_PORT"] = str(server_args_dict["_disaggregation_bootstrap_port"])

    return env


def _run_vllm_server(kwargs: dict, env: dict) -> None:
    os.environ.update(env)

    from vllm.entrypoints.cli.serve import ServeSubcommand
    from vllm.entrypoints.openai.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    ns = argparse.Namespace(**kwargs)
    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(args=[], namespace=ns)
    validate_parsed_serve_args(args)
    ServeSubcommand.cmd(args)


def _wait_server_healthy(base_url, is_process_alive):
    while True:
        try:
            response = requests.get(f"{base_url}/health")
            if response.status_code == 200:
                break
        except requests.RequestException:
            pass

        if not is_process_alive():
            raise Exception("Server process terminated unexpectedly.")

        time.sleep(2)


class VLLMEngine(RayActor):
    def __init__(
        self,
        args,
        rank: int,
        worker_type: str = "regular",
        base_gpu_id: int | None = None,
        vllm_overrides: dict | None = None,
        num_gpus_per_engine: int | None = None,
    ):
        self.args = args
        self.rank = rank
        self.worker_type = worker_type
        self.base_gpu_id = base_gpu_id
        self.vllm_overrides = vllm_overrides or {}
        self.num_gpus_per_engine = num_gpus_per_engine
        self._weight_version: str | None = None

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
        del nccl_port

        self.router_ip = router_ip
        self.router_port = router_port

        host = host or get_host_info()[1]

        def _format_v6_uri(addr):
            if not addr or addr.startswith("["):
                return addr
            try:
                if ipaddress.ip_address(addr).version == 6:
                    return f"[{addr}]"
            except ValueError:
                pass
            return addr

        host = _format_v6_uri(host)
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        dist_init_addr = f"{_format_v6_uri(ip_part)}:{port_part}"

        server_args_dict, external_engine_need_check_fields = _compute_server_args(
            self.args,
            self.rank,
            dist_init_addr,
            host,
            port,
            self.worker_type,
            disaggregation_bootstrap_port,
            base_gpu_id=self.base_gpu_id,
            vllm_overrides=self.vllm_overrides,
            num_gpus_per_engine=self.num_gpus_per_engine,
        )

        self.node_rank = server_args_dict["node_rank"]
        self.server_host = server_args_dict["host"]
        self.server_port = server_args_dict["port"]

        if self.args.rollout_external:
            self._init_external(server_args_dict, external_engine_need_check_fields=external_engine_need_check_fields)
        else:
            self._init_normal(server_args_dict)

    def _init_external(self, expect_server_args, external_engine_need_check_fields):
        logger.info(f"Use external vLLM engine (rank={self.rank}, expect_server_args={expect_server_args})")

        def _sanity_check_server_args(actual_server_args, expect_server_args):
            for name in external_engine_need_check_fields:
                expect_value = expect_server_args.get(name)
                actual_value = actual_server_args.get(name)
                assert (
                    actual_value == expect_value
                ), f"{name=} {expect_value=} {actual_value=} {expect_server_args=} {actual_server_args=}"

        actual_server_args = get_server_info(f"http://{self.server_host}:{self.server_port}")
        _sanity_check_server_args(actual_server_args, expect_server_args)
        self._register_to_router(expect_server_args)

    def _init_normal(self, server_args_dict):
        logger.info(f"Launch vLLM api_server at: {self.server_host}:{self.server_port}")
        self.process = launch_server_process(server_args_dict)
        self._register_to_router(server_args_dict)

    def _register_to_router(self, server_args_dict):
        if self.worker_type == "encoder":
            return

        if self.node_rank == 0 and self.router_ip and self.router_port:
            import vllm_router
            from packaging.version import parse

            worker_url = f"http://{self.server_host}:{self.server_port}"
            if parse(vllm_router.__version__) <= parse("0.2.1"):
                assert self.worker_type == "regular", "pd disaggregation is not supported in old router."
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/add_worker?url={worker_url}",
                )
            else:
                payload = {
                    "url": worker_url,
                    "worker_type": self.worker_type,
                }
                if self.worker_type == "prefill":
                    bootstrap_port = server_args_dict.get("disaggregation_bootstrap_port")
                    if bootstrap_port is None:
                        raise RuntimeError(
                            f"Prefill worker {worker_url} does not have disaggregation_bootstrap_port; "
                            "cannot register it to the PD router."
                        )
                    payload["bootstrap_port"] = bootstrap_port
                response = requests.post(
                    f"http://{self.router_ip}:{self.router_port}/workers",
                    json=payload,
                )
            response.raise_for_status()

    def _make_request(self, endpoint: str, payload: dict | None = None):
        """Make a POST request to the specified endpoint with the given payload.

        Args:
            endpoint: The API endpoint to call
            payload: The JSON payload to send (default: empty dict)

        Returns:
            The JSON response from the server
        """
        if self.node_rank != 0:
            return

        url = f"http://{self.server_host}:{self.server_port}/{endpoint}"
        response = requests.post(url, json=payload or {})
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        if not response.content or not response.content.strip():
            return {"ok": True}
        return response.json()

    def health_generate(self, timeout: float = 5.0) -> bool:
        if self.node_rank != 0:
            return True

        response = requests.get(
            f"http://{self.server_host}:{self.server_port}/health",
            timeout=timeout,
        )
        response.raise_for_status()
        return True

    def update_weights_from_tensor(
        self,
        *,
        names: list[str],
        dtype_names: list[str],
        shapes: list[list[int]],
        ipc_handles: list[dict] | None = None,
        weight_version: str,
        flush_cache: bool = False,
    ):
        payload: dict = {"names": names, "dtype_names": dtype_names, "shapes": shapes}
        if ipc_handles is not None:
            payload["ipc_handles_pickled"] = base64.b64encode(cloudpickle.dumps(ipc_handles)).decode("utf-8")
        if flush_cache:
            self.flush_cache()
        result = self._make_request("update_weights", {"update_info": payload})
        self._weight_version = str(weight_version)
        return result

    def flush_cache(self):
        if self.node_rank != 0:
            return
        params = {"reset_running_requests": False}
        requests.post(
            f"http://{self.server_host}:{self.server_port}/reset_prefix_cache", params=params
        ).raise_for_status()

    def get_url(self):
        if self.node_rank != 0:
            return None
        return f"http://{self.server_host}:{self.server_port}"

    def shutdown(self):
        if self.args.rollout_external:
            return

        logger.info(f"Shutdown engine {self.server_host}:{self.server_port}...")
        if self.worker_type != "encoder" and self.node_rank == 0:
            worker_url = f"http://{self.server_host}:{self.server_port}"
            try:
                all_workers = requests.get(f"http://{self.router_ip}:{self.router_port}/workers").json()["workers"]
                for worker in all_workers:
                    if worker["url"] == worker_url:
                        response = requests.delete(
                            f"http://{self.router_ip}:{self.router_port}/workers/{quote(worker_url, safe='')}",
                        )
                        response.raise_for_status()
                        break
                else:
                    logger.warning(f"Worker {worker_url} not found in vllm-router during shutdown.")
            except Exception as e:
                logger.warning(f"Failed to fetch workers list or remove worker: {e}")

        kill_process_tree(self.process.pid)

    def get_weight_version(self):
        if self.node_rank != 0:
            return
        if self._weight_version is None:
            raise RuntimeError(
                "VLLMEngine.get_weight_version called before any successful " "weight transfer recorded a version."
            )
        return self._weight_version

    def set_weight_version(self, new_version: str):
        self._weight_version = str(new_version)

    def release_memory_occupation(self, level: int = 2):
        self.flush_cache()
        response = requests.post(f"http://{self.server_host}:{self.server_port}/sleep", params={"level": level})
        response.raise_for_status()
        if not response.content or not response.content.strip():
            return {"ok": True}
        return response.json()

    def resume_memory_occupation(self, tags: list[str] = None):
        tags = _normalize_vllm_wake_tags(tags)
        wake_params: list[tuple[str, str]] | None = [("tags", t) for t in tags] if tags else None
        response = requests.post(f"http://{self.server_host}:{self.server_port}/wake_up", params=wake_params)
        response.raise_for_status()
        if not response.content or not response.content.strip():
            return {"ok": True}
        return response.json()

    def check_weights(self, action: str):
        del action
        return {"ok": True, "supported": False}

    def init_weight_transfer_engine(self, payload: dict) -> dict:
        return self._make_request("init_weight_transfer_engine", payload)

    def start_weight_update(self, is_checkpoint_format: bool = False) -> dict:
        return self._make_request("start_weight_update", {"is_checkpoint_format": is_checkpoint_format})

    def finish_weight_update(self) -> dict:
        return self._make_request("finish_weight_update", {})

    def update_weights_from_disk(self, model_path: str, load_format: str | None = None):
        del load_format
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/collective_rpc",
            json={"method": "reload_weights", "kwargs": {"weights_path": model_path, "is_checkpoint_format": True}},
        )
        try:
            response.raise_for_status()
        except requests.exceptions.HTTPError as e:
            e.add_note(f"{response.text=}")
            raise
        return response.json()

    def init_weights_update_group(self, master_address, master_port, rank_offset, world_size, group_name, backend):
        del group_name, backend
        return self._make_request(
            "init_weight_transfer_engine",
            {
                "init_info": {
                    "master_address": master_address,
                    "master_port": master_port,
                    "rank_offset": rank_offset,
                    "world_size": world_size,
                }
            },
        )

    def destroy_weights_update_group(self, group_name):
        del group_name
        return None

    def update_weights_from_distributed(
        self,
        names,
        dtypes,
        shapes,
        group_name,
        *,
        flush_cache=False,
        weight_version: str,
        packed: bool = True,
    ):
        del group_name
        if flush_cache:
            self.flush_cache()
        dtype_names = [str(d).replace("torch.", "") for d in dtypes]
        update_info = {
            "names": names,
            "dtype_names": dtype_names,
            "shapes": [list(s) for s in shapes],
            "packed": bool(packed),
        }
        result = self._make_request("update_weights", {"update_info": update_info})
        self._weight_version = str(weight_version)
        return result

    def pause_generation(self):
        response = requests.post(
            f"http://{self.server_host}:{self.server_port}/pause",
            params={"mode": "keep", "clear_cache": "false"},
            json={},
        )
        response.raise_for_status()
        return response

    def continue_generation(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/resume", json={})
        response.raise_for_status()
        return response

    def post_process_weights(
        self,
        restore_weights_before_load: bool = False,
        post_process_quantization: bool = False,
    ):
        del restore_weights_before_load, post_process_quantization
        return {"ok": True, "noop": True}

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
        response = requests.post(f"http://{self.server_host}:{self.server_port}/start_profile", json={})
        response.raise_for_status()
        return response

    def stop_profile(self):
        response = requests.post(f"http://{self.server_host}:{self.server_port}/stop_profile", json={})
        response.raise_for_status()
        return response

    def simulate_crash(self):
        if self.args.rollout_external or not getattr(self, "process", None):
            logger.info(
                "simulate_crash called but no local engine process exists (rollout_external=%s); skip kill",
                self.args.rollout_external,
            )
            return

        logger.info(f"Simulating crash on engine {self.server_host}:{self.server_port}...")
        self.shutdown()


def _normalize_vllm_wake_tags(tags: list[str] | None) -> list[str] | None:
    if not tags:
        return tags
    normalized = [t for t in tags if t in _VLLM_WAKE_TAGS]
    dropped = set(tags) - set(normalized)
    if dropped:
        logger.debug("vLLM wake_up: dropped tags not supported by vLLM: %s", sorted(dropped))
    return normalized or None


def _resolve_parallel_sizes(args, *, gpus_per_engine: int) -> tuple[int, int, int]:
    pp = int(getattr(args, "vllm_pipeline_parallel_size", 1) or 1)
    dp = int(getattr(args, "vllm_dp_size", None) or getattr(args, "vllm_data_parallel_size", 1) or 1)
    if gpus_per_engine % (pp * dp) != 0:
        raise ValueError(
            f"num_gpus_per_engine ({gpus_per_engine}) must be divisible by "
            f"vllm_pipeline_parallel_size * vllm_data_parallel_size ({pp} * {dp} = {pp * dp})"
        )
    tp = gpus_per_engine // (pp * dp)
    return tp, pp, dp


def _compute_server_args(
    args,
    rank,
    dist_init_addr,
    host,
    port,
    worker_type: str = "regular",
    disaggregation_bootstrap_port: int | None = None,
    base_gpu_id: int | None = None,
    vllm_overrides: dict | None = None,
    num_gpus_per_engine: int | None = None,
):
    _gpus_per_engine = num_gpus_per_engine or args.rollout_num_gpus_per_engine
    nnodes = max(1, _gpus_per_engine // args.num_gpus_per_node)
    node_rank = rank % nnodes
    if nnodes == 1:
        local_num_gpus = min(args.num_gpus_per_node, _gpus_per_engine)
    else:
        if _gpus_per_engine % nnodes != 0:
            raise ValueError(
                f"rollout_num_gpus_per_engine ({_gpus_per_engine}) must be divisible by "
                f"the number of nodes per engine ({nnodes})"
            )
        local_num_gpus = _gpus_per_engine // nnodes

    tp, pp, dp = _resolve_parallel_sizes(args, gpus_per_engine=_gpus_per_engine)
    base = base_gpu_id if base_gpu_id is not None else get_base_gpu_id(args, rank)

    master_addr: str | None = None
    master_port: int | None = None
    if nnodes > 1:
        if not dist_init_addr:
            raise ValueError("dist_init_addr is required when launching a multi-node vLLM engine")
        ip_part, port_part = dist_init_addr.rsplit(":", 1)
        master_addr = ip_part.strip("[]")
        master_port = int(port_part)

    host_for_subprocess = (host or "127.0.0.1").strip("[]")

    kwargs: dict[str, Any] = {
        "model": str(args.hf_checkpoint),
        "trust_remote_code": True,
        "seed": args.seed + rank,
        "host": host_for_subprocess,
        "port": port,
        "nnodes": nnodes,
        "node_rank": node_rank,
        "tensor_parallel_size": tp,
        "logprobs_mode": "processed_logprobs",
        "enable_prompt_tokens_details": True,
        "enable_server_load_tracking": True,
    }

    if pp > 1:
        kwargs["pipeline_parallel_size"] = pp

    if nnodes > 1:
        kwargs["master_addr"] = master_addr
        kwargs["master_port"] = master_port
        kwargs["data_parallel_backend"] = "mp"
        kwargs["distributed_executor_backend"] = "mp"
        if node_rank != 0:
            kwargs["headless"] = True

    if worker_type == "prefill":
        kwargs["disaggregation_mode"] = "prefill"
        assert (
            disaggregation_bootstrap_port is not None
        ), "disaggregation_bootstrap_port must be set for prefill worker"
    elif worker_type == "decode":
        kwargs["disaggregation_mode"] = "decode"

    if args.use_rollout_routing_replay:
        kwargs["enable_return_routed_experts"] = True
    if args.fp16:
        kwargs["dtype"] = "float16"

    if args.offload_rollout and not getattr(args, "vllm_enable_sleep_mode", False):
        kwargs["enable_sleep_mode"] = True
        args.vllm_enable_sleep_mode = True

    if (
        getattr(args, "rollout_max_context_len", None) is not None
        and getattr(args, "vllm_max_model_len", None) is None
    ):
        kwargs["max_model_len"] = args.rollout_max_context_len

    if args.colocate:
        kwargs["weight_transfer_config"] = {"backend": "ipc"}
    else:
        kwargs["weight_transfer_config"] = {"backend": "nccl"}

    external_engine_need_check_fields = [k for k in kwargs.keys() if k not in _EXTERNAL_ENGINE_SKIP_CHECK_FIELDS]

    global _VLLM_SERVER_FIELDS  # noqa: PLW0603
    if _VLLM_SERVER_FIELDS is None:
        _VLLM_SERVER_FIELDS = _vllm_server_field_names()

    for key, value in vars(args).items():
        if not key.startswith("vllm_"):
            continue
        field_name = key[len("vllm_") :]
        if field_name not in _VLLM_SERVER_FIELDS:
            continue
        if field_name in kwargs:
            continue
        if value is None:
            continue
        kwargs[field_name] = value

    # Per-server-group overrides from --vllm-config YAML.
    # Applied after base args so they take highest priority.
    if vllm_overrides:
        for key, value in vllm_overrides.items():
            normalized_key = key.replace("-", "_")
            if normalized_key != key:
                logger.warning(
                    f"vllm_overrides key '{key}' normalized to '{normalized_key}' (rank={rank}). "
                    "Please use underscore style in YAML overrides."
                )
            if normalized_key in ("model_path",) or normalized_key.startswith("disaggregation"):
                continue
            if normalized_key in kwargs:
                logger.info(
                    f"vllm_overrides: overriding {normalized_key}={kwargs[normalized_key]} -> {value} (rank={rank})"
                )
            kwargs[normalized_key] = value
        if "model_path" in {k.replace("-", "_") for k in vllm_overrides}:
            kwargs["model"] = str(vllm_overrides.get("model_path") or vllm_overrides.get("model-path"))

    # vLLM-specific: topology metadata consumed by launch_server_process / _build_subprocess_env.
    # These keys are stripped before passing to vLLM's argparse.
    kwargs["_args"] = args
    kwargs["_rank"] = rank
    kwargs["_worker_type"] = worker_type
    kwargs["_visible_devices"] = ",".join(str(base + i) for i in range(local_num_gpus))
    kwargs["_tp_size"] = tp
    kwargs["_pp_size"] = pp
    kwargs["_dp_size"] = dp
    kwargs["_disaggregation_bootstrap_port"] = disaggregation_bootstrap_port

    return kwargs, external_engine_need_check_fields


def _vllm_server_field_names() -> frozenset[str]:
    """Valid vLLM server-arg field names: ``AsyncEngineArgs`` ∪ ``FrontendArgs``. vLLM has no
    single ``ServerArgs`` class (sglang does); their union is the faithful translation. Single
    source of truth for ``--vllm-*`` flag generation and ``--vllm-config`` override validation.
    """
    from vllm.engine.arg_utils import AsyncEngineArgs
    from vllm.entrypoints.openai.cli_args import FrontendArgs

    return frozenset(f.name for f in (*dataclasses.fields(AsyncEngineArgs), *dataclasses.fields(FrontendArgs)))


_VLLM_SERVER_FIELDS: frozenset[str] | None = None


_EXTERNAL_ENGINE_SKIP_CHECK_FIELDS = [
    "model",
    "trust_remote_code",
    "seed",
    "host",
    "port",
    "tensor_parallel_size",
    "logprobs_mode",
    "enable_prompt_tokens_details",
    "enable_server_load_tracking",
]
