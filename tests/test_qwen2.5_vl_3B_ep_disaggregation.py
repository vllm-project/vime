"""Three-GPU vLLM-only acceptance test for encoder-prefill disaggregation."""

from __future__ import annotations

import asyncio
import base64
import io
import os
import shutil
import sys
import tempfile
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import ray
import requests
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import vime.utils.external_utils.command_utils as U
from vime.backends.vllm_utils.arguments import vllm_parse_args
from vime.ray.placement_group import _create_placement_group
from vime.ray.rollout import start_rollout_servers
from vime.rollout import vllm_rollout
from vime.utils.http_utils import init_http_client, is_port_available, post
from vime.utils.processing_utils import load_tokenizer

NUM_GPUS = 3
MODEL_NAME = "Qwen2.5-VL-3B-Instruct"
MODEL_REVISION = "66285546d2b821cf421d4f5eb2576359d3770cd3"
TEST_ROOT = Path(os.environ.get("VIME_TEST_ROOT", "/root"))
MODEL_PATH = TEST_ROOT / "models" / MODEL_NAME
MAX_MODEL_LEN = 4096
MAX_NEW_TOKENS = 32
SEED = 2525

GROUP_OVERRIDES = {
    "gpu_memory_utilization": 0.55,
    "max_model_len": MAX_MODEL_LEN,
    "max_num_seqs": 4,
    "enforce_eager": True,
    "generation_config": "vllm",
}


def prepare() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --revision {MODEL_REVISION} --local-dir {MODEL_PATH}")


def _write_config(worker_types: tuple[str, ...]) -> str:
    config = {
        "vllm": [
            {
                "name": "default",
                "model_path": str(MODEL_PATH),
                "update_weights": False,
                "server_groups": [
                    {
                        "worker_type": worker_type,
                        "num_gpus": 1,
                        "num_gpus_per_engine": 1,
                        "overrides": dict(GROUP_OVERRIDES),
                    }
                    for worker_type in worker_types
                ],
            }
        ]
    }
    handle = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="vime_epd_", delete=False)
    with handle:
        yaml.safe_dump(config, handle)
    return handle.name


def _make_args(config_path: str, worker_types: tuple[str, ...]):
    args = vllm_parse_args()
    args.vllm_config = config_path
    args.rollout_external = False
    args.rollout_num_gpus = len(worker_types)
    args.rollout_num_gpus_per_engine = 1
    args.rollout_num_engines = max(1, sum(worker_type != "encoder" for worker_type in worker_types))
    args.num_gpus_per_node = NUM_GPUS
    args.debug_train_only = False
    args.debug_rollout_only = True
    args.colocate = False
    args.actor_num_nodes = 0
    args.actor_num_gpus_per_node = 0
    args.offload_rollout = False
    args.use_critic = False
    args.critic_num_nodes = 0
    args.critic_num_gpus_per_node = 0
    args.hf_checkpoint = str(MODEL_PATH)
    args.seed = SEED
    args.fp16 = False
    args.use_rollout_routing_replay = False
    args.rollout_max_context_len = MAX_MODEL_LEN
    args.use_distributed_post = False
    args.vllm_server_concurrency = 4
    args.vllm_enable_deterministic_inference = True
    args.vllm_router_ip = None
    args.vllm_router_port = None
    args.vllm_pipeline_parallel_size = 1
    args.vllm_data_parallel_size = 1
    args.vllm_dp_size = 1
    return args


def _shutdown_servers(servers: dict[str, Any]) -> None:
    engines = [engine for server in servers.values() for engine in server.all_engines if engine is not None]
    try:
        if engines:
            ray.get([engine.shutdown.remote() for engine in engines], timeout=120)
    finally:
        for engine in engines:
            try:
                ray.kill(engine, no_restart=True)
            except Exception:
                pass

    deadline = time.monotonic() + 60
    while not is_port_available(15000) and time.monotonic() < deadline:
        time.sleep(1)


@contextmanager
def _deployment(pg, worker_types: tuple[str, ...]):
    config_path = _write_config(worker_types)
    args = _make_args(config_path, worker_types)
    servers: dict[str, Any] = {}
    ec_path: Path | None = None
    try:
        servers, init_handles = start_rollout_servers(args, pg)
        if init_handles:
            ray.get(init_handles)
        init_http_client(args)

        for group in servers["default"].server_groups:
            config = group.vllm_overrides.get("ec_transfer_config") or {}
            extra = config.get("ec_connector_extra_config") or {}
            if path := extra.get("shared_storage_path"):
                ec_path = Path(path)
                break
        yield args, servers["default"]
    finally:
        try:
            _shutdown_servers(servers)
        finally:
            if ec_path is not None:
                shutil.rmtree(ec_path, ignore_errors=True)
            Path(config_path).unlink(missing_ok=True)


def _image_data_url() -> str:
    image = Image.new("RGB", (64, 64))
    pixels = image.load()
    for y in range(64):
        for x in range(64):
            pixels[x, y] = (220, 40, 40) if (x // 16 + y // 16) % 2 == 0 else (30, 90, 220)
    output = io.BytesIO()
    image.save(output, format="PNG")
    return "data:image/png;base64," + base64.b64encode(output.getvalue()).decode("ascii")


async def _generate(args, messages: list[dict[str, Any]], *, epd_server=None) -> list[int]:
    base_url = f"http://{args.vllm_router_ip}:{args.vllm_router_port}"
    await vllm_rollout.prime_encoder(args, messages)
    if epd_server is not None:
        assert getattr(args, "vllm_model_encoder_endpoints", {}), "EPD deployment did not expose an encoder endpoint"
        storage_paths = [
            group.vllm_overrides["ec_transfer_config"]["ec_connector_extra_config"]["shared_storage_path"]
            for group in epd_server.server_groups
            if group.worker_type == "encoder"
        ]
        assert any(
            Path(path).glob("*/encoder_cache.safetensors") for path in storage_paths
        ), "encoder did not publish an external EC cache"
    render_data = await post(
        f"{base_url}/v1/chat/completions/render",
        {"model": str(MODEL_PATH), "messages": messages},
    )
    body = vllm_rollout._mm_render_response_to_generate_body(render_data, str(MODEL_PATH))
    body["sampling_params"] = vllm_rollout._build_inference_sampling_params(
        {
            "max_new_tokens": MAX_NEW_TOKENS,
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "seed": SEED,
            "skip_special_tokens": False,
        }
    )
    output = await post(f"{base_url}/inference/v1/generate", body)
    token_ids = output["choices"][0].get("token_ids") or []
    assert token_ids, output
    return [int(token_id) for token_id in token_ids]


def _find_config(value: Any, key: str) -> dict[str, Any] | None:
    if isinstance(value, dict):
        candidate = value.get(key)
        if isinstance(candidate, dict):
            return candidate
        for child in value.values():
            if found := _find_config(child, key):
                return found
    elif isinstance(value, list):
        for child in value:
            if found := _find_config(child, key):
                return found
    return None


def _engine_info(server) -> dict[str, tuple[str, dict[str, Any]]]:
    result = {}
    for group in server.server_groups:
        endpoint = ray.get(group.engines[0].get_url.remote())
        response = requests.get(f"{endpoint}/server_info?config_format=json", timeout=30)
        response.raise_for_status()
        result[group.worker_type] = (endpoint, response.json())
    return result


def _validate_epd_services(server) -> None:
    info = _engine_info(server)
    encoder_ec = _find_config(info["encoder"][1], "ec_transfer_config")
    prefill_ec = _find_config(info["prefill"][1], "ec_transfer_config")
    prefill_kv = _find_config(info["prefill"][1], "kv_transfer_config")
    decode_ec = _find_config(info["decode"][1], "ec_transfer_config")
    decode_kv = _find_config(info["decode"][1], "kv_transfer_config")

    assert encoder_ec is not None
    assert encoder_ec["ec_connector"] == "ECExampleConnector"
    assert encoder_ec["ec_role"] == "ec_producer"
    assert prefill_ec is not None
    assert prefill_ec["ec_connector"] == "ECExampleConnector"
    assert prefill_ec["ec_role"] == "ec_consumer"
    assert prefill_kv is not None and prefill_kv["kv_role"] == "kv_producer"
    assert decode_ec is None or decode_ec.get("ec_role") is None
    assert decode_kv is not None and decode_kv["kv_role"] == "kv_consumer"

    response = requests.get(f"http://{server.router_ip}:{server.router_port}/workers", timeout=30)
    response.raise_for_status()
    workers = response.json()["workers"]
    assert {worker["worker_type"] for worker in workers} == {"prefill", "decode"}
    assert info["encoder"][0] not in {worker["url"] for worker in workers}

    config = server.server_groups[0].vllm_overrides["ec_transfer_config"]
    path = Path(config["ec_connector_extra_config"]["shared_storage_path"])
    assert path.is_dir()


def execute() -> None:
    ray.init()
    pg = _create_placement_group(NUM_GPUS)
    image_messages = [
        {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": _image_data_url()}},
                {"type": "text", "text": "Name the two dominant colors. Answer briefly."},
            ],
        }
    ]
    try:
        with _deployment(pg, ("regular",)) as (baseline_args, _):
            baseline_tokens = asyncio.run(_generate(baseline_args, image_messages))

        with _deployment(pg, ("encoder", "prefill", "decode")) as (epd_args, epd_server):
            epd_tokens = asyncio.run(_generate(epd_args, image_messages, epd_server=epd_server))
            assert epd_tokens == baseline_tokens
            tokenizer = load_tokenizer(str(MODEL_PATH), trust_remote_code=True)
            assert tokenizer.decode(epd_tokens, skip_special_tokens=False) == tokenizer.decode(
                baseline_tokens, skip_special_tokens=False
            )

            _validate_epd_services(epd_server)
    finally:
        if pg[0] is not None:
            ray.util.remove_placement_group(pg[0])
        ray.shutdown()


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
