"""Single-GPU end-to-end regression for Geo3K Issue #331."""

from __future__ import annotations

import asyncio
import os
import shlex
import subprocess
import sys
import time
from argparse import Namespace
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.error import URLError
from urllib.parse import urlsplit
from urllib.request import urlopen

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vllm.utils.system_utils import kill_process_tree

import vime.utils.external_utils.command_utils as U
from vime.utils.http_utils import is_port_available
from vime.utils.misc import load_function

NUM_GPUS = 1
MODEL_NAME = "Qwen3-VL-2B-Instruct"
MODEL_REVISION = "89644892e4d85e24eaac8bacfd4f463576704203"
TEST_ROOT = Path(os.environ.get("VIME_TEST_ROOT", "/root"))
MODEL_PATH = TEST_ROOT / "models" / MODEL_NAME
DATASET_NAME = "VeraIsHere/geo3k_imgurl_processed"
DATASET_REVISION = "53ff8fbb1f9758b9efbc005e9487e1fdd874364a"
DATA_ROOT = TEST_ROOT / "datasets/geo3k_imgurl_processed"
TRAIN_DATA_PATH = DATA_ROOT / "train.parquet"
CUSTOM_GENERATE_PATH = "examples.geo3k_vlm_multi_turn.rollout.generate"
HOST = "127.0.0.1"
WORKER_PORT = 10090
ROUTER_PORT = 30000
PROMETHEUS_PORT = 29001
WORKER_URL = f"http://{HOST}:{WORKER_PORT}"
ROUTER_URL = f"http://{HOST}:{ROUTER_PORT}"
RENDER_ENDPOINT = "/v1/chat/completions/render"
GENERATE_ENDPOINT = "/inference/v1/generate"
SEED = 331
ROW_INDEX = 0
MAX_NEW_TOKENS = 1024
MAX_TURNS = 2
TEMPERATURE = 0.0
TOP_P = 1.0
TOP_K = -1
GPU_MEMORY_UTILIZATION = 0.70
MAX_MODEL_LEN = 4096
MAX_NUM_SEQS = 1
ROUTER_REQUEST_TIMEOUT_S = 600
HTTP_MAX_RETRIES = 60
SERVICE_START_TIMEOUT_S = 180
SERVICE_POLL_INTERVAL_S = 2
PROCESS_STOP_TIMEOUT_S = 30
WORKER_LOG = Path("/tmp/geo3k-vllm-worker.log")
ROUTER_LOG = Path("/tmp/geo3k-vllm-router.log")
WORKER_COMMAND = shlex.split(
    f"vllm serve {MODEL_PATH} --host {HOST} --port {WORKER_PORT} --dtype bfloat16 "
    f"--tensor-parallel-size {NUM_GPUS} --gpu-memory-utilization {GPU_MEMORY_UTILIZATION} "
    f"--max-model-len {MAX_MODEL_LEN} --max-num-seqs {MAX_NUM_SEQS} --enforce-eager "
    "--generation-config vllm --logprobs-mode processed_logprobs"
)
ROUTER_COMMAND = shlex.split(
    f"vllm-router --host {HOST} --port {ROUTER_PORT} --worker-urls {WORKER_URL} "
    f"--policy consistent_hash --prometheus-port {PROMETHEUS_PORT} --prometheus-host {HOST} "
    f"--request-timeout-secs {ROUTER_REQUEST_TIMEOUT_S}"
)


@dataclass(frozen=True)
class RequestEvent:
    endpoint: str
    request_token_ids: tuple[int, ...]
    response_token_ids: tuple[int, ...]
    has_features: bool


class TwoTurnEnvironment:
    def reset(self) -> None:
        pass

    def step(self, _response: str) -> tuple[str, bool, dict]:
        return "Continue the reasoning.", False, {}

    def format_observation(self, observation: str) -> dict[str, str]:
        return {"role": "user", "content": observation}

    def close(self) -> None:
        pass


def build_env(*, sample: Any, args: Namespace) -> TwoTurnEnvironment:
    del sample, args
    return TwoTurnEnvironment()


def prepare() -> None:
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    DATA_ROOT.parent.mkdir(parents=True, exist_ok=True)
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --revision {MODEL_REVISION} --local-dir {MODEL_PATH}")
    U.exec_command(
        f"hf download --repo-type dataset {DATASET_NAME} --revision {DATASET_REVISION} --local-dir {DATA_ROOT}"
    )
    if not TRAIN_DATA_PATH.is_file():
        raise FileNotFoundError(f"Dataset not found at {TRAIN_DATA_PATH}")


def _wait_for_health(process: subprocess.Popen, health_url: str, log_path: Path) -> None:
    deadline = time.monotonic() + SERVICE_START_TIMEOUT_S
    while time.monotonic() < deadline:
        if process.poll() is not None:
            break
        try:
            with urlopen(f"{health_url}/health", timeout=SERVICE_POLL_INTERVAL_S):
                return
        except URLError:
            time.sleep(SERVICE_POLL_INTERVAL_S)
    logs = log_path.read_text(encoding="utf-8", errors="replace")
    raise RuntimeError(f"Service failed to become healthy at {health_url}:\n{logs}")


def _stop_process(process: subprocess.Popen | None) -> None:
    if process is None or process.poll() is not None:
        return
    kill_process_tree(process.pid)
    process.wait(timeout=PROCESS_STOP_TIMEOUT_S)


def _start_process(
    command: list[str],
    health_url: str,
    *,
    log_path: Path,
    env: dict[str, str] | None = None,
) -> subprocess.Popen:
    with log_path.open("w", encoding="utf-8") as log_file:
        process = subprocess.Popen(command, stdout=log_file, stderr=subprocess.STDOUT, env=env)
    try:
        _wait_for_health(process, health_url, log_path)
    except BaseException:
        _stop_process(process)
        raise
    return process


@contextmanager
def _vllm_services() -> Iterator[None]:
    ports = (WORKER_PORT, ROUTER_PORT, PROMETHEUS_PORT)
    if unavailable := [port for port in ports if not is_port_available(port)]:
        raise RuntimeError(f"Required test ports are already in use: {unavailable}")
    worker_env = os.environ.copy()
    worker_env.update(VLLM_SERVER_DEV_MODE="1", VLLM_BATCH_INVARIANT="1")
    worker = _start_process(WORKER_COMMAND, WORKER_URL, log_path=WORKER_LOG, env=worker_env)
    router: subprocess.Popen | None = None
    try:
        router = _start_process(ROUTER_COMMAND, ROUTER_URL, log_path=ROUTER_LOG)
        yield
    finally:
        try:
            _stop_process(router)
        finally:
            _stop_process(worker)


@contextmanager
def _trace_http_requests() -> Iterator[list[RequestEvent]]:
    from vime.utils import http_utils

    original_post = http_utils.post
    events: list[RequestEvent] = []

    async def traced_post(url, body, max_retries=HTTP_MAX_RETRIES, *, headers=None):
        output = await original_post(url, body, max_retries=max_retries, headers=headers)
        choices = output.get("choices") if isinstance(output, dict) else None
        choice = choices[0] if isinstance(choices, list) and choices else {}
        events.append(
            RequestEvent(
                endpoint=urlsplit(url).path,
                request_token_ids=tuple(int(token) for token in body.get("token_ids") or ()),
                response_token_ids=tuple(int(token) for token in choice.get("token_ids") or ()),
                has_features=body.get("features") is not None,
            )
        )
        return output

    http_utils.post = traced_post
    try:
        yield events
    finally:
        http_utils.post = original_post


def _load_sample() -> Any:
    from vime.utils.data import Dataset
    from vime.utils.processing_utils import load_processor, load_tokenizer

    data_slice = f"{TRAIN_DATA_PATH}@[{ROW_INDEX}:{ROW_INDEX + 1}]"
    tokenizer = load_tokenizer(str(MODEL_PATH), trust_remote_code=True)
    processor = load_processor(str(MODEL_PATH), trust_remote_code=True)
    dataset = Dataset(
        data_slice,
        tokenizer,
        processor,
        None,
        prompt_key="problem",
        multimodal_keys={"image": "images"},
    )
    return dataset[0]


async def _run_rollout() -> tuple[Any, list[RequestEvent]]:
    from vime.utils import http_utils

    args = Namespace(
        partial_rollout=False,
        max_turns=MAX_TURNS,
        rollout_interaction_env_path=__name__,
        vllm_router_ip=HOST,
        vllm_router_port=ROUTER_PORT,
        router_policy="consistent_hash",
        hf_checkpoint=str(MODEL_PATH),
        rollout_max_context_len=None,
        use_rollout_routing_replay=False,
        vllm_server_concurrency=NUM_GPUS,
        rollout_num_engines=NUM_GPUS,
        use_distributed_post=False,
        rollout_temperature=TEMPERATURE,
        rollout_top_p=TOP_P,
        rollout_top_k=TOP_K,
        rollout_max_response_len=MAX_NEW_TOKENS,
        rollout_stop=None,
        rollout_stop_token_ids=None,
        rollout_skip_special_tokens=False,
        vllm_dp_size=NUM_GPUS,
    )
    sample = _load_sample()
    http_utils.init_http_client(args)
    sampling = {
        "max_new_tokens": MAX_NEW_TOKENS,
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "skip_special_tokens": False,
        "seed": SEED,
    }
    with _trace_http_requests() as events:
        generate = load_function(CUSTOM_GENERATE_PATH)
        result = await generate(args, sample, sampling)
    return result, events


def _validate_result(sample: Any, events: list[RequestEvent]) -> None:
    assert [event.endpoint for event in events] == [RENDER_ENDPOINT, GENERATE_ENDPOINT, GENERATE_ENDPOINT]
    first, second = events[1:]
    assert first.request_token_ids and first.response_token_ids
    generated_start = len(first.request_token_ids)
    generated_end = generated_start + len(first.response_token_ids)
    assert second.request_token_ids[:generated_start] == first.request_token_ids
    assert second.request_token_ids[generated_start:generated_end] == first.response_token_ids
    assert len(second.request_token_ids) > generated_end
    assert sample.tokens == list(second.request_token_ids + second.response_token_ids)
    assert first.has_features and second.has_features
    assert sample.response_length == len(sample.loss_mask) == len(sample.rollout_log_probs)


def execute() -> None:
    with _vllm_services():
        sample, events = asyncio.run(_run_rollout())
    _validate_result(sample, events)


if __name__ == "__main__":
    prepare()
    execute()
