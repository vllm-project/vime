"""
Runtime check for the vLLM rollout data path.

This test launches a real vLLM server through ``vllm_engine.launch_server_process``
and then calls ``vllm_rollout.generate``. It specifically verifies that the
rollout client can use vLLM's token-only ``/inference/v1/generate`` endpoint.
"""

import asyncio
import os
import socket
from argparse import Namespace
from dataclasses import dataclass

import vime.utils.external_utils.command_utils as U
from vime.backends.vllm_utils.vllm_engine import _compute_server_args, _wait_server_healthy, launch_server_process
from vime.rollout import vllm_rollout
from vime.utils import http_utils
from vime.utils.types import Sample


@dataclass(frozen=True)
class VLLMGenerateCase:
    name: str
    hf_repo: str
    model_path: str
    num_gpus: int
    prompt: str
    use_rollout_routing_replay: bool = False
    max_model_len: int = 1024
    max_new_tokens: int = 8
    timeout_s: float = 600.0


CASES = [
    VLLMGenerateCase(
        name="qwen3-0.6b",
        hf_repo="Qwen/Qwen3-0.6B",
        model_path="/root/models/Qwen3-0.6B",
        num_gpus=1,
        prompt="The capital of France is",
    ),
    VLLMGenerateCase(
        name="qwen3-30b-a3b",
        hf_repo="Qwen/Qwen3-30B-A3B",
        model_path="/root/models/Qwen3-30B-A3B",
        num_gpus=1,
        prompt="Solve: 1 + 1 =",
    ),
    VLLMGenerateCase(
        name="qwen3-30b-a3b-r3",
        hf_repo="Qwen/Qwen3-30B-A3B",
        model_path="/root/models/Qwen3-30B-A3B",
        num_gpus=1,
        prompt="Solve: 2 + 3 =",
        use_rollout_routing_replay=True,
    ),
]
CASES_BY_NAME = {case.name: case for case in CASES}


def _free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _stop_process_tree(process) -> None:
    if not process.is_alive():
        return
    try:
        from vllm.utils.system_utils import kill_process_tree

        kill_process_tree(process.pid)
    except Exception:
        process.terminate()
    process.join(timeout=30)
    if process.is_alive():
        process.kill()
        process.join(timeout=10)


async def _generate_with_http_client(rollout_args, sample: Sample, sampling_params: dict):
    http_utils.init_http_client(rollout_args)
    try:
        return await vllm_rollout.generate(rollout_args, sample, sampling_params)
    finally:
        if http_utils._http_client is not None:
            await http_utils._http_client.aclose()
            http_utils._http_client = None


def prepare():
    U.exec_command("mkdir -p /root/models")
    seen_model_paths = set()
    for case in CASES:
        if case.model_path in seen_model_paths:
            continue
        seen_model_paths.add(case.model_path)
        if not os.path.exists(case.model_path):
            U.exec_command(f"hf download {case.hf_repo} --local-dir {case.model_path}")


def _execute_case(case: VLLMGenerateCase):
    if not os.path.exists(case.model_path):
        U.exec_command(f"hf download {case.hf_repo} --local-dir {case.model_path}")

    server_port = _free_port()
    args = Namespace(
        rollout_num_gpus_per_engine=case.num_gpus,
        num_gpus_per_node=case.num_gpus,
        hf_checkpoint=case.model_path,
        seed=1234,
        vllm_gpu_memory_utilization=0.9,
        vllm_async_scheduling=False,
        vllm_enforce_eager=False,
        vllm_enable_sleep_mode=False,
        rollout_max_context_len=case.max_model_len,
        use_rollout_routing_replay=case.use_rollout_routing_replay,
        vllm_dp_size=1,
        # Placement attrs so _compute_server_args derives the GPU base through the
        # real get_base_gpu_id() path rather than a hardcoded id. A single
        # colocate engine at rank 0 -> local base 0, and the child server's
        # CUDA_VISIBLE_DEVICES is computed exactly as the production VLLMEngine
        # does (honoring an externally set CUDA_VISIBLE_DEVICES instead of always
        # grabbing physical GPU 0).
        colocate=True,
        actor_num_nodes=1,
        actor_num_gpus_per_node=case.num_gpus,
        use_critic=False,
        debug_rollout_only=False,
    )

    # ``launch_server_process`` consumes a single ``server_args`` dict built by
    # ``_compute_server_args`` since the PR #68 multi-node topology refactor
    # (this test predates it). Let it derive GPU placement via
    # get_base_gpu_id()/_to_local_gpu_id() — identical to VLLMEngine — so the
    # launched server tracks CUDA_VISIBLE_DEVICES.
    server_args = _compute_server_args(
        args,
        rank=0,
        dist_init_addr=None,
        host="127.0.0.1",
        port=server_port,
    )
    process = launch_server_process(server_args)

    try:
        _wait_server_healthy(f"http://127.0.0.1:{server_port}", process)

        rollout_args = Namespace(
            ci_test=False,
            hf_checkpoint=case.model_path,
            vllm_router_ip="127.0.0.1",
            vllm_router_port=server_port,
            vllm_server_concurrency=512,
            rollout_num_gpus=case.num_gpus,
            rollout_num_gpus_per_engine=case.num_gpus,
            rollout_temperature=0.0,
            rollout_top_p=1.0,
            rollout_top_k=-1,
            rollout_max_response_len=case.max_new_tokens,
            rollout_stop=None,
            rollout_stop_token_ids=None,
            rollout_skip_special_tokens=True,
            vllm_dp_size=1,
            use_rollout_routing_replay=case.use_rollout_routing_replay,
            vllm_speculative_config=None,
            use_distributed_post=False,
        )
        sampling_params = {
            "temperature": 0.0,
            "top_p": 1.0,
            "top_k": -1,
            "max_new_tokens": case.max_new_tokens,
            "stop": None,
            "stop_token_ids": None,
            "skip_special_tokens": True,
        }

        # Ensure a previous GenerateState singleton from an in-process runner
        # cannot keep stale tokenizer/router args.
        vllm_rollout.GenerateState.clear_instances()
        sample = asyncio.run(
            _generate_with_http_client(
                rollout_args,
                Sample(index=0, prompt=case.prompt),
                sampling_params,
            )
        )

        assert sample.response_length > 0
        assert sample.rollout_log_probs is not None
        assert len(sample.rollout_log_probs) == sample.response_length
        assert sample.status in (Sample.Status.COMPLETED, Sample.Status.TRUNCATED)
        if case.use_rollout_routing_replay:
            re = sample.rollout_routed_experts
            assert re is not None
            assert re.ndim == 3
            expected_rows = len(sample.tokens) - 1
            assert (
                re.shape[0] == expected_rows
            ), f"rollout_routed_experts rows {re.shape[0]} != len(tokens)-1 ({expected_rows})"
    finally:
        _stop_process_tree(process)


def test_qwen3_0_6b_vllm_inference_generate_endpoint():
    _execute_case(CASES_BY_NAME["qwen3-0.6b"])


def test_qwen3_30b_a3b_vllm_inference_generate_endpoint():
    _execute_case(CASES_BY_NAME["qwen3-30b-a3b"])


def test_qwen3_30b_a3b_r3_vllm_inference_generate_endpoint():
    _execute_case(CASES_BY_NAME["qwen3-30b-a3b-r3"])


def execute():
    for case in CASES:
        _execute_case(case)


if __name__ == "__main__":
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    prepare()
    execute()
