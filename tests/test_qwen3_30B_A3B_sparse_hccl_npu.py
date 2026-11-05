"""Six-NPU training, sparse weight update and checkpoint inference comparison.

MODEL_PATH and PROMPT_DATA_PATH reuse local inputs. SPARSE_HCCL_RESULT_DIR
stores an independent full checkpoint, sync metrics and both inference results.
SPARSE_HCCL_BASELINE_PROMPT selects the prompt (default: a short arithmetic task).
Token IDs and every generated-token logprob must match exactly; no tolerance.
"""

import copy
import json
import math
import os
import shlex
import socket
import sys
import time
from pathlib import Path

TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = os.environ.get("MODEL_PATH", f"{TEST_ROOT}/models/Qwen3-30B-A3B")
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"
PROMPT_DATA = os.environ.get("PROMPT_DATA_PATH", f"{DATASET_DIR}/dapo-math-17k.jsonl")
RESULT_DIR = os.environ.get("SPARSE_HCCL_RESULT_DIR", f"{TEST_ROOT}/sparse-hccl-e2e-{time.time_ns()}")
BASELINE_PROMPT = os.environ.get("SPARSE_HCCL_BASELINE_PROMPT", "What is 17 + 25? Give the answer in one sentence.")


def prepare():
    import vime.utils.external_utils.command_utils as U

    U.exec_command(f"mkdir -p {shlex.quote(f'{TEST_ROOT}/models')} {shlex.quote(f'{TEST_ROOT}/datasets')}")
    if not Path(MODEL_DIR, "config.json").is_file():
        U.exec_command(f"hf download Qwen/Qwen3-30B-A3B --local-dir {shlex.quote(MODEL_DIR)}")
    if not Path(PROMPT_DATA).is_file():
        U.exec_command(
            "hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {shlex.quote(DATASET_DIR)}"
        )


def execute():
    import vime.utils.external_utils.command_utils as U

    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(PROMPT_DATA)
    checkpoint_args = (
        f"--hf-checkpoint {model_dir} " f"--load {model_dir} " f"--ref-load {model_dir} " "--no-load-optim "
    )

    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {int(os.environ.get('SPARSE_HCCL_NUM_ROLLOUT', '3'))} "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 64 "
        "--rollout-temperature 1 "
        "--global-batch-size 8 "
        "--balance-data "
    )

    parallel_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 2 "
        "--expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--micro-batch-size 1 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.001 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 2 "
        "--vllm-pipeline-parallel-size 1 "
        "--vllm-enable-expert-parallel "
        "--vllm-expert-placement-strategy linear "
        "--vllm-gpu-memory-utilization 0.6 "
        "--vllm-enforce-eager "
        "--vllm-enable-scale-out "
        """--vllm-additional-config '{"weight_nz_mode":0}' """
    )

    update_weight_args = (
        "--update-weight-mode delta "
        "--update-weight-transport sparse_hccl "
        "--update-weight-delta-batch-diff 32 "
        "--update-weight-delta-batch-gather 32 "
    )

    model_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--use-flash-attn "
        "--no-gradient-accumulation-fusion "
    )

    runtime_args = (
        "--train-backend megatron "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 2 "
        "--ci-test "
    )

    train_args = (
        checkpoint_args
        + rollout_args
        + parallel_args
        + grpo_args
        + optimizer_args
        + vllm_args
        + update_weight_args
        + model_args
        + runtime_args
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=6,
        train_script="tests/test_qwen3_30B_A3B_sparse_hccl_npu.py --e2e-worker",
        megatron_model_type="qwen3-30B-A3B",
        extra_env_vars={
            "DISABLE_L2_CACHE": "1",
            "VLLM_USE_AOT_COMPILE": "0",
            "SPARSE_HCCL_RESULT_DIR": RESULT_DIR,
            "SPARSE_HCCL_BASELINE_PROMPT": BASELINE_PROMPT,
            "SPARSE_HCCL_BASELINE_TOKENS": os.environ.get("SPARSE_HCCL_BASELINE_TOKENS", "64"),
        },
    )


def main():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


def _write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _infer(url, prompt_ids, sampling_params):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        models = session.get(f"{url}/v1/models", timeout=30)
        models.raise_for_status()
        response = session.post(
            f"{url}/inference/v1/generate",
            json={
                "model": models.json()["data"][0]["id"],
                "token_ids": prompt_ids,
                "sampling_params": sampling_params,
            },
            timeout=300,
        )
        response.raise_for_status()
        body = response.json()
    choices = body["choices"]
    if len(choices) != 1:
        raise AssertionError(f"Expected one generation, got {len(choices)}")
    choice = choices[0]
    tokens = choice["token_ids"]
    rows = choice["logprobs"]["content"]
    if not tokens or len(tokens) != len(rows):
        raise AssertionError("Missing generated token IDs or aligned logprobs")
    logprobs = [float(row["logprob"]) for row in rows]
    if not all(math.isfinite(value) for value in logprobs):
        raise AssertionError("Non-finite generated-token logprob")
    return {"token_ids": tokens, "logprobs": logprobs, "finish_reason": choice["finish_reason"]}


def _compare(actual, expected):
    tokens_equal = actual["token_ids"] == expected["token_ids"]
    logprobs_equal = actual["logprobs"] == expected["logprobs"]
    same_length = len(actual["logprobs"]) == len(expected["logprobs"])
    max_error = max((abs(a - b) for a, b in zip(actual["logprobs"], expected["logprobs"], strict=False)), default=0.0)
    return {
        "token_ids_equal": tokens_equal,
        "logprobs_equal": logprobs_equal,
        "finish_reason_equal": actual["finish_reason"] == expected["finish_reason"],
        "max_abs_logprob_error": max_error if same_length else None,
        "passed": tokens_equal and logprobs_equal and actual["finish_reason"] == expected["finish_reason"],
    }


def run_e2e(args):
    import ray
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from vime.backends.megatron_utils.actor import MegatronTrainRayActor
    from vime.backends.megatron_utils.hf_checkpoint_saver import (
        _copy_hf_assets,
        _raise_if_rank_zero_failed,
        _SafetensorShardWriter,
    )
    from vime.backends.vllm_utils.vllm_engine import _compute_server_args, launch_server_process
    from vime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
    from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models

    class OracleTrainActor(MegatronTrainRayActor):
        def update_weights(self):
            updater = self.weight_updater
            publish = updater._publish_flush
            updates = 0

            def observe(patches):
                nonlocal updates
                publish(patches)
                updates += sum(patch.indices.numel() for patch in patches)

            updater._publish_flush = observe
            try:
                super().update_weights()
            finally:
                updater._publish_flush = publish
            if dist.get_rank() == 0 and updates == 0:
                raise AssertionError("Expected a nonempty weight update")

        def export_checkpoint(self, directory):
            # Full Bridge export from the frozen training model, independent of
            # sparse patches and their snapshots. Every trainer rank participates.
            path = Path(directory)
            root = dist.get_rank() == 0
            error = None
            if root:
                try:
                    path.mkdir(parents=True, exist_ok=False)
                    _copy_hf_assets(self.args.hf_checkpoint, path)
                except Exception as exc:
                    error = repr(exc)
            _raise_if_rank_zero_failed("prepare comparison checkpoint", error)
            writer = _SafetensorShardWriter(path, enabled=root)
            for number, chunk in enumerate(self.weight_updater._iterator.get_hf_weight_chunks()):
                error = None
                if root:
                    try:
                        writer.write(chunk, number)
                    except Exception as exc:
                        error = repr(exc)
                _raise_if_rank_zero_failed("write comparison checkpoint shard", error)
            error = None
            if root:
                try:
                    writer.finalize()
                except Exception as exc:
                    error = repr(exc)
            _raise_if_rank_zero_failed("finalize comparison checkpoint", error)
            return self.weight_updater.weight_version if root else None

    if args.use_critic or args.offload_train or args.offload_rollout or args.release_train:
        raise ValueError("This comparison requires a resident actor and one resident rollout engine")
    if (
        args.num_rollout < 1
        or args.actor_num_nodes != 1
        or args.actor_num_gpus_per_node != 4
        or args.rollout_num_gpus != 2
    ):
        raise ValueError("Use at least one training step on four training plus two rollout NPUs")
    output = Path(RESULT_DIR).resolve()
    output.mkdir(parents=True, exist_ok=True)
    checkpoint = output / "updated-checkpoint"
    if checkpoint.exists() or (output / "comparison.json").exists():
        raise FileExistsError(f"Use a fresh SPARSE_HCCL_RESULT_DIR: {output}")
    tokenizer = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)
    prompt_ids = tokenizer.apply_chat_template(
        [{"role": "user", "content": BASELINE_PROMPT}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    sampling = {
        "temperature": 0.0,
        "top_p": 1.0,
        "top_k": -1,
        "seed": args.seed,
        "max_tokens": int(os.environ.get("SPARSE_HCCL_BASELINE_TOKENS", "64")),
        "logprobs": 1,
    }
    _write_json(
        output / "baseline.json",
        {"prompt": BASELINE_PROMPT, "prompt_token_ids": prompt_ids, "sampling_params": sampling},
    )
    configure_logger()
    init_tracking(args)
    manager = actor = process = None
    try:
        pgs = create_placement_groups(args)
        manager, _ = create_rollout_manager(args, pgs["rollout"])
        actor, _ = create_training_models(args, pgs, manager, actor_cls=OracleTrainActor)
        actor.update_weights()
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            data = ray.get(manager.generate.remote(rollout_id))
            ray.get(actor.async_train(rollout_id, data))
            actor.clear_memory()
            actor.update_weights()
        ray.get(manager.health_monitoring_pause.remote())
        engines, *_ = ray.get(manager.get_updatable_engines_and_lock.remote())
        if len(engines) != 1:
            raise AssertionError("The comparison requires exactly one rollout engine")
        url = ray.get(engines[0].get_url.remote())
        version = ray.get(engines[0].get_weight_version.remote())
        export_version = ray.get(
            [worker.export_checkpoint.remote(str(checkpoint)) for worker in actor._actor_handlers]
        )[0]
        if str(version) != str(export_version):
            raise AssertionError("Inference weight version does not match the exported training model")
        ray.get(engines[0].flush_cache.remote())
        actual = _infer(url, prompt_ids, sampling)
        _write_json(output / "sparse-inference.json", actual)
        ray.get(engines[0].flush_cache.remote())
        repeat = _infer(url, prompt_ids, sampling)
        if not _compare(actual, repeat)["passed"]:
            raise AssertionError("Sparse inference is not exactly repeatable on the baseline prompt")
        ray.get(manager.dispose.remote())
        manager = None
        actor.release()
        actor = None

        # Start a fresh serving process on the same two devices and with the
        # same engine options. It loads the exported checkpoint from disk.
        reference_args = copy.deepcopy(args)
        reference_args.hf_checkpoint = str(checkpoint)
        with socket.socket() as reserved:
            reserved.bind(("127.0.0.1", 0))
            port = reserved.getsockname()[1]
        kwargs, _ = _compute_server_args(
            reference_args,
            0,
            f"127.0.0.1:{port}",
            "127.0.0.1",
            port,
            "regular",
            None,
            base_gpu_id=int(pgs["rollout"][2][0]),
        )
        process = launch_server_process(kwargs)
        expected = _infer(f"http://127.0.0.1:{port}", prompt_ids, sampling)
        _write_json(output / "checkpoint-inference.json", expected)
        comparison = {"weight_version": version, "checkpoint": str(checkpoint), **_compare(actual, expected)}
        _write_json(output / "comparison.json", comparison)
        if not comparison["passed"]:
            raise AssertionError(f"Sparse/checkpoint token IDs or logprobs differ: {comparison}")
        print("SPARSE HCCL E2E PASS: token IDs and logprobs match exactly", flush=True)
    finally:
        from vllm.utils.system_utils import kill_process_tree

        if process is not None:
            kill_process_tree(process.pid)
            process.join(timeout=30)
        try:
            if manager is not None:
                ray.get(manager.dispose.remote())
        finally:
            if actor is not None:
                actor.release()
            finish_tracking(args)


def test_exact_inference_comparison():
    reference = {"token_ids": [17, 25], "logprobs": [-0.25, -1.5], "finish_reason": "length"}
    assert _compare(reference, copy.deepcopy(reference))["passed"]
    for field, changed in [
        ("token_ids", [17, 26]),
        ("logprobs", [-0.25, -1.5000001]),
        ("logprobs", [-0.25]),
        ("finish_reason", "stop"),
    ]:
        candidate = {**reference, field: changed}
        assert not _compare(candidate, reference)["passed"]


if __name__ == "__main__":
    if "--e2e-worker" in sys.argv:
        sys.argv.remove("--e2e-worker")
        from vime.platforms import current_platform

        if current_platform().is_npu:
            import vime.backends.megatron_utils  # noqa: F401
        from vime.utils.arguments import parse_args

        run_e2e(parse_args())
    else:
        main()
