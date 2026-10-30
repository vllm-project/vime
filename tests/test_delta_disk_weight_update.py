"""Qwen3-0.6B disk-delta E2E and full-checkpoint inference oracle.

Each delta encoding runs a baseline pull, two training/update rounds, then
compares the reloaded rollout engine with a fresh engine loaded from an
independently exported full checkpoint. Token IDs and generated-token
logprobs must match exactly.
"""

import copy
import json
import math
import multiprocessing
import os
import shlex
import shutil
import socket
import sys
import time
from contextlib import contextmanager
from pathlib import Path

from safetensors import safe_open

import vime.utils.external_utils.command_utils as U
from vime.utils.disk_delta import checksum, make_tensor_reader


MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
TRAIN_GPUS = 1
ENGINE_GPUS = 1
NUM_GPUS = TRAIN_GPUS + ENGINE_GPUS
TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = os.environ.get("MODEL_PATH", f"{TEST_ROOT}/models/{MODEL_NAME}")
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"
PROMPT_DATA = os.environ.get("PROMPT_DATA_PATH", f"{DATASET_DIR}/dapo-math-17k.jsonl")
RESULT_DIR = os.environ.get("DISK_DELTA_RESULT_DIR", f"{TEST_ROOT}/disk-delta-0.6b-{time.time_ns()}")
BASELINE_PROMPT = os.environ.get("DISK_DELTA_BASELINE_PROMPT", "What is 17 + 25? Give the answer in one sentence.")


def prepare():
    model_dir = shlex.quote(MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)
    U.exec_command(f"mkdir -p {shlex.quote(f'{TEST_ROOT}/models')} {shlex.quote(f'{TEST_ROOT}/datasets')}")
    if not Path(MODEL_DIR, "config.json").is_file():
        U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {model_dir}")
    if not Path(PROMPT_DATA).is_file():
        U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {dataset_dir}")


def _write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8")


def _infer(url, prompt_ids, sampling):
    import requests

    with requests.Session() as session:
        session.trust_env = False
        models = session.get(f"{url}/v1/models", timeout=30)
        models.raise_for_status()
        payload = models.json()
        data = payload.get("data") if isinstance(payload, dict) else None
        if not isinstance(data, list) or not data or not isinstance(data[0], dict) or not data[0].get("id"):
            raise AssertionError(f"Invalid model list from {url}/v1/models: {payload!r}")
        response = session.post(
            f"{url}/inference/v1/generate",
            json={
                "model": data[0]["id"],
                "token_ids": prompt_ids,
                "sampling_params": sampling,
            },
            timeout=300,
        )
        response.raise_for_status()
    choices = response.json()["choices"]
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


def _serve_reference(kwargs, env, listener):
    import argparse

    os.environ.update(env)
    import uvloop
    from vllm.entrypoints.launchers.cli_args import make_arg_parser, validate_parsed_serve_args
    from vllm.entrypoints.openai.api_server import run_server_worker
    from vllm.utils.argparse_utils import FlexibleArgumentParser

    parser = make_arg_parser(FlexibleArgumentParser())
    args = parser.parse_args(args=[], namespace=argparse.Namespace(**kwargs))
    args.api_server_count = 1
    validate_parsed_serve_args(args)
    uvloop.run(run_server_worker(f"http://{args.host}:{args.port}", listener, args))


@contextmanager
def _reference_server(reference_args):
    import requests
    from vllm.utils.system_utils import kill_process_tree

    from vime.backends.vllm_utils.vllm_engine import _build_subprocess_env, _compute_server_args

    # Keep the socket bound throughout spawn and serving; never release and rebind its port.
    with socket.socket() as listener:
        listener.bind(("127.0.0.1", 0))
        port = listener.getsockname()[1]
        url = f"http://127.0.0.1:{port}"
        kwargs, _ = _compute_server_args(
            reference_args,
            0,
            f"127.0.0.1:{port}",
            "127.0.0.1",
            port,
            "regular",
            None,
            base_gpu_id=TRAIN_GPUS,
            num_gpus_per_engine=ENGINE_GPUS,
        )
        env = _build_subprocess_env(kwargs)
        process = multiprocessing.get_context("spawn").Process(
            target=_serve_reference,
            args=({k: v for k, v in kwargs.items() if not k.startswith("_")}, env, listener),
        )
        try:
            process.start()
            deadline = time.monotonic() + 600
            with requests.Session() as session:
                session.trust_env = False
                while True:
                    if not process.is_alive():
                        raise RuntimeError(f"Fresh-checkpoint server exited with code {process.exitcode}: {url}")
                    try:
                        if session.get(f"{url}/health", timeout=2).status_code == 200:
                            break
                    except requests.RequestException:
                        pass
                    if time.monotonic() >= deadline:
                        raise TimeoutError(f"Fresh-checkpoint server did not become healthy: {url}")
                    time.sleep(1)
            yield url
        finally:
            if process.pid is not None:
                kill_process_tree(process.pid)
                process.join(timeout=30)


def _compare(actual, expected):
    same_length = len(actual["logprobs"]) == len(expected["logprobs"])
    errors = [abs(a - b) for a, b in zip(actual["logprobs"], expected["logprobs"], strict=False)]
    result = {
        "token_ids_equal": actual["token_ids"] == expected["token_ids"],
        "logprobs_equal": actual["logprobs"] == expected["logprobs"],
        "finish_reason_equal": actual["finish_reason"] == expected["finish_reason"],
        "max_abs_logprob_error": max(errors, default=0.0) if same_length else None,
    }
    result["passed"] = all(result[key] for key in ("token_ids_equal", "logprobs_equal", "finish_reason_equal"))
    return result


def verify_checkpoints(case_dir: Path, encoding: str, round_metrics: list[dict]):
    comparison_path = case_dir / "comparison.json"
    assert comparison_path.is_file(), f"Missing fresh-checkpoint inference comparison: {comparison_path}"
    inference_comparison = json.loads(comparison_path.read_text(encoding="utf-8"))
    assert inference_comparison.get("passed") is True, inference_comparison
    assert inference_comparison.get("max_abs_logprob_error") == 0.0, inference_comparison

    published = case_dir / "published"
    local = case_dir / "local"
    indexes = []
    for version in (1, 2):
        path = published / f"weight_v{version:06d}"
        index = json.loads((path / "model.safetensors.index.json").read_text(encoding="utf-8"))
        metadata = index["metadata"]
        assert metadata["version"] == f"{version:06d}", metadata
        assert metadata["base_version"] == f"{version - 1:06d}", metadata
        assert metadata["delta_encoding"] == encoding, metadata
        assert metadata["compression_format"] == "zstd", metadata
        assert metadata["checksum_format"] == "adler32", metadata
        assert index["weight_map"], f"No changed tensors in delta version {version}"
        for filename in set(index["weight_map"].values()):
            assert (path / filename).stat().st_size > 0
        indexes.append(index)

    state = json.loads((local / ".weight_sync/state.json").read_text(encoding="utf-8"))
    assert state == {"version": "000002"}, state
    read_tensor = make_tensor_reader(str(local))
    final_index = indexes[-1]
    checked_delta_tensors = 0
    for filename in set(final_index["weight_map"].values()):
        with safe_open(str(published / "weight_v000002" / filename), framework="np") as delta:
            digests = delta.metadata()
            for name in delta.keys():
                assert final_index["weight_map"][name] == filename
                assert (
                    checksum("adler32", read_tensor(name)) == digests[name]
                ), f"Recovered weight checksum mismatch: {name}"
                checked_delta_tensors += 1
    assert checked_delta_tensors == len(final_index["weight_map"])

    # Independently compare the complete delta-reconstructed checkpoint to the
    # Bridge-exported final model, covering both changed and unchanged tensors.
    oracle = make_tensor_reader(str(case_dir / "updated-checkpoint"))
    checked_full_tensors = 0
    for name in json.loads(
        (case_dir / "updated-checkpoint" / "model.safetensors.index.json").read_text(encoding="utf-8")
    )["weight_map"]:
        actual = read_tensor(name)
        expected = oracle(name)
        assert actual.shape == expected.shape, f"Reconstructed shape mismatch: {name}"
        assert (actual == expected).all(), f"Reconstructed bytes differ from full checkpoint: {name}"
        checked_full_tensors += 1

    assert len(round_metrics) == 2, round_metrics
    assert all(round_.get("changed_bytes", 0) > 0 for round_ in round_metrics), round_metrics
    summary = {
        "model": MODEL_NAME,
        "encoding": encoding,
        "rounds": round_metrics,
        "local_version": state["version"],
        "checked_delta_tensors": checked_delta_tensors,
        "checked_full_checkpoint_tensors": checked_full_tensors,
        "full_checkpoint_byte_match": True,
        "inference_comparison": inference_comparison,
    }
    _write_json(case_dir / "validation.json", summary)
    # Keep the compact functional evidence; the independent full checkpoint,
    # local base and versioned payloads can each occupy tens of GiB.
    for path in (published, local, case_dir / "updated-checkpoint"):
        shutil.rmtree(path)
    print(f"Disk delta {MODEL_NAME} checkpoint validation passed: {summary}", flush=True)


def execute():
    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(PROMPT_DATA)
    output = Path(RESULT_DIR).resolve()
    output.mkdir(parents=True, exist_ok=False)
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    for encoding in ("xor", "overwrite"):
        case_dir = output / encoding
        case_dir.mkdir()
        ckpt_args = (
            f"--hf-checkpoint {model_dir} " f"--load {model_dir} " f"--ref-load {model_dir} " "--no-load-optim "
        )

        rollout_args = (
            f"--prompt-data {shlex.quote(prompt_data)} "
            "--input-key prompt "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type deepscaler "
            "--num-rollout 2 "
            "--rollout-batch-size 2 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 64 "
            "--rollout-temperature 1 "
            "--global-batch-size 8 "
            "--balance-data "
        )

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
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
            f"--rollout-num-gpus-per-engine {ENGINE_GPUS} "
            f"--rollout-num-gpus {ENGINE_GPUS} "
            "--vllm-enable-scale-out "
            "--vllm-pipeline-parallel-size 1 "
            "--vllm-gpu-memory-utilization 0.6 "
            "--vllm-enforce-eager "
            "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        )

        disk_update_args = (
            "--update-weight-mode delta "
            "--update-weight-transport disk "
            f"--update-weight-disk-dir {shlex.quote(str(case_dir / 'published'))} "
            f"--update-weight-local-checkpoint-dir {shlex.quote(str(case_dir / 'local'))} "
            f"--update-weight-delta-encoding {encoding} "
            "--update-weight-delta-checksum adler32 "
            "--update-weight-disk-keep-files "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--use-flash-attn "
            "--no-gradient-accumulation-fusion "
            "--train-backend megatron "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {TRAIN_GPUS} "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{perf_args} "
            f"{vllm_args} "
            f"{disk_update_args} "
            f"{ci_args} "
            f"{misc_args} "
        )
        extra_env_vars = {
            "DISK_DELTA_RESULT_DIR": str(case_dir),
            "DISK_DELTA_BASELINE_PROMPT": BASELINE_PROMPT,
            "DISK_DELTA_BASELINE_TOKENS": os.environ.get("DISK_DELTA_BASELINE_TOKENS", "64"),
            "DISK_DELTA_ENCODING": encoding,
        }
        if "ASCEND_RT_VISIBLE_DEVICES" in os.environ:
            extra_env_vars["ASCEND_RT_VISIBLE_DEVICES"] = os.environ["ASCEND_RT_VISIBLE_DEVICES"]
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
            train_script=f"tests/{Path(__file__).name} --e2e-worker",
            extra_env_vars=extra_env_vars,
        )
        metrics_path = case_dir / "round-metrics.json"
        if not metrics_path.is_file():
            raise AssertionError(f"Training job completed without writing update metrics: {case_dir}")
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        verify_checkpoints(case_dir, encoding, metrics)


def run_e2e(args):
    import ray
    import torch
    import torch.distributed as dist
    from transformers import AutoTokenizer

    from vime.backends.megatron_utils.actor import MegatronTrainRayActor
    from vime.backends.megatron_utils.hf_checkpoint_saver import save_hf_model_to_path
    from vime.observability.logging_utils import configure_logger, finish_tracking, init_tracking
    from vime.ray.placement_group import create_placement_groups, create_rollout_manager, create_training_models
    from vime.utils import accelerator

    case_dir = Path(RESULT_DIR).resolve()
    checkpoint = case_dir / "updated-checkpoint"
    if checkpoint.exists():
        raise FileExistsError(f"Use a fresh DISK_DELTA_RESULT_DIR: {case_dir}")
    round_metrics = []

    class DiskDeltaOracleActor(MegatronTrainRayActor):
        def update_weights(self):
            updater = self.weight_updater
            started = time.monotonic()
            super().update_weights()
            if updater._baseline_captured and updater.weight_version > 0:
                counts = torch.tensor(
                    [updater.changed_bytes, updater.total_bytes, updater.wire_bytes],
                    dtype=torch.int64,
                    device=accelerator.current_device(),
                )
                dist.all_reduce(counts)
            else:
                counts = None
            if dist.get_rank() == 0 and counts is not None:
                changed_bytes, total_bytes, wire_bytes = counts.tolist()
                round_metrics.append(
                    {
                        "version": updater.weight_version,
                        "changed_bytes": changed_bytes,
                        "total_bytes": total_bytes,
                        "wire_bytes": wire_bytes,
                        "update_seconds": time.monotonic() - started,
                    }
                )
                _write_json(case_dir / "round-metrics.json", round_metrics)
            if updater.weight_version == 2:
                save_hf_model_to_path(self.args, str(checkpoint), self.model)
            return updater.weight_version

    if args.use_critic or args.offload_train or args.offload_rollout or args.release_train:
        raise ValueError("The full-checkpoint oracle requires one resident actor and rollout engine")
    if (
        args.num_rollout != 2
        or args.actor_num_nodes != 1
        or args.actor_num_gpus_per_node != TRAIN_GPUS
        or args.rollout_num_gpus != ENGINE_GPUS
        or args.rollout_num_gpus_per_engine != ENGINE_GPUS
    ):
        raise ValueError("Use two rollouts with one training and one serving NPU")

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
        "max_tokens": int(os.environ.get("DISK_DELTA_BASELINE_TOKENS", "64")),
        "logprobs": 1,
    }
    _write_json(
        case_dir / "baseline.json",
        {
            "model": MODEL_NAME,
            "encoding": os.environ["DISK_DELTA_ENCODING"],
            "prompt": BASELINE_PROMPT,
            "prompt_token_ids": prompt_ids,
            "sampling_params": sampling,
        },
    )

    configure_logger()
    init_tracking(args)
    manager = actor = None
    try:
        pgs = create_placement_groups(args)
        manager, _ = create_rollout_manager(args, pgs["rollout"])
        actor, _ = create_training_models(args, pgs, manager, actor_cls=DiskDeltaOracleActor)
        actor.update_weights()  # First disk-delta call materializes the base checkpoint.
        export_versions = []
        for rollout_id in range(args.start_rollout_id, args.num_rollout):
            data = ray.get(manager.generate.remote(rollout_id))
            ray.get(actor.async_train(rollout_id, data))
            actor.clear_memory()
            export_versions = actor.update_weights()

        ray.get(manager.health_monitoring_pause.remote())
        engines, *_ = ray.get(manager.get_updatable_engines_and_lock.remote())
        if len(engines) != 1:
            raise AssertionError("The disk delta oracle requires exactly one rollout engine")
        engine = engines[0]
        url = ray.get(engine.get_url.remote())
        version = ray.get(engine.get_weight_version.remote())
        if str(version) != "2" or not export_versions or any(str(v) != "2" for v in export_versions):
            raise AssertionError(f"Rollout/export version mismatch: serving={version}, export={export_versions}")

        ray.get(engine.flush_cache.remote())
        actual = _infer(url, prompt_ids, sampling)
        _write_json(case_dir / "delta-inference.json", actual)
        ray.get(engine.flush_cache.remote())
        repeated = _infer(url, prompt_ids, sampling)
        if not _compare(actual, repeated)["passed"]:
            raise AssertionError("Disk-delta inference is not repeatable for the fixed greedy prompt")

        ray.get(manager.dispose.remote())
        manager = None
        actor.release()
        actor = None

        reference_args = copy.deepcopy(args)
        reference_args.hf_checkpoint = str(checkpoint)
        with _reference_server(reference_args) as reference_url:
            expected = _infer(reference_url, prompt_ids, sampling)
        _write_json(case_dir / "checkpoint-inference.json", expected)
        comparison = {"weight_version": version, **_compare(actual, expected)}
        _write_json(case_dir / "comparison.json", comparison)
        if not comparison["passed"]:
            raise AssertionError(f"Disk-delta/full-checkpoint inference mismatch: {comparison}")
        print(f"Disk delta {MODEL_NAME} E2E PASS: token IDs and generated-token logprobs match", flush=True)
    finally:
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
        assert not _compare({**reference, field: changed}, reference)["passed"]


def test_infer_rejects_invalid_model_lists(monkeypatch):
    from unittest.mock import MagicMock

    import pytest
    import requests

    for payload in ({}, {"data": []}, {"data": {}}, {"data": [None]}, {"data": [{}]}, []):
        session = MagicMock()
        session.get.return_value.json.return_value = payload
        factory = MagicMock()
        factory.return_value.__enter__.return_value = session
        monkeypatch.setattr(requests, "Session", factory)
        with pytest.raises(AssertionError, match="Invalid model list from http://reference/v1/models"):
            _infer("http://reference", [1], {})
        session.post.assert_not_called()


def test_reference_server_reserves_port_and_cleans_up(monkeypatch):
    from types import SimpleNamespace
    from unittest.mock import MagicMock

    import pytest
    import requests
    import vllm.utils.system_utils as system_utils

    from vime.backends.vllm_utils import vllm_engine

    monkeypatch.setattr(vllm_engine, "_compute_server_args", lambda *a, **k: ({"host": a[3], "port": a[4]}, None))
    monkeypatch.setattr(vllm_engine, "_build_subprocess_env", lambda _: {})
    for alive in (True, False):
        process = MagicMock(pid=123, exitcode=1)
        process.is_alive.return_value = alive
        factory = MagicMock(return_value=process)
        monkeypatch.setattr(
            multiprocessing, "get_context", lambda _, factory=factory: SimpleNamespace(Process=factory)
        )
        cleanup = MagicMock()
        monkeypatch.setattr(system_utils, "kill_process_tree", cleanup)
        session = MagicMock()
        session.get.return_value.status_code = 200
        session_factory = MagicMock()
        session_factory.return_value.__enter__.return_value = session
        monkeypatch.setattr(requests, "Session", session_factory)
        if alive:
            with _reference_server(SimpleNamespace()) as url:
                listener = factory.call_args.kwargs["args"][2]
                address = listener.getsockname()
                assert url == f"http://{address[0]}:{address[1]}"
                with socket.socket() as contender, pytest.raises(OSError):
                    contender.bind(address)
            with socket.socket() as contender:
                contender.bind(address)
        else:
            with pytest.raises(RuntimeError, match="Fresh-checkpoint server exited with code 1"):
                with _reference_server(SimpleNamespace()):
                    pytest.fail("An exited server must not be yielded")
        cleanup.assert_called_once_with(123)
        process.join.assert_called_once_with(timeout=30)
        assert factory.call_args.kwargs["args"][2].fileno() == -1


if __name__ == "__main__":
    if "--e2e-worker" in sys.argv:
        sys.argv.remove("--e2e-worker")
        from vime.platforms import current_platform

        if current_platform().is_npu:
            import vime.backends.megatron_utils  # noqa: F401
        from vime.utils.arguments import parse_args

        run_e2e(parse_args())
    else:
        prepare()
        execute()
