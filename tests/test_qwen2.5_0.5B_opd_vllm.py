import os
import subprocess
import time
import urllib.request
from shlex import quote

from vllm.utils.system_utils import kill_process_tree

import vime.utils.external_utils.command_utils as U
from vime.platforms import current_platform
from vime.utils.http_utils import is_port_available

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
NUM_TRAIN_GPUS = 4
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"

TEACHER_HOST = "127.0.0.1"
TEACHER_PORT = 13141


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/gsm8k", dir_dst=DATASET_DIR)


def _get_gpu_split():
    """Keep the teacher outside the four-device training/Ray allocation."""
    visible_env = current_platform().ray.visible_devices_env
    visible = os.environ.get(visible_env, ",".join(str(i) for i in range(NUM_GPUS)))
    devices = [value.strip() for value in visible.split(",") if value.strip()]
    assert len(devices) >= NUM_GPUS and len(set(devices)) == len(
        devices
    ), f"Expected at least {NUM_GPUS} distinct devices; {visible_env}={visible!r}"
    return devices[:NUM_TRAIN_GPUS], devices[NUM_TRAIN_GPUS]


def _stop_teacher(process):
    if process is not None:
        kill_process_tree(process.pid)
        process.wait(timeout=30)


def _launch_teacher_server(teacher_gpu: str, log_path):
    if not is_port_available(TEACHER_PORT):
        raise RuntimeError(f"Teacher port {TEACHER_PORT} is already in use")
    platform = current_platform()
    env = os.environ.copy()
    env[platform.ray.visible_devices_env] = teacher_gpu
    env = platform.vllm.subprocess_env(env, visible_devices=teacher_gpu, colocate=False)
    env.update(VLLM_SERVER_DEV_MODE="1", VLLM_USE_V2_MODEL_RUNNER="1")
    command = [
        "python3",
        "-m",
        "vllm.entrypoints.openai.api_server",
        "--model",
        str(MODEL_DIR),
        "--host",
        TEACHER_HOST,
        "--port",
        str(TEACHER_PORT),
        "--tensor-parallel-size",
        "1",
        "--gpu-memory-utilization",
        "0.6",
        "--trust-remote-code",
        "--enable-scale-out",
    ]
    if platform.is_npu:
        command += ["--max-model-len", "8192", "--max-num-seqs", "16", "--additional-config", '{"weight_nz_mode": 0}']
    with log_path.open("w") as log_file:
        process = subprocess.Popen(command, env=env, stdout=log_file, stderr=subprocess.STDOUT)
    print(f"Starting teacher on {platform.name}:{teacher_gpu} (pid={process.pid}), log: {log_path}")
    try:
        for _ in range(120):
            if process.poll() is not None:
                raise RuntimeError(f"Teacher server exited with code {process.returncode}")
            try:
                with urllib.request.urlopen(f"http://{TEACHER_HOST}:{TEACHER_PORT}/health", timeout=2) as response:
                    if response.status == 200:
                        return process
            except (OSError, TimeoutError):
                pass
            time.sleep(5)
        raise TimeoutError("Teacher server did not become healthy within 10 minutes")
    except BaseException:
        print(f"Teacher startup failed:\n{log_path.read_text(errors='replace')[-16000:]}", flush=True)
        _stop_teacher(process)
        raise


def execute():
    with U.test_work_directory("vime_opd_") as directory:
        _execute(directory)


def _execute(directory):
    train_gpus, teacher_gpu = _get_gpu_split()
    teacher_process = None

    visible_env = current_platform().ray.visible_devices_env
    previous_visible = os.environ.get(visible_env)
    # Restrict the platform's visibility before Ray starts; the teacher uses
    # an independent physical device and never enters the training pool.
    os.environ[visible_env] = ",".join(train_gpus)
    log_path = directory / "teacher.log"

    def launch_teacher():
        nonlocal teacher_process
        teacher_process = _launch_teacher_server(teacher_gpu, log_path)

    try:
        ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(str(MODEL_DIR))} "

        rollout_args = (
            f"--prompt-data {quote(str(DATASET_DIR / 'gsm8k/train.parquet'))} "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            "--num-rollout 2 "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
        )

        eval_args = (
            f"--eval-prompt-data gsm8k {quote(str(DATASET_DIR / 'gsm8k/test.parquet'))} "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 1024 "
            "--eval-top-k 1 "
        )

        if not U.get_bool_env_var("VIME_TEST_ENABLE_EVAL", "1"):
            eval_args = ""

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        )

        rm_args = (
            "--custom-rm-path vime.rollout.on_policy_distillation.reward_func "
            "--custom-reward-post-process-path vime.rollout.on_policy_distillation.post_process_rewards "
            f"--rm-url http://{TEACHER_HOST}:{TEACHER_PORT}/inference/v1/generate "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            # OPD with vLLM teacher (self-distillation for CI test)
            "--use-opd "
            "--opd-type vllm "
            "--opd-kl-coef 1.0 "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.00 "
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
        )

        vllm_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--vllm-gpu-memory-utilization 0.7 "
            "--vllm-max-num-seqs 32 "
            "--vllm-max-cudagraph-capture-size 16 "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--actor-num-nodes 1 "
            f"--actor-num-gpus-per-node {NUM_TRAIN_GPUS} "
            "--colocate "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{eval_args} "
            f"{vllm_args} "
            f"{ci_args} "
            f"{misc_args} "
            f"{rm_args} "
            f"{U.get_test_npu_model_args(max_model_len=8192)} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_TRAIN_GPUS,
            megatron_model_type=MODEL_TYPE,
            before_ray_job_submit=launch_teacher,
        )
    except BaseException:
        if log_path.exists():
            print(f"OPD failed; teacher log tail:\n{log_path.read_text(errors='replace')[-16000:]}", flush=True)
        raise
    finally:
        try:
            _stop_teacher(teacher_process)
        finally:
            if previous_visible is None:
                os.environ.pop(visible_env, None)
            else:
                os.environ[visible_env] = previous_visible


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
