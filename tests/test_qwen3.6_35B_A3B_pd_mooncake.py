import os
from shlex import quote

import vime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3.6-35B-A3B"
MODEL_TYPE = "qwen3.5-35B-A3B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"
TORCH_DIST_CKPT = f"/root/models/{MODEL_NAME}_torch_dist"


def prepare():
    U.hf_download_public_model(f"Qwen/{MODEL_NAME}", dir_dst=MODEL_DIR)
    U.hf_download_dataset("zhuzilin/dapo-math-17k", dir_dst=DATASET_DIR)
    if U.get_bool_env_var("VIME_TEST_ENABLE_EVAL", "1"):
        U.hf_download_dataset("zhuzilin/aime-2024", dir_dst=DATASET_DIR)
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=str(MODEL_DIR.parent),
        hf_checkpoint=str(MODEL_DIR),
    )


def execute():
    with U.test_work_directory("vime_qwen36_pd_") as directory:
        _execute(directory)


def _execute(directory):
    npu = U.current_platform().name == "npu"
    if npu and U.get_bool_env_var("VIME_TEST_USE_DEEPEP", "0"):
        raise ValueError("DeepEP requires a separate NPU expert-communication adaptation")
    os.environ.setdefault("VLLM_SSM_CONV_STATE_LAYOUT", "DS")
    debug_data_path = os.environ.get("DEBUG_ROLLOUT_DATA") or str(directory / "rollout_{rollout_id}.pt")
    ref = MODEL_DIR if npu else MODEL_DIR.parent / (MODEL_NAME + "_torch_dist")
    ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(str(ref))} "
    rollout_args = (
        "--rollout-data-transport straw "
        f"--prompt-data {quote(str(DATASET_DIR / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 16384 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 16 "
    )

    eval_args = (
        f"--eval-prompt-data aime24 {quote(str(DATASET_DIR / 'aime-2024/aime-2024.jsonl'))} "
        "--n-samples-per-eval-prompt 2 "
        "--eval-max-response-len 16384 "
        "--eval-temperature 0.6 "
        "--eval-top-p 0.95 "
    )

    if not U.get_bool_env_var("VIME_TEST_ENABLE_EVAL", "1"):
        eval_args = ""

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--max-tokens-per-gpu 8192 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
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
        "--rollout-num-gpus-per-engine 4 "
        "--vllm-gpu-memory-utilization 0.75 "
        "--vllm-data-parallel-size 4 "
        "--vllm-enable-expert-parallel "
        "--vllm-max-num-seqs 512 "
        "--vllm-cudagraph-capture-sizes 5 10 20 40 80 "
        '--vllm-speculative-config \'{"method":"mtp","num_speculative_tokens":4}\' '
        "--prefill-num-servers 1 "
    )

    misc_args = (
        "--ci-test "
        f"--save-debug-rollout-data {quote(debug_data_path)} "
        "--update-weight-buffer-size 2147483648 "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type flex "
        "--moe-enable-deepep "
    )

    if npu:
        # Explicit BF16/alltoall variant; this does not validate GPU DeepEP.
        misc_args = misc_args.replace(
            "--moe-token-dispatcher-type flex --moe-enable-deepep ", "--moe-token-dispatcher-type alltoall "
        )
        vllm_args = vllm_args.replace("--vllm-cudagraph-capture-sizes 5 10 20 40 80 ", "")

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{vllm_args} "
        f"{misc_args} "
        f"{U.get_test_npu_model_args(max_model_len=32768)} "
    )

    train_args += f"--rollout-data-dir {quote(str(directory / 'queue'))} "
    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
