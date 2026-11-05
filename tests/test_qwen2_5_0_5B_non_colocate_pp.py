"""E2E smoke test for non-colocated PP distributed weight updates."""

import os
from shlex import quote

import vime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 4
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/gsm8k", dir_dst=DATASET_DIR)

    # This migration case intentionally initializes NPU training from HF weights.
    if U.current_platform().name != "npu":
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=NUM_GPUS,
            dir_dst=str(TEST_ROOT / "models"),
            hf_checkpoint=str(MODEL_DIR),
        )


def execute():
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
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0.8 "
        "--over-sampling-batch-size 8 "
        "--dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
        "--global-batch-size 16 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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
        "--rollout-num-gpus 2 "
        "--rollout-num-gpus-per-engine 1 "
        "--vllm-gpu-memory-utilization 0.7 "
        "--vllm-max-cudagraph-capture-size 16 "
    )
    if U.current_platform().name == "npu":
        # Keep graph execution; bound serving memory for the CI request load.
        vllm_args = (
            "--rollout-num-gpus 2 --rollout-num-gpus-per-engine 1 "
            "--vllm-gpu-memory-utilization 0.5 --vllm-max-num-seqs 16 "
            "--vllm-max-cudagraph-capture-size 8 "
            "--vllm-max-model-len 8192 "
        )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 2 "
    )

    ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} "
    if U.current_platform().name != "npu":
        checkpoint = str(TEST_ROOT / "models" / f"{MODEL_NAME}_torch_dist")
        ckpt_args += f"--load {quote(checkpoint)} --ref-load {quote(checkpoint)} "
    else:
        # Native HF loading is the NPU baseline; this case tests PP weight sync.
        ckpt_args += f"--ref-load {quote(str(MODEL_DIR))} "
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{U.get_test_platform_args()} "
        f"{'--log-probs-chunk-size 128 ' if U.current_platform().name == 'npu' else ''}"
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
