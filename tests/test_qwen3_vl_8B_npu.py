import os
import shlex

import vime.utils.external_utils.command_utils as U


# Single-turn Qwen3-VL GRPO on geo3k (mirrors examples/geo3k_vlm/run_geo3k_vlm_npu.sh).
# Qwen3-VL-8B maps to the qwen3-8B megatron config; the vision tower is handled by bridge.
MODEL_NAME = "Qwen3-VL-8B-Instruct"
MODEL_TYPE = "qwen3-8B"
TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/{MODEL_NAME}"
DATASET_DIR = f"{TEST_ROOT}/datasets/geo3k_imgurl"


def prepare():
    models_dir = shlex.quote(f"{TEST_ROOT}/models")
    datasets_dir = shlex.quote(f"{TEST_ROOT}/datasets")
    model_dir = shlex.quote(MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)

    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {model_dir}")
    U.exec_command(f"hf download --repo-type dataset chenhegu/geo3k_imgurl --local-dir {dataset_dir}")


def execute():
    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(f"{DATASET_DIR}/train.parquet")

    checkpoint_args = (
        f"--hf-checkpoint {model_dir} " f"--load {model_dir} " "--megatron-to-hf-mode bridge " "--no-load-optim "
    )

    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key problem "
        "--label-key answer "
        '--multimodal-keys \'{"image": "images"}\' '
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 4096 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
    )

    parallel_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
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
        "--vllm-gpu-memory-utilization 0.8 "
        "--vllm-max-model-len 16384 "
        "--vllm-generation-config auto "
        "--vllm-logprobs-mode processed_logprobs "
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
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
        "--colocate "
        "--ci-test "
    )

    train_args = (
        checkpoint_args
        + rollout_args
        + parallel_args
        + grpo_args
        + optimizer_args
        + vllm_args
        + model_args
        + runtime_args
    )
    # qwen3-8B.sh builds MODEL_ARGS with --rotary-base ${MODEL_ARGS_ROTARY_BASE}; Qwen3-VL needs 5e6.
    os.environ["MODEL_ARGS_ROTARY_BASE"] = "5000000"
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={},
    )


def main():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    main()
