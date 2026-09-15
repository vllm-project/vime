import os
import shlex

import vime.utils.external_utils.command_utils as U


TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/GLM-4.7-Flash"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"


def prepare():
    models_dir = shlex.quote(f"{TEST_ROOT}/models")
    datasets_dir = shlex.quote(f"{TEST_ROOT}/datasets")
    model_dir = shlex.quote(MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)

    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download zai-org/GLM-4.7-Flash --local-dir {model_dir}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {dataset_dir}")


def execute():
    # Default to G1; G2 and eager diagnostics are explicit, independent opt-ins.
    enable_mtp = os.environ.get("VIME_TEST_GLM_MTP", "0") == "1"
    enforce_eager = os.environ.get("VIME_TEST_GLM_EAGER", "0") == "1"
    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(f"{DATASET_DIR}/dapo-math-17k.jsonl")

    # G1 loads HF weights through the native loader, without torch_dist conversion.
    checkpoint_args = f"--hf-checkpoint {model_dir} --load {model_dir} --ref-load {model_dir} --no-load-optim "

    # Smoke-scaled rollout (num-rollout/batch/n-samples trimmed like test_qwen3_30B_A3B_npu).
    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 128 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    # TP=4/EP=8 mirrors scripts/run-glm4.7-30B-A3B-npu.sh.
    parallel_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 20480 "
        "--micro-batch-size 1 "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    vllm_args = (
        "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        "--rollout-num-gpus-per-engine 4 "
        "--vllm-gpu-memory-utilization 0.7 "
        "--vllm-enable-expert-parallel "
        "--vllm-cudagraph-capture-sizes 1 2 4 8 "
    )
    mtp_args = ""
    if enable_mtp:
        mtp_args = "--mtp-num-layers 1 --enable-mtp-training --mtp-loss-scaling-factor 0.2 "
        vllm_args += '--vllm-speculative-config \'{"method":"mtp","num_speculative_tokens":1}\' '
    if enforce_eager:
        vllm_args += "--vllm-enforce-eager "

    model_args = (
        # GLM-4.7-Flash has no HF rope_scaling; MLA otherwise defaults to YaRN.
        "--rope-type rope "
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
        "--ci-test "
    )

    train_args = (
        checkpoint_args
        + rollout_args
        + parallel_args
        + grpo_args
        + optimizer_args
        + mtp_args
        + vllm_args
        + model_args
        + runtime_args
    )
    # Model architecture (num-experts, moe-*, multi-latent-attention, q-lora-rank,
    # kv-lora-rank, ...) is injected by sourcing scripts/models/glm4.7-30B-A3B.sh
    # via ${MODEL_ARGS[@]}, so only runtime/training args are passed here.
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=16,
        megatron_model_type="glm4.7-30B-A3B",
        extra_env_vars={
            "DISABLE_L2_CACHE": "1",
            "VLLM_USE_AOT_COMPILE": "0",
        },
    )


def main():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    main()
