import json
import os
import shlex

import vime.utils.external_utils.command_utils as U


TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/Kimi-K2-Thinking"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"


def prepare():
    models_dir = shlex.quote(f"{TEST_ROOT}/models")
    datasets_dir = shlex.quote(f"{TEST_ROOT}/datasets")
    model_dir = shlex.quote(MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)

    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download moonshotai/Kimi-K2-Thinking --local-dir {model_dir}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {dataset_dir}")


def generate_dummy_config(model_dir):
    model_config = os.path.join(model_dir, "config.json")
    with open(model_config, "r", encoding="utf-8") as f:
        data = json.load(f)

    data["num_hidden_layers"] = 2
    data["quantization_config"]["ignore"].extend([
        "model.embed_tokens",
        "re:.*mlp\\.gate\\.weights$"
    ])

    with open(model_config, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def execute():
    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(f"{DATASET_DIR}/dapo-math-17k.jsonl")

    generate_dummy_config(model_dir)

    # NPU skips torch_dist conversion; HF weights load directly via bridge mode.
    checkpoint_args = (
        f"--hf-checkpoint {model_dir} "
        f"--load {model_dir} "
        f"--ref-load {model_dir} "
        "--megatron-to-hf-mode bridge "
        "--no-load-optim "
    )

    # Smoke-scaled rollout (num-rollout/batch/n-samples trimmed like test_kimi_k2_thinking_npu).
    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    # TP=8/EP=8 mirrors scripts/run-kimi-k2-thinking-npu.sh.
    parallel_args = (
        "--tensor-model-parallel-size 8 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--micro-batch-size 1 "
        "--max-tokens-per-gpu 2048 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
        "--use-tis"
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
        "--rollout-backend vllm "
        "--rollout-num-gpus-per-engine 16 "
        "--vllm-gpu-memory-utilization 0.45 "
        "--vllm-enable-expert-parallel "
        "--vllm-enable-sleep-mode "
        "--vllm-weight-sync-mode native "
        "--vllm-enforce-eager "
        "--vllm-load-format dummy "
    )

    model_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--use-flash-attn "
    )

    runtime_args = (
        "--train-backend megatron "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
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
    # Model architecture (--spec, --attention-output-gate, --moe-shared-expert-gate,
    # num-experts, moe-* ...) is injected by sourcing scripts/models/kimi-k2-thinking.sh
    # via ${MODEL_ARGS[@]}, so only runtime/training args are passed here.
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=16,
        megatron_model_type="kimi-k2-thinking-2layer",
        extra_env_vars={
            "DISABLE_L2_CACHE": "1",
            "VLLM_USE_AOT_COMPILE": "0",
            "ASCEND_CUSTOM_OPP_PATH": "/vllm-workspace/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer:/usr/local/Ascend/cann-9.0.0/opp/vendors/fla_npu_transformer",
        },
    )


def main():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()


if __name__ == "__main__":
    main()
