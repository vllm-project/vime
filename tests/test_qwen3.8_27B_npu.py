"""Two-round Qwen3.8-27B GRPO smoke test with GDN head TP and HCCL updates."""

import os
import shlex
import sys

import vime.utils.external_utils.command_utils as U


TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_NAME = "Qwen3.8-27B"
MODEL_DIR = f"{TEST_ROOT}/models/{MODEL_NAME}"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"


def prepare():
    # Install the optional GDN kernels without replacing the image's serving stack.
    U.exec_command(
        f"{shlex.quote(sys.executable)} -m pip install --disable-pip-version-check --no-deps "
        "einops 'fla-core==0.5.2' 'flash-linear-attention==0.5.2'"
    )
    U.exec_command(f"mkdir -p {shlex.quote(MODEL_DIR)} {shlex.quote(DATASET_DIR)}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {shlex.quote(MODEL_DIR)}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {shlex.quote(DATASET_DIR)}")


def execute():
    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(f"{DATASET_DIR}/dapo-math-17k.jsonl")
    # The second rollout exercises the weights updated after the first train step.
    train_args = (
        f"--hf-checkpoint {model_dir} --load {model_dir} --ref-load {model_dir} --no-load-optim "
        f"--prompt-data {prompt_data} "
        "--input-key prompt --label-key label --apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        "--rollout-shuffle --rm-type dapo --reward-key score "
        "--num-rollout 2 --rollout-batch-size 4 --n-samples-per-prompt 4 "
        "--rollout-max-prompt-len 1024 --rollout-max-response-len 128 "
        "--rollout-max-context-len 1152 --rollout-temperature 1 "
        "--global-batch-size 16 --balance-data "
        "--tensor-model-parallel-size 4 --pipeline-model-parallel-size 2 "
        "--sequence-parallel --qwen-gdn-tensor-parallel --context-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 4096 --seq-length 1152 "
        "--advantage-estimator grpo --entropy-coef 0 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
        "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        '--vllm-compilation-config \'{"cudagraph_mode":"PIECEWISE"}\' '
        "--vllm-cudagraph-capture-sizes 8 16 32 64 96 128 "
        "--rollout-num-gpus-per-engine 4 --vllm-gpu-memory-utilization 0.85 "
        "--attention-dropout 0 --hidden-dropout 0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash --use-flash-attn "
        "--no-gradient-accumulation-fusion --train-backend megatron "
        "--actor-num-nodes 1 --actor-num-gpus-per-node 8 --rollout-num-gpus 8 "
        "--no-offload-train --no-offload-rollout --ci-test "
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=16,
        megatron_model_type="qwen3.8-27B",
        extra_env_vars={
            "DISABLE_L2_CACHE": "1",
            "VLLM_USE_AOT_COMPILE": "0",
            "VLLM_USE_V2_MODEL_RUNNER": "1",
            "VLLM_ENGINE_ITERATION_TIMEOUT_S": "900",
        },
    )


def main():
    prepare()
    for name in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(name, None)
    execute()


if __name__ == "__main__":
    main()
