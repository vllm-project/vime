"""Qwen3-4B GRPO convergence run on dapo-math-17k -> AIME-2024 eval.

Mirrors the proven miles/slime sibling recipe (scripts/run-qwen3-4B.sh):
GRPO + clip-higher 0.28 + deepscaler reward, train response 8192, AIME eval.
Topology: 2-host 16-GPU COLOCATE, pure data-parallel — Megatron TP=1 PP=1 CP=1
=> DP=16, and vLLM rollout TP=1 => 16 single-GPU engines (DP=16). Each H200
holds a full Qwen3-4B replica; optimizer offloaded to CPU during rollout.

Launch cross-host with SLIME_SCRIPT_EXTERNAL_RAY=1 + MASTER_ADDR + WANDB_API_KEY.
"""

import os

import slime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8  # per node; actor-num-nodes=2 => 16 GPU total


def prepare():
    # Model, datasets and torch_dist ckpt are pre-staged in the mounted
    # models-shared (as HF-cache symlinks), so guard every download to skip
    # when present — a bare `hf download --local-dir` trips on the symlinks.
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(
        f"test -f /root/models/{MODEL_NAME}/config.json || hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}"
    )
    U.exec_command(
        "test -d /root/datasets/dapo-math-17k || hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /root/datasets/dapo-math-17k"
    )
    U.exec_command(
        "test -d /root/datasets/aime-2024 || hf download --repo-type dataset zhuzilin/aime-2024 --local-dir /root/datasets/aime-2024"
    )
    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load /root/models/{MODEL_NAME}_torch_dist "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template "
        "--rollout-shuffle --rm-type deepscaler "
        "--num-rollout 1000 "
        "--rollout-batch-size 32 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 256 "
        "--balance-data "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data aime24 /root/datasets/aime-2024/aime-2024.jsonl "
        "--n-samples-per-eval-prompt 16 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    # DP=16: pure data parallel, no tensor/pipeline/context split (4B fits per GPU).
    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 12288 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl --kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 --eps-clip-high 0.28 "
    )

    optimizer_args = (
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d --use-precision-aware-optimizer "
    )

    # vLLM rollout: TP=1 => 16 engines across 2 hosts (DP=16).
    vllm_args = (
        "--rollout-num-gpus 16 "
        "--rollout-num-gpus-per-engine 1 "
        "--num-gpus-per-node 8 "
        "--vllm-gpu-memory-utilization 0.55 "
    )

    misc_args = (
        "--attention-dropout 0.0 --hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 --attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 2 --actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} {rollout_args} {eval_args} {optimizer_args} {grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} {vllm_args} {misc_args} "
    )

    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    for v in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(v, None)
    execute()
