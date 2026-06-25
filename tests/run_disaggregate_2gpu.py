"""Minimal 2-GPU train/inference disaggregation smoke test with Mooncake rollout transfer.

GPU layout (non-colocate):
  - GPU 0: Megatron training (actor)
  - GPU 1: vLLM rollout (inference)

Rollout tensors are transferred via Mooncake instead of Ray object store.
"""

import os

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
MODELSCOPE_MODEL_ID = "Qwen/Qwen2.5-0.5B-Instruct"
MODELSCOPE_DATASET_ID = "AI-ModelScope/gsm8k"
DATASET_TRAIN_PATH = "/root/datasets/gsm8k/main/train-00000-of-00001.parquet"
NUM_GPUS = 2


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    model_dir = f"/root/models/{MODEL_NAME}"
    if not os.path.exists(f"{model_dir}/config.json"):
        U.ms_download_model(MODELSCOPE_MODEL_ID, model_dir)
    dataset_dir = "/root/datasets/gsm8k"
    if not os.path.exists(DATASET_TRAIN_PATH):
        U.ms_download_dataset(MODELSCOPE_DATASET_ID, dataset_dir)


def execute():
    from vime.utils.mooncake_store_service import ensure_mooncake_master

    ensure_mooncake_master()

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        f"--prompt-data {DATASET_TRAIN_PATH} "
        "--input-key question "
        "--label-key answer "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 4 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
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
        "--rollout-num-gpus-per-engine 1 "
        "--vllm-gpu-memory-utilization 0.25 "
        "--vllm-max-cudagraph-capture-size 8 "
        "--vllm-max-num-seqs 16 "
    )

    mooncake_args = "--transfer-backend mooncake_store "

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--no-gradient-accumulation-fusion "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 1 "
        "--rollout-num-gpus 1 "
        "--megatron-to-hf-mode bridge "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{mooncake_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        train_script="train_async.py",
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    # Use free GPUs when 0/1 are occupied (e.g. sglang). Override: CUDA_VISIBLE_DEVICES=4,5
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "4,5")
    execute()
