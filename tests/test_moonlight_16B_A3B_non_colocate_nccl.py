"""Eight-GPU non-colocated MoE E2E for native vLLM NCCL updates."""

import os

import vime.utils.external_utils.command_utils as U


os.environ.setdefault("NCCL_NVLS_ENABLE", "0")


MODEL_NAME = "Moonlight-16B-A3B-Instruct"
MODEL_TYPE = "moonlight"
NUM_GPUS = 8
TRAIN_GPUS = 4
ROLLOUT_GPUS = 4
TORCH_DIST_CKPT = f"/dev/shm/{MODEL_NAME}_torch_dist"


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(
        "hf download moonshotai/Moonlight-16B-A3B-Instruct " "--local-dir /root/models/Moonlight-16B-A3B-Instruct"
    )
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=TRAIN_GPUS,
        dir_dst="/dev/shm",
    )


def execute():
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load {TORCH_DIST_CKPT} "

    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 2 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1.0 "
        "--global-batch-size 8 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
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
        "--optimizer-cpu-offload "
        "--overlap-cpu-optimizer-d2h-h2d "
        "--use-precision-aware-optimizer "
    )

    vllm_args = (
        f"--rollout-num-gpus {ROLLOUT_GPUS} "
        f"--rollout-num-gpus-per-engine {ROLLOUT_GPUS} "
        f"--vllm-data-parallel-size {ROLLOUT_GPUS} "
        "--vllm-enable-expert-parallel "
        "--vllm-all2all-backend naive "
        "--vllm-gpu-memory-utilization 0.65 "
        "--vllm-max-model-len 4096 "
        "--vllm-max-num-seqs 8 "
        "--vllm-enforce-eager "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {TRAIN_GPUS} "
        "--ci-test "
    )

    U.execute_train(
        train_args=(
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{perf_args} "
            f"{vllm_args} "
            f"{misc_args} "
        ),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
