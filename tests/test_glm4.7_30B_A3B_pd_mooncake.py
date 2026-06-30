"""GLM-4.7-Flash colocated training test with single-node PD + Mooncake."""

import os
import tempfile

import yaml

import vime.utils.external_utils.command_utils as U


MODEL_REPO = "zai-org/GLM-4.7-Flash"
MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-30B-A3B"
NUM_GPUS = 8


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download {MODEL_REPO} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/root/models",
        hf_checkpoint=f"/root/models/{MODEL_NAME}",
    )


def write_vllm_config() -> str:
    config = {
        "vllm": [
            {
                "name": "default",
                "server_groups": [
                    {
                        "worker_type": "prefill",
                        "num_gpus": 4,
                        "num_gpus_per_engine": 4,
                    },
                    {
                        "worker_type": "decode",
                        "num_gpus": 4,
                        "num_gpus_per_engine": 4,
                    },
                ],
            }
        ]
    }
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", prefix="vllm_pd_mooncake_", delete=False)
    with f:
        yaml.safe_dump(config, f, sort_keys=False)
    return f.name


def execute():
    vllm_config = write_vllm_config()

    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME} " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 2 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 1.0 "
        "--rollout-top-p 0.95 "
        "--global-batch-size 8 "
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
    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
        "--eps-clip-high 0.28 "
    )
    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 4 "
        "--expert-tensor-parallel-size 1 "
        "--decoder-last-pipeline-num-layers 23 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )
    vllm_args = (
        "--rollout-num-gpus 8 "
        "--rollout-num-gpus-per-engine 4 "
        "--vllm-data-parallel-size 4 "
        "--vllm-enable-expert-parallel "
        "--vllm-gpu-memory-utilization 0.45 "
        "--vllm-max-num-seqs 16 "
        "--vllm-max-cudagraph-capture-size 8 "
        '--vllm-speculative-config \'{"method":"mtp","num_speculative_tokens":3}\' '
        "--router-request-timeout-secs 1200 "
        f"--vllm-config {vllm_config} "
    )
    misc_args = (
        "--ci-test "
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--moe-token-dispatcher-type alltoall "
    )
    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{misc_args} "
    )
    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
