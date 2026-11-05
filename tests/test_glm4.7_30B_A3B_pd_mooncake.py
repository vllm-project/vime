"""GLM-4.7-Flash colocated training test with single-node PD + Mooncake."""

import os
from pathlib import Path
from shlex import quote

import yaml

import vime.utils.external_utils.command_utils as U

MODEL_REPO = "zai-org/GLM-4.7-Flash"
MODEL_NAME = "GLM-4.7-Flash"
MODEL_TYPE = "glm4.7-30B-A3B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"


def prepare():
    U.exec_command(f"hf download {MODEL_REPO} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", dir_dst=DATASET_DIR)
    # This migration case intentionally initializes NPU training from HF weights.
    if U.current_platform().name != "npu":
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=NUM_GPUS,
            dir_dst=str(MODEL_DIR.parent),
            hf_checkpoint=str(MODEL_DIR),
        )


def write_vllm_config(directory) -> str:
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
    path = Path(directory) / "serving.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False))
    return str(path)


def execute():
    with U.test_work_directory("vime_glm_pd_") as directory:
        _execute(directory)


def _execute(directory):
    vllm_config = write_vllm_config(directory)
    ref = MODEL_DIR if U.current_platform().name == "npu" else MODEL_DIR.parent / (MODEL_NAME + "_torch_dist")
    ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(str(ref))} "
    rollout_args = (
        f"--prompt-data {quote(str(DATASET_DIR / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
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
        "--rollout-top-p 1.0 "
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
        "--vllm-max-cudagraph-capture-size 40 "
        '--vllm-speculative-config \'{"method":"mtp","num_speculative_tokens":4}\' '
        "--router-request-timeout-secs 1200 "
        f"--vllm-config {quote(vllm_config)} "
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
        f"{U.get_test_npu_model_args(max_model_len=8192)} "
    )
    U.execute_train(train_args=train_args, num_gpus_per_node=NUM_GPUS, megatron_model_type=MODEL_TYPE)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
