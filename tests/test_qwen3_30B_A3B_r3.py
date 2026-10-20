import os
from shlex import quote

import vime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("VIME_TEST_ENABLE_EVAL", "1")))
USE_DEEPEP = bool(int(os.environ.get("VIME_TEST_USE_DEEPEP", "1")))
USE_FP8_ROLLOUT = bool(int(os.environ.get("VIME_TEST_USE_FP8_ROLLOUT", "1")))

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"


def validate_backend_options():
    if U.current_platform().name == "npu" and (USE_DEEPEP or USE_FP8_ROLLOUT):
        raise ValueError(
            "This NPU R3 case covers BF16/alltoall; set VIME_TEST_USE_DEEPEP=0 and VIME_TEST_USE_FP8_ROLLOUT=0. DeepEP/FP8 require separate adaptation."
        )


def prepare():
    validate_backend_options()
    U.exec_command(f"mkdir -p {quote(str(MODEL_DIR.parent))} {quote(str(DATASET_DIR))}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    if USE_FP8_ROLLOUT:
        U.exec_command(f"hf download Qwen/{MODEL_NAME}-FP8 --local-dir {quote(str(MODEL_DIR) + '-FP8')}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", dir_dst=DATASET_DIR)
    if ENABLE_EVAL:
        U.hf_download_dataset("zhuzilin/aime-2024", dir_dst=DATASET_DIR)

    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=str(TEST_ROOT),
        hf_checkpoint=str(MODEL_DIR),
    )


def execute():
    validate_backend_options()
    ref_load = TEST_ROOT / f"{MODEL_NAME}_torch_dist" if U.current_platform().torch_dist_convert else MODEL_DIR
    hf_checkpoint = str(MODEL_DIR) + ("-FP8" if USE_FP8_ROLLOUT else "")
    ckpt_args = f"--hf-checkpoint {quote(hf_checkpoint)} --ref-load {quote(str(ref_load))} "

    rollout_args = (
        "--rollout-data-transport straw "
        "--rollout-queue-online-gc "
        f"--prompt-data {quote(str(DATASET_DIR / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    eval_args = (
        (
            "--eval-interval 20 "
            f"--eval-prompt-data aime24 {quote(str(DATASET_DIR / 'aime-2024/aime-2024.jsonl'))} "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 16384 "
            "--eval-top-k 1 "
        )
        if ENABLE_EVAL
        else ""
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-tis "
        "--use-rollout-routing-replay "
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
        "--rollout-num-gpus-per-engine 8 "
        "--vllm-gpu-memory-utilization 0.8 "
        "--vllm-max-num-seqs 512 "
        "--vllm-max-cudagraph-capture-size 32 "
    )

    if USE_DEEPEP:
        vllm_args += "--vllm-all2all-backend deepep_high_throughput --vllm-enable-expert-parallel "

    ci_args = "--ci-test "

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    if USE_DEEPEP:
        misc_args += "--moe-token-dispatcher-type flex --moe-enable-deepep "
    else:
        misc_args += "--moe-token-dispatcher-type alltoall "

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{grpo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{eval_args} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{U.get_test_npu_model_args()} "
    )

    with U.test_work_directory("vime_qwen3_r3_") as rollout_dir:
        train_args += f"--rollout-data-dir {quote(str(rollout_dir))} "
        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )


if __name__ == "__main__":
    # TODO also use typer
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
