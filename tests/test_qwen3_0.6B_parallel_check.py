import os
import tempfile
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("VIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"

# Cover pure DP scaling, all dense parallel dimensions together, and one
# size-4 case per dimension.
PARALLEL_CONFIGS = (
    # num_gpus, tp, pp, cp
    (2, 1, 1, 1),
    (8, 2, 2, 2),
    (8, 4, 1, 1),
    (8, 1, 4, 1),
    (8, 1, 1, 4),
)


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", dir_dst=DATASET_DIR)

    # This migration case intentionally initializes NPU training from HF weights.
    if U.current_platform().name != "npu":
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=NUM_GPUS,
            dir_dst=str(TEST_ROOT / "models"),
            hf_checkpoint=str(MODEL_DIR),
        )


def execute():
    ref_load = (
        str(TEST_ROOT / "models" / f"{MODEL_NAME}_torch_dist")
        if (U.current_platform().name != "npu")
        else str(MODEL_DIR)
    )
    ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(ref_load)} "

    rollout_args = (
        f"--prompt-data {quote(str(DATASET_DIR / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
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
        "--rollout-num-gpus-per-engine 2 "
        "--rollout-num-gpus 8 "
        "--vllm-gpu-memory-utilization 0.8 "
        "--vllm-max-cudagraph-capture-size 16 "
        '--vllm-compilation-config \'{"cudagraph_mode":"FULL_DECODE_ONLY"}\' '
        "--vllm-enable-deterministic-inference "
    )
    if U.current_platform().name == "npu":
        # Keep graph execution; bound serving memory for the CI request load.
        # Batch-invariant CUDA inference is not required for this comparison:
        # all twelve launches consume the same saved rollout after the first.
        vllm_args = (
            "--rollout-num-gpus 8 --rollout-num-gpus-per-engine 2 "
            "--vllm-gpu-memory-utilization 0.5 --vllm-max-num-seqs 16 "
            "--vllm-max-cudagraph-capture-size 8 "
            "--vllm-max-model-len 32768 "
        )

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
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{U.get_test_platform_args()} "
        f"{'--log-probs-chunk-size 128 ' if U.current_platform().name == 'npu' else ''}"
    )

    with tempfile.TemporaryDirectory(prefix="vime_parallel_check_", dir=U.get_test_tmp_dir()) as work_dir:
        _compare_parallel_configs(train_args, Path(work_dir))


def _compare_parallel_configs(train_args, work_dir):
    rollout_data_path = quote(str(work_dir / "rollout-data_{rollout_id}.pt"))
    configs = PARALLEL_CONFIGS
    if U.current_platform().name == "npu":
        # Isolate each dimension before the TP2/PP2/CP2 combination.
        configs = tuple(config for config in configs if config != (8, 2, 2, 2)) + ((8, 2, 2, 2),)
    for i, calculate_per_token_loss in enumerate((False, True)):
        grad_norm_path = quote(str(work_dir / f"grad_norms-{i}.pt"))
        print(f"Parallel check baseline: per_token_loss={calculate_per_token_loss}", flush=True)
        loss_args = "--calculate-per-token-loss " if calculate_per_token_loss else ""
        rollout_data_args = (
            f"--save-debug-rollout-data {rollout_data_path} "
            if i == 0
            else f"--load-debug-rollout-data {rollout_data_path} "
        )
        U.execute_train(
            train_args=(
                train_args
                + loss_args
                + rollout_data_args
                + f"--ci-save-grad-norm {grad_norm_path} "
                + f"--actor-num-gpus-per-node {NUM_GPUS} "
            ),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )
        for num_gpus, tp_size, pp_size, cp_size in configs:
            print(
                f"Parallel check: devices={num_gpus}, TP={tp_size}, PP={pp_size}, CP={cp_size}, "
                f"per_token_loss={calculate_per_token_loss}",
                flush=True,
            )
            args = (
                train_args
                + loss_args
                + f"--load-debug-rollout-data {rollout_data_path} "
                + f"--ci-load-grad-norm {grad_norm_path} "
                + f"--context-parallel-size {cp_size} "
                + f"--tensor-model-parallel-size {tp_size} "
                + f"--pipeline-model-parallel-size {pp_size} "
                + "--sequence-parallel "
                + f"--actor-num-gpus-per-node {num_gpus} "
                + "--use-dynamic-batch-size "
                + "--max-tokens-per-gpu 8192 "
            )

            U.execute_train(
                train_args=args,
                num_gpus_per_node=num_gpus,
                megatron_model_type=MODEL_TYPE,
            )


if __name__ == "__main__":
    # TODO also use typer
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
