import os
import tempfile
from shlex import quote

import vime.utils.external_utils.command_utils as U

ENABLE_EVAL = bool(int(os.environ.get("VIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
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
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", dir=U.get_test_tmp_dir()) as megatron_config:
        megatron_config.write(
            """
    megatron:
      - name: default
        role: critic
        overrides:
          lr: 1e-5
      - name: default
        role: actor
        overrides:
          lr: 1e-6
    """
        )
        megatron_config.flush()

        ref_load = (
            str(TEST_ROOT / f"{MODEL_NAME}_torch_dist") if U.current_platform().torch_dist_convert else str(MODEL_DIR)
        )
        ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(ref_load)} "

        rollout_args = (
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
            "--rollout-temperature 0.8 "
            "--global-batch-size 16 "
            "--balance-data "
        )

        eval_args = (
            f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
            f"--eval-prompt-data aime24 {quote(str(DATASET_DIR / 'aime-2024/aime-2024.jsonl'))} "
            "--n-samples-per-eval-prompt 1 "
            "--eval-max-response-len 16384 "
            "--eval-top-k 1 "
        )

        perf_args = (
            "--tensor-model-parallel-size 2 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 2 "
            "--recompute-granularity full "
            "--recompute-method uniform "
            "--recompute-num-layers 1 "
            "--use-dynamic-batch-size "
            f"--max-tokens-per-gpu {8192 if U.current_platform().name == 'npu' else 16384} "
        )

        ppo_args = (
            "--advantage-estimator ppo "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type k1 "
            "--kl-coef 0.00 "
            "--entropy-coef 0.00 "
            "--eps-clip 4e-4 "
            "--num-critic-only-steps 2 "
            "--normalize-advantages "
        )

        optimizer_args = (
            "--optimizer adam "
            "--lr 1e-6 "
            "--lr-decay-style constant "
            "--weight-decay 0.1 "
            "--adam-beta1 0.9 "
            "--adam-beta2 0.98 "
        )

        if U.current_platform().name == "npu":
            # Actor and critic coexist at TP2/CP2; use the established NPU Adam
            # offload policy without changing either role's learning rate.
            optimizer_args += (
                "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d " "--use-precision-aware-optimizer "
            )

        vllm_args = (
            "--rollout-num-gpus-per-engine 2 "
            "--rollout-num-gpus 8 "
            "--vllm-gpu-memory-utilization 0.8 "
            "--vllm-max-num-seqs 512 "
            "--vllm-max-cudagraph-capture-size 16 "
        )
        if U.current_platform().name == "npu":
            # Keep graph execution; bound serving memory for the CI request load.
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
            "--actor-num-gpus-per-node 8 "
            "--colocate "
        )

        train_args = (
            f"--megatron-config-path {quote(megatron_config.name)} "
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{ppo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{eval_args} "
            f"{vllm_args} "
            f"{ci_args} "
            f"{misc_args} "
            f"{U.get_test_platform_args()} "
            f"{'--log-probs-chunk-size 128 ' if U.current_platform().name == 'npu' else ''}"
        )

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
