"""
Task 2 - DP + EP with DeepEP comm kernel (Qwen3-30B-A3B) P1.

Validates that Qwen3-30B-A3B trains stably with the DeepEP all-to-all
backend in vLLM across two parallel layouts and two dispatcher phases.

Layouts (selected via SLIME_TEST_DEEPEP_LAYOUT):
  production  TP=4 CP=2 DP=1 EP=8   production layout from test_qwen3_30B_A3B.py
  dp          TP=4 CP=1 DP=2 EP=8   adds data parallelism (DP=2) on top of EP=8

Phases (selected via SLIME_TEST_DEEPEP_PHASE):
  alltoall  --moe-token-dispatcher-type alltoall
  deepep    --moe-token-dispatcher-type flex --moe-enable-deepep
            --vllm-all2all-backend deepep_high_throughput

Full matrix (default LAYOUT=both PHASE=both) is 4 runs:
  L1 x alltoall, L1 x deepep, L2 x alltoall, L2 x deepep

Acceptance (per issue #11 Task 2), applied PER LAYOUT (within-layout A/B):
  - Both phases complete >= 3 rollouts without NCCL / HTTP-400 / OOM errors.
  - DeepEP reward within 10% of alltoall reward (same layout, same rollout).
  - DeepEP step time <= 115% of alltoall step time (same layout).
  Cross-layout comparisons (e.g. L2 step time vs L1 step time) are NOT in scope.

Usage:
  python tests/test_qwen3_30B_A3B_deepep.py
  SLIME_TEST_DEEPEP_LAYOUT=production python tests/test_qwen3_30B_A3B_deepep.py
  SLIME_TEST_DEEPEP_LAYOUT=dp python tests/test_qwen3_30B_A3B_deepep.py
  SLIME_TEST_DEEPEP_PHASE=deepep python tests/test_qwen3_30B_A3B_deepep.py
"""

import os
from pathlib import Path

import vime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("SLIME_TEST_ENABLE_EVAL", "0")))
TIGHT_HOST_MEMORY = bool(int(os.environ.get("SLIME_TEST_TIGHT_HOST_MEMORY", "1")))
USE_FP8_ROLLOUT = bool(int(os.environ.get("SLIME_TEST_USE_FP8_ROLLOUT", "0")))
DEEPEP_PHASE = os.environ.get("SLIME_TEST_DEEPEP_PHASE", "both")
DEEPEP_LAYOUT = os.environ.get("SLIME_TEST_DEEPEP_LAYOUT", "both")
DATA_ROOT = Path(os.environ.get("SLIME_TEST_DATA_ROOT", "/root/vime-data"))

MODEL_NAME = "Qwen3-30B-A3B"
MODEL_TYPE = "qwen3-30B-A3B"
NUM_GPUS = 8

# world_size = TP * CP * DP * PP = 8 (PP=1). EP is independent of world_size
# and groups GPUs for expert routing only.
_LAYOUTS = {
    "production": {"TP": 4, "CP": 2, "DP": 1, "EP": 8},
    "dp": {"TP": 4, "CP": 1, "DP": 2, "EP": 8},
}


def _select_layouts():
    if DEEPEP_LAYOUT not in ("both", *_LAYOUTS):
        raise ValueError(f"Unsupported SLIME_TEST_DEEPEP_LAYOUT={DEEPEP_LAYOUT!r}")
    return [(name, layout) for name, layout in _LAYOUTS.items() if DEEPEP_LAYOUT in ("both", name)]


def _select_phases():
    if DEEPEP_PHASE not in ("both", "alltoall", "deepep"):
        raise ValueError(f"Unsupported SLIME_TEST_DEEPEP_PHASE={DEEPEP_PHASE!r}")
    phases = []
    if DEEPEP_PHASE in ("both", "alltoall"):
        phases.append(False)
    if DEEPEP_PHASE in ("both", "deepep"):
        phases.append(True)
    return phases


def prepare():
    models_dir = DATA_ROOT / "models"
    datasets_dir = DATA_ROOT / "datasets"
    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {models_dir / MODEL_NAME}")
    if USE_FP8_ROLLOUT:
        U.exec_command(f"hf download Qwen/{MODEL_NAME}-FP8 --local-dir {models_dir / f'{MODEL_NAME}-FP8'}")

    U.exec_command(
        "hf download --repo-type dataset zhuzilin/dapo-math-17k "
        f"--local-dir {datasets_dir / 'dapo-math-17k'}"
    )
    U.exec_command(
        "hf download --repo-type dataset zhuzilin/aime-2024 " f"--local-dir {datasets_dir / 'aime-2024'}"
    )
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst=str(DATA_ROOT),
        hf_checkpoint=str(models_dir / MODEL_NAME),
    )


def execute(use_deepep: bool, layout_name: str, layout: dict):
    phase_tag = "deepep" if use_deepep else "alltoall"
    tp, cp, dp, ep = layout["TP"], layout["CP"], layout["DP"], layout["EP"]
    run_tag = f"{layout_name}/{phase_tag}"
    print(f"\n{'=' * 60}")
    print(f"Run: {run_tag}  (TP={tp} CP={cp} DP={dp} EP={ep}, use_deepep={use_deepep})")
    print(f"{'=' * 60}")

    models_dir = DATA_ROOT / "models"
    datasets_dir = DATA_ROOT / "datasets"
    hf_ckpt = models_dir / (f"{MODEL_NAME}-FP8" if USE_FP8_ROLLOUT else MODEL_NAME)
    ckpt_args = f"--hf-checkpoint {hf_ckpt} --ref-load {DATA_ROOT / f'{MODEL_NAME}_torch_dist'} "

    rollout_args = (
        f"--prompt-data {datasets_dir / 'dapo-math-17k' / 'dapo-math-17k.jsonl'} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 3 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 8192 "
        "--rollout-temperature 1 "
        "--global-batch-size 32 "
        "--balance-data "
    )

    eval_args = (
        f"{'--eval-interval 20 ' if ENABLE_EVAL else ''}"
        f"--eval-prompt-data aime24 {datasets_dir / 'aime-2024' / 'aime-2024.jsonl'} "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 16384 "
        "--eval-top-k 1 "
    )

    perf_args = (
        f"--tensor-model-parallel-size {tp} "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        f"--context-parallel-size {cp} "
        f"--expert-model-parallel-size {ep} "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if TIGHT_HOST_MEMORY else 16384} "
    )

    grpo_args = (
        "--advantage-estimator gspo "
        f"{'' if TIGHT_HOST_MEMORY else '--use-kl-loss '}"
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 4e-4 "
        "--use-tis "
        "--use-routing-replay "
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

    # 30B uses a single 8-GPU vLLM engine. In the DP layout, validate rollout
    # DP+EP by splitting that engine into TP=4 and DP=2; vLLM derives EP as
    # tensor_parallel_size x data_parallel_size, so rollout EP stays 8.
    vllm_args = (
        "--rollout-num-gpus-per-engine 8 "
        "--vllm-enable-expert-parallel "
        "--vllm-gpu-memory-utilization 0.8 "
        "--vllm-max-num-seqs 512 "
        "--vllm-max-cudagraph-capture-size 64 "
    )
    if dp > 1:
        vllm_args += f"--vllm-data-parallel-size {dp} "
    if use_deepep:
        vllm_args += "--vllm-all2all-backend deepep_high_throughput "

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
    )
    if use_deepep:
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
        f"{misc_args} "
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

    print(f"\nRun {run_tag}: PASSED")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)

    for layout_name, layout in _select_layouts():
        for use_deepep in _select_phases():
            execute(use_deepep=use_deepep, layout_name=layout_name, layout=layout)
