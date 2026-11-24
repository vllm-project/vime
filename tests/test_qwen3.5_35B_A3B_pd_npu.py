"""Qwen3.5-35B-A3B colocate PD smoke for Ascend NPU (1P1D, TP4 engines)."""

import os
import shlex
import sys
import tempfile
from pathlib import Path

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3.5-35B-A3B"
MODEL_TYPE = "qwen3.5-35B-A3B"
NUM_GPUS = 8
NUM_ROLLOUT = 2
TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/{MODEL_NAME}"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"


def prepare(checkpoint_root):
    U.exec_command(f"mkdir -p {shlex.quote(MODEL_DIR)} {shlex.quote(DATASET_DIR)} {shlex.quote(f'{TEST_ROOT}/tmp')}")
    if not (Path(MODEL_DIR) / "model.safetensors.index.json").is_file():
        U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {shlex.quote(MODEL_DIR)}")
    if not (Path(DATASET_DIR) / "dapo-math-17k.jsonl").is_file():
        U.exec_command(
            "hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {shlex.quote(DATASET_DIR)}"
        )
    override = os.environ.get("MOE_TORCH_DIST")
    if override:
        return override
    checkpoint_path = Path(checkpoint_root)
    U.exec_command(
        "source scripts/models/qwen3.5-35B-A3B.sh && "
        "TRANSFORMERS_VERBOSITY=error "
        f"VIME_PLATFORM=npu PYTHONPATH={shlex.quote(str(U.repo_base_dir))}:/root/Megatron-LM:${{PYTHONPATH:-}} "
        f"{shlex.quote(sys.executable)} -m torch.distributed.run --nproc-per-node 8 "
        "tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        "--qwen-gdn-backend fla_npu "
        "--qwen-gdn-implementation fused_tp "
        f"--hf-checkpoint {shlex.quote(MODEL_DIR)} --save {shlex.quote(str(checkpoint_path))}"
    )
    tracker = checkpoint_path / "latest_checkpointed_iteration.txt"
    assert tracker.read_text().strip() == "release"
    return str(checkpoint_path)


def _train_args(*, ref_load: str) -> str:
    ckpt_args = f"--hf-checkpoint {shlex.quote(MODEL_DIR)} --ref-load {shlex.quote(ref_load)} --no-load-optim "

    rollout_args = (
        f"--prompt-data {shlex.quote(f'{DATASET_DIR}/dapo-math-17k.jsonl')} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 4 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
        "--log-probs-chunk-size 1024 "
        "--micro-batch-size 1 "
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
        "--router-kv-connector nixl "
        "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        "--rollout-num-gpus-per-engine 4 "
        "--vllm-enable-sleep-mode "
        "--vllm-enable-expert-parallel "
        "--vllm-gpu-memory-utilization 0.7 "
        "--vllm-max-model-len 16384 "
        "--prefill-num-servers 1 "
    )

    verify_args = "--ci-test " "--ci-disable-kl-checker "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--use-flash-attn "
        "--no-gradient-accumulation-fusion "
        "--train-backend megatron "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
        "--qwen-gdn-backend fla_npu "
        "--qwen-gdn-implementation fused_tp "
        "--colocate "
    )

    return (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{perf_args} "
        f"{grpo_args} "
        f"{optimizer_args} "
        f"{vllm_args} "
        f"{verify_args} "
        f"{misc_args} "
    )


def execute(ref_load: str):
    U.execute_train(
        train_args=_train_args(ref_load=ref_load),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={"DISABLE_L2_CACHE": "1", "VLLM_USE_AOT_COMPILE": "0"},
    )


def main():
    scratch = Path(TEST_ROOT) / "tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_NAME}_torch_dist_", dir=scratch) as checkpoint_root:
        ref_load = prepare(checkpoint_root)
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        execute(ref_load)


if __name__ == "__main__":
    main()
