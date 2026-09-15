import os
import shlex
import sys
import tempfile
from pathlib import Path

import vime.utils.external_utils.command_utils as U


TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/Qwen3-30B-A3B"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"


def prepare(torch_dist_ref_load=True):
    models_dir = shlex.quote(f"{TEST_ROOT}/models")
    datasets_dir = shlex.quote(f"{TEST_ROOT}/datasets")
    model_dir = shlex.quote(MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)

    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download Qwen/Qwen3-30B-A3B --local-dir {model_dir}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {dataset_dir}")
    if not torch_dist_ref_load:
        return None

    # Retain conversion artifacts for inspection; never delete an existing checkpoint.
    checkpoint_path = Path(tempfile.mkdtemp(prefix="Qwen3-30B-A3B_torch_dist_", dir=f"{TEST_ROOT}/models"))
    checkpoint_dir = shlex.quote(str(checkpoint_path))
    U.exec_command(
        "source scripts/models/qwen3-30B-A3B.sh && "
        "TRANSFORMERS_VERBOSITY=error "
        f"VIME_PLATFORM=npu PYTHONPATH={shlex.quote(str(U.repo_base_dir))}:/root/Megatron-LM:${{PYTHONPATH:-}} "
        f"{shlex.quote(sys.executable)} -m torch.distributed.run --nproc-per-node 8 "
        "tools/convert_hf_to_torch_dist.py "
        "${MODEL_ARGS[@]} "
        f"--hf-checkpoint {model_dir} --save {checkpoint_dir}"
    )

    tracker = checkpoint_path / "latest_checkpointed_iteration.txt"
    assert tracker.read_text().strip() == "release"
    weight_files = [
        path
        for path in checkpoint_path.rglob("*")
        if path.is_file() and path.name != "latest_checkpointed_iteration.txt"
    ]
    assert weight_files, f"No checkpoint weights found under {checkpoint_path}"
    return str(checkpoint_path)


def execute(torch_dist_checkpoint=None):
    model_dir = shlex.quote(MODEL_DIR)
    prompt_data = shlex.quote(f"{DATASET_DIR}/dapo-math-17k.jsonl")

    checkpoint_args = f"--hf-checkpoint {model_dir} --load {model_dir} --ref-load {model_dir} --no-load-optim "
    if torch_dist_checkpoint is not None:
        checkpoint_args = (
            f"--hf-checkpoint {model_dir} --ref-load {shlex.quote(torch_dist_checkpoint)} --no-load-optim "
        )

    rollout_args = (
        f"--prompt-data {prompt_data} "
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

    parallel_args = (
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
        "--max-tokens-per-gpu 20480 "
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
        "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        "--rollout-num-gpus-per-engine 4 "
        "--vllm-enable-sleep-mode "
        "--vllm-enable-expert-parallel "
        "--vllm-gpu-memory-utilization 0.7 "
    )

    model_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--use-flash-attn "
        "--no-gradient-accumulation-fusion "
    )

    runtime_args = (
        "--train-backend megatron "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--rollout-num-gpus 8 "
        "--ci-test "
        "--colocate "
    )

    train_args = (
        checkpoint_args
        + rollout_args
        + parallel_args
        + grpo_args
        + optimizer_args
        + vllm_args
        + model_args
        + runtime_args
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=16,
        megatron_model_type="qwen3-30B-A3B",
        extra_env_vars={
            "DISABLE_L2_CACHE": "1",
            "VLLM_USE_AOT_COMPILE": "0",
        },
    )


def main():
    checkpoint = prepare(torch_dist_ref_load=os.environ.get("VIME_TEST_TORCH_DIST_REF_LOAD", "1") == "1")
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(checkpoint)


if __name__ == "__main__":
    main()
