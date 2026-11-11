import os
import shlex
import sys
import tempfile
from pathlib import Path

import vime.utils.external_utils.command_utils as U


TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_NAME = "Qwen3.6-35B-A3B"
# Qwen3.6 reuses the Qwen3.5-35B-A3B model spec (same GDN + MoE topology).
MODEL_TYPE = "qwen3.5-35B-A3B"
MODEL_DIR = f"{TEST_ROOT}/models/{MODEL_NAME}"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"


def prepare(checkpoint_root, torch_dist_ref_load=True):
    models_dir = shlex.quote(f"{TEST_ROOT}/models")
    datasets_dir = shlex.quote(f"{TEST_ROOT}/datasets")
    model_dir = shlex.quote(MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)

    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {model_dir}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {dataset_dir}")
    if not torch_dist_ref_load:
        return None

    checkpoint_path = Path(checkpoint_root)
    checkpoint_dir = shlex.quote(str(checkpoint_path))
    # Colocate runs PP 2 across 16 NPUs. The converter pipeline-splits the 40
    # layers across 8 ranks (TP 1) so each rank holds a contiguous layer block;
    # the trainer then re-shards under TP 2 / PP 2 at load time.
    U.exec_command(
        "source scripts/models/qwen3.5-35B-A3B.sh && "
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

    # Match the verified colocate run: enable_thinking=false keeps responses
    # inside the 14336-token window and avoids the <think> block that the
    # --ci-test logprob/entropy bounds reject.
    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--apply-chat-template-kwargs '{\"enable_thinking\": false}' "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 2 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 8 "
        "--rollout-max-response-len 2048 "
        "--rollout-max-prompt-len 2048 "
        "--rollout-temperature 1 "
        "--global-batch-size 64 "
        "--balance-data "
    )

    # Colocate parallelism: TP 2 / PP 2 across 16 NPUs, EP 8 for the 256 experts.
    # --decoder-last-pipeline-num-layers 18 balances the 40 layers as 22/18
    # across the two pipeline stages (the last stage holds fewer layers because
    # it also drives the rollout engine on the same devices).
    parallel_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--decoder-last-pipeline-num-layers 18 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 8 "
        "--expert-tensor-parallel-size 1 "
        "--moe-token-dispatcher-type alltoall "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--max-tokens-per-gpu 16384 "
        "--micro-batch-size 1 "
        "--use-dynamic-batch-size "
        "--log-probs-chunk-size 1024 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
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

    # Colocate rollout: 4 GPUs per engine, DP 2, expert-parallel so the 256
    # experts shard across engines. weight_nz_mode=0 matches the verified run.
    vllm_args = (
        "--vllm-additional-config '{\"weight_nz_mode\":0}' "
        "--rollout-num-gpus-per-engine 4 "
        "--vllm-gpu-memory-utilization 0.7 "
        "--vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256) "
        "--vllm-enable-expert-parallel "
        "--vllm-data-parallel-size 2 "
        "--vllm-max-model-len 16384 "
    )

    model_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--loss-mask-type qwen3_5 "
        "--use-flash-attn "
        "--no-gradient-accumulation-fusion "
    )

    # Colocate: rollout shares the 16 NPUs with training. The memory margin
    # reserves ~8 GiB per device for the rollout engine's KV cache / weights.
    runtime_args = (
        "--train-backend megatron "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 16 "
        "--colocate "
        "--train-memory-margin-bytes 8589934596 "
        "--ci-test "
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
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "DISABLE_L2_CACHE": "1",
            "VLLM_USE_AOT_COMPILE": "0",
            "VLLM_ASCEND_ENABLE_NZ": "0",
            "HCCL_BUFFSIZE": "512",
            "HCCL_BUFFSIZE_EP": "512",
            "HCCL_CONNECT_TIMEOUT": "7200",
        },
    )


def main():
    checkpoint_tmp_root = Path(TEST_ROOT) / "tmp"
    checkpoint_tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f"{MODEL_NAME}_torch_dist_", dir=checkpoint_tmp_root) as checkpoint_root:
        checkpoint = prepare(
            checkpoint_root,
            torch_dist_ref_load=os.environ.get("VIME_TEST_TORCH_DIST_REF_LOAD", "1") == "1",
        )
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        execute(checkpoint)


if __name__ == "__main__":
    main()
