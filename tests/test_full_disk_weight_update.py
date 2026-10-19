"""E2E smoke test for full checkpoint weight updates through disk.

Runs a tiny Qwen3.5-0.8B job where each weight sync writes a complete HF
checkpoint and rollout engines reload it through ``update_weights_from_disk``.
"""

import json
import os
import tempfile
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U


MODEL_NAME = "Qwen3.5-0.8B"
MODEL_TYPE = "qwen3.5-0.8B"
NUM_GPUS = 4
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"
TORCH_DIST_CKPT = f"/dev/shm/{MODEL_NAME}_torch_dist"


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/gsm8k", dir_dst=DATASET_DIR)
    U.convert_checkpoint(
        model_name=MODEL_NAME,
        megatron_model_type=MODEL_TYPE,
        num_gpus_per_node=NUM_GPUS,
        dir_dst="/dev/shm",
        hf_checkpoint=str(MODEL_DIR),
    )


def verify_checkpoints(disk_dir, num_rollout):
    checkpoint_dirs = sorted(Path(disk_dir).glob("weight_v[0-9]*"))
    assert len(checkpoint_dirs) >= num_rollout + 1, "Expected initial weights and an update after every rollout"
    versions = [int(path.name.removeprefix("weight_v")) for path in checkpoint_dirs]
    assert versions == list(range(1, len(versions) + 1)), versions
    for path in checkpoint_dirs:
        index = json.loads((path / "model.safetensors.index.json").read_text())
        assert "delta_encoding" not in index.get("metadata", {}), "Expected full checkpoints"
        assert index["weight_map"], f"No weights in {path}"
        for filename in set(index["weight_map"].values()):
            assert (path / filename).stat().st_size > 0, f"Missing or empty shard: {path / filename}"
    print(f"Full disk validation passed: {num_rollout} rollouts, published versions {versions}", flush=True)


def execute():
    num_rollout = int(os.environ.get("VIME_TEST_NUM_ROLLOUT", "1"))
    assert num_rollout >= 1
    with tempfile.TemporaryDirectory(prefix="vime_full_disk_weight_update_", dir=U.get_test_tmp_dir()) as disk_dir:
        # The NPU launcher skips automatic conversion; use the actual HF
        # checkpoint as in the existing delta-disk test, not a missing path.
        ref_load = TORCH_DIST_CKPT if U.current_platform().torch_dist_convert else str(MODEL_DIR)
        ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(ref_load)} "

        rollout_args = (
            f"--prompt-data {quote(str(DATASET_DIR / 'gsm8k/train.parquet'))} "
            "--input-key messages "
            "--label-key label "
            "--apply-chat-template "
            "--rollout-shuffle "
            "--rm-type math "
            f"--num-rollout {num_rollout} "
            "--rollout-batch-size 4 "
            "--n-samples-per-prompt 4 "
            "--rollout-max-response-len 1024 "
            "--rollout-temperature 0.8 "
            "--over-sampling-batch-size 8 "
            "--dynamic-sampling-filter-path vime.rollout.filter_hub.dynamic_sampling_filters.check_reward_nonzero_std "
            "--global-batch-size 16 "
        )

        perf_args = (
            "--tensor-model-parallel-size 1 "
            "--sequence-parallel "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            "--expert-model-parallel-size 1 "
            "--expert-tensor-parallel-size 1 "
            "--use-dynamic-batch-size "
            "--max-tokens-per-gpu 9216 "
        )

        grpo_args = (
            "--advantage-estimator grpo "
            "--use-kl-loss "
            "--kl-loss-coef 0.00 "
            "--kl-loss-type low_var_kl "
            "--entropy-coef 0.01 "
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
        )

        vllm_args = (
            "--rollout-num-gpus-per-engine 1 "
            "--rollout-num-gpus 3 "
            "--vllm-gpu-memory-utilization 0.7 "
            "--vllm-max-cudagraph-capture-size 32 "
        )

        disk_update_args = (
            "--update-weight-mode full "
            "--update-weight-transport disk "
            f"--update-weight-disk-dir {quote(disk_dir)} "
            "--update-weight-disk-keep-files "
        )

        ci_args = "--ci-test "

        misc_args = (
            "--attention-dropout 0.0 "
            "--hidden-dropout 0.0 "
            "--accumulate-allreduce-grads-in-fp32 "
            "--attention-softmax-in-fp32 "
            "--attention-backend flash "
            "--loss-mask-type qwen3_5 "
            "--actor-num-nodes 1 "
            "--actor-num-gpus-per-node 1 "
        )

        train_args = (
            f"{ckpt_args} "
            f"{rollout_args} "
            f"{optimizer_args} "
            f"{grpo_args} "
            f"{U.get_default_wandb_args(__file__)} "
            f"{perf_args} "
            f"{vllm_args} "
            f"{disk_update_args} "
            f"{ci_args} "
            f"{misc_args} "
            f"{U.get_test_platform_args()} "
        )

        U.execute_train(
            train_args=train_args,
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
        )

        verify_checkpoints(disk_dir, num_rollout)


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
