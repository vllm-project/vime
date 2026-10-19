"""End-to-end debug-dump alignment test with TP/PP/CP all enabled.

Runs a single rollout+train pass on Qwen2.5-0.5B with tensor / pipeline /
context parallel all > 1 and ``--dump-details``, which writes both the rollout
debug dump (``rollout_data/{id}.pt``) and the train debug dump
(``train_data/{id}.pt``). It then joins the two dumps by ``rollout_position``
and asserts that the train dump's CP-reassembled ``rollout_log_probs`` match the
per-sample ``rollout_log_probs`` stored on the rollout side.

This is the strongest correctness check for the train dump: it exercises the
writer selection (only the last PP stage + TP rank 0 write), the cross-CP
reassembly of response-token fields, and the sample-index/position ordering all
at once — a mismatch in any of them makes the compared log-probs diverge.

Uses Qwen2.5-0.5B-Instruct with 8 GPUs: TP=2, PP=2, CP=2 (DP=1).
"""

import os
import tempfile
from shlex import quote

import torch

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"
NUM_ROLLOUT = 1


def prepare():
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/gsm8k", dir_dst=DATASET_DIR)


def _train_args(dump_dir: str) -> str:
    ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(str(MODEL_DIR))} "

    rollout_args = (
        f"--prompt-data {quote(str(DATASET_DIR / 'gsm8k/train.parquet'))} "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 256 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
    )

    # TP=2, PP=2, CP=2 -> 8 GPUs, DP=1. Exercises the dump's writer selection
    # (last PP stage + TP0 + CP0) and the cross-CP response-field reassembly.
    parallel_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 2 "
        "--context-parallel-size 2 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = "--advantage-estimator grpo --eps-clip 0.2 "

    optimizer_args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        "--ci-test "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 2 " "--vllm-gpu-memory-utilization 0.7 " "--vllm-max-cudagraph-capture-size 16 "
    )
    if U.current_platform().name == "npu":
        # Keep graph execution; bound serving memory for the CI request load.
        vllm_args = (
            "--rollout-num-gpus-per-engine 2 "
            "--vllm-gpu-memory-utilization 0.5 --vllm-max-num-seqs 16 "
            "--vllm-max-cudagraph-capture-size 8 "
            "--vllm-max-model-len 8192 "
        )

    return (
        f"{ckpt_args} {rollout_args} {optimizer_args} {grpo_args} "
        f"{parallel_args} {misc_args} {vllm_args} "
        f"--dump-details {quote(dump_dir)} "
        f"{U.get_test_platform_args()} "
        f"{'--log-probs-chunk-size 128 ' if U.current_platform().name == 'npu' else ''}"
    )


def _verify(dump_dir: str):
    """Join the rollout and train dumps by position and compare rollout_log_probs."""
    for rollout_id in range(NUM_ROLLOUT):
        rollout_path = f"{dump_dir}/rollout_data/{rollout_id}.pt"
        train_path = f"{dump_dir}/train_data/{rollout_id}.pt"

        rollout = torch.load(rollout_path, weights_only=False)
        train = torch.load(train_path, weights_only=True)
        assert train["format_version"] == 2, train["format_version"]

        rollout_samples = rollout["samples"]
        train_samples = train["samples"]
        assert train_samples, f"empty train dump for rollout {rollout_id}"

        compared = 0
        for sample in train_samples:
            pos = sample["rollout_position"]
            assert pos is not None, "rollout_position missing; cannot align train dump to rollout dump"
            train_lp = sample.get("rollout_log_probs")
            if train_lp is None:
                continue  # rollout log-probs not carried into training; nothing to compare
            ref_lp = rollout_samples[pos]["rollout_log_probs"]
            assert ref_lp is not None, f"rollout dump sample {pos} has no rollout_log_probs"
            ref_lp = torch.as_tensor(ref_lp, dtype=train_lp.dtype)
            assert train_lp.shape == ref_lp.shape, (
                f"rollout {rollout_id} pos {pos}: reassembled shape {tuple(train_lp.shape)} "
                f"!= rollout dump {tuple(ref_lp.shape)}"
            )
            torch.testing.assert_close(
                train_lp,
                ref_lp,
                rtol=1e-3,
                atol=1e-3,
                msg=lambda m, pos=pos, rid=rollout_id: f"rollout {rid} pos {pos}: rollout_log_probs mismatch\n{m}",
            )
            compared += 1

        assert compared > 0, (
            f"rollout {rollout_id}: no rollout_log_probs were compared; the train dump did not carry "
            "rollout_log_probs, so this test would silently pass without checking anything."
        )
        print(
            f"rollout {rollout_id}: verified {compared} samples' reassembled rollout_log_probs match the rollout dump"
        )


def execute():
    dump_dir = tempfile.mkdtemp(prefix="vime_dump_details_", dir=U.get_test_tmp_dir())
    print(f"Using dump-details dir: {dump_dir}")

    U.execute_train(
        train_args=_train_args(dump_dir),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

    print("=" * 60)
    print("Verifying train dump aligns with rollout dump (join by rollout_position)")
    print("=" * 60)
    _verify(dump_dir)
    print("Train/rollout debug-dump alignment verified.")


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
