"""Megatron vs fsdp2 alignment on the same weights and the same batch.

Its tolerances are first estimates. On the first H100 run every response was
truncated, so all rewards and GRPO advantages were zero and both backends
reported a zero gradient; only the log-prob comparison was informative.
A small entropy term now gives the loss a gradient independent of rewards,
and the test fails if Megatron's gradient is zero.

1. Megatron (colocate, 4 GPUs) generates one rollout with vLLM, saves it,
   trains on it, and records its grad norms and train debug dump.
2. fsdp2 replays the saved rollout from the same HF weights with the same DP
   schedule (DP = 4) and asserts every step's grad norm is within 1% of
   Megatron's (``--ci-load-grad-norm``), then writes its own train dump.
3. The old-policy log-probs of both dumps (computed before any update, so
   from identical weights) must agree per token within bf16 noise.

Two optimizer steps per rollout keep the separate log-prob pass enabled on
both backends, so both dumps carry ``log_probs``.
"""

import os
import tempfile
from pathlib import Path

import torch

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-0.6B"
MODEL_TYPE = "qwen3-0.6B"
NUM_GPUS = 4
LOG_PROB_MEAN_ABS_DIFF = 2e-2


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def _common_args() -> str:
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        "--num-rollout 1 "
        "--rollout-batch-size 8 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )
    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        # Nonzero so the gradient compared below never vanishes with the rewards.
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
        "--rollout-num-gpus-per-engine 1 " "--vllm-gpu-memory-utilization 0.6 " "--vllm-max-cudagraph-capture-size 16 "
    )
    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        f"--actor-num-gpus-per-node {NUM_GPUS} "
        "--colocate "
        "--ci-test "
    )
    return (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"{rollout_args}{ppo_args}{optimizer_args}{vllm_args}{misc_args}"
        f"{U.get_default_wandb_args(__file__)} "
    )


def _log_prob_diffs(megatron_dump: str, fsdp_dump: str) -> list[float]:
    megatron = {
        s["rollout_position"]: s for s in torch.load(megatron_dump, map_location="cpu", weights_only=True)["samples"]
    }
    fsdp = {s["rollout_position"]: s for s in torch.load(fsdp_dump, map_location="cpu", weights_only=True)["samples"]}
    assert megatron.keys() == fsdp.keys() and megatron, "the two backends trained on different samples"
    diffs = []
    for position, sample in megatron.items():
        assert "log_probs" in sample and "log_probs" in fsdp[position], "a dump has no old-policy log_probs"
        mask = sample["loss_masks"].float()
        diff = (sample["log_probs"].float() - fsdp[position]["log_probs"].float()).abs()
        diffs.append(((diff * mask).sum() / mask.sum().clamp_min(1)).item())
    return diffs


def execute():
    work_dir = tempfile.mkdtemp(prefix="vime_fsdp2_align_")
    rollout_data = f"{work_dir}/rollout_{{rollout_id}}.pt"
    grad_norms = f"{work_dir}/grad_norm_{{rollout_id}}_{{step_id}}.pt"
    common = _common_args()

    U.execute_train(
        train_args=(
            common + f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
            "--tensor-model-parallel-size 1 "
            "--pipeline-model-parallel-size 1 "
            "--context-parallel-size 1 "
            f"--save-debug-rollout-data {rollout_data} "
            f"--save-debug-train-data {work_dir}/megatron/{{rollout_id}}.pt "
            f"--ci-save-grad-norm {grad_norms} "
        ),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )
    norms = [float(torch.load(path)) for path in sorted(Path(work_dir).glob("grad_norm_*.pt"))]
    assert norms and min(norms) > 0, f"Megatron grad norms {norms}: a zero gradient makes the 1% check vacuous"
    U.execute_train(
        train_args=(
            common + f"--ref-load /root/models/{MODEL_NAME}/ "
            "--train-backend fsdp2 "
            "--fsdp-attn-implementation flash_attention_2 "
            f"--load-debug-rollout-data {rollout_data} "
            f"--save-debug-train-data {work_dir}/fsdp2/{{rollout_id}}.pt "
            f"--ci-load-grad-norm {grad_norms} "
        ),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )

    diffs = _log_prob_diffs(f"{work_dir}/megatron/0.pt", f"{work_dir}/fsdp2/0.pt")
    mean_diff = sum(diffs) / len(diffs)
    print(f"log_probs mean abs diff over {len(diffs)} samples: mean {mean_diff:.3e}, max {max(diffs):.3e}")
    assert mean_diff <= LOG_PROB_MEAN_ABS_DIFF, f"{mean_diff=} > {LOG_PROB_MEAN_ABS_DIFF}"


if __name__ == "__main__":
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute()
