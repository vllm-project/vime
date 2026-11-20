"""E2E test: mixed offload with updatable + frozen models.

Deploys two models via --vllm-config in colocate mode:
  - "actor": update_weights=true, 4 GPUs → overlaps with megatron, gets offloaded
    and weights updated from training.
  - "ref":   update_weights=false, 4 GPUs → overlaps with megatron, gets offloaded
    via level-1 sleep (weights kept in CPU RAM) and restored by memory wake_up.
    NOT a disk round-trip: update_weights_from_disk only runs for the updatable
    actor; frozen engines never take the disk path.

Key coverage:
  - Per-group needs_offload (both overlap with megatron in colocate mode)
  - Frozen model offload/restore via level-1 sleep + memory resume
  - Selective flush_cache (only for offloaded / updatable engines)
  - Offload/onload cycle completes without crash
"""

import os
import tempfile

import vime.utils.external_utils.command_utils as U

# The slime sync (#386) removed command_utils.is_rocm(); these ROCm test files
# still gate on U.is_rocm(). Re-provide it here (test-file only, merge-safe).
if not hasattr(U, "is_rocm"):
    import torch as _torch

    U.is_rocm = lambda: _torch.version.hip is not None

MODEL_NAME = "Qwen2.5-0.5B-Instruct"
MODEL_TYPE = "qwen2.5-0.5B"
NUM_GPUS = 8

# ROCm converts HF->Megatron (no modelopt bridge) into the host-mounted
# models dir, so the converted checkpoint is cached and reused across runs.
MG_PATH = f"/root/models/{MODEL_NAME}_torch_dist"

# Two models on 8 GPUs (colocate): actor gets weight updates, ref is frozen.
VLLM_CONFIG_YAML = """\
vllm:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 1
  - name: ref
    update_weights: false
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 1
"""


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.hf_download_dataset("zhuzilin/gsm8k")
    if U.is_rocm():
        U.convert_checkpoint(
            MODEL_NAME,
            MODEL_TYPE,
            num_gpus_per_node=1,
            extra_args="--no-gradient-accumulation-fusion --attention-backend flash",
            dir_dst="/root/models",
        )


def execute():
    config_file = tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", prefix="vllm_mixed_offload_", delete=False)
    config_file.write(VLLM_CONFIG_YAML)
    config_file.flush()
    config_path = config_file.name

    if U.is_rocm():
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ --ref-load {MG_PATH}/ "
    else:
        ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}/ "

    rollout_args = (
        "--prompt-data /root/datasets/gsm8k/train.parquet "
        "--input-key messages "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        "--num-rollout 2 "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 512 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
    )

    eval_args = (
        "--eval-interval 20 "
        "--eval-prompt-data gsm8k /root/datasets/gsm8k/test.parquet "
        "--n-samples-per-eval-prompt 1 "
        "--eval-max-response-len 512 "
        "--eval-top-k 1 "
    )

    perf_args = (
        "--tensor-model-parallel-size 1 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 4096 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--use-kl-loss "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type low_var_kl "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
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
        f"--vllm-gpu-memory-utilization {'0.3' if U.is_rocm() else '0.7'} "
        "--vllm-max-num-seqs 32 "
        f"{'' if U.is_rocm() else '--vllm-max-cudagraph-capture-size 16 '}"
        f"--vllm-config {config_path} "
    )

    ci_args = "--ci-test "

    misc_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
        f'{"--no-gradient-accumulation-fusion --no-offload-train " if U.is_rocm() else ""}'
        f'{"--update-weight-transport disk --update-weight-disk-dir /tmp/vime_wsync_mixed_offload " if U.is_rocm() else ""}'
    )

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
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
    )


if __name__ == "__main__":
    prepare()
    os.environ.pop("http_proxy", None)
    os.environ.pop("https_proxy", None)
    os.environ.pop("HTTP_PROXY", None)
    os.environ.pop("HTTPS_PROXY", None)
    execute()
