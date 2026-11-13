"""Qwen3-4B DSpark external-draft E2E smoke (NPU).

Chain under test: vLLM serves Qwen3-4B with a DSpark draft, rollout hidden
states feed the external draft collector, the draft trains and saves
draft_latest.pt, and publishes back into the live engine (weight-update
session).

Draft weights come from HF (deepseek-ai/dspark_qwen3_4b_block7, public).
Unlike the EAGLE-3 AngelSlim checkpoint it declares everything vLLM and the
training side need (target_layer_ids, block_size, markov head), so prepare()
needs no config patching: the layer-id resolver adopts the declaration and
the block size is read from the checkpoint. First-round acceptance near the
published vLLM measurement (4.1 at block 7) is the layer-alignment oracle.
"""

import os
import shlex
import shutil

import vime.utils.external_utils.command_utils as U

import draft_update_checks


TEST_ROOT = os.environ.get("HF_HOME") or "/root"
MODEL_DIR = f"{TEST_ROOT}/models/Qwen3-4B"
DRAFT_MODEL_DIR = f"{TEST_ROOT}/models/dspark_qwen3_4b_block7"
DATASET_DIR = f"{TEST_ROOT}/datasets/dapo-math-17k"
DRAFT_CHECKPOINT_DIR = f"{TEST_ROOT}/dspark-draft-smoke-output/draft"
JOB_LOG_PATH = f"{TEST_ROOT}/dspark-draft-smoke-output/job.log"

# Post-run evidence-check thresholds (tests/draft_update_checks.py): healthy
# rounds measured 2.44-3.64, while the acceptance-collapse failure mode sits
# at ~1.0-1.1 with every other signal green, so the floor is what actually
# detects it. 2.0 splits the two regimes.
NUM_ROLLOUT = 3
MIN_ACCEPT_LENGTH = 2.0


def ensure_speculators():
    # speculators 0.9.0 declares numpy>=2/transformers>=5 floors that the
    # pinned NPU stack cannot satisfy; the two pure-python wheels run on
    # the stack's own versions (E2E-validated), so install them without
    # dependency resolution when the image lacks them. CI sets
    # PIP_INDEX_URL because the default mirror is unreachable from the
    # buildkite agent.
    try:
        import speculators  # noqa: F401
    except ImportError:
        U.exec_command("pip install --no-deps speculators==0.9.0 hs-connectors==0.9.0")


def prepare():
    ensure_speculators()

    models_dir = shlex.quote(f"{TEST_ROOT}/models")
    datasets_dir = shlex.quote(f"{TEST_ROOT}/datasets")
    model_dir = shlex.quote(MODEL_DIR)
    draft_model_dir = shlex.quote(DRAFT_MODEL_DIR)
    dataset_dir = shlex.quote(DATASET_DIR)

    U.exec_command(f"mkdir -p {models_dir} {datasets_dir}")
    U.exec_command(f"hf download Qwen/Qwen3-4B --local-dir {model_dir}")
    U.exec_command(f"hf download deepseek-ai/dspark_qwen3_4b_block7 --local-dir {draft_model_dir}")
    U.exec_command("hf download --repo-type dataset zhuzilin/dapo-math-17k " f"--local-dir {dataset_dir}")

    # A leftover draft checkpoint remembers its next rollout while a fresh
    # actor always restarts at 0 -- the draft/actor alignment guard in
    # create_draft_model rejects the run before training starts.
    shutil.rmtree(DRAFT_CHECKPOINT_DIR, ignore_errors=True)


def execute():
    model_dir = shlex.quote(MODEL_DIR)
    draft_model_dir = shlex.quote(DRAFT_MODEL_DIR)
    draft_checkpoint_dir = shlex.quote(DRAFT_CHECKPOINT_DIR)
    prompt_data = shlex.quote(f"{DATASET_DIR}/dapo-math-17k.jsonl")

    # No spaces inside: launch_commands forwards train_args as one shell
    # string, and the speculative config must stay a single shell word.
    # block 7 matches the checkpoint's declared block_size.
    speculative_config = (
        '{"method":"dspark","model":"' + DRAFT_MODEL_DIR + '","num_speculative_tokens":7}'
    )

    checkpoint_args = f"--hf-checkpoint {model_dir} --ref-load {model_dir} --load {model_dir} --no-load-optim "

    rollout_args = (
        f"--prompt-data {prompt_data} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type math "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 2048 "
        "--rollout-temperature 1 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    parallel_args = (
        "--tensor-model-parallel-size 4 "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 1 "
        "--expert-model-parallel-size 1 "
        "--expert-tensor-parallel-size 1 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 8192 "
    )

    grpo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.0 "
        "--kl-loss-type low_var_kl "
        "--kl-coef 0.00 "
        "--entropy-coef 0.0 "
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
        # The block-7 DSpark draft with 512 anchors needs a larger CUDA-graph
        # workspace than the EAGLE-3 run; 0.6 and 0.5 both OOM during
        # PIECEWISE capture, 0.4 leaves the capture pool enough headroom.
        "--vllm-gpu-memory-utilization 0.4 "
        "--vllm-max-model-len 4096 "
        f"--vllm-speculative-config '{speculative_config}' "
    )

    draft_args = (
        "--enable-external-draft-training "
        "--draft-algorithm dspark "
        f"--draft-model-path {draft_model_dir} "
        # Layer ids are omitted on purpose: the checkpoint's target_layer_ids
        # [1, 9, 17, 25, 33] declaration drives both the vLLM serving hooks
        # and (translated to the captured hidden states) the training-side
        # collector, keeping the two sides on the same layers by construction.
        "--draft-dspark-loss-fn '{\"ce\":0.1,\"tv\":0.9}' "
        "--draft-hidden-window-tokens 64 "
        "--draft-max-samples-per-rollout-per-dp 2 "
        "--draft-max-tokens-per-rollout-per-dp 256 "
        "--draft-collect-interval 1 "
        "--draft-train-interval 1 "
        "--draft-publish-interval 1 "
        "--draft-train-steps-per-trigger 1 "
        "--draft-batch-size-per-gpu 1 "
        f"--draft-checkpoint-path {draft_checkpoint_dir} "
        "--draft-save-interval 1 "
        "--update-weight-mode full "
        "--update-weight-transport nccl "
    )

    model_args = (
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        "--attention-backend flash "
        "--micro-batch-size 1 "
        "--use-flash-attn "
    )

    runtime_args = (
        "--train-backend megatron "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 4 "
        "--rollout-num-gpus 4 "
        "--ci-test "
        # DSpark block verification records engine logprobs on a different
        # kernel path than the Megatron re-forward, so the bit-exact
        # first-step ppo_kl==0 checker does not apply with a draft in the
        # loop; the abs-diff checker (threshold 0.1) stays on.
        "--ci-disable-kl-checker "
    )

    train_args = (
        checkpoint_args
        + rollout_args
        + parallel_args
        + grpo_args
        + optimizer_args
        + vllm_args
        + draft_args
        + model_args
        + runtime_args
    )
    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=8,
        megatron_model_type="qwen3-4B",
        # Engine-behavior envs the drafter form relies on (both were present
        # in every validated dspark run): spawn dodges the CANN 9.2 issue
        # where fork-derived engine workers inherit a device table built
        # from the wide job env; NZ off matches weight_nz_mode=0 above.
        extra_env_vars={
            "VLLM_WORKER_MULTIPROC_METHOD": "spawn",
            "VLLM_ASCEND_ENABLE_NZ": "0",
        },
    )


def main():
    prepare()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    # exec_command lets the ray job write straight to this process's fd 1,
    # so the job log is captured at the descriptor level (tee_output).
    with draft_update_checks.tee_output(JOB_LOG_PATH):
        execute()
    draft_update_checks.check_draft_update_evidence(
        JOB_LOG_PATH,
        num_rollouts=NUM_ROLLOUT,
        min_accept_length=MIN_ACCEPT_LENGTH,
    )


if __name__ == "__main__":
    main()
