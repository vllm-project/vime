"""DSpark speculative rollout with online draft training (Qwen3-4B, 8 GPUs).

Trains the policy with GRPO and the DSpark draft at the same time for a few
steps, in colocate or non-colocate placement (``--layout``), and checks from
the run's TensorBoard log that

- a draft push to vLLM was issued after every training step (the vLLM engine
  does not support ``check_weights``, so this cannot confirm vLLM applied it;
  tests/test_dspark_weight_export.py checks the pushed names and shapes), and
- speculation stayed effective: the acceptance rate of every rollout is above
  ``MIN_SPEC_ACCEPT_RATE``.

``--ci-test`` additionally bounds the train/rollout log-prob difference.
"""

import argparse
import importlib.util
import os
import tempfile
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
DRAFT_REPO = "deepseek-ai/dspark_qwen3_4b_block7"
DRAFT_DIR = "/root/models/dspark_qwen3_4b_block7"
NUM_GPUS = 8
NUM_ROLLOUT = 3
# Provisional floor: the pretrained draft accepts roughly 30-40% on dapo-math
# (examples/dspark/README.md). Replace with the measured benchmark baseline.
MIN_SPEC_ACCEPT_RATE = 0.1


def prepare():
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"hf download {DRAFT_REPO} --local-dir {DRAFT_DIR}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.convert_checkpoint(model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS)


def train_args(layout: str, rollout_dir: Path) -> str:
    speculative_config = '{"method":"dspark","model":"' + DRAFT_DIR + '","num_speculative_tokens":7}'
    if layout == "colocate":
        placement = (
            "--tensor-model-parallel-size 1 "
            "--actor-num-gpus-per-node 8 "
            "--colocate "
            "--rollout-num-gpus-per-engine 1 "
            "--vllm-gpu-memory-utilization 0.5 "
        )
    else:
        placement = (
            "--tensor-model-parallel-size 4 "
            "--sequence-parallel "
            "--actor-num-gpus-per-node 4 "
            "--rollout-num-gpus 4 "
            "--rollout-num-gpus-per-engine 4 "
            "--vllm-gpu-memory-utilization 0.85 "
        )
    return (
        f"--hf-checkpoint /root/models/{MODEL_NAME}/ "
        f"--ref-load /root/{MODEL_NAME}_torch_dist "
        "--rollout-data-transport straw "
        f"--rollout-data-dir {quote(str(rollout_dir))} "
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt --label-key label --apply-chat-template --rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {NUM_ROLLOUT} "
        "--rollout-batch-size 8 --n-samples-per-prompt 4 --global-batch-size 32 "
        "--rollout-max-response-len 2048 --rollout-temperature 1 "
        "--advantage-estimator grpo --use-kl-loss --kl-loss-coef 0.00 --kl-loss-type low_var_kl "
        "--entropy-coef 0.00 --eps-clip 0.2 --eps-clip-high 0.28 "
        "--optimizer adam --lr 1e-6 --lr-decay-style constant --weight-decay 0.1 "
        "--adam-beta1 0.9 --adam-beta2 0.98 "
        "--pipeline-model-parallel-size 1 --context-parallel-size 1 "
        "--recompute-granularity full --recompute-method uniform --recompute-num-layers 1 "
        "--use-dynamic-batch-size --max-tokens-per-gpu 8192 "
        f"--vllm-speculative-config '{speculative_config}' "
        "--dspark-block-size 7 --dspark-num-draft-layers 5 --dspark-target-layer-ids 1,9,17,25,33 "
        "--dspark-markov-rank 256 --dspark-markov-head-type vanilla --dspark-num-anchors 512 "
        "--dspark-mask-token-id 151669 --dspark-ce-loss-alpha 0.5 --dspark-l1-loss-alpha 0.5 "
        "--dspark-confidence-head-alpha 0.1 --dspark-loss-decay-gamma 4.0 --dspark-draft-loss-weight 1.0 "
        f"--dspark-intermediate-size 9728 --dspark-pretrained-model {DRAFT_DIR}/model.safetensors "
        "--attention-dropout 0.0 --hidden-dropout 0.0 --accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 --attention-backend flash "
        "--actor-num-nodes 1 "
        f"{placement}"
        "--use-tensorboard "
        "--ci-test "
        f"{U.get_default_wandb_args(__file__, run_name_prefix=layout)} "
    )


def _read_tensorboard(directory: Path) -> dict[str, dict[int, float]]:
    path = U.repo_base_dir / "tools" / "bench_spec_rollout.py"
    spec = importlib.util.spec_from_file_location("bench_spec_rollout", path)
    bench = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bench)
    return bench.read_tensorboard(directory)


def verify(tensorboard_dir: Path) -> None:
    scalars = _read_tensorboard(tensorboard_dir)

    accept = scalars.get("rollout/spec_accept_rate", {})
    assert len(accept) == NUM_ROLLOUT, f"expected {NUM_ROLLOUT} rollouts with spec metrics, got {accept}"
    assert all(rate >= MIN_SPEC_ACCEPT_RATE for rate in accept.values()), accept

    # Perf of step k is logged before the weight update that follows step k, so
    # the draft pushes after steps 0..N-2 appear on steps 1..N-1.
    draft_sync = scalars.get("perf/update_draft_weights_time", {})
    perf_steps = sorted(scalars.get("perf/actor_train_time", {}))
    assert perf_steps, "no perf metrics logged"
    missing = [step for step in perf_steps[1:] if step not in draft_sync]
    assert not missing, f"no draft push before steps {missing}: {draft_sync}"
    print(f"DSpark verified: accept rates {accept}, draft push times {draft_sync}")


def execute(layout: str) -> None:
    with tempfile.TemporaryDirectory(prefix="vime_dspark_") as tmp:
        tmp = Path(tmp)
        U.execute_train(
            train_args=train_args(layout, tmp / "rollout"),
            num_gpus_per_node=NUM_GPUS,
            megatron_model_type=MODEL_TYPE,
            extra_env_vars={"TENSORBOARD_DIR": str(tmp / "tensorboard")},
        )
        verify(tmp / "tensorboard")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", choices=["colocate", "non-colocate"], default="colocate")
    options = parser.parse_args()
    prepare()
    for key in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(key, None)
    execute(options.layout)
