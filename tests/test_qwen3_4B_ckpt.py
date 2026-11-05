import json
import os
from argparse import ArgumentParser
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")
parser.add_argument(
    "--save-optimizer",
    choices=["cpu", "gpu"],
    default="cpu",
    help="Optimizer placement for save; gpu means the current accelerator (NPU on Ascend).",
)
parser.add_argument(
    "--load-optimizer",
    choices=["cpu", "gpu"],
    default="cpu",
    help="Optimizer placement for load; gpu means the current accelerator (NPU on Ascend).",
)
parser.add_argument("--checkpoint-dir", default=None, help="Explicit checkpoint directory; always preserved.")
parser.add_argument(
    "--keep-artifacts",
    action="store_true",
    default=None,
    help="Preserve automatic scratch; also VIME_TEST_KEEP_ARTIFACTS=1.",
)


def prepare(checkpoint_dir: str):
    checkpoint = Path(checkpoint_dir)
    if checkpoint.exists() and any(checkpoint.iterdir()):
        raise FileExistsError(f"Checkpoint directory must be empty: {checkpoint}")
    U.exec_command(f"mkdir -p {quote(str(TEST_ROOT / 'models'))} {quote(str(DATASET_DIR))}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k", dir_dst=DATASET_DIR)

    # This migration case intentionally initializes NPU training from HF weights.
    if U.current_platform().name != "npu":
        U.convert_checkpoint(
            model_name=MODEL_NAME,
            megatron_model_type=MODEL_TYPE,
            num_gpus_per_node=NUM_GPUS,
            dir_dst=str(TEST_ROOT / "models"),
            hf_checkpoint=str(MODEL_DIR),
        )


def optimizer_args(optimizer: str):
    # Historical CLI spelling: gpu disables CPU offload; it never selects CUDA.
    args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
        "--use-precision-aware-optimizer "
    )
    if optimizer == "cpu":
        args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    return args


def execute(mode: str = "", optimizer: str = "cpu", checkpoint_dir: str = ""):
    ref_load = TEST_ROOT / "models" / f"{MODEL_NAME}_torch_dist" if (U.current_platform().name != "npu") else MODEL_DIR
    ckpt_args = f"--hf-checkpoint {quote(str(MODEL_DIR))} --ref-load {quote(str(ref_load))} "
    checkpoint_dir_arg = quote(checkpoint_dir)
    if mode in ("save", "async_save", "load"):
        ckpt_args += f"--save {checkpoint_dir_arg} --save-interval 2 "
    if mode == "async_save":
        ckpt_args += "--async-save "
    elif mode == "load":
        ckpt_args += f"--load {checkpoint_dir_arg} --ckpt-step 1 --use-checkpoint-opt-param-scheduler "

    rollout_args = (
        "--rollout-data-transport straw "
        f"--rollout-data-dir {quote(str(Path(checkpoint_dir) / 'rollout_data'))} "
        f"--prompt-data {quote(str(DATASET_DIR / 'dapo-math-17k/dapo-math-17k.jsonl'))} "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {3 if mode == 'load' else 2} "
        "--rollout-batch-size 4 "
        "--n-samples-per-prompt 4 "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        f"--max-tokens-per-gpu {2048 if U.current_platform().name == 'npu' else 16384} "
    )

    ppo_args = (
        "--advantage-estimator grpo "
        "--kl-loss-coef 0.00 "
        "--kl-loss-type k1 "
        "--kl-coef 0.00 "
        "--entropy-coef 0.00 "
        "--eps-clip 0.2 "
    )

    vllm_args = (
        "--rollout-num-gpus-per-engine 2 --vllm-gpu-memory-utilization 0.8 --vllm-max-cudagraph-capture-size 16 "
    )

    if U.current_platform().name == "npu":
        vllm_args = (
            "--rollout-num-gpus-per-engine 2 --vllm-gpu-memory-utilization 0.5 "
            "--vllm-max-num-seqs 16 --vllm-max-cudagraph-capture-size 8 --vllm-max-model-len 32768 "
        )
    phase = "load" if mode == "load" else "save"
    check_dir = Path(checkpoint_dir) / "checks" / phase
    check_dir.mkdir(parents=True, exist_ok=True)
    ci_args = (
        "--ci-test --custom-megatron-before-train-step-hook-path checkpoint_test_helpers.before_train_step "
        f"--ci-save-grad-norm {quote(str(check_dir / 'grad_{rollout_id}_{step_id}.pt'))} "
    )

    misc_args = (
        # default dropout in megatron is 0.1
        "--attention-dropout 0.0 "
        "--hidden-dropout 0.0 "
        # should be good for model performance
        "--accumulate-allreduce-grads-in-fp32 "
        "--attention-softmax-in-fp32 "
        # need to comment this when using model with MLA
        "--attention-backend flash "
        "--actor-num-nodes 1 "
        "--actor-num-gpus-per-node 8 "
        "--colocate "
    )

    train_args = (
        f"{ckpt_args} "
        f"{rollout_args} "
        f"{optimizer_args(optimizer)} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
        f"{U.get_test_platform_args()} "
        f"{'--log-probs-chunk-size 128 ' if U.current_platform().name == 'npu' else ''}"
    )

    U.execute_train(
        train_args=train_args,
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        extra_env_vars={
            "PYTHONPATH": f"{Path(__file__).resolve().parent}:{U.repo_base_dir}:/root/Megatron-LM",
            "VIME_CKPT_TEST_PHASE": phase,
            "VIME_CKPT_TEST_DIR": str(check_dir),
        },
    )


def verify_phase(checkpoint_dir, phase, expected_rollouts, optimizer=None):
    import torch

    check_dir = Path(checkpoint_dir) / "checks" / phase
    gradients = sorted(check_dir.glob("grad_*.pt"))
    assert len(gradients) == len(expected_rollouts), (phase, gradients)
    for rollout in expected_rollouts:
        gradient = check_dir / f"grad_{rollout}_0.pt"
        assert gradient.is_file(), gradient
        assert torch.isfinite(torch.as_tensor(torch.load(gradient, weights_only=False))).all(), gradient
        report = json.loads((check_dir / f"before_{rollout}.json").read_text())
        assert report["rollout_id"] == rollout and report["scheduler_num_steps"] == rollout * 16
        if optimizer is not None:
            assert report["optimizer_cpu_offload"] == (optimizer == "cpu"), report
        if phase == "load":
            assert not report["no_load_optim"] and not report["finetune"], report
    print(f"Checkpoint {phase} verified: rollouts {list(expected_rollouts)}, finite gradients and scheduler progress.")


def run_roundtrip(args):
    save_mode = "async" if args.async_save else "sync"
    prefix = f"vime_ckpt_{save_mode}_{args.save_optimizer}_{args.load_optimizer}_"
    with U.test_work_directory(
        prefix, directory=args.checkpoint_dir, keep=getattr(args, "keep_artifacts", None)
    ) as work_dir:
        checkpoint_dir = str(work_dir if args.checkpoint_dir is not None else work_dir / "checkpoint")
        prepare(checkpoint_dir)
        for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
            os.environ.pop(proxy_var, None)
        execute("async_save" if args.async_save else "save", args.save_optimizer, checkpoint_dir)
        for filename in ("queue_state_1.json", "builder_state_1.json"):
            assert (Path(checkpoint_dir) / "rollout" / filename).is_file(), f"Missing straw checkpoint: {filename}"
        verify_phase(checkpoint_dir, "save", (0, 1), args.save_optimizer)
        # Loading checkpoint 1 resumes at rollout 2; a horizon of 2 would train zero steps.
        execute("load", args.load_optimizer, checkpoint_dir)
        verify_phase(checkpoint_dir, "load", (2,), args.load_optimizer)
        print(
            f"Checkpoint roundtrip passed: save optimizer={args.save_optimizer}, load optimizer={args.load_optimizer}, async={args.async_save}"
        )


if __name__ == "__main__":
    args = parser.parse_args()
    run_roundtrip(args)
