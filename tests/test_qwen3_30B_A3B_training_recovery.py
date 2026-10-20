"""MoE/R3 recovery with stateless Adam: OOM, manager loss, resharding and route replay."""

import argparse
import runpy
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U

MODEL_NAME = "Qwen3-30B-A3B"
NUM_GPUS = 8
TEST_ROOT = U.get_test_root()
MODEL_DIR = TEST_ROOT / "models" / MODEL_NAME
DATASET_DIR = TEST_ROOT / "datasets"


def prepare():
    U.exec_command(f"mkdir -p {quote(str(MODEL_DIR.parent))} {quote(str(DATASET_DIR))}")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir {quote(str(MODEL_DIR))}")
    U.hf_download_dataset("zhuzilin/gsm8k", dir_dst=DATASET_DIR)


def execute(
    directory=None, model_path=None, dataset_path=None, pd=False, keep_artifacts=None, job_timeout_seconds=None
):
    npu = U.current_platform().name == "npu"
    suite = runpy.run_path(str(Path(__file__).with_name("test_qwen2.5_0.5B_training_recovery.py")))
    suite["execute"](
        directory=directory,
        model_path=model_path or str(MODEL_DIR),
        dataset_path=dataset_path or str(DATASET_DIR / "gsm8k/train.parquet"),
        failure_rollout=1,
        save_interval=1,
        kill_manager=True,
        manager_crash_phase="training",
        fault_tolerance=False,
        model_type="qwen3-30B-A3B",
        # ETP1 experts shard over EP, not dense TP. Eight expert ranks halve
        # the FP32 expert-gradient buffer that OOMed before fault injection.
        train_gpus=8 if npu else 4,
        rollout_gpus=4,
        pd=pd,
        keep_artifacts=keep_artifacts,
        job_timeout_seconds=(
            job_timeout_seconds
            if job_timeout_seconds is not None
            else (10800 if U.current_platform().name == "npu" else 3600)
        ),
        extra_args=(
            ("--num-gpus-per-node 16 " if npu else "")
            + f"--rollout-num-gpus-per-engine 2 --expert-model-parallel-size {8 if npu else 4} "
            # Exercise recovery without Adam moments; scheduler/RNG still resume
            # with the model, and accepted R3 routes must replay byte-for-byte.
            "--use-stateless-adam --no-save-optim "
            "--use-rollout-routing-replay --optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
            "--use-precision-aware-optimizer --recompute-granularity full --recompute-method uniform "
            "--recompute-num-layers 1 --vllm-server-concurrency 32 "
            # Parallelize the CPU master-parameter updates for this full-size model.
            '--train-env-vars \'{"OMP_NUM_THREADS":"8"}\' '
            # Keep communication scratch bounded across repeated training calls.
            "--overlap-grad-reduce --ddp-bucket-size 40000000 "
            "--moe-token-dispatcher-type alltoall "
        ),
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-prepare", action="store_true")
    parser.add_argument("--directory", help="Explicit artifact directory; always preserved.")
    parser.add_argument("--keep-artifacts", action="store_true", default=None)
    parser.add_argument(
        "--job-timeout-seconds",
        type=int,
        default=None,
        help="Per Ray job; NPU defaults to 10800 for full-size model checkpoints.",
    )
    parser.add_argument("--model-path")
    parser.add_argument("--dataset-path")
    parser.add_argument("--pd", action="store_true")
    args = parser.parse_args()
    if not args.no_prepare:
        prepare()
    execute(args.directory, args.model_path, args.dataset_path, args.pd, args.keep_artifacts, args.job_timeout_seconds)
