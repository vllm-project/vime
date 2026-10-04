import json
import math
import os
from argparse import ArgumentParser
from pathlib import Path
from shlex import quote

import vime.utils.external_utils.command_utils as U


ENABLE_EVAL = bool(int(os.environ.get("VIME_TEST_ENABLE_EVAL", "1")))

MODEL_NAME = "Qwen3-4B"
MODEL_TYPE = "qwen3-4B"
NUM_GPUS = 8


parser = ArgumentParser()
parser.add_argument("--async-save", action="store_true", help="Whether to test async save/load.")
parser.add_argument(
    "--save-optimizer", choices=["cpu", "gpu", "nvme"], default="cpu", help="Optimizer placement for save."
)
parser.add_argument(
    "--load-optimizer", choices=["cpu", "gpu", "nvme"], default="cpu", help="Optimizer placement for load."
)
parser.add_argument("--checkpoint-dir", default=None, help="Directory used for the save/load checkpoint roundtrip.")
parser.add_argument(
    "--skip-prepare", action="store_true", help="Use already downloaded and converted model/data caches."
)


def default_checkpoint_dir(args):
    save_mode = "async" if args.async_save else "sync"
    return f"/root/models/{MODEL_NAME}_vime_{save_mode}_{args.save_optimizer}_save_{args.load_optimizer}_load"


def prepare(checkpoint_dir: str):
    U.exec_command("mkdir -p /root/models /root/datasets")
    U.exec_command(f"hf download Qwen/{MODEL_NAME} --local-dir /root/models/{MODEL_NAME}")
    U.exec_command(f"rm -rf {quote(checkpoint_dir)}")
    U.hf_download_dataset("zhuzilin/dapo-math-17k")
    U.hf_download_dataset("zhuzilin/aime-2024")

    U.convert_checkpoint(
        model_name=MODEL_NAME, megatron_model_type=MODEL_TYPE, num_gpus_per_node=NUM_GPUS, dir_dst="/root/models"
    )


def optimizer_args(optimizer: str, checkpoint_dir: str):
    args = (
        "--optimizer adam "
        "--lr 1e-6 "
        "--lr-decay-style constant "
        "--weight-decay 0.1 "
        "--adam-beta1 0.9 "
        "--adam-beta2 0.98 "
    )
    if optimizer == "cpu":
        args += "--use-precision-aware-optimizer "
        args += "--optimizer-cpu-offload --overlap-cpu-optimizer-d2h-h2d "
    elif optimizer == "gpu":
        args += "--use-precision-aware-optimizer "
    elif optimizer == "nvme":
        nvme_dir = quote(f"{checkpoint_dir}_nvme_scratch")
        args += (
            "--stream-optimizer-state-to-disk "
            f"--offload-train-disk-dir {nvme_dir} "
            "--offload-train-disk-chunk-mb 64 "
            "--stream-optimizer-state-moment-dtype fp32 "
        )
    return args


def assert_nvme_checkpoint(checkpoint_dir: str, iteration: int, expected_steps: int):
    base = Path(checkpoint_dir) / f"iter_{iteration:07d}"
    manifests = sorted(base.glob("rank*/opt*/manifest.json"))
    assert len(manifests) == NUM_GPUS, f"Expected {NUM_GPUS} NVMe manifests below {base}, got {len(manifests)}"
    assert {p.parent.parent.name for p in manifests} == {f"rank{rank:05d}" for rank in range(NUM_GPUS)}
    bucket_files = 0
    total_bytes = 0
    for manifest_path in manifests:
        manifest = json.loads(manifest_path.read_text())
        assert manifest["dtypes"] == {name: "torch.float32" for name in ("main", "exp_avg", "exp_avg_sq")}
        assert manifest.get("buckets"), f"Empty NVMe bucket manifest: {manifest_path}"
        for bucket in manifest["buckets"]:
            path = manifest_path.parent / bucket["file"]
            assert path.is_file(), path
            # Each parameter segment is padded to the store's 4096-byte I/O alignment.
            expected_bytes = 3 * sum((n * 4 + 4095) // 4096 * 4096 for n in bucket["entry_numels"])
            assert path.stat().st_size == expected_bytes, f"Invalid NVMe bucket size: {path}"
            assert sum(bucket["entry_numels"]) == bucket["numel"]
            assert len(bucket["state_steps"]) == len(bucket["entry_numels"])
            # Torch Adam counts per parameter; Apex FusedAdam counts per group.
            steps = bucket["state_steps"] if any(bucket["state_steps"]) else bucket["steps"]
            assert set(steps) == {expected_steps}, (path, steps)
            bucket_files += 1
            total_bytes += path.stat().st_size
    summary = dict(
        iteration=iteration,
        optimizer_steps=expected_steps,
        ranks=len(manifests),
        bucket_files=bucket_files,
        bytes=total_bytes,
    )
    print(f"NVMe checkpoint validated: {json.dumps(summary, sort_keys=True)}", flush=True)
    return summary


async def nvme_test_reward(args, sample, **kwargs):
    """Synthetic reward: keep the smoke test independent of math-task accuracy.

    A short response limit can give every completion reward zero, making GRPO
    gradients zero and leaving Adam moments untested. Alternating rewards within
    each prompt group ensure that real generated responses produce advantages.
    """
    if isinstance(sample, list):
        return [float(item.index % args.n_samples_per_prompt) for item in sample]
    return float(sample.index % args.n_samples_per_prompt)


def assert_nvme_grad_norms(checkpoint_dir: str, mode: str, expected_count: int):
    import torch

    paths = sorted(Path(checkpoint_dir + "_grad_norms").glob(f"{mode}_*.pt"))
    assert len(paths) == expected_count, (mode, paths)
    norms = [float(torch.load(path, map_location="cpu", weights_only=True)) for path in paths]
    assert all(math.isfinite(norm) and norm > 0 for norm in norms), norms
    print(f"NVMe {mode}: {len(norms)} finite, nonzero gradient norms: {norms}", flush=True)


def build_train_args(mode: str = "", optimizer: str = "cpu", checkpoint_dir: str = ""):
    ckpt_args = f"--hf-checkpoint /root/models/{MODEL_NAME}/ " f"--ref-load /root/models/{MODEL_NAME}_torch_dist "
    checkpoint_dir_arg = quote(checkpoint_dir)
    if mode == "save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
    elif mode == "async_save":
        ckpt_args += f"--save {checkpoint_dir_arg} "
        ckpt_args += "--save-interval 2 "
        ckpt_args += "--async-save "
    elif mode == "load":
        ckpt_args += f"--load {checkpoint_dir_arg} "
        ckpt_args += "--ckpt-step 1 "
        if optimizer == "nvme":
            # Persist the resumed update too: a successful process exit alone does
            # not demonstrate that Adam counters were restored rather than reset.
            ckpt_args += f"--save {quote(checkpoint_dir + '_resumed')} --save-interval 1 "

    nvme = optimizer == "nvme"
    rollout_args = (
        "--prompt-data /root/datasets/dapo-math-17k/dapo-math-17k.jsonl "
        "--input-key prompt "
        "--label-key label "
        "--apply-chat-template "
        "--rollout-shuffle "
        "--rm-type deepscaler "
        f"--num-rollout {3 if nvme else 2} "
        f"--rollout-batch-size {12 if nvme else 4} "
        f"--n-samples-per-prompt {2 if nvme else 4} "
        "--rollout-max-response-len 1024 "
        "--rollout-temperature 0.8 "
        "--global-batch-size 16 "
        "--balance-data "
    )

    if nvme:
        # 24 samples split into optimizer steps of 16 and 8 each rollout.
        rollout_args += "--variable-global-batch-size "
        rollout_args += "--custom-rm-path test_qwen3_4B_ckpt.nvme_test_reward "

    perf_args = (
        "--tensor-model-parallel-size 2 "
        "--sequence-parallel "
        "--pipeline-model-parallel-size 1 "
        "--context-parallel-size 2 "
        "--recompute-granularity full "
        "--recompute-method uniform "
        "--recompute-num-layers 1 "
        "--use-dynamic-batch-size "
        "--max-tokens-per-gpu 16384 "
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

    ci_args = "--ci-test "
    if nvme:
        grad_path = checkpoint_dir + "_grad_norms/" + mode + "_{rollout_id}_{step_id}.pt"
        ci_args += f"--ci-save-grad-norm {quote(grad_path)} "

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
        f"{optimizer_args(optimizer, checkpoint_dir)} "
        f"{ppo_args} "
        f"{U.get_default_wandb_args(__file__)} "
        f"{perf_args} "
        f"{vllm_args} "
        f"{ci_args} "
        f"{misc_args} "
    )

    return train_args


def execute(mode: str = "", optimizer: str = "cpu", checkpoint_dir: str = ""):
    U.execute_train(
        train_args=build_train_args(mode, optimizer, checkpoint_dir),
        num_gpus_per_node=NUM_GPUS,
        megatron_model_type=MODEL_TYPE,
        # Expose this test-only reward hook to the Ray workers, not production code.
        extra_env_vars=(
            {"PYTHONPATH": f"{Path(__file__).resolve().parent}:/root/Megatron-LM/"} if optimizer == "nvme" else {}
        ),
    )


if __name__ == "__main__":
    args = parser.parse_args()
    if "nvme" in (args.save_optimizer, args.load_optimizer):
        if args.async_save or args.save_optimizer != args.load_optimizer:
            parser.error("NVMe roundtrip requires synchronous saving and nvme for both optimizer placements")
    checkpoint_dir = args.checkpoint_dir or default_checkpoint_dir(args)
    if args.skip_prepare:
        for required in (
            f"/root/models/{MODEL_NAME}/config.json",
            f"/root/models/{MODEL_NAME}_torch_dist/latest_checkpointed_iteration.txt",
            "/root/datasets/dapo-math-17k/dapo-math-17k.jsonl",
        ):
            if not Path(required).is_file():
                parser.error(f"Missing prepared cache: {required}")
        if Path(checkpoint_dir).exists():
            parser.error("--skip-prepare requires a fresh --checkpoint-dir")
    else:
        prepare(checkpoint_dir)
    if args.load_optimizer == "nvme" and Path(checkpoint_dir + "_resumed").exists():
        parser.error("Resumed checkpoint directory already exists; choose a fresh --checkpoint-dir")
    if args.save_optimizer == "nvme":
        Path(checkpoint_dir + "_grad_norms").mkdir()
    for proxy_var in ("http_proxy", "https_proxy", "HTTP_PROXY", "HTTPS_PROXY"):
        os.environ.pop(proxy_var, None)
    execute(
        "save" if not args.async_save else "async_save",
        optimizer=args.save_optimizer,
        checkpoint_dir=checkpoint_dir,
    )
    if args.save_optimizer == "nvme":
        assert_nvme_grad_norms(checkpoint_dir, "save", expected_count=6)
        assert_nvme_checkpoint(checkpoint_dir, iteration=1, expected_steps=4)
        assert_nvme_checkpoint(checkpoint_dir, iteration=2, expected_steps=6)
    execute("load", optimizer=args.load_optimizer, checkpoint_dir=checkpoint_dir)
    if args.load_optimizer == "nvme":
        assert_nvme_grad_norms(checkpoint_dir, "load", expected_count=2)
        assert_nvme_checkpoint(checkpoint_dir + "_resumed", iteration=2, expected_steps=6)
        print("NVMe train/rollout/checkpoint/resume roundtrip: PASS", flush=True)
