"""Fixed-batch checkpoint resume runner; see tests/checkpoint_resume_parity.md."""

import argparse
import importlib.util
import json
import os
import sys
import tempfile
from pathlib import Path


def record(args):
    import ray

    checkout = Path(__file__).resolve().parents[1]
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=False)
    tokens = json.loads(args.train_args.read_text())
    assert isinstance(tokens, list) and all(isinstance(t, str) for t in tokens)
    assert "--load-debug-rollout-data" in tokens, "Use saved rollout batches for resume parity"
    assert "--num-rollout" in tokens and "--load" in tokens

    def setarg(flag, value):
        if flag in tokens:
            tokens[tokens.index(flag) + 1] = str(value)
        else:
            tokens.extend([flag, str(value)])

    setarg("--custom-megatron-before-train-step-hook-path", "tests.checkpoint_resume_audit.before_train")
    setarg("--dump-details", output / "dumps")
    setarg("--save", output / "checkpoints")
    setarg("--save-interval", args.split)
    setarg("--num-gpus-per-node", 1)
    setarg("--actor-num-gpus-per-node", 1)
    for flag in ["--tensor-model-parallel-size", "--pipeline-model-parallel-size", "--context-parallel-size"]:
        setarg(flag, 1)
    os.environ["VIME_RESUME_AUDIT_DIR"] = str(output)
    (output / "train-args.json").write_text(json.dumps(tokens, indent=2) + "\n")
    sys.argv = [str(checkout / "train.py")] + tokens
    print(f"Recording training and audit logs in {output}", flush=True)
    saved_fds = [os.dup(1), os.dup(2)]
    log = (output / "train.log").open("w")
    os.dup2(log.fileno(), 1)
    os.dup2(log.fileno(), 2)
    # Each invocation owns a fresh Ray head. Shared-machine supervisors should
    # bind an available GPU before launching and clean up only this invocation.
    try:
        ray_temp = tempfile.mkdtemp(prefix="vime-resume-ray-")
        (output / "ray-temp-dir.txt").write_text(ray_temp + "\n")
        ray.init(
            address="local",
            num_cpus=12,
            num_gpus=1,
            include_dashboard=False,
            object_store_memory=2 * 1024**3,
            namespace=output.name,
            _temp_dir=ray_temp,
            _node_ip_address="127.0.0.1",
            runtime_env={"env_vars": {"VIME_RESUME_AUDIT_DIR": str(output)}},
        )
        spec = importlib.util.spec_from_file_location("resume_parity_train", checkout / "train.py")
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        train_args = module.parse_args()
        if args.stop_after is not None:
            assert 0 < args.stop_after < train_args.num_rollout
            # Do not change num_rollout: the scheduler horizon must be identical
            # in the continuous run, split prefix, and resumed suffix.
            module.range = lambda start, stop: range(start, min(stop, args.stop_after))
        module.train(train_args)
        (output / "train-returned.json").write_text(json.dumps({"completed": True}))
    finally:
        ray.shutdown()
        sys.stdout.flush()
        sys.stderr.flush()
        for fd, original in zip([1, 2], saved_fds, strict=True):
            os.dup2(original, fd)
            os.close(original)
        log.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="action", required=True)
    run = subparsers.add_parser("record")
    run.add_argument("--train-args", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--split", type=int, default=4)
    run.add_argument("--stop-after", type=int)
    compare = subparsers.add_parser("compare")
    for name in ["continuous", "first", "resumed", "output"]:
        compare.add_argument("--" + name, type=Path, required=True)
    compare.add_argument("--split", type=int, default=4)
    compare.add_argument("--steps", type=int, default=8)
    args = parser.parse_args()
    if args.action == "record":
        record(args)
    else:
        from tests.checkpoint_resume_audit import verify

        result = verify(args.continuous, args.first, args.resumed, args.split, args.steps)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
