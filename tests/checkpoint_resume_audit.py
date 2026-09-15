"""Observation-only fixed-batch resume audit, loaded through Vime's public hook."""

import ast
import hashlib
import json
import math
import os
import pickle
import random
import re
import time
from pathlib import Path


def tensor_record(tensor):
    import torch

    value = tensor.detach().cpu().contiguous()
    return {
        "shape": list(value.shape),
        "dtype": str(value.dtype),
        "sha256": hashlib.sha256(value.reshape(-1).view(torch.uint8).numpy().tobytes()).hexdigest(),
    }


def tree_record(value):
    import torch

    if torch.is_tensor(value):
        return tensor_record(value)
    if isinstance(value, dict):
        return {str(k): tree_record(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [tree_record(v) for v in value]
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise TypeError(f"Unsupported state: {type(value)}")


def emit(kind, **data):
    path = Path(os.environ["VIME_RESUME_AUDIT_DIR"]) / f"resume-{os.getpid()}.jsonl"
    with path.open("a") as stream:
        stream.write(json.dumps({"kind": kind, "pid": os.getpid(), "time": time.time(), **data}) + "\n")


def rng_record():
    import numpy as np
    import torch
    from megatron.core.tensor_parallel.random import get_cuda_rng_tracker

    return {
        "python": hashlib.sha256(pickle.dumps(random.getstate())).hexdigest(),
        "numpy": hashlib.sha256(pickle.dumps(np.random.get_state())).hexdigest(),
        "torch_cpu": tensor_record(torch.get_rng_state()),
        "torch_cuda": tensor_record(torch.cuda.get_rng_state()),
        "megatron": tree_record(get_cuda_rng_tracker().get_states()),
    }


def state_record(model, optimizer, scheduler):
    optimizers = getattr(optimizer, "chained_optimizers", [optimizer])
    states = []
    for opt in optimizers:
        inner = opt.optimizer
        states.append(
            {
                "class": type(opt).__name__,
                "inner_class": type(inner).__name__,
                "state": tree_record(inner.state_dict()),
                "master_parameters": tree_record([p for group in inner.param_groups for p in group["params"]]),
            }
        )
    return {
        "model": {f"{i}/{name}": tensor_record(p) for i, m in enumerate(model) for name, p in m.named_parameters()},
        "optimizer": states,
        "scheduler": tree_record(scheduler.state_dict()),
        "rng": rng_record(),
    }


def before_train(args, rollout_id, step_id, model, optimizer, scheduler):
    import torch

    emit(
        "before_update",
        rollout_id=rollout_id,
        step_id=step_id,
        state=state_record(model, optimizer, scheduler),
        load_flags={
            k: getattr(args, k, None) for k in ["load", "no_load_optim", "no_load_rng", "finetune", "offload_train"]
        },
    )
    optimizer._resume_audit_id = (rollout_id, step_id)
    if getattr(optimizer, "_resume_audit_installed", False):
        return
    optimizer._resume_audit_installed = True
    original_step = optimizer.step
    original_scheduler_step = scheduler.step

    def audited_step(*a, **kw):
        rid, sid = optimizer._resume_audit_id
        gradients = {}
        for i, m in enumerate(model):
            for name, p in m.named_parameters():
                grad = getattr(p, "main_grad", None)
                if grad is None:
                    grad = p.grad
                if grad is not None:
                    gradients[f"{i}/{name}"] = tensor_record(grad)
        emit("gradients", rollout_id=rid, step_id=sid, tensors=gradients)
        result = original_step(*a, **kw)
        emit("optimizer_result", rollout_id=rid, step_id=sid, result=tree_record(result))
        return result

    def audited_scheduler_step(*a, **kw):
        result = original_scheduler_step(*a, **kw)
        rid, sid = optimizer._resume_audit_id
        torch.cuda.synchronize()
        emit("after_update", rollout_id=rid, step_id=sid, state=state_record(model, optimizer, scheduler))
        return result

    optimizer.step = audited_step
    scheduler.step = audited_scheduler_step


def require_equal(left, right, path="root"):
    if type(left) is not type(right):
        raise AssertionError(f"{path}: types differ: {type(left).__name__} / {type(right).__name__}")
    if isinstance(left, dict):
        if left.keys() != right.keys():
            raise AssertionError(f"{path}: keys differ: {left.keys() ^ right.keys()}")
        for key in left:
            require_equal(left[key], right[key], f"{path}.{key}")
    elif isinstance(left, list):
        if len(left) != len(right):
            raise AssertionError(f"{path}: lengths differ")
        for index, (a, b) in enumerate(zip(left, right, strict=True)):
            require_equal(a, b, f"{path}[{index}]")
    elif left != right:
        raise AssertionError(f"{path}: values differ: {str(left)[:90]} / {str(right)[:90]}")


def read_run(root, expected_ids):
    records = [json.loads(line) for p in root.glob("resume-*.jsonl") for line in p.read_text().splitlines()]
    by_kind = {}
    pids = set()
    for kind in ["before_update", "gradients", "optimizer_result", "after_update"]:
        rows = [row for row in records if row["kind"] == kind]
        ids = [row["rollout_id"] for row in rows]
        require_equal(sorted(ids), list(expected_ids), f"{root.name}.{kind}.rollout_ids")
        for row in rows:
            assert row["step_id"] == 0, "This fixture expects one optimizer update per rollout"
            pids.add(row["pid"])
        by_kind[kind] = {row["rollout_id"]: row for row in rows}
    assert len(pids) == 1, f"Expected a single trainer process, got {pids}"
    assert json.loads((root / "train-returned.json").read_text())["completed"]
    metrics = {}
    for line in (root / "train.log").read_text().splitlines():
        match = re.search(r"step (\d+): (\{'train/loss'.*)", line)
        if match:
            row = ast.literal_eval(match.group(2))
            metrics[int(match.group(1))] = row
    require_equal(sorted(metrics), list(expected_ids), f"{root.name}.metric_steps")
    for rid in expected_ids:
        row = metrics[rid]
        assert all(math.isfinite(v) for v in row.values() if isinstance(v, (float, int)))
        assert row["train/grad_norm"] > 0
        assert by_kind["gradients"][rid]["tensors"], "Missing gradient evidence"
        assert by_kind["before_update"][rid]["state"]["model"], "Missing model evidence"
        assert by_kind["after_update"][rid]["state"]["optimizer"], "Missing optimizer evidence"
        assert by_kind["optimizer_result"][rid]["result"][0] is True
        assert (root / f"dumps/train_data/{rid}.pt").is_file(), "Missing fixed-batch evidence"
        assert (
            by_kind["after_update"][rid]["state"]["model"] != by_kind["before_update"][rid]["state"]["model"]
        ), "No parameter update"
    return by_kind, metrics, pids.pop()


def verify(continuous, first, resumed, split=4, steps=8):
    import torch

    assert 0 < split < steps
    baseline, base_metrics, a_pid = read_run(continuous, range(steps))
    before, before_metrics, b_pid = read_run(first, range(split))
    after, after_metrics, c_pid = read_run(resumed, range(split, steps))
    assert len({a_pid, b_pid, c_pid}) == 3, "Runs must use distinct trainer processes"
    for rid in range(steps):
        branch, metrics, root = (before, before_metrics, first) if rid < split else (after, after_metrics, resumed)
        for kind, value_key in [
            ("before_update", "state"),
            ("gradients", "tensors"),
            ("optimizer_result", "result"),
            ("after_update", "state"),
        ]:
            require_equal(baseline[kind][rid][value_key], branch[kind][rid][value_key], f"rollout{rid}.{kind}")
        require_equal(base_metrics[rid], metrics[rid], f"rollout{rid}.metrics")
        a = torch.load(continuous / f"dumps/train_data/{rid}.pt", map_location="cpu", weights_only=False)
        b = torch.load(root / f"dumps/train_data/{rid}.pt", map_location="cpu", weights_only=False)
        require_equal(tree_record(a), tree_record(b), f"rollout{rid}.training_data")
    flags = after["before_update"][split]["load_flags"]
    for name in ["no_load_optim", "no_load_rng", "finetune"]:
        assert flags[name] is False, f"Restore bypassed {name}: {flags}"
    require_equal(
        before["after_update"][split - 1]["state"], after["before_update"][split]["state"], "checkpoint_boundary"
    )
    return {
        "status": "FIXED_BATCH_FRESH_PROCESS_RESUME_EXACT",
        "steps": steps,
        "split": split,
        "trainer_pids": [a_pid, b_pid, c_pid],
        "model_parameters": len(baseline["before_update"][0]["state"]["model"]),
        "compared": [
            "model",
            "FP32 master parameters",
            "Adam moments and step",
            "scheduler",
            "RNG",
            "gradients",
            "optimizer results",
            "loss metrics",
            "training tokens masks logprobs advantages and schedule",
        ],
        "offload": flags["offload_train"],
        "tolerance": "bitwise state and tensor hashes; exact scalar equality",
        "scope": "TP=PP=CP=1, dense fixed-batch training. Live serving synchronization is a separate integration check.",
    }
