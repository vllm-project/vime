"""Negative controls for the resume evidence verifier; CPU only."""

import json

import pytest
import torch

from tests.checkpoint_resume_audit import tensor_record, verify


@pytest.fixture
def evidence(tmp_path):
    roots = [tmp_path / name for name in ["continuous", "first", "resumed"]]
    for root, ids, pid in zip(roots, [range(2), range(1), range(1, 2)], [101, 102, 103], strict=True):
        (root / "dumps/train_data").mkdir(parents=True)
        (root / "train-returned.json").write_text('{"completed": true}')
        records, logs = [], []

        def state(step):
            return {
                "model": {"weight": tensor_record(torch.tensor([step], dtype=torch.bfloat16))},
                "optimizer": [
                    {
                        "class": "DistributedOptimizer",
                        "inner_class": "AdamW",
                        "master_parameters": [tensor_record(torch.tensor([step], dtype=torch.float32))],
                        "state": {"step": step, "exp_avg": step / 10, "exp_avg_sq": step / 100},
                    }
                ],
                "scheduler": {"num_steps": step * 8, "max_lr": 0.001},
                "rng": {"torch_cpu": "a", "torch_cuda": "b", "python": "c", "numpy": "d"},
            }

        for rid in ids:
            common = {"pid": pid, "rollout_id": rid, "step_id": 0}
            records.extend(
                [
                    {
                        **common,
                        "kind": "before_update",
                        "state": state(rid),
                        "load_flags": {
                            "no_load_optim": False,
                            "no_load_rng": False,
                            "finetune": False,
                            "offload_train": True,
                        },
                    },
                    {**common, "kind": "gradients", "tensors": {"weight": tensor_record(torch.tensor([rid + 0.5]))}},
                    {**common, "kind": "optimizer_result", "result": [True, 0.5, 0]},
                    {**common, "kind": "after_update", "state": state(rid + 1)},
                ]
            )
            logs.append(f"step {rid}: {{'train/loss': 0.25, 'train/grad_norm': 0.5, 'train/step': {rid}}}")
            torch.save(
                {
                    "tokens": torch.tensor([1, 2]),
                    "loss_masks": torch.tensor([0, 1]),
                    "log_probs": torch.tensor([-0.5]),
                    "advantages": torch.tensor([0.3]),
                    "rollout_id": rid,
                },
                root / f"dumps/train_data/{rid}.pt",
            )
        (root / "resume-audit.jsonl").write_text("\n".join(json.dumps(x) for x in records) + "\n")
        (root / "train.log").write_text("\n".join(logs) + "\n")
    return roots


def mutate_record(root, kind, mutate):
    path = root / "resume-audit.jsonl"
    records = [json.loads(line) for line in path.read_text().splitlines()]
    row = next(row for row in records if row["kind"] == kind)
    mutate(row)
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")


def test_accepts_complete_exact_resume(evidence):
    assert verify(*evidence, split=1, steps=2)["status"] == "FIXED_BATCH_FRESH_PROCESS_RESUME_EXACT"


@pytest.mark.parametrize("component", ["model", "optimizer", "scheduler", "rng"])
@pytest.mark.parametrize("kind", ["before_update", "after_update"])
def test_rejects_state_changes(evidence, component, kind):
    mutate_record(evidence[2], kind, lambda row: row["state"].__setitem__(component, {"corrupted": True}))
    with pytest.raises(AssertionError, match=component):
        verify(*evidence, split=1, steps=2)


@pytest.mark.parametrize("name", ["no_load_optim", "no_load_rng", "finetune"])
def test_rejects_resume_bypass(evidence, name):
    mutate_record(evidence[2], "before_update", lambda row: row["load_flags"].__setitem__(name, True))
    with pytest.raises(AssertionError, match="Restore bypassed"):
        verify(*evidence, split=1, steps=2)


def test_rejects_missing_gradient_evidence(evidence):
    mutate_record(evidence[2], "gradients", lambda row: row.__setitem__("tensors", {}))
    with pytest.raises(AssertionError, match="Missing gradient"):
        verify(*evidence, split=1, steps=2)


def test_rejects_duplicate_update(evidence):
    path = evidence[2] / "resume-audit.jsonl"
    lines = path.read_text().splitlines()
    path.write_text("\n".join(lines + [lines[0]]) + "\n")
    with pytest.raises(AssertionError, match="rollout_ids"):
        verify(*evidence, split=1, steps=2)


@pytest.mark.parametrize("key", ["tokens", "loss_masks", "log_probs", "advantages"])
def test_rejects_changed_training_batch(evidence, key):
    path = evidence[2] / "dumps/train_data/1.pt"
    batch = torch.load(path, weights_only=False)
    batch[key] = batch[key] + 1
    torch.save(batch, path)
    with pytest.raises(AssertionError, match="training_data"):
        verify(*evidence, split=1, steps=2)


def test_rejects_changed_loss(evidence):
    path = evidence[2] / "train.log"
    path.write_text(path.read_text().replace("0.25", "0.26"))
    with pytest.raises(AssertionError, match="metrics"):
        verify(*evidence, split=1, steps=2)


def test_scalar_and_bfloat16_hashing():
    assert tensor_record(torch.tensor(1.0))["shape"] == []
    assert tensor_record(torch.tensor([1.0], dtype=torch.bfloat16)) != tensor_record(
        torch.tensor([2.0], dtype=torch.bfloat16)
    )
