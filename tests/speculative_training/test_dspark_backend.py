from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vime.backends.speculative_training.backends import dspark
from vime.backends.speculative_training.backends.dspark import (
    collate_dspark_samples,
    compute_dspark_loss,
    dspark_trainer_kwargs,
    sync_speculator_lm_heads,
)
from vime.backends.speculative_training.feature_schema import DraftFeatureSample


def _sample(sample_id: str, offset: int = 0) -> DraftFeatureSample:
    rows = 10
    positions = torch.arange(offset, offset + rows)
    return DraftFeatureSample(
        input_ids=torch.arange(10, 10 + rows),
        loss_mask=torch.tensor([0] + [1] * (rows - 1), dtype=torch.float32),
        position_ids=positions,
        aux_hidden_states=torch.randn(rows, 6),
        final_hidden_states=torch.randn(rows, 2),
        rollout_id=1,
        target_weight_version="4",
        original_sample_id=sample_id,
        prompt_length=1,
        response_length=rows - 1,
        window_start=offset,
        window_end=offset + rows,
        aux_layer_ids=(2, 8, 13),
        algorithm="dspark",
    )


@pytest.mark.unit
def test_dspark_collator_isolates_documents_and_masks_each_tail():
    batch = collate_dspark_samples([_sample("a"), _sample("b", 20)], "cpu", block_size=3)

    assert tuple(batch["input_ids"].shape) == (1, 20)
    assert batch["document_ids"].tolist() == [[0] * 10 + [1] * 10]
    assert batch["loss_mask"][0, 7:10].tolist() == [0, 0, 0]
    assert batch["loss_mask"][0, 17:20].tolist() == [0, 0, 0]
    assert batch["position_ids"].tolist() == [list(range(10)) + list(range(20, 30))]


class _FakeDSpark(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = torch.nn.Parameter(torch.tensor(2.0))

    def forward(self, **batch):
        loss = self.scale.square()
        device = batch["input_ids"].device
        metrics = {
            "loss_sum": torch.tensor(99.0, device=device),
            "loss_total": torch.tensor(1.0, device=device),
            "full_acc_sum": torch.tensor(3.0, device=device),
            "full_acc_total": torch.tensor(5.0, device=device),
            "accept_rate_sum": torch.tensor(4.0, device=device),
            "accept_rate_total": torch.tensor(5.0, device=device),
            "ce_loss_sum": torch.tensor(2.0, device=device),
            "ce_loss_total": torch.tensor(1.0, device=device),
            "tv_loss_sum": torch.tensor(6.0, device=device),
            "tv_loss_total": torch.tensor(2.0, device=device),
            "confidence_loss_sum": torch.tensor(3.0, device=device),
            "confidence_loss_total": torch.tensor(1.0, device=device),
        }
        return None, loss, metrics


@pytest.mark.unit
def test_dspark_loss_adapts_speculators_metrics():
    batch = collate_dspark_samples([_sample("a")], "cpu", block_size=3)

    loss, metrics = compute_dspark_loss(_FakeDSpark(), batch, {})

    assert loss.item() == 4.0
    assert metrics["loss_sum"].item() == 20.0
    assert metrics["token_count"].item() == 5.0
    assert metrics["top1_correct"].item() == 3.0
    assert metrics["accept_rate_sum"].item() == 4.0
    assert metrics["ce_loss_sum"].item() == 2.0
    assert metrics["ce_loss_total"].item() == 1.0
    assert metrics["tv_loss_sum"].item() == 6.0
    assert metrics["tv_loss_total"].item() == 2.0
    assert metrics["confidence_loss_sum"].item() == 3.0
    assert metrics["confidence_loss_total"].item() == 1.0


@pytest.mark.unit
def test_dspark_syncs_restricted_draft_and_verifier_heads():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 3, bias=False)
    model.verifier_lm_head = torch.nn.Linear(2, 3, bias=False)
    target = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    sync_speculator_lm_heads(model, target, torch.tensor([0, 2, 4]))

    expected = target[[0, 2, 4]]
    assert torch.equal(model.lm_head.weight, expected)
    assert torch.equal(model.verifier_lm_head.weight, expected)


@pytest.mark.unit
def test_eagle3_sync_touches_only_the_verifier_head():
    model = torch.nn.Module()
    model.lm_head = torch.nn.Linear(2, 3, bias=False)
    model.verifier_lm_head = torch.nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.lm_head.weight.fill_(7.0)
        model.verifier_lm_head.weight.fill_(0.0)
    target = torch.arange(10, dtype=torch.float32).reshape(5, 2)

    sync_speculator_lm_heads(
        model,
        target,
        torch.tensor([1, 3]),
        heads=("verifier_lm_head",),
    )

    assert torch.equal(model.verifier_lm_head.weight, target[[1, 3]])
    assert bool((model.lm_head.weight == 7.0).all())


class _TrainerKwargsModel(torch.nn.Module):
    def __init__(self, loss_config, *, supports_explicit_loss=True):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.loss_config = loss_config
        self.supports_explicit_loss = supports_explicit_loss
        self.resolved_kwargs = None

    def get_trainer_kwargs(self, **kwargs):
        self.resolved_kwargs = kwargs
        train_kwargs = {
            "loss_config": self.loss_config,
            "max_anchors": kwargs["max_anchors"],
        }
        if self.supports_explicit_loss:
            train_kwargs["tv_loss_fn"] = kwargs["loss_implementation"]
        return train_kwargs, {}


def _trainer_args():
    return type(
        "Args",
        (),
        {
            "draft_dspark_loss_fn": '{"ce": 0.1, "tv": 0.9}',
            "draft_dspark_decay_gamma": 4.0,
            "draft_dspark_max_anchors": 64,
            "draft_dspark_confidence_head_alpha": 1.0,
            "draft_dspark_per_position_loss_weight": "fixed-exp-decay",
            "draft_dspark_dpace_alpha": 0.5,
        },
    )()


@pytest.mark.unit
@pytest.mark.parametrize(("npu", "implementation"), [(True, "eager"), (False, "fused")])
def test_dspark_selects_native_loss_implementation(monkeypatch, npu, implementation):
    model = _TrainerKwargsModel({})
    monkeypatch.setattr(dspark, "current_platform", lambda: SimpleNamespace(is_npu=npu))

    train_kwargs = dspark_trainer_kwargs(model, _trainer_args())

    assert model.resolved_kwargs["loss_implementation"] == implementation
    assert train_kwargs["tv_loss_fn"] == implementation


@pytest.mark.unit
def test_dspark_rejects_legacy_fused_loss_resolver_on_npu(monkeypatch):
    model = _TrainerKwargsModel({}, supports_explicit_loss=False)
    monkeypatch.setattr(dspark, "current_platform", lambda: SimpleNamespace(is_npu=True))

    with pytest.raises(RuntimeError, match="explicit eager DSpark losses"):
        dspark_trainer_kwargs(model, _trainer_args())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
