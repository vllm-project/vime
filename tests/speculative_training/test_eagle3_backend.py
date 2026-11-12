from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from vime.backends.speculative_training.backends import eagle3
from vime.backends.speculative_training.backends.eagle3 import (
    align_eagle3_sample,
    collate_eagle3_samples,
    compute_eagle3_loss,
    eagle3_trainer_kwargs,
)
from vime.backends.speculative_training.feature_schema import DraftFeatureSample


def _sample(offset: int = 0) -> DraftFeatureSample:
    rows = 6
    aux = torch.arange(rows * 4, dtype=torch.float32).reshape(rows, 4) + offset
    final = torch.arange(rows * 3, dtype=torch.float32).reshape(rows, 3) + offset
    return DraftFeatureSample(
        input_ids=torch.tensor([10, 11, 12, 13, 14, 15]) + offset,
        loss_mask=torch.tensor([0, 0, 1, 1, 1, 1], dtype=torch.float32),
        position_ids=torch.arange(20 + offset, 20 + offset + rows),
        aux_hidden_states=aux,
        final_hidden_states=final,
        rollout_id=0,
        target_weight_version="1",
        original_sample_id=f"s{offset}",
        prompt_length=2,
        response_length=4,
        window_start=offset,
        window_end=offset + rows,
        aux_layer_ids=(1,),
    )


@pytest.mark.unit
def test_eagle3_alignment_is_p_p1_p2():
    sample = _sample()
    aligned = align_eagle3_sample(sample)

    assert aligned.input_ids.tolist() == [11, 12, 13, 14]
    assert torch.equal(aligned.hidden_states, sample.aux_hidden_states[:4])
    assert torch.equal(aligned.last_hidden_states, sample.final_hidden_states[1:5])
    assert aligned.loss_mask.tolist() == [1, 1, 1, 1]
    assert aligned.position_ids.tolist() == [20, 21, 22, 23]


@pytest.mark.unit
def test_eagle3_collator_packs_documents_into_one_sequence():
    batch = collate_eagle3_samples([_sample(0), _sample(6)], device="cpu")

    assert batch["input_ids"].shape == (1, 8)
    assert batch["input_ids"].tolist() == [[11, 12, 13, 14, 17, 18, 19, 20]]
    assert batch["document_ids"].tolist() == [[0, 0, 0, 0, 1, 1, 1, 1]]
    assert batch["position_ids"].tolist() == [[20, 21, 22, 23, 26, 27, 28, 29]]
    assert batch["hidden_states"].shape == (1, 8, 4)
    assert batch["verifier_last_hidden_states"].shape == (1, 8, 3)
    assert batch["loss_mask"].dtype == torch.float32
    assert batch["loss_mask"].tolist() == [[1, 1, 1, 1, 1, 1, 1, 1]]
    assert torch.equal(
        batch["verifier_last_hidden_states"][0, 4:],
        _sample(6).final_hidden_states[1:5],
    )


@pytest.mark.unit
def test_eagle3_collator_masks_leading_rows_of_later_documents():
    batch = collate_eagle3_samples([_sample(0), _sample(6)], device="cpu", ttt_steps=3)

    assert batch["loss_mask"].tolist() == [[1, 1, 1, 1, 0, 0, 1, 1]]


@pytest.mark.unit
def test_eagle3_collator_rejects_other_algorithms():
    sample = _sample()
    sample.algorithm = "dspark"
    with pytest.raises(ValueError, match="dspark"):
        collate_eagle3_samples([sample], device="cpu")


@pytest.mark.unit
def test_eagle3_collator_rejects_empty_batches_and_bad_steps():
    with pytest.raises(ValueError, match="empty"):
        collate_eagle3_samples([], device="cpu")
    with pytest.raises(ValueError, match="TTT steps"):
        collate_eagle3_samples([_sample()], device="cpu", ttt_steps=0)


class _SpeculatorsDraft(torch.nn.Module):
    """Mimics Eagle3DraftModel.forward returning (tokens, loss, metrics)."""

    def __init__(self, raw_metrics: dict) -> None:
        super().__init__()
        self.raw_metrics = raw_metrics
        self.seen_batches: list[dict] = []

    def forward(self, **kwargs):
        self.seen_batches.append(kwargs)
        loss = torch.tensor(2.0, requires_grad=True)
        return kwargs["input_ids"].clone(), loss, self.raw_metrics


@pytest.mark.unit
def test_eagle3_loss_adapts_speculators_metrics():
    raw_metrics = {
        "full_acc_0_sum": torch.tensor(3.0),
        "full_acc_0_total": torch.tensor(4.0),
        "full_acc_1_sum": torch.tensor(1.0),
        "full_acc_1_total": torch.tensor(4.0),
    }
    model = _SpeculatorsDraft(raw_metrics)
    batch = collate_eagle3_samples([_sample()], device="cpu")
    train_kwargs = {"ttt_steps": 2, "loss_config": {"kl_div": (None, 1.0)}}

    loss, metrics = compute_eagle3_loss(model, batch, train_kwargs)

    assert loss.item() == 2.0
    assert metrics["token_count"].item() == 8.0
    assert metrics["top1_correct"].item() == 4.0
    assert metrics["top5_correct"].item() == 0.0
    assert metrics["loss_sum"].item() == 16.0
    seen = model.seen_batches[0]
    for key in ("input_ids", "hidden_states", "loss_mask", "document_ids", "position_ids"):
        assert key in seen
    assert seen["ttt_steps"] == 2
    assert seen["loss_config"] == {"kl_div": (None, 1.0)}


@pytest.mark.unit
def test_eagle3_loss_falls_back_to_mask_token_count():
    model = _SpeculatorsDraft({})
    batch = collate_eagle3_samples([_sample()], device="cpu")

    loss, metrics = compute_eagle3_loss(model, batch, {"ttt_steps": 1})

    assert metrics["token_count"].item() == 4.0
    assert metrics["loss_sum"].item() == 8.0
    assert metrics["top1_correct"].item() == 0.0


class _TrainerKwargsModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))
        self.resolved_kwargs = None

    def get_trainer_kwargs(self, **kwargs):
        self.resolved_kwargs = kwargs
        train_kwargs = {
            "ttt_steps": kwargs["ttt_steps"],
            "ttt_step_loss_decay": kwargs["ttt_step_loss_decay"],
            "loss_config": {"kl_div": (None, 1.0)},
        }
        return train_kwargs, {}


def _trainer_args():
    return type(
        "Args",
        (),
        {
            "draft_eagle3_loss_fn": "kl_div",
            "draft_ttt_length": 1,
            "draft_temporal_decay": 0.8,
        },
    )()


@pytest.mark.unit
@pytest.mark.parametrize(("npu", "implementation"), [(True, "eager"), (False, "fused")])
def test_eagle3_selects_native_loss_implementation(monkeypatch, npu, implementation):
    model = _TrainerKwargsModel()
    monkeypatch.setattr(eagle3, "current_platform", lambda: SimpleNamespace(is_npu=npu))

    train_kwargs = eagle3_trainer_kwargs(model, _trainer_args())

    assert model.resolved_kwargs["loss_implementation"] == implementation
    assert model.resolved_kwargs["loss_fn"] == "kl_div"
    assert train_kwargs["ttt_steps"] == 1
    assert train_kwargs["ttt_step_loss_decay"] == 0.8


@pytest.mark.unit
def test_eagle3_rejects_models_without_trainer_kwargs_resolver():
    model = torch.nn.Module()
    with pytest.raises(TypeError, match="get_trainer_kwargs"):
        eagle3_trainer_kwargs(model, _trainer_args())


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
