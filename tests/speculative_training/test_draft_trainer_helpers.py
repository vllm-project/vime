import json
from argparse import Namespace
from types import SimpleNamespace

import pytest
import torch

from vime.backends.speculative_training.draft_trainer import (
    ExternalDraftTrainer,
    _architecture_fingerprint,
    _load_target_embedding,
)


class _Draft(torch.nn.Module):
    def __init__(self, rows=4, hidden=3):
        super().__init__()
        self.embed_tokens = torch.nn.Embedding(rows, hidden)
        self.proj = torch.nn.Linear(hidden, hidden, bias=False)


def _serialized_dspark_config() -> str:
    return json.dumps(
        {
            "speculators_model_type": "dspark",
            "speculators_config": None,
            "aux_hidden_state_layer_ids": [2, 4],
            "block_size": 7,
            "mask_token_id": 3,
            "markov_rank": 2,
            "markov_head_type": "vanilla",
            "enable_confidence_head": True,
            "confidence_head_with_markov": True,
            "sample_from_anchor": True,
            "transformer_layer_config": {
                "model_type": "qwen3",
                "hidden_size": 3,
                "intermediate_size": 6,
                "num_hidden_layers": 1,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "vocab_size": 4,
            },
        }
    )


def _flat_dspark_config() -> dict:
    return {
        "architectures": ["Qwen3DSparkModel"],
        "attention_bias": False,
        "block_size": 7,
        "confidence_head_with_markov": True,
        "dtype": "bfloat16",
        "enable_confidence_head": True,
        "hidden_size": 3,
        "model_type": "qwen3",
        "num_anchors": 512,
        "num_target_layers": 4,
        "rope_parameters": {"rope_theta": 1000000, "rope_type": "default"},
        "target_layer_ids": [1, 3],
        "transformers_version": "5.10.2",
        "vocab_size": 4,
    }


@pytest.mark.unit
def test_generic_target_embedding_loader_reads_pytorch_checkpoint(tmp_path):
    source = torch.arange(12, dtype=torch.float32).reshape(4, 3)
    torch.save({"model.embed_tokens.weight": source}, tmp_path / "pytorch_model.bin")
    model = _Draft()
    args = Namespace(
        draft_target_embedding_path=str(tmp_path),
        draft_target_embedding_key="model.embed_tokens.weight",
        hf_checkpoint=None,
    )

    _load_target_embedding(model, args)

    assert torch.equal(model.embed_tokens.weight, source)


@pytest.mark.unit
def test_architecture_fingerprint_changes_with_parameter_layout():
    assert _architecture_fingerprint(_Draft(rows=4)) != _architecture_fingerprint(_Draft(rows=5))


@pytest.mark.unit
def test_actor_colocated_trainer_does_not_join_actor_process_group(monkeypatch):
    import vime.backends.speculative_training.draft_trainer as module

    model = _Draft()
    monkeypatch.setattr(module, "current_platform", lambda: SimpleNamespace(is_npu=False))
    monkeypatch.setattr(module.torch.cuda, "current_device", lambda: 0)
    monkeypatch.setattr(module, "_load_draft_model", lambda args, device: model)
    monkeypatch.setattr(module, "_load_target_embedding", lambda model, args: None)
    monkeypatch.setattr(module.dist, "get_rank", lambda: pytest.fail("must not query Actor rank"))
    monkeypatch.setattr(module.dist, "get_world_size", lambda: pytest.fail("must not query Actor world size"))
    args = Namespace(
        draft_freeze_embeddings=True,
        draft_vocab_mapping_path=None,
        draft_learning_rate=1e-5,
        draft_weight_decay=0.0,
        draft_lr_warmup_steps=0,
        draft_lr_total_steps=1,
        draft_lr_scheduler_type="constant",
        draft_train_interval=1,
        draft_train_steps_per_trigger=1,
        draft_queue_max_samples=4,
        num_rollout=1,
        draft_checkpoint_path=None,
    )

    trainer = ExternalDraftTrainer(args, distributed=False)

    assert trainer.rank == 0
    assert trainer.world_size == 1
    assert trainer.model is model


@pytest.mark.unit
def test_dspark_train_reports_compound_loss_metrics(monkeypatch):
    import vime.backends.speculative_training.draft_trainer as module

    model = torch.nn.Linear(1, 1, bias=False)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.target_weight_version = "9"
    trainer.target_lm_head_weight = None
    trainer.algorithm = "dspark"
    trainer.device = torch.device("cpu")
    trainer.device_type = "cpu"
    trainer.world_size = 1
    trainer.model = model
    trainer.algorithm_train_kwargs = {"loss_config": {"ce": (None, 0.1), "tv": (None, 0.9)}}
    trainer.args = Namespace(
        draft_train_steps_per_trigger=2,
        draft_batch_size_per_gpu=1,
        draft_dspark_block_size=2,
        draft_max_grad_norm=1.0,
    )
    trainer.queue = type(
        "Queue",
        (),
        {
            "count": lambda self, version: 1,
            "take": lambda self, version, batch_size, repeat: [object()],
        },
    )()
    trainer.optimizer = optimizer
    trainer.scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    trainer.trainable_parameters = list(model.parameters())
    trainer.optimizer_steps = 0
    trainer.draft_version = 0

    metric_steps = iter(
        [
            (2.0, 1.0, 6.0, 2.0, 3.0, 1.0),
            (4.0, 1.0, 10.0, 2.0, 5.0, 1.0),
        ]
    )

    def compute_loss(model, batch, train_kwargs):
        ce_sum, ce_total, tv_sum, tv_total, confidence_sum, confidence_total = next(metric_steps)
        loss = model.weight.square().sum()
        scalar = loss.detach().new_tensor
        return loss, {
            "loss_sum": loss.detach() * 5,
            "token_count": scalar(5.0),
            "top1_correct": scalar(3.0),
            "top5_correct": scalar(0.0),
            "ce_loss_sum": scalar(ce_sum),
            "ce_loss_total": scalar(ce_total),
            "tv_loss_sum": scalar(tv_sum),
            "tv_loss_total": scalar(tv_total),
            "confidence_loss_sum": scalar(confidence_sum),
            "confidence_loss_total": scalar(confidence_total),
        }

    monkeypatch.setattr(module, "collate_dspark_samples", lambda *args, **kwargs: {})
    monkeypatch.setattr(module, "compute_dspark_loss", compute_loss)

    result = trainer.train(4)

    assert result["ce_loss"] == pytest.approx(3.0)
    assert result["tv_loss"] == pytest.approx(4.0)
    assert result["confidence_loss"] == pytest.approx(4.0)


@pytest.mark.unit
def test_dspark_publish_includes_synced_frozen_lm_head():
    model = _Draft(rows=4, hidden=3)
    model.embed_tokens.weight.requires_grad_(False)
    model.lm_head = torch.nn.Linear(3, 4, bias=False)
    model.lm_head.weight.requires_grad_(False)
    model.verifier_lm_head = torch.nn.Linear(3, 4, bias=False)
    model.verifier_norm = torch.nn.LayerNorm(3)
    model.confidence_head = torch.nn.Linear(3, 1)
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.draft_version = 2
    trainer.target_weight_version = "9"
    trainer.algorithm = "dspark"
    trainer.args = Namespace(draft_publish_dtype="bf16")
    trainer.model = model
    trainer.architecture_fingerprint = "fingerprint"

    snapshot = trainer.prepare_publish_snapshot()

    assert snapshot["algorithm"] == "dspark"
    names = {name for name, _ in snapshot["named_tensors"]}
    assert "lm_head.weight" in names
    assert "embed_tokens.weight" not in names
    assert "verifier_lm_head.weight" not in names
    assert "verifier_norm.weight" not in names
    assert "verifier_norm.bias" not in names
    tensors = dict(snapshot["named_tensors"])
    assert tensors["lm_head.weight"].dtype == torch.bfloat16
    assert tensors["confidence_head.weight"].dtype == torch.float32
    assert tensors["confidence_head.bias"].dtype == torch.float32


@pytest.mark.unit
def test_dspark_export_uses_custom_rollout_directory(tmp_path):
    model = _Draft()
    saved_tensors = {}
    source = tmp_path / "original"
    source.mkdir()
    source_config = _flat_dspark_config()
    source_with_runtime_metadata = {**source_config, "_commit_hash": "test-revision"}
    (source / "config.json").write_text(json.dumps(source_with_runtime_metadata), encoding="utf-8")

    def save_pretrained(path, **kwargs):
        assert kwargs["safe_serialization"] is True
        assert kwargs["max_shard_size"] == "100GB"
        saved_tensors.update(
            {name: (tensor.device.type, tensor.is_contiguous()) for name, tensor in kwargs["state_dict"].items()}
        )
        (path / "config.json").write_text(
            _serialized_dspark_config(),
            encoding="utf-8",
        )
        torch.save(kwargs["state_dict"], path / "model.safetensors")
        (path / "generation_config.json").write_text("{}", encoding="utf-8")

    model.save_pretrained = save_pretrained
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(source),
        draft_save_hf=str(tmp_path / "export-{rollout_id}"),
        save_hf=None,
    )

    result = trainer.export_hf_model(7)

    expected = tmp_path / "export-7"
    assert result == str(expected.resolve())
    assert expected.is_dir()
    assert set(saved_tensors) == set(model.state_dict())
    assert all(device == "cpu" and contiguous for device, contiguous in saved_tensors.values())
    saved_state_dict = torch.load(expected / "model.safetensors", map_location="cpu", weights_only=True)
    assert set(saved_state_dict) == set(model.state_dict())
    assert all(torch.equal(saved_state_dict[name], value.cpu()) for name, value in model.state_dict().items())
    saved_config = json.loads((expected / "config.json").read_text(encoding="utf-8"))
    assert saved_config == source_config
    assert {path.name for path in expected.iterdir()} == {"config.json", "model.safetensors"}
    assert not list(tmp_path.glob(".export-7.tmp-*"))
    assert not list(tmp_path.glob(".export-7.backup-*"))


@pytest.mark.unit
def test_failed_dspark_export_preserves_previous_valid_directory(tmp_path):
    output = tmp_path / "export"
    output.mkdir()
    marker = output / "previous-model.bin"
    marker.write_bytes(b"previous weights")
    model = _Draft()
    model.save_pretrained = lambda path, **kwargs: (path / "config.json").write_text(
        '{"speculators_model_type":"dspark"}', encoding="utf-8"
    )
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.model = model
    trainer.args = Namespace(
        draft_model_path=str(tmp_path / "original"),
        draft_save_hf=str(output),
        save_hf=None,
    )

    with pytest.raises(RuntimeError, match="non-empty model.safetensors"):
        trainer.export_hf_model(2)

    assert marker.read_bytes() == b"previous weights"


@pytest.mark.unit
@pytest.mark.parametrize("collision", ["source", "actor"])
def test_dspark_export_rejects_unsafe_output_directory(tmp_path, collision):
    source = tmp_path / "original"
    source.mkdir()
    model = _Draft()
    model.save_pretrained = lambda *args, **kwargs: pytest.fail("must not overwrite another model")
    trainer = ExternalDraftTrainer.__new__(ExternalDraftTrainer)
    trainer.rank = 0
    trainer.algorithm = "dspark"
    trainer.model = model
    output_template = str(source if collision == "source" else tmp_path / "actor-{rollout_id}")
    trainer.args = Namespace(
        draft_model_path=str(source),
        draft_save_hf=output_template,
        save_hf=output_template if collision == "actor" else None,
    )

    with pytest.raises(ValueError, match="must not overwrite"):
        trainer.export_hf_model(3)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
