import ast
import json
from argparse import Namespace
from contextlib import nullcontext
from pathlib import Path

import pytest
import torch
from safetensors.torch import load_file

from vime.backends.megatron_utils import draft_feature_collector as sink
from vime.utils.draft_feature_contract import manifest_from_dict

pytestmark = pytest.mark.unit
NUM_GPUS = 0


class Head(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.randn(8, 3))

    def forward(self, hidden):
        return torch.nn.functional.linear(hidden, self.weight)


class Target(torch.nn.Module):
    def __init__(self, tied=False):
        super().__init__()
        self.output_layer = Head()
        self.share_embeddings_and_output_weights = tied
        self.embedding = self.output_layer.weight
        self.skip_head = False
        self.fail = False

    def shared_embedding_or_output_weight(self):
        return self.embedding

    def forward(self, hidden):
        if self.fail:
            raise RuntimeError("forward fault")
        return hidden if self.skip_head else self.output_layer(hidden)


@pytest.fixture
def fixture(tmp_path):
    checkpoint = tmp_path / "model"
    checkpoint.mkdir()
    (checkpoint / "config.json").write_text("{}")
    (checkpoint / "tokenizer.json").write_text("{}")
    args = Namespace(
        hf_checkpoint=str(checkpoint),
        draft_feature_output_dir=str(tmp_path / "exports"),
        draft_feature_run_id="fresh",
        draft_feature_max_tokens=8,
        draft_feature_max_batches=2,
        draft_feature_max_bytes=120,
    )
    batch = {
        "total_lengths": [5],
        "response_lengths": [2],
        "loss_masks": [torch.tensor([1, 1])],
        "unconcat_tokens": [torch.tensor([10, 11, 12, 13, 14])],
    }
    rollout = {
        "sample_indices": [7, 3],
        "group_indices": [1, 2],
        "rollout_ids": [4, 4],
        "weight_versions": [None, ["2", "3"]],
    }
    hidden = torch.arange(18, dtype=torch.float32).reshape(6, 1, 3)
    return args, batch, rollout, hidden


def collect(collector, target, fixture, index=0):
    _, batch, rollout, hidden = fixture
    return collector.forward(target, {"hidden": hidden}, batch, rollout, [index])


@pytest.mark.parametrize("tied", [False, True])
def test_owned_storage_and_padding_alignment(fixture, tied):
    args, _, _, hidden = fixture
    target = Target(tied)
    # A bias-free wrapper need not register a bias attribute at all.
    collector = sink.DraftFeatureCollector(args, target, 4, "3")
    output = collect(collector, target, fixture, index=1)
    collector.finish()
    manifest = manifest_from_dict(json.loads((collector.root / "batch-0000.json").read_text()))
    assert manifest.token_map.sequences[0].sample_id == 3
    assert manifest.token_map.sequences[0].weight_versions == ("2", "3")
    assert [(t.packed_position, t.target_token_id) for t in manifest.token_map.selected_tokens] == [(2, 13), (3, 14)]
    features = load_file(str(collector.root / manifest.payload_ref))["features"]
    head = load_file(str(collector.head_ref))["weight"]
    torch.testing.assert_close(features @ head.T, output.detach()[[2, 3], 0])
    payload_bytes = (collector.root / manifest.payload_ref).read_bytes()
    head_bytes = collector.head_ref.read_bytes()
    hidden.add_(100)
    with torch.no_grad():
        target.output_layer.weight.add_(100)
    assert (collector.root / manifest.payload_ref).read_bytes() == payload_bytes
    assert collector.head_ref.read_bytes() == head_bytes
    torch.testing.assert_close(features, torch.tensor([[6.0, 7.0, 8.0], [9.0, 10.0, 11.0]]))


@pytest.mark.parametrize("budget,success", [(120, True), (119, False), (95, False)])
def test_exact_tensor_budget(fixture, budget, success):
    args, *_ = fixture
    args.draft_feature_max_bytes = budget
    target = Target()
    collector = sink.DraftFeatureCollector(args, target, 4, "3")
    collect(collector, target, fixture)
    assert not target.output_layer._forward_pre_hooks
    if success:
        collector.finish()
        assert collector.bytes_used == 120
    else:
        with pytest.raises(ValueError, match="byte budget"):
            collector.finish()
        assert not list(collector.root.iterdir())


def test_empty_microbatch_then_valid_and_empty_round(fixture):
    args, batch, *_ = fixture
    target = Target()
    collector = sink.DraftFeatureCollector(args, target, 0, "3")
    batch["loss_masks"][0].zero_()
    collect(collector, target, fixture)
    with pytest.raises(ValueError, match="no selected target tokens"):
        collector.finish()
    batch["loss_masks"][0].fill_(1)
    collect(collector, target, fixture)
    collector.finish()


@pytest.mark.parametrize("failure", ["forward", "hook", "copy", "save", "rename", "missing_head"])
def test_faults_preserve_exception_and_remove_hook(fixture, monkeypatch, failure):
    args, *_ = fixture
    target = Target()
    collector = sink.DraftFeatureCollector(args, target, 0, "3")
    target.fail = failure == "forward"
    target.skip_head = failure == "missing_head"
    if failure == "hook":
        fixture[3].resize_(6, 2, 3)
    if failure == "copy":
        monkeypatch.setattr(torch.Tensor, "cpu", lambda _: (_ for _ in ()).throw(RuntimeError("copy fault")))
    if failure == "save":
        monkeypatch.setattr(sink, "save_file", lambda *args: (_ for _ in ()).throw(RuntimeError("save fault")))
    if failure == "rename":
        replace = sink.os.replace

        def fail_manifest(source, destination):
            if str(destination).endswith(".json"):
                raise RuntimeError("rename fault")
            replace(source, destination)

        monkeypatch.setattr(sink.os, "replace", fail_manifest)
    with pytest.raises((RuntimeError, ValueError)):
        collect(collector, target, fixture)
    assert not target.output_layer._forward_pre_hooks
    assert not list(collector.root.iterdir())
    target.fail = target.skip_head = False
    target(torch.zeros(6, 1, 3))


def test_completed_batch_survives_later_sink_failure(fixture, monkeypatch):
    args, *_ = fixture
    args.draft_feature_max_bytes = 1000
    target = Target()
    collector = sink.DraftFeatureCollector(args, target, 0, "3")
    collect(collector, target, fixture)
    old = {p.name: p.read_bytes() for p in collector.root.iterdir()}
    monkeypatch.setattr(sink, "save_file", lambda *args: (_ for _ in ()).throw(RuntimeError("sink fault")))
    with pytest.raises(RuntimeError, match="sink fault"):
        collect(collector, target, fixture)
    assert {p.name: p.read_bytes() for p in collector.root.iterdir()} == old


@pytest.mark.parametrize(
    "field,value,reason",
    [("draft_feature_max_tokens", 1, "token budget"), ("draft_feature_max_batches", 0, "batch budget")],
)
def test_empty_budget_reason(fixture, field, value, reason):
    args, *_ = fixture
    setattr(args, field, value)
    target = Target()
    collector = sink.DraftFeatureCollector(args, target, 0, "3")
    collect(collector, target, fixture)
    with pytest.raises(ValueError, match=reason):
        collector.finish()


def test_flat_head_then_padding_head_keeps_the_real_sequence(fixture):
    args, _, _, hidden = fixture

    class PackedTarget(Target):
        def forward(self, hidden):
            real = self.output_layer(hidden[:5, 0])
            padding = self.output_layer(hidden[5:, 0])
            return torch.cat((real, padding)).unsqueeze(0)

    target = PackedTarget()
    collector = sink.DraftFeatureCollector(args, target, 0, "3")
    collect(collector, target, fixture)
    collector.finish()
    features = load_file(str(collector.root / "batch-0000.safetensors"))["features"]
    torch.testing.assert_close(features, hidden[[2, 3], 0])


def _actor_methods():
    source = Path(__file__).resolve().parents[1] / "vime/backends/megatron_utils/actor.py"
    actor = next(
        node
        for node in ast.parse(source.read_text()).body
        if isinstance(node, ast.ClassDef) and node.name == "MegatronTrainRayActor"
    )
    return {node.name: node for node in actor.body if isinstance(node, ast.FunctionDef)}


def test_compute_log_prob_passes_the_collector_to_forward_only():
    seen = []

    def fake_forward_only(*_args, draft_feature_collector=None, **_kwargs):
        seen.append(draft_feature_collector)
        return {}

    # Importing the whole actor needs Ray and GPU Megatron; run the real method alone.
    isolated = ast.Module(body=[_actor_methods()["compute_log_prob"]], type_ignores=[])
    namespace = {
        "DataIterator": object,
        "torch": torch,
        "timer": lambda _name: nullcontext(),
        "forward_only": fake_forward_only,
        "get_log_probs_and_entropy": None,
    }
    exec(compile(ast.fix_missing_locations(isolated), "actor.py", "exec"), namespace)
    actor = Namespace(args=None, model=None)
    collector = object()
    namespace["compute_log_prob"](actor, [], [1], draft_feature_collector=collector)
    namespace["compute_log_prob"](actor, [], [1], store_prefix="ref_")
    assert seen == [collector, None]


def test_train_actor_creates_the_collector_it_passes():
    train_actor = _actor_methods()["train_actor"]
    passed = [
        (node.lineno, keyword.value.id)
        for node in ast.walk(train_actor)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute) and node.func.attr == "compute_log_prob"
        for keyword in node.keywords
        if keyword.arg == "draft_feature_collector" and isinstance(keyword.value, ast.Name)
    ]
    assigned = {
        target.id: node.lineno
        for node in ast.walk(train_actor)
        if isinstance(node, ast.Assign)
        for target in node.targets
        if isinstance(target, ast.Name)
    }
    assert passed, "the actor log-prob forward no longer receives the draft feature collector"
    for lineno, name in passed:
        assert name in assigned and assigned[name] < lineno, f"{name} is passed before it is assigned"
