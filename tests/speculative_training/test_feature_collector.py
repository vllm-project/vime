from argparse import Namespace

import pytest
import torch

from vime.backends.speculative_training.feature_collector import DraftFeatureCollector, find_target_output_weight


class _Layer(torch.nn.Module):
    def forward(self, hidden):
        return hidden + 1


class _Target(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = torch.nn.Module()
        self.decoder.layers = torch.nn.ModuleList([_Layer() for _ in range(5)])
        self.output_layer = torch.nn.Linear(4, 8, bias=False)

    def forward(self, hidden):
        for layer in self.decoder.layers:
            hidden = layer(hidden)
        return self.output_layer(hidden)


class _TiedTarget(_Target):
    def __init__(self):
        super().__init__()
        self.output_layer.register_parameter("weight", None)
        self.embedding_weight = torch.nn.Parameter(torch.randn(8, 4))

    def shared_embedding_or_output_weight(self):
        return self.embedding_weight


def _args(**overrides):
    values = {
        "num_layers": 5,
        "draft_feature_layer_ids": [0, 2, 4],
        "draft_collection_sample_rate": 1.0,
        "draft_max_samples_per_rollout_per_dp": 4,
        "draft_max_tokens_per_rollout_per_dp": 32,
        "draft_hidden_window_tokens": 6,
        "draft_hidden_window_mode": "front",
        "draft_random_seed": 1,
        "sequence_parallel": False,
    }
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.unit
def test_find_target_output_weight_supports_tied_megatron_embeddings():
    target = _TiedTarget()

    assert find_target_output_weight([target]) is target.embedding_weight


@pytest.mark.unit
def test_collector_reconstructs_sample_window_and_aux_layers(monkeypatch):
    target = _Target()
    collector = DraftFeatureCollector(_args(), [target], rollout_id=2, target_weight_version="11")
    monkeypatch.setattr(collector, "_gather_sequence_parallel", lambda tensor: tensor)
    monkeypatch.setattr(collector, "_is_export_rank", lambda: True)
    monkeypatch.setattr(collector, "_dp_rank", lambda: 3)
    tokens = torch.tensor([10, 11, 12, 13, 14, 15])
    batch = {
        "unconcat_tokens": [tokens],
        "total_lengths": [6],
        "response_lengths": [3],
        "loss_masks": [torch.tensor([1, 1, 1])],
    }

    try:
        collector.begin_microbatch(batch, [7])
        target(torch.zeros(6, 1, 4))
        collector.end_microbatch()
        payloads = collector.pop_payloads()
    finally:
        collector.close()

    assert len(payloads) == 1
    payload = payloads[0]
    assert payload["input_ids"].tolist() == [12, 13, 14, 15]
    assert payload["loss_mask"].tolist() == [0.0, 1.0, 1.0, 1.0]
    assert tuple(payload["aux_hidden_states"].shape) == (4, 12)
    assert tuple(payload["final_hidden_states"].shape) == (4, 4)
    assert payload["position_ids"].tolist() == [2, 3, 4, 5]
    assert payload["original_sample_id"] == "dp3-sample7"
    assert payload["target_weight_version"] == "11"


@pytest.mark.unit
@pytest.mark.parametrize(("response_length", "expected_payloads"), [(3, 0), (4, 1)])
def test_dspark_collector_requires_one_complete_future_block(
    monkeypatch,
    response_length,
    expected_payloads,
):
    target = _Target()
    collector = DraftFeatureCollector(
        _args(draft_algorithm="dspark", draft_dspark_block_size=3, draft_hidden_window_tokens=8),
        [target],
        rollout_id=2,
        target_weight_version="11",
    )
    monkeypatch.setattr(collector, "_gather_sequence_parallel", lambda tensor: tensor)
    monkeypatch.setattr(collector, "_is_export_rank", lambda: True)
    tokens = torch.arange(8)
    batch = {
        "unconcat_tokens": [tokens],
        "total_lengths": [8],
        "response_lengths": [response_length],
        "loss_masks": [torch.ones(response_length)],
    }

    try:
        collector.begin_microbatch(batch, [0])
        target(torch.zeros(8, 1, 4))
        collector.end_microbatch()
        payloads = collector.pop_payloads()
    finally:
        collector.close()

    assert len(payloads) == expected_payloads
    if payloads:
        assert payloads[0]["algorithm"] == "dspark"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
