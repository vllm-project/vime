import pytest
import torch

from vime.backends.megatron_utils.draft_feature_collector import DraftFeatureCollector
from vime.utils.draft_feature_contract import build_token_map

NUM_GPUS = 0
pytestmark = pytest.mark.unit


def test_reordered_samples_preserve_identity_and_unknown_provenance():
    collector = object.__new__(DraftFeatureCollector)
    rollout = {
        "sample_indices": [8, 3],
        "group_indices": [4, 7],
        "rollout_ids": [1, 2],
        "weight_versions": [None, ["v1", "v2"]],
    }
    batch = {
        "total_lengths": [3, 5],
        "response_lengths": [1, 2],
        "loss_masks": [torch.tensor([1]), torch.tensor([0, 1])],
        "unconcat_tokens": [torch.tensor([20, 21, 22]), torch.tensor([10, 11, 12, 13, 14])],
    }
    sequences = collector._sequences(batch, rollout, [1, 0])
    assert [(s.sample_id, s.group_id, s.rollout_id, s.weight_versions) for s in sequences] == [
        (3, 7, 2, ("v1", "v2")),
        (8, 4, 1, ()),
    ]
    mapping = build_token_map(sequences, 2)
    assert [(t.sample_id, t.packed_position, t.target_token_id) for t in mapping.selected_tokens] == [
        (3, 1, 22),
        (8, 6, 14),
    ]
