from types import SimpleNamespace

import pytest
import torch

from vime.backends.megatron_utils.draft_feature_collector import DraftFeatureCollector
from vime.data import batch_builder
from vime.utils.draft_feature_contract import build_token_map
from vime.utils.types import Sample

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


def test_collect_only_training_data_carries_the_collector_identity(monkeypatch):
    # The collector reads group_indices and weight_versions from each DP partition, so
    # conversion must produce them and the object-store split must keep them per sample.
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=1,
        rollout_batch_size=2,
        reward_key=None,
        draft_feature_mode="collect-only",
        rollout_data_transport="object-store",
        global_batch_size=2,
    )
    samples = [
        Sample(
            group_index=4,
            index=8,
            rollout_id=8,
            reward=1.0,
            tokens=[100, 20],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.5],
            weight_versions=["v1"],
        ),
        Sample(
            group_index=7,
            index=3,
            rollout_id=3,
            reward=0.0,
            tokens=[100, 21],
            response_length=1,
            loss_mask=[1],
            rollout_log_probs=[-0.5],
            weight_versions=[],
        ),
    ]
    builder = batch_builder.BatchBuilder(args)
    data = builder.convert(samples)
    assert data["group_indices"] == [4, 7]
    assert data["weight_versions"] == [("v1",), ()]

    monkeypatch.setattr(batch_builder, "build_dp_schedule", lambda *a, **kw: ([[1], [0]], [[[0]], [[0]]], [1], [2]))
    monkeypatch.setattr(batch_builder.ray, "put", lambda value: value)
    builder.train_parallel_config = {"dp_size": 2}
    parts = [box.inner for box in builder.split_by_dp(data)]
    assert [(p["sample_indices"], p["group_indices"], p["weight_versions"]) for p in parts] == [
        ([3], [7], [()]),
        ([8], [4], [("v1",)]),
    ]


def test_training_data_without_collection_has_no_collector_identity():
    args = SimpleNamespace(
        custom_reward_post_process_path=None,
        custom_convert_samples_to_train_data_path=None,
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=False,
        n_samples_per_prompt=1,
        rollout_batch_size=1,
        reward_key=None,
        draft_feature_mode="off",
    )
    sample = Sample(
        group_index=4,
        index=8,
        rollout_id=8,
        reward=1.0,
        tokens=[100, 20],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.5],
    )
    data = batch_builder.BatchBuilder(args).convert([sample])
    assert "group_indices" not in data and "weight_versions" not in data
