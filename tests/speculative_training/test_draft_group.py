from argparse import Namespace

import pytest

from vime.backends.speculative_training.draft_group import ExternalDraftTrainGroup


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _Actor:
    def __init__(self):
        self.calls = []
        self.get_external_draft_start_rollout = _RemoteMethod(lambda: 0)
        self.collect_external_draft_features = _RemoteMethod(self._collect)
        self.train_external_draft = _RemoteMethod(self._train)
        self.prepare_external_draft_publish_snapshot = _RemoteMethod(self._snapshot)
        self.save_external_draft = _RemoteMethod(lambda rollout_id: f"draft-{rollout_id}.pt")
        self.export_external_draft = _RemoteMethod(lambda rollout_id: f"draft-hf-{rollout_id}")

    def _collect(self, feature_refs, target_head, target_version):
        self.calls.append((feature_refs, target_head, target_version))
        return {"accepted": 2, "received": 2, "queued": 2, "rejected_version_mismatch": 0}

    @staticmethod
    def _train(rollout_id):
        return {"trained": 1, "draft_version": 1, "target_weight_version": "7", "rollout_id": rollout_id}

    @staticmethod
    def _snapshot():
        return {"named_tensors": [("weight", 1)]}


@pytest.mark.unit
def test_draft_group_delegates_to_actor_rank_zero(monkeypatch):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "put", lambda value: ("object-ref", value))
    rank_zero = _Actor()
    unused_rank = _Actor()
    actor_group = Namespace(_actor_handlers=[rank_zero, unused_rank])
    group = ExternalDraftTrainGroup(Namespace(), actor_group)

    assert group.create() == [0]
    result = group.collect_actor_results(
        [
            {
                "target_weight_version": "7",
                "draft_features_ref": "features",
                "draft_target_lm_head_ref": "head",
            }
        ]
    )
    assert result["placement"] == "actor_rank0"
    assert result["accepted"] == 2
    assert rank_zero.calls == [(["features"], "head", "7")]
    assert unused_rank.calls == []

    assert group.train_draft(3)["trained"] == 1
    snapshot_ref, version = group.prepare_publish_snapshot()
    assert snapshot_ref[0] == "object-ref"
    assert version == "1"
    assert group.save_draft(3) == ["draft-3.pt"]
    assert group.save_draft(4, export_hf=True) == ["draft-4.pt", "draft-hf-4"]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
