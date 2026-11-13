import logging
from argparse import Namespace

import pytest

from vime.backends.speculative_training.draft_group import DraftTrainingLoop, ExternalDraftTrainGroup


class _RemoteMethod:
    def __init__(self, fn):
        self.fn = fn

    def remote(self, *args, **kwargs):
        return self.fn(*args, **kwargs)


class _Actor:
    def __init__(self, accepted=2):
        self.calls = []
        self._accepted = accepted
        self.get_external_draft_start_rollout = _RemoteMethod(lambda: 0)
        self.collect_external_draft_features = _RemoteMethod(self._collect)
        self.train_external_draft = _RemoteMethod(self._train)
        self.prepare_external_draft_publish_snapshot = _RemoteMethod(self._snapshot)
        self.save_external_draft = _RemoteMethod(lambda rollout_id: f"draft-{rollout_id}.pt")
        self.export_external_draft = _RemoteMethod(lambda rollout_id: f"draft-hf-{rollout_id}")

    def _collect(self, feature_refs, target_head, target_version):
        self.calls.append((feature_refs, target_head, target_version))
        return {"accepted": self._accepted, "received": 2, "queued": 2, "rejected_version_mismatch": 0}

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


def _manifest():
    return {
        "target_weight_version": "7",
        "draft_features_ref": "features",
        "draft_target_lm_head_ref": "head",
    }


def _make_loop(monkeypatch, accepted=2, **arg_values):
    import vime.backends.speculative_training.draft_group as module

    monkeypatch.setattr(module.ray, "get", lambda value: value)
    monkeypatch.setattr(module.ray, "put", lambda value: ("object-ref", value))
    args = Namespace(
        use_wandb=False,
        use_tensorboard=False,
        wandb_always_use_train_step=False,
        draft_train_interval=1,
        draft_publish_interval=1,
        draft_save_interval=None,
        draft_save_hf=None,
        num_rollout=3,
    )
    args.__dict__.update(arg_values)
    group = ExternalDraftTrainGroup(args, Namespace(_actor_handlers=[_Actor(accepted=accepted)]))
    return DraftTrainingLoop(args, group), group


@pytest.mark.unit
def test_draft_loop_sequences_stages_and_clears_pending(monkeypatch):
    draft_loop, group = _make_loop(monkeypatch)
    assert draft_loop.pending_snapshot is None
    draft_loop.collect(0, [_manifest()])
    assert draft_loop.should_train(0)
    draft_loop.train(0)
    assert draft_loop.should_publish(0)
    draft_loop.prepare_snapshot()
    snapshot_ref, version = draft_loop.pending_snapshot
    assert snapshot_ref[0] == "object-ref"
    assert version == "1"
    draft_loop.commit_publish(0)
    assert draft_loop.pending_snapshot is None
    assert group.last_published_draft_version == 1

    # Without a new training round the already-published version is not
    # re-published, even when the publish interval fires again.
    draft_loop.collect(1, [_manifest()])
    assert draft_loop.should_publish(1)
    draft_loop.prepare_snapshot()
    assert draft_loop.pending_snapshot is None


@pytest.mark.unit
def test_draft_loop_train_requires_fresh_collection(monkeypatch):
    draft_loop, _ = _make_loop(monkeypatch, accepted=0)
    draft_loop.collect(0, [_manifest()])
    # The train interval is due, but nothing was collected this rollout.
    assert not draft_loop.should_train(0)


@pytest.mark.unit
def test_draft_loop_save_schedule_matrix(monkeypatch):
    follows_actor, _ = _make_loop(monkeypatch, draft_save_interval=None, num_rollout=3)
    assert not follows_actor.should_save(0, actor_save_due=False)
    assert follows_actor.should_save(1, actor_save_due=True)
    # The final rollout always flushes a draft checkpoint.
    assert follows_actor.should_save(2, actor_save_due=False)

    explicit, _ = _make_loop(monkeypatch, draft_save_interval=2, num_rollout=10)
    # An explicit draft interval ignores the Actor save cadence.
    assert not explicit.should_save(0, actor_save_due=True)
    assert explicit.should_save(1, actor_save_due=False)
    assert not explicit.should_save(2, actor_save_due=False)
    assert explicit.should_save(9, actor_save_due=False)


@pytest.mark.unit
def test_draft_loop_save_uses_export_flag(monkeypatch, caplog):
    plain, _ = _make_loop(monkeypatch)
    with caplog.at_level(logging.INFO):
        plain.save(2)
    assert "External Draft save completed: ['draft-2.pt']" in caplog.text

    exporting, _ = _make_loop(monkeypatch, draft_save_hf="/tmp/draft-{rollout_id}")
    with caplog.at_level(logging.INFO):
        exporting.save(2)
    assert "'draft-2.pt', 'draft-hf-2'" in caplog.text


@pytest.mark.unit
def test_draft_loop_log_contract(monkeypatch, caplog):
    # The NPU smoke post-run evidence check (tests/draft_update_checks.py)
    # greps these exact log wordings; keep them stable.
    draft_loop, _ = _make_loop(monkeypatch)
    with caplog.at_level(logging.INFO):
        draft_loop.collect(0, [_manifest()])
        draft_loop.train(0)
        draft_loop.save(2)
    assert "External Draft feature collection: " in caplog.text
    assert "External Draft training: " in caplog.text
    assert "External Draft save completed: " in caplog.text


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
