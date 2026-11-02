from __future__ import annotations

import ray


class ExternalDraftTrainGroup:
    """Draft control plane backed by Actor rank zero.

    Draft optimization is serialized after the Actor phase and uses the same
    device already owned by Actor rank zero.  This wrapper intentionally keeps
    the driver-facing API used by ``train.py`` while avoiding another Ray actor
    and another placement group.
    """

    def __init__(self, _args, actor_group) -> None:
        if not actor_group._actor_handlers:
            raise RuntimeError("Actor-colocated Draft training requires an initialized Actor group")
        self._draft_actor = actor_group._actor_handlers[0]
        self.last_train_result = None
        self.last_published_draft_version = -1

    def create(self):
        """Return checkpoint state from the Draft trainer embedded in Actor rank zero."""
        return [ray.get(self._draft_actor.get_external_draft_start_rollout.remote())]

    def release(self) -> None:
        """The Draft trainer has the same lifetime as its owning Actor worker."""
        return None

    def collect_actor_results(self, actor_results) -> dict[str, int | str]:
        results = [value for value in actor_results if isinstance(value, dict)]
        if not results:
            return {"accepted": 0, "received": 0, "reason": "no_actor_feature_manifests"}
        versions = {str(value["target_weight_version"]) for value in results}
        if len(versions) != 1:
            raise RuntimeError(f"Actor Draft feature versions diverged: {sorted(versions)}")
        target_version = versions.pop()
        feature_refs = [
            value["draft_features_ref"] for value in results if value.get("draft_features_ref") is not None
        ]
        head_refs = [
            value["draft_target_lm_head_ref"] for value in results if value.get("draft_target_lm_head_ref") is not None
        ]
        if not head_refs:
            raise RuntimeError("No Actor rank exported the Target LM Head for Draft supervision")
        worker_result = ray.get(
            self._draft_actor.collect_external_draft_features.remote(
                feature_refs,
                head_refs[0],
                target_version,
            )
        )
        return {
            "accepted": int(worker_result.get("accepted", 0)),
            "received": int(worker_result.get("received", 0)),
            "queued": int(worker_result.get("queued", 0)),
            "rejected_version_mismatch": int(worker_result.get("rejected_version_mismatch", 0)),
            "target_weight_version": target_version,
            "placement": "actor_rank0",
        }

    def train_draft(self, rollout_id: int):
        self.last_train_result = ray.get(self._draft_actor.train_external_draft.remote(rollout_id))
        return self.last_train_result

    def prepare_publish_snapshot(self):
        if not isinstance(self.last_train_result, dict) or not int(self.last_train_result.get("trained", 0)):
            return None
        candidate_version = int(self.last_train_result["draft_version"])
        if candidate_version <= self.last_published_draft_version:
            return None
        snapshot = ray.get(self._draft_actor.prepare_external_draft_publish_snapshot.remote())
        if snapshot is None:
            return None
        snapshot_ref = snapshot if isinstance(snapshot, ray.ObjectRef) else ray.put(snapshot)
        return snapshot_ref, str(candidate_version)

    def mark_published(self, draft_version: str) -> None:
        draft_version = int(draft_version)
        if draft_version < self.last_published_draft_version:
            raise RuntimeError(
                f"Draft publish version regressed: {draft_version} < {self.last_published_draft_version}"
            )
        self.last_published_draft_version = draft_version

    def save_draft(self, rollout_id: int, export_hf: bool = False) -> list[str]:
        saved_paths = [ray.get(self._draft_actor.save_external_draft.remote(rollout_id))]
        if export_hf:
            saved_paths.append(ray.get(self._draft_actor.export_external_draft.remote(rollout_id)))
        return [result for result in saved_paths if result is not None]
