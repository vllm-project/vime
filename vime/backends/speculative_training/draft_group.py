from __future__ import annotations

import logging
from argparse import Namespace

import ray

from vime.backends.speculative_training.config import should_run_draft_interval
from vime.observability.logging_utils import log
from vime.observability.metric_utils import compute_rollout_step

logger = logging.getLogger(__name__)


def _log_draft_result(args, rollout_id, prefix, result):
    if not isinstance(result, dict):
        return
    metrics = {
        f"draft/{prefix}_{key}": value
        for key, value in result.items()
        if isinstance(value, (int, float)) and not isinstance(value, bool)
    }
    if not metrics:
        return
    metrics["rollout/step"] = compute_rollout_step(args, rollout_id)
    log(args, metrics, step_key="rollout/step")


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


class DraftTrainingLoop:
    """Driver-side schedule policy for the external Draft stages.

    ``train.py`` keeps only the stage sequence -- collect -> train -> prepare ->
    save -> attach -- and the two interleavings that are training-loop
    knowledge: the publish snapshot is attached before ``update_weights`` and
    publication is committed immediately after it.  Interval decisions, the
    save-cadence fallback to the Actor schedule, and publish-snapshot
    bookkeeping live here, so Draft saving/publication policy changes do not
    touch the shared training loop.
    """

    def __init__(self, args: Namespace, group: ExternalDraftTrainGroup) -> None:
        self._args = args
        self._group = group
        self._collected_this_rollout = False
        self._pending_snapshot: tuple[object, str] | None = None
        if args.draft_save_hf:
            logger.info(
                "External DSpark export enabled: template=%s, interval=%s. Artifacts are written on "
                "Actor rank zero; use a shared path when Ray workers run on multiple hosts.",
                args.draft_save_hf,
                args.draft_save_interval,
            )

    def collect(self, rollout_id: int, actor_train_results) -> None:
        collect_result = self._group.collect_actor_results(actor_train_results)
        self._collected_this_rollout = int(collect_result.get("accepted", 0)) > 0
        if self._collected_this_rollout:
            logger.info("External Draft feature collection: %s", collect_result)
            _log_draft_result(self._args, rollout_id, "collect", collect_result)

    def should_train(self, rollout_id: int) -> bool:
        return self._collected_this_rollout and should_run_draft_interval(rollout_id, self._args.draft_train_interval)

    def train(self, rollout_id: int) -> None:
        draft_train_result = self._group.train_draft(rollout_id)
        logger.info("External Draft training: %s", draft_train_result)
        _log_draft_result(self._args, rollout_id, "train", draft_train_result)

    def should_publish(self, rollout_id: int) -> bool:
        return should_run_draft_interval(rollout_id, self._args.draft_publish_interval)

    def prepare_snapshot(self) -> None:
        prepared_snapshot = self._group.prepare_publish_snapshot()
        if prepared_snapshot is not None:
            self._pending_snapshot = prepared_snapshot

    def should_save(self, rollout_id: int, actor_save_due: bool) -> bool:
        return (
            (self._args.draft_save_interval is None and actor_save_due)
            or should_run_draft_interval(rollout_id, self._args.draft_save_interval)
            or rollout_id == self._args.num_rollout - 1
        )

    def save(self, rollout_id: int) -> None:
        draft_save_results = self._group.save_draft(
            rollout_id,
            export_hf=bool(self._args.draft_save_hf),
        )
        logger.info("External Draft save completed: %s", draft_save_results)

    @property
    def pending_snapshot(self) -> tuple[object, str] | None:
        return self._pending_snapshot

    def commit_publish(self, rollout_id: int) -> None:
        snapshot_ref, draft_snapshot_version = self._pending_snapshot
        self._group.mark_published(draft_snapshot_version)
        _log_draft_result(
            self._args,
            rollout_id,
            "publish",
            {"published": 1, "draft_version": int(draft_snapshot_version)},
        )
        self._pending_snapshot = None

    def release(self) -> None:
        self._group.release()
