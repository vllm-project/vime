from __future__ import annotations

import logging
from collections.abc import Callable, Sequence
from typing import TYPE_CHECKING

import ray
import torch.distributed as dist

from vime.utils.distributed_utils import get_gloo_group

if TYPE_CHECKING:
    from ray.actor import ActorHandle

logger = logging.getLogger(__name__)

WeightTransfer = Callable[[int], None]
WeightCommit = Callable[[], None]


def post_process_weights(
    restore_weights_before_load: bool,
    post_process_quantization: bool,
    rollout_engines: Sequence[ActorHandle],
) -> None:
    """Run compressed-weight pre/post processing on every rollout engine."""
    ray.get(
        [
            engine.post_process_weights.remote(
                restore_weights_before_load=restore_weights_before_load,
                post_process_quantization=post_process_quantization,
            )
            for engine in rollout_engines
        ]
    )


class WeightUpdateCoordinator:
    """Coordinate one committed weight update across rollout engines.

    Transport-specific work stays in ``transfer_target``/``transfer_draft``.
    This class owns only the shared control plane:

    pause -> flush -> quant pre-process -> transfer -> quant post-process
    -> source commit -> publish committed version -> resume.

    The candidate version is returned only after every phase succeeds. Before
    resume, failures leave generation paused. If a batched resume partially
    succeeds, the coordinator best-effort pauses every engine again and restores
    its version metadata; external recovery is still required because vLLM applies
    weight chunks in place and cannot roll them back atomically. Transfer
    callbacks are distributed operations and must surface failures collectively;
    this coordinator does not turn a rank-local exception into a collective one.
    """

    def __init__(
        self,
        rollout_engines: Sequence[ActorHandle],
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.rollout_engines = tuple(rollout_engines)
        self.quantization_config = quantization_config

    def run(
        self,
        *,
        current_version: int,
        transfer_target: WeightTransfer,
        transfer_draft: WeightTransfer | None = None,
        commit: WeightCommit | None = None,
    ) -> int:
        candidate_version = current_version + 1
        try:
            self._quiesce()
            transfer_target(candidate_version)
            if transfer_draft is not None:
                transfer_draft(candidate_version)

            # Keep all trainer ranks aligned before rank 0 publishes the new
            # committed version to the rollout engines.
            self._barrier()
            self._publish_and_resume(
                candidate_version=candidate_version,
                current_version=current_version,
                commit=commit,
            )
        except BaseException:
            logger.exception(
                "Weight update to version %s failed; caller retains version %s "
                "and rollout engines require fail-stop recovery.",
                candidate_version,
                current_version,
            )
            raise
        return candidate_version

    def _quiesce(self) -> None:
        def quiesce_rank_zero() -> None:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
            ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
            if self._uses_compressed_tensors():
                post_process_weights(
                    restore_weights_before_load=True,
                    post_process_quantization=False,
                    rollout_engines=self.rollout_engines,
                )

        self._run_rank_zero_phase("quiesce", quiesce_rank_zero)

    def _publish_and_resume(
        self,
        *,
        candidate_version: int,
        current_version: int,
        commit: WeightCommit | None,
    ) -> None:
        def publish_and_resume_rank_zero() -> None:
            if self._uses_compressed_tensors():
                post_process_weights(
                    restore_weights_before_load=False,
                    post_process_quantization=True,
                    rollout_engines=self.rollout_engines,
                )

            # Commit transport-side source state while generation is still
            # paused. A commit failure is broadcast to every trainer rank before
            # the candidate version is published or any engine resumes.
            if commit is not None:
                commit()

            # Chunk RPCs carry a candidate version for compatibility, but do not
            # publish it. Publish once after the source and every rollout worker
            # have completed the candidate successfully.
            try:
                self._set_engine_version(candidate_version)
                ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
            except BaseException:
                self._restore_fail_stop_state(current_version)
                raise

        self._run_rank_zero_phase("publish/resume", publish_and_resume_rank_zero)

    def _restore_fail_stop_state(self, current_version: int) -> None:
        # A list of Ray calls is not an atomic fanout: continue_generation may
        # have succeeded on only a subset before ray.get reports an error.
        # Re-pause first so partially resumed engines stop serving mixed state.
        try:
            ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
        except BaseException:
            logger.exception("Failed to re-pause every rollout engine after publish/resume error")

        # GPU weights and transport-side source state cannot be rolled back here.
        # Restore only the version metadata so no engine advertises a partially
        # resumed candidate. Backend-specific fail-stop recovery is still needed.
        try:
            self._set_engine_version(current_version)
        except BaseException:
            logger.exception(
                "Failed to restore rollout-engine version marker to %s",
                current_version,
            )

    def _set_engine_version(self, version: int) -> None:
        ray.get([engine.set_weight_version.remote(str(version)) for engine in self.rollout_engines])

    def _uses_compressed_tensors(self) -> bool:
        return bool(self.quantization_config and self.quantization_config.get("quant_method") == "compressed-tensors")

    @staticmethod
    def _run_rank_zero_phase(phase: str, operation: Callable[[], None]) -> None:
        """Run a control-plane phase once and propagate its result to all ranks."""
        rank_zero_error: BaseException | None = None
        status: list[str | None] = [None]
        if dist.get_rank() == 0:
            try:
                operation()
            except BaseException as exc:
                rank_zero_error = exc
                status[0] = f"{type(exc).__name__}: {exc}"

        dist.broadcast_object_list(status, src=0, group=get_gloo_group())
        if rank_zero_error is not None:
            raise rank_zero_error
        if status[0] is not None:
            raise RuntimeError(f"Rank-0 weight-update {phase} failed: {status[0]}")

    @staticmethod
    def _barrier() -> None:
        dist.barrier(group=get_gloo_group())
