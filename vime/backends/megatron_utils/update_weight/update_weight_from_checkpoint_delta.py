from __future__ import annotations

import logging
import time

import ray
import torch
from vllm.distributed.weight_transfer.nccl_engine import NCCLTrainerSendWeightsArgs, NCCLWeightTransferEngine

from .checkpoint_delta import CheckpointDeltaChunk, CheckpointDeltaSource
from .update_weight_from_distributed import UpdateWeightFromDistributed

logger = logging.getLogger(__name__)


class UpdateWeightFromCheckpointDelta(UpdateWeightFromDistributed):
    """VIME direct DWU: full HF export, GPU diff, NCCL checkpoint patches."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        if self.quantization_config is not None:
            raise NotImplementedError("VIME direct DWU does not support quantized models")
        self._delta_source = CheckpointDeltaSource()

    def _get_weight_update_commit(self):
        return self._delta_source.commit

    @torch.no_grad()
    def update_weights(self) -> None:
        is_source = getattr(self, "_is_pp_src_rank", False)
        if is_source:
            base_version = self.weight_version
            self._delta_source.begin_update(
                base_version=base_version,
                target_version=base_version + 1,
            )
        started_at = time.perf_counter()
        try:
            super().update_weights()
        except BaseException:
            if is_source:
                self._delta_source.abort()
            raise
        if not is_source:
            return
        density = (
            self._delta_source.changed_elements / self._delta_source.total_elements
            if self._delta_source.total_elements
            else 0.0
        )
        elapsed = time.perf_counter() - started_at
        self.update_weight_metrics.update(
            {
                "weight_sync/is_dense_seed": float(self._delta_source.is_seed_update),
                "weight_sync/total_elements": float(self._delta_source.total_elements),
                "weight_sync/changed_elements": float(self._delta_source.changed_elements),
                "weight_sync/delta_density": density,
                "weight_sync/wire_bytes": float(self._delta_source.wire_bytes),
                "weight_sync/seconds": elapsed,
            }
        )
        if getattr(self, "_is_pp_src_rank", False):
            logger.info(
                "Direct DWU committed version=%d dense_seed=%s changed=%d/%d density=%.6f wire_bytes=%d seconds=%.3f",
                self.weight_version,
                self._delta_source.is_seed_update,
                self._delta_source.changed_elements,
                self._delta_source.total_elements,
                density,
                self._delta_source.wire_bytes,
                elapsed,
            )

    def _update_bucket_weights_from_distributed(
        self,
        converted_named_tensors,
        pbar=None,
    ) -> None:
        chunks = self._delta_source.encode_chunk(converted_named_tensors)
        converted_named_tensors.clear()

        for chunk in chunks:
            while not ray.get(self.rollout_engine_lock.acquire.remote()):
                time.sleep(0.1)
            try:
                refs = self._send_delta_chunk(chunk)
                ray.get(refs)
            finally:
                ray.get(self.rollout_engine_lock.release.remote())
        if pbar is not None:
            pbar.update(1)

    def _send_weights_to_rollout_engines(self) -> None:
        super()._send_weights_to_rollout_engines()
        if not self._is_pp_src_rank:
            return
        final_chunk = self._delta_source.finish_update()

        while not ray.get(self.rollout_engine_lock.acquire.remote()):
            time.sleep(0.1)
        try:
            ray.get(self._send_delta_chunk(final_chunk))
        finally:
            ray.get(self.rollout_engine_lock.release.remote())

    def _send_delta_chunk(self, chunk: CheckpointDeltaChunk):
        refs = [
            engine.update_checkpoint_delta_from_distributed.remote(
                update_info=chunk.update_info(),
            )
            for engine in self.rollout_engines
        ]
        NCCLWeightTransferEngine.trainer_send_weights(
            chunk.wire_tensors(),
            NCCLTrainerSendWeightsArgs(
                group=self._model_update_groups,
                packed=False,
            ),
        )
        return refs
