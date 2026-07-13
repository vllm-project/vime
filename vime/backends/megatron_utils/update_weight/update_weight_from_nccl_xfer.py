from __future__ import annotations

import logging
from argparse import Namespace
from collections.abc import Callable, Mapping, Sequence

import torch
from ray.actor import ActorHandle

from .common import named_params_and_buffers
from .nccl_xfer_bindings import get_nccl_xfer_availability
from .nccl_xfer_layout import analyze_nccl_xfer_layout
from .update_weight_from_distributed import UpdateWeightFromDistributed

logger = logging.getLogger(__name__)


class UpdateWeightFromNcclXfer:
    """Opt-in NCCL Xfer updater with explicit broadcast fallback.

    The first implementation wires the control plane and layout checks but does
    not fake native payload transfer. Until a Python bridge for
    ``ncclXferReshardWithWindow`` exists, updates delegate to the existing
    non-colocated broadcast backend with a clear log message.
    """

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        weights_getter: Callable[[], Mapping[str, torch.Tensor]],
        *,
        model_name: str,
        quantization_config: dict[str, int | str | list[str]] | None,
    ) -> None:
        self.args = args
        self.model = model
        self.quantization_config = quantization_config
        self._fallback = UpdateWeightFromDistributed(
            args,
            model,
            weights_getter,
            model_name=model_name,
            quantization_config=quantization_config,
        )
        self._fallback_logged = False
        self._fallback_reason_cache = self._determine_fallback_reason()

    def connect_rollout_engines(
        self,
        rollout_engines: Sequence[ActorHandle],
        rollout_engine_lock: ActorHandle,
        engine_gpu_counts: Sequence[int] | None = None,
        engine_gpu_offsets: Sequence[int] | None = None,
    ) -> None:
        self._fallback.connect_rollout_engines(
            rollout_engines,
            rollout_engine_lock,
            engine_gpu_counts=engine_gpu_counts,
            engine_gpu_offsets=engine_gpu_offsets,
        )

    def disconnect_rollout_engines(self) -> None:
        self._fallback.disconnect_rollout_engines()

    @torch.no_grad()
    def update_weights(self) -> None:
        reason = self._fallback_reason()
        if reason is not None:
            if not self._fallback_logged:
                logger.warning("NCCL Xfer weight sync unavailable; falling back to broadcast: %s", reason)
                self._fallback_logged = True
            self._fallback.update_weights()
            return

        raise NotImplementedError("native NCCL Xfer weight transfer is not implemented")

    def _fallback_reason(self) -> str | None:
        return self._fallback_reason_cache

    def _determine_fallback_reason(self) -> str | None:
        availability = get_nccl_xfer_availability()
        if not availability.available:
            return availability.reason or "native NCCL Xfer bridge is unavailable"

        for name, tensor in named_params_and_buffers(self.args, self.model):
            decision = analyze_nccl_xfer_layout(
                name,
                tuple(tensor.shape),
                tensor.dtype,
                quantization_config=self.quantization_config,
            )
            if not decision.supported:
                return decision.reason
        return None
