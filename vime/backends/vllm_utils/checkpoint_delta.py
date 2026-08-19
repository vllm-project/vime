from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from vllm.distributed.weight_transfer import WeightTransferEngineFactory
from vllm.distributed.weight_transfer.base import WeightTransferUpdateInfo
from vllm.distributed.weight_transfer.nccl_engine import NCCLWeightTransferEngine

if TYPE_CHECKING:
    from vllm.config import VllmConfig

VIME_DELTA_NCCL_BACKEND = "vime_delta_nccl"
VIME_DELTA_WORKER_EXTENSION = "vime.backends.vllm_utils.checkpoint_delta.VimeDeltaWorkerExtension"
_REGISTERED = False


def _checkpoint_patch_api():
    try:
        from vllm.model_executor.model_loader.checkpoint_weight_patch import (
            CheckpointWeightPatch,
            load_checkpoint_weight_patches,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.model_executor.model_loader.checkpoint_weight_patch":
            raise
        raise RuntimeError("VIME direct DWU requires a vLLM build containing PR #50723") from exc
    return CheckpointWeightPatch, load_checkpoint_weight_patches


def _layerwise_reload_api():
    try:
        from vllm.model_executor.model_loader.reload import (
            finalize_layerwise_reload,
            initialize_layerwise_reload,
        )
    except ModuleNotFoundError as exc:
        if exc.name != "vllm.model_executor.model_loader.reload":
            raise
        raise RuntimeError(
            "VIME direct DWU dense seeding requires vLLM's layerwise reload API"
        ) from exc
    return initialize_layerwise_reload, finalize_layerwise_reload


# GPU staging bound for the checkpoint patch API: full checkpoint-shaped
# tensors accumulate up to this target per internal load_weights call. Peak
# transient memory is this target plus the largest single checkpoint tensor,
# because a tensor bigger than the target is still staged whole.
_PATCH_CHUNK_BYTES = 256 << 20


@dataclass
class VimeDeltaNCCLUpdateInfo(WeightTransferUpdateInfo):
    schema_version: int
    base_version: int
    target_version: int
    sequence_no: int
    is_final: bool
    encoding: str
    patches: list[dict[str, Any]]
    position_count: int
    value_count: int
    value_dtype_name: str

    def __post_init__(self) -> None:
        if self.schema_version != 1:
            raise ValueError(f"Unsupported checkpoint delta schema: {self.schema_version}")
        if self.target_version != self.base_version + 1:
            raise ValueError("target_version must be base_version + 1")
        if self.sequence_no < 0:
            raise ValueError("sequence_no must be non-negative")
        if self.encoding not in {"dense", "indices"}:
            raise ValueError(f"Unsupported checkpoint delta encoding: {self.encoding!r}")
        if self.position_count < 0 or self.value_count < 0:
            raise ValueError("Checkpoint delta tensor sizes must be non-negative")
        if self.is_final:
            if self.patches or self.position_count or self.value_count:
                raise ValueError("A final checkpoint delta manifest cannot carry data")
            return
        if not self.patches or self.value_count == 0:
            raise ValueError("A non-final checkpoint delta chunk must carry patches")
        if self.encoding == "dense" and self.position_count:
            raise ValueError("Dense checkpoint delta chunks must not contain positions")
        if self.encoding == "indices" and self.position_count != self.value_count:
            raise ValueError("Sparse checkpoint delta chunks require one position per value")


class VimeDeltaNCCLWeightTransferEngine(NCCLWeightTransferEngine):
    """Receive VIME checkpoint-coordinate patches over the stock NCCL group."""

    update_info_cls = VimeDeltaNCCLUpdateInfo
    supports_draft_weight_update = False

    def __init__(
        self,
        config,
        vllm_config: VllmConfig,
        device: torch.device,
        model: torch.nn.Module,
    ) -> None:
        super().__init__(config, vllm_config, device, model)
        # Fail at engine construction, not mid-session, when the vLLM build
        # lacks either required model-loader API.
        _checkpoint_patch_api()
        _layerwise_reload_api()
        if self.device.type != "cuda":
            raise NotImplementedError("VIME direct DWU requires CUDA")
        if self.model_config.dtype != torch.bfloat16:
            raise NotImplementedError("VIME direct DWU currently supports BF16 only")
        if getattr(self.vllm_config, "quant_config", None) is not None:
            raise NotImplementedError("VIME direct DWU does not support quantized models")
        if getattr(self.vllm_config, "speculative_config", None) is not None:
            raise NotImplementedError("VIME direct DWU does not update speculative draft models")
        self._committed_version = 0
        self._session_base_version: int | None = None
        self._session_target_version: int | None = None
        self._session_encoding: str | None = None
        self._next_sequence_no = 0
        self._reload_initialized = False
        self._final_received = False
        self._update_failed = False

    def init_transfer_engine(self, init_info) -> None:
        super().init_transfer_engine(init_info)

    def update_weights(self, update_info: dict[str, Any]) -> None:
        try:
            super().update_weights(update_info)
        except BaseException:
            self._update_failed = True
            raise

    def start_weight_update(self) -> None:
        if self._update_failed:
            raise RuntimeError("A previous direct DWU session failed; restart and dense-seed the vLLM workers")
        self._session_base_version = None
        self._session_target_version = None
        self._session_encoding = None
        self._next_sequence_no = 0
        self._reload_initialized = False
        self._final_received = False

    def receive_weights(self, update_info: VimeDeltaNCCLUpdateInfo) -> None:
        if self.model_update_group is None:
            raise RuntimeError("VIME direct DWU NCCL group is not initialized")

        try:
            if update_info.is_final:
                self._accept_chunk_metadata(update_info)
                if self._final_received:
                    raise ValueError("A direct DWU session received two final manifests")
                if update_info.encoding == "dense" and not self._reload_initialized:
                    raise ValueError("A dense direct DWU session did not carry any weights")
                self._final_received = True
                return

            # Drain this chunk from NCCL before consulting worker-local session
            # state. If one TP worker has a stale version or failed earlier, all
            # ranks still complete the same collectives and fail-stop cleanly
            # instead of stranding the trainer and its healthy peers.
            positions = torch.empty(
                update_info.position_count,
                dtype=torch.int32,
                device=self.device,
            )
            if positions.numel():
                self.model_update_group.broadcast(
                    positions,
                    src=0,
                    stream=torch.cuda.current_stream(),
                )

            values = torch.empty(
                update_info.value_count,
                dtype=torch.bfloat16,
                device=self.device,
            )
            self.model_update_group.broadcast(
                values,
                src=0,
                stream=torch.cuda.current_stream(),
            )

            self._accept_chunk_metadata(update_info)
            if self._final_received:
                raise ValueError("Checkpoint delta data arrived after the final manifest")
            if update_info.value_dtype_name != "bfloat16":
                raise ValueError("VIME direct DWU wire values must be BF16")

            if update_info.encoding == "dense" and not self._reload_initialized:
                initialize_layerwise_reload, _ = _layerwise_reload_api()
                self._reload_initialized = True
                try:
                    initialize_layerwise_reload(self.model)
                except BaseException:
                    self._reload_initialized = False
                    raise

            CheckpointWeightPatch, load_checkpoint_weight_patches = _checkpoint_patch_api()
            patches = []
            for spec in update_info.patches:
                patch_indices = None
                if update_info.encoding == "indices":
                    patch_indices = positions[spec["position_start"] : spec["position_end"]]
                patches.append(
                    CheckpointWeightPatch(
                        name=spec["name"],
                        shape=tuple(spec["shape"]),
                        dtype=getattr(torch, spec["dtype_name"]),
                        values=values[spec["value_start"] : spec["value_end"]],
                        indices=patch_indices,
                    )
                )
            # Indices come from one torch.nonzero over a bitwise compare on the
            # source, so they are unique by construction; skip the per-patch
            # duplicate sort.
            load_checkpoint_weight_patches(
                self.model,
                patches,
                max_chunk_bytes=_PATCH_CHUNK_BYTES,
                validate_unique_indices=False,
            )
        except BaseException:
            self._update_failed = True
            raise

    def _accept_chunk_metadata(self, update_info: VimeDeltaNCCLUpdateInfo) -> None:
        if self._session_base_version is None:
            if update_info.base_version != self._committed_version:
                raise RuntimeError(
                    f"Checkpoint delta base version mismatch: worker={self._committed_version}, update={update_info.base_version}"
                )
            self._session_base_version = update_info.base_version
            self._session_target_version = update_info.target_version
        elif (
            update_info.base_version != self._session_base_version
            or update_info.target_version != self._session_target_version
        ):
            raise ValueError("One direct DWU session cannot mix update versions")

        if update_info.sequence_no != self._next_sequence_no:
            raise ValueError(
                f"Checkpoint delta sequence mismatch: expected {self._next_sequence_no}, got {update_info.sequence_no}"
            )
        self._next_sequence_no += 1

        if self._session_encoding is None:
            self._session_encoding = update_info.encoding
        elif self._session_encoding != update_info.encoding:
            raise ValueError("One direct DWU session cannot mix dense and sparse chunks")

    def finish_weight_update(self) -> None:
        try:
            if not self._final_received:
                raise RuntimeError("Direct DWU session ended without a final manifest")
            if self._session_encoding == "dense":
                _, finalize_layerwise_reload = _layerwise_reload_api()
                finalize_layerwise_reload(self.model, self.model_config)
            assert self._session_target_version is not None
            self._committed_version = self._session_target_version
        except BaseException:
            self._update_failed = True
            raise
        finally:
            self._session_base_version = None
            self._session_target_version = None
            self._session_encoding = None
            self._next_sequence_no = 0
            self._reload_initialized = False
            self._final_received = False

    def shutdown(self) -> None:
        self._session_encoding = None
        super().shutdown()


def register_vime_delta_weight_transfer_engine() -> None:
    global _REGISTERED
    if _REGISTERED:
        return
    WeightTransferEngineFactory.register_engine(
        VIME_DELTA_NCCL_BACKEND,
        VimeDeltaNCCLWeightTransferEngine,
    )
    _REGISTERED = True


class VimeDeltaWorkerExtension:
    """Register the VIME WTE before the vLLM worker loads its model."""


# Resolving ``worker_extension_cls`` imports this module before GPUWorker loads
# the model and asks the factory to create its configured transfer engine.
register_vime_delta_weight_transfer_engine()
