from __future__ import annotations

import hashlib
import logging
from argparse import Namespace
from collections.abc import Sequence
from typing import Any

import torch

from .config import resolve_feature_layer_ids
from .feature_schema import DraftFeatureSample

logger = logging.getLogger(__name__)


def _unwrap_module(module: torch.nn.Module) -> torch.nn.Module:
    while hasattr(module, "module") and isinstance(module.module, torch.nn.Module):
        module = module.module
    return module


def _getattr_path(root: object, path: str) -> object | None:
    value: object | None = root
    for name in path.split("."):
        value = getattr(value, name, None)
        if value is None:
            return None
    return value


def _find_transformer_layers(root: torch.nn.Module) -> list[torch.nn.Module]:
    for path in ("decoder.layers", "transformer.layers", "language_model.decoder.layers"):
        value = _getattr_path(root, path)
        if isinstance(value, (torch.nn.ModuleList, list, tuple)):
            return list(value)
    raise RuntimeError(
        "Cannot locate Megatron transformer layers for external Draft feature collection. "
        "Expected decoder.layers, transformer.layers, or language_model.decoder.layers."
    )


def _find_output_layer(root: torch.nn.Module) -> torch.nn.Module:
    for path in ("output_layer", "lm_head", "language_model.output_layer"):
        value = _getattr_path(root, path)
        if isinstance(value, torch.nn.Module):
            return value
    raise RuntimeError("Cannot locate the Target output layer used to capture final normalized hidden states")


def find_target_output_weight(model: Sequence[torch.nn.Module]) -> torch.Tensor:
    if len(model) != 1:
        raise RuntimeError("External Draft MVP requires one virtual pipeline model chunk")
    output_layer = _find_output_layer(_unwrap_module(model[0]))
    weight = getattr(output_layer, "weight", None)
    if not torch.is_tensor(weight):
        # Megatron allocates no separate output weight when input/output
        # embeddings are tied.  In that case the canonical accessor returns
        # the embedding tensor used by the output projection.
        root = _unwrap_module(model[0])
        shared_weight = getattr(root, "shared_embedding_or_output_weight", None)
        if callable(shared_weight):
            weight = shared_weight()
    if not torch.is_tensor(weight):
        raise RuntimeError(
            "Target output layer does not expose a weight tensor and the model "
            "does not provide a shared embedding/output weight"
        )
    return weight


def _first_tensor(value: Any) -> torch.Tensor:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (list, tuple)):
        for item in value:
            if torch.is_tensor(item):
                return item
    if isinstance(value, dict):
        for item in value.values():
            if torch.is_tensor(item):
                return item
    raise TypeError(f"Draft feature hook expected a tensor output, got {type(value).__name__}")


def _as_token_hidden(tensor: torch.Tensor) -> torch.Tensor:
    tensor = tensor.detach()
    if tensor.dim() == 3:
        if tensor.size(1) == 1:  # Megatron THD/TBH packed path: [T, 1, H]
            tensor = tensor[:, 0]
        elif tensor.size(0) == 1:  # [1, T, H]
            tensor = tensor[0]
        else:
            raise ValueError(f"Cannot flatten hidden tensor with shape {tuple(tensor.shape)}")
    if tensor.dim() != 2:
        raise ValueError(f"Draft hidden tensor must reduce to [tokens, hidden], got {tuple(tensor.shape)}")
    return tensor


def _hash_fraction(key: str) -> float:
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") / float(1 << 64)


def _hash_offset(key: str, inclusive_max: int) -> int:
    if inclusive_max <= 0:
        return 0
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % (inclusive_max + 1)


class DraftFeatureCollector:
    """Collect selected Target layer activations from a Megatron forward-only pass."""

    def __init__(
        self,
        args: Namespace,
        model: Sequence[torch.nn.Module],
        *,
        rollout_id: int,
        target_weight_version: str,
    ) -> None:
        if len(model) != 1:
            raise RuntimeError("External Draft MVP requires VPP=1")
        self.args = args
        self.rollout_id = int(rollout_id)
        self.target_weight_version = str(target_weight_version)
        self.algorithm = str(getattr(args, "draft_algorithm", "eagle3")).lower()
        self.layer_ids = tuple(resolve_feature_layer_ids(args))
        self.block_size = int(getattr(args, "draft_dspark_block_size", 0) or 0)
        self.sample_rate = float(getattr(args, "draft_collection_sample_rate", 1.0))
        self.max_samples = int(getattr(args, "draft_max_samples_per_rollout_per_dp", 16))
        self.max_tokens = int(getattr(args, "draft_max_tokens_per_rollout_per_dp", 16384))
        self.window_tokens = int(getattr(args, "draft_hidden_window_tokens", 512))
        self.window_mode = str(getattr(args, "draft_hidden_window_mode", "front"))
        self.seed = int(getattr(args, "draft_random_seed", 1234))
        self._handles: list[Any] = []
        self._captured_layers: dict[int, torch.Tensor] = {}
        self._captured_final: torch.Tensor | None = None
        self._current_batch: dict[str, Any] | None = None
        self._current_indices: list[int] = []
        self._payloads: list[dict[str, Any]] = []
        self._collected_tokens = 0

        root = _unwrap_module(model[0])
        layers = _find_transformer_layers(root)
        if max(self.layer_ids) >= len(layers):
            raise RuntimeError(
                f"Configured Draft layer ids {self.layer_ids} exceed local Target layer count {len(layers)}"
            )
        for layer_id in self.layer_ids:
            self._handles.append(layers[layer_id].register_forward_hook(self._make_layer_hook(layer_id)))
        self._handles.append(_find_output_layer(root).register_forward_pre_hook(self._final_hidden_hook))

    def _make_layer_hook(self, layer_id: int):
        def hook(_module, _inputs, output):
            if self._current_batch is not None:
                self._captured_layers[layer_id] = _as_token_hidden(_first_tensor(output))

        return hook

    def _final_hidden_hook(self, _module, inputs):
        if self._current_batch is not None:
            self._captured_final = _as_token_hidden(_first_tensor(inputs))

    def begin_microbatch(self, batch: dict[str, Any], original_indices: Sequence[int]) -> None:
        if self._current_batch is not None:
            raise RuntimeError("DraftFeatureCollector.begin_microbatch called before the previous batch ended")
        self._current_batch = batch
        self._current_indices = [int(value) for value in original_indices]
        self._captured_layers = {}
        self._captured_final = None

    def _gather_sequence_parallel(self, tensor: torch.Tensor) -> torch.Tensor:
        try:
            from megatron.core import mpu
        except ImportError:
            return tensor
        tp_size = int(mpu.get_tensor_model_parallel_world_size())
        if tp_size <= 1 or not bool(getattr(self.args, "sequence_parallel", False)):
            return tensor
        gathered = [torch.empty_like(tensor) for _ in range(tp_size)]
        torch.distributed.all_gather(gathered, tensor, group=mpu.get_tensor_model_parallel_group())
        return torch.cat(gathered, dim=0)

    @staticmethod
    def _is_export_rank() -> bool:
        try:
            from megatron.core import mpu
        except ImportError:
            return True
        return int(mpu.get_tensor_model_parallel_rank()) == 0

    @staticmethod
    def _dp_rank() -> int:
        try:
            from megatron.core import mpu
        except ImportError:
            return 0
        return int(mpu.get_data_parallel_rank(with_context_parallel=False))

    def end_microbatch(self) -> None:
        batch = self._current_batch
        try:
            if batch is None:
                raise RuntimeError("DraftFeatureCollector.end_microbatch called without begin_microbatch")
            missing = [layer_id for layer_id in self.layer_ids if layer_id not in self._captured_layers]
            if missing or self._captured_final is None:
                raise RuntimeError(
                    f"Draft feature hooks did not capture all tensors: missing_layers={missing}, "
                    f"missing_final={self._captured_final is None}"
                )
            layers = [self._gather_sequence_parallel(self._captured_layers[layer_id]) for layer_id in self.layer_ids]
            final_hidden = self._gather_sequence_parallel(self._captured_final)
            if not self._is_export_rank():
                return
            packed_aux = torch.cat(layers, dim=-1)
            tokens_list = batch["unconcat_tokens"]
            total_lengths = [int(value) for value in batch["total_lengths"]]
            response_lengths = [int(value) for value in batch["response_lengths"]]
            response_masks = batch["loss_masks"]
            packed_rows = sum(total_lengths)
            if packed_aux.size(0) < packed_rows or final_hidden.size(0) < packed_rows:
                raise RuntimeError(
                    "Captured hidden state is shorter than the packed token stream: "
                    f"aux={packed_aux.size(0)}, final={final_hidden.size(0)}, tokens={packed_rows}"
                )
            offset = 0
            for local_index, (tokens, total_length, response_length, response_mask) in enumerate(
                zip(tokens_list, total_lengths, response_lengths, response_masks, strict=True)
            ):
                original_index = self._current_indices[local_index]
                sample_aux = packed_aux[offset : offset + total_length]
                sample_final = final_hidden[offset : offset + total_length]
                offset += total_length
                self._maybe_collect_sample(
                    tokens=tokens,
                    response_mask=response_mask,
                    aux_hidden=sample_aux,
                    final_hidden=sample_final,
                    total_length=total_length,
                    response_length=response_length,
                    original_index=original_index,
                )
        finally:
            self._current_batch = None
            self._current_indices = []
            self._captured_layers = {}
            self._captured_final = None

    def _maybe_collect_sample(
        self,
        *,
        tokens: torch.Tensor,
        response_mask: torch.Tensor,
        aux_hidden: torch.Tensor,
        final_hidden: torch.Tensor,
        total_length: int,
        response_length: int,
        original_index: int,
    ) -> None:
        if len(self._payloads) >= self.max_samples or self._collected_tokens >= self.max_tokens:
            return
        prompt_length = total_length - response_length
        min_rows = self.block_size + 2 if self.algorithm == "dspark" else 3
        min_response = self.block_size + 1 if self.algorithm == "dspark" else 2
        if response_length < min_response or total_length < min_rows:
            return
        sample_id = f"dp{self._dp_rank()}-sample{original_index}"
        key = f"{self.seed}:{self.rollout_id}:{sample_id}"
        if _hash_fraction(key) >= self.sample_rate:
            return
        feature_start = max(prompt_length - 1, 0)
        available = total_length - feature_start
        window_rows = min(self.window_tokens, available, self.max_tokens - self._collected_tokens)
        if window_rows < min_rows:
            return
        max_offset = max(available - window_rows, 0)
        window_offset = _hash_offset(key, max_offset) if self.window_mode == "random" else 0
        start = feature_start + window_offset
        end = start + window_rows

        token_loss_mask = torch.zeros(total_length, dtype=torch.float32, device=response_mask.device)
        response_mask = response_mask.reshape(-1)[:response_length].float()
        token_loss_mask[prompt_length : prompt_length + response_mask.numel()] = response_mask
        positions = torch.arange(start, end, dtype=torch.long, device=tokens.device)
        sample = DraftFeatureSample(
            input_ids=tokens.reshape(-1)[start:end],
            loss_mask=token_loss_mask[start:end],
            position_ids=positions,
            aux_hidden_states=aux_hidden[start:end],
            final_hidden_states=final_hidden[start:end],
            rollout_id=self.rollout_id,
            target_weight_version=self.target_weight_version,
            original_sample_id=sample_id,
            prompt_length=prompt_length,
            response_length=response_length,
            window_start=start,
            window_end=end,
            aux_layer_ids=self.layer_ids,
            algorithm=self.algorithm,
        )
        self._payloads.append(sample.to_payload())
        self._collected_tokens += window_rows

    def pop_payloads(self) -> list[dict[str, Any]]:
        payloads, self._payloads = self._payloads, []
        self._collected_tokens = 0
        return payloads

    def abort_microbatch(self) -> None:
        self._current_batch = None
        self._current_indices = []
        self._captured_layers = {}
        self._captured_final = None

    def close(self) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []

    def __enter__(self) -> DraftFeatureCollector:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()
