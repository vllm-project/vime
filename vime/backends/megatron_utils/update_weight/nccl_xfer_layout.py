from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

NCCL_XFER_MAX_TENSOR_DIMS = 3

_COLUMN_PARALLEL_SUFFIXES = (
    "q_proj.weight",
    "k_proj.weight",
    "v_proj.weight",
    "gate_proj.weight",
    "up_proj.weight",
    "q_a_proj.weight",
    "q_b_proj.weight",
    "kv_a_proj_with_mqa.weight",
    "kv_b_proj.weight",
)
_ROW_PARALLEL_SUFFIXES = ("o_proj.weight", "down_proj.weight")
_VOCAB_PARALLEL_NAMES = ("embed_tokens.weight", "lm_head.weight")

_SUPPORTED_DTYPES = {
    dtype
    for dtype in (
        torch.int8,
        torch.uint8,
        getattr(torch, "float8_e4m3fn", None),
        getattr(torch, "float8_e5m2", None),
        torch.float16,
        torch.bfloat16,
        torch.int32,
        torch.uint32,
        torch.float32,
        torch.int64,
        torch.uint64,
        torch.float64,
    )
    if dtype is not None
}


@dataclass(frozen=True)
class NcclXferLayoutDecision:
    supported: bool
    reason: str | None = None
    shard_tensor_dim: int | None = None
    replicated: bool = False


def analyze_nccl_xfer_layout(
    name: str,
    shape: Sequence[int],
    dtype: torch.dtype,
    *,
    quantization_config: Mapping[str, Any] | object | None = None,
) -> NcclXferLayoutDecision:
    """Classify whether a model weight can use the MVP NCCL Xfer layout mapping.

    This mirrors the RFC's first-pass placement rules. It is intentionally
    conservative: unsupported cases should fall back to the existing broadcast
    path instead of entering a partially implemented native transfer.
    """

    if _get_quant_method(quantization_config) == "compressed-tensors":
        return NcclXferLayoutDecision(False, "compressed-tensors quantized weights require broadcast fallback")

    ndim = len(shape)
    if ndim == 0:
        return NcclXferLayoutDecision(False, f"{name}: scalar tensors are not supported by NCCL Xfer reshard")
    if ndim > NCCL_XFER_MAX_TENSOR_DIMS:
        return NcclXferLayoutDecision(False, f"{name}: tensor rank {ndim} exceeds NCCL Xfer limit of 3")
    if dtype not in _SUPPORTED_DTYPES:
        return NcclXferLayoutDecision(False, f"{name}: dtype {dtype} is not supported by NCCL Xfer")

    if ".experts." in name:
        if ndim != 3:
            return NcclXferLayoutDecision(
                False,
                f"{name}: MoE expert tensors must be grouped as rank-3 [num_experts, out, in]",
            )
        return NcclXferLayoutDecision(True, shard_tensor_dim=0)

    if ndim < 2:
        return NcclXferLayoutDecision(True, shard_tensor_dim=None, replicated=True)

    if _endswith_any(name, _COLUMN_PARALLEL_SUFFIXES) or _contains_any(name, _VOCAB_PARALLEL_NAMES):
        return NcclXferLayoutDecision(True, shard_tensor_dim=0)

    if _endswith_any(name, _ROW_PARALLEL_SUFFIXES):
        return NcclXferLayoutDecision(True, shard_tensor_dim=1)

    return NcclXferLayoutDecision(True, shard_tensor_dim=None, replicated=True)


def _endswith_any(name: str, suffixes: Sequence[str]) -> bool:
    return any(name.endswith(suffix) for suffix in suffixes)


def _contains_any(name: str, needles: Sequence[str]) -> bool:
    return any(needle in name for needle in needles)


def _get_quant_method(quantization_config: Mapping[str, Any] | object | None) -> str | None:
    if quantization_config is None:
        return None
    if isinstance(quantization_config, Mapping):
        return quantization_config.get("quant_method")
    return getattr(quantization_config, "quant_method", None)
