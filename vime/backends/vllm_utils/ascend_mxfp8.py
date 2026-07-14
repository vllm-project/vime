"""Ascend W8A8-MXFP8 helpers for online rollout weight updates.

Optional Ascend dependencies stay lazily imported so importing Vime on CPU,
CUDA, or ROCm does not require ``vllm-ascend`` or ``torch-npu``.
"""

from __future__ import annotations

from collections.abc import Iterable, Iterator

import torch

MXFP8_QUANT_TYPE = "W8A8_MXFP8"


def is_ascend_mxfp8_config(quant_config: object) -> bool:
    """Return whether *quant_config* is a ModelSlim config containing MXFP8."""
    try:
        from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig
    except ImportError:
        return False

    if not isinstance(quant_config, AscendModelSlimConfig):
        return False
    quant_description = getattr(quant_config, "quant_description", {})
    return MXFP8_QUANT_TYPE in quant_description.values()


def _mxfp8_scheme(module: object) -> object | None:
    quant_method = getattr(module, "quant_method", None)
    scheme = getattr(quant_method, "quant_method", quant_method)
    if callable(getattr(scheme, "restore_weights_for_rl_loading", None)) and callable(
        getattr(scheme, "process_weights_after_loading", None)
    ):
        return scheme
    return None


def _module_from_param_name(model: object, name: str) -> object | None:
    module_path = name.split(".")[:-1]
    if not module_path:
        return None

    packed_mapping = getattr(model, "packed_modules_mapping", {})
    reversed_mapping = {
        original_name: fused_name
        for fused_name, original_names in packed_mapping.items()
        for original_name in original_names
    }
    if module_path[-1] in reversed_mapping:
        module_path[-1] = reversed_mapping[module_path[-1]]

    current = model
    try:
        for part in module_path:
            # Fused MoE checkpoints contain deeper expert paths even though the
            # target vLLM module is already the fused quantized module.
            if _mxfp8_scheme(current) is not None and hasattr(current, "w13_weight"):
                return current
            if isinstance(current, (torch.nn.ModuleList, torch.nn.Sequential)):
                current = current[int(part)]
            else:
                current = getattr(current, part)
    except (AttributeError, IndexError, KeyError, TypeError, ValueError):
        return None
    return current


def _is_mxfp8_weight(name: str, model: object) -> bool:
    if not name.endswith("weight"):
        return False
    module = _module_from_param_name(model, name)
    return module is not None and _mxfp8_scheme(module) is not None


def quantize_mxfp8_weights(
    weights: Iterable[tuple[str, torch.Tensor]],
    model: object,
    dtype: torch.dtype = torch.bfloat16,
) -> Iterator[tuple[str, torch.Tensor]]:
    """Convert eligible high-precision weights to Ascend MXFP8 on demand."""
    import torch_npu

    for name, value in weights:
        if not _is_mxfp8_weight(name, model):
            yield name, value
            continue

        quantized, scale = torch_npu.npu_dynamic_mx_quant(
            value.to(dtype),
            axis=-1,
            dst_type=torch_npu.float8_e4m3fn,
        )
        scale = scale.flatten(-2, -1).squeeze(-1)
        yield name, quantized
        yield name + "_scale", scale


def _has_original_shapes(module: object) -> bool:
    original_shapes = getattr(module, "_mxfp8_original_shapes", None)
    if not isinstance(original_shapes, dict) or not original_shapes:
        return False
    for name, expected_shape in original_shapes.items():
        tensor = getattr(module, name, None)
        if tensor is None or tuple(tensor.shape) != tuple(expected_shape):
            return False
    return True


def prepare_mxfp8_modules_for_reload(model: torch.nn.Module) -> int:
    """Put MXFP8 modules into model-format layout before loading weights.

    Current vLLM restores model-format tensors from recorded metadata during
    ``start_weight_update``. In that case only the vLLM-Ascend idempotence
    marker needs resetting. Older/non-native paths still use the scheme's
    explicit restore operation.
    """
    prepared = 0
    for module in model.modules():
        scheme = _mxfp8_scheme(module)
        if scheme is None:
            continue
        if getattr(module, "_mxfp8_transformed", False):
            if _has_original_shapes(module):
                module._mxfp8_transformed = False
            else:
                scheme.restore_weights_for_rl_loading(module)
        prepared += 1
    return prepared


def finalize_mxfp8_modules_after_reload(model: torch.nn.Module) -> int:
    """Reapply vLLM-Ascend inference layouts after a successful reload."""
    finalized = 0
    for module in model.modules():
        scheme = _mxfp8_scheme(module)
        if scheme is None or getattr(module, "_mxfp8_transformed", False):
            continue
        scheme.process_weights_after_loading(module)
        finalized += 1
    return finalized


__all__ = [
    "MXFP8_QUANT_TYPE",
    "finalize_mxfp8_modules_after_reload",
    "is_ascend_mxfp8_config",
    "prepare_mxfp8_modules_for_reload",
    "quantize_mxfp8_weights",
]
