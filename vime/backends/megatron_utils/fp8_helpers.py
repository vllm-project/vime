"""FP8 / UE8M0 quantization helpers for megatron → vLLM weight transfer.

All symbols fall back to ``None`` when vLLM's deep_gemm helpers are not
available, which disables the UE8M0 requantization path.
"""

import torch

# deep_gemm is a third-party DeepSeek MoE GEMM library; not all images ship it.
# When missing, the UE8M0 requantization path falls back to None so callers can
# skip it gracefully (matches the module docstring contract).
try:
    import deep_gemm.utils.layout as _deep_gemm_layout
    from vllm.utils.deep_gemm import get_tma_aligned_size as _get_tma_aligned_size

    _HAS_DEEP_GEMM = True
except ImportError:
    _deep_gemm_layout = None
    _get_tma_aligned_size = None
    _HAS_DEEP_GEMM = False

try:
    from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used as _vllm_is_e8m0
    from vllm.utils.deep_gemm import per_block_cast_to_fp8 as _vllm_per_block_cast
except ImportError:
    _vllm_is_e8m0 = lambda: False  # noqa: E731
    _vllm_per_block_cast = None


def should_deepgemm_weight_requant_ue8m0(weight_block_size) -> bool:
    return weight_block_size is not None and _vllm_is_e8m0()


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: list[int],
):
    assert weight_block_size == [128, 128]
    assert weight_dequant.dtype == torch.bfloat16, f"{weight_dequant.dtype=} {weight_dequant.shape=}"
    *batch_dims, n, k = weight_dequant.shape
    flat = weight_dequant.view(-1, k)
    out_w_flat, out_s_flat = _vllm_per_block_cast(flat, block_size=[128, 128], use_ue8m0=True)
    out_w = out_w_flat.view(*batch_dims, n, k)
    from math import ceil

    out_s = out_s_flat.view(
        *batch_dims,
        ceil(n / weight_block_size[0]),
        ceil(k / weight_block_size[1]),
    )
    return out_w, out_s


def transform_scale_ue8m0(sf: torch.Tensor, mn: int, use_torch_impl: bool = False):
    if _deep_gemm_layout is None:
        raise RuntimeError("deep_gemm not installed; UE8M0 requantization unavailable.")
    get_fn = _deep_gemm_layout.get_mn_major_tma_aligned_packed_ue8m0_tensor
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = get_fn(sf)
    if sf.shape[-1] == 1:
        get_tma_aligned_size = _get_tma_aligned_size  # pre-imported with fallback

        aligned_mn = get_tma_aligned_size(sf.shape[-2], sf.element_size())
        if sf.stride(-1) != aligned_mn:
            new_stride = list(sf.stride())
            new_stride[-1] = aligned_mn
            sf = sf.as_strided(sf.shape, tuple(new_stride))
    return sf


__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
]
