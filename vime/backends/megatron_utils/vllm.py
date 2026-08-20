"""vLLM FP8 helpers used by Megatron weight conversion."""

from math import ceil

import torch
from vllm.utils.deep_gemm import (
    get_mn_major_tma_aligned_packed_ue8m0_tensor,
    get_tma_aligned_size,
    is_deep_gemm_e8m0_used,
    per_block_cast_to_fp8,
)


def should_deepgemm_weight_requant_ue8m0(weight_block_size) -> bool:
    return weight_block_size is not None and is_deep_gemm_e8m0_used()


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: list[int],
):
    assert weight_block_size == [128, 128]
    assert weight_dequant.dtype == torch.bfloat16, f"{weight_dequant.dtype=} {weight_dequant.shape=}"
    *batch_dims, n, k = weight_dequant.shape
    flat = weight_dequant.view(-1, k)
    out_w_flat, out_s_flat = per_block_cast_to_fp8(flat, block_size=[128, 128], use_ue8m0=True)
    out_w = out_w_flat.view(*batch_dims, n, k)
    out_s = out_s_flat.view(
        *batch_dims,
        ceil(n / weight_block_size[0]),
        ceil(k / weight_block_size[1]),
    )
    return out_w, out_s


def transform_scale_ue8m0(sf: torch.Tensor, mn: int):
    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    if sf.shape[-1] == 1:
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
