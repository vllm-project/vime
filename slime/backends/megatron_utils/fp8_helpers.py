"""FP8 / UE8M0 quantization helpers for megatron → vLLM weight transfer.

``quant_weight_ue8m0`` / ``transform_scale_ue8m0`` (and their DeepGEMM-derived
helpers ``per_block_cast_to_fp8`` / ``ceil_to_ue8m0`` / ``ceil_div`` /
``ceil_align`` / ``_get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl``)
are copied verbatim from SGLang's
``sglang/srt/layers/quantization/fp8_utils.py``. They depend only on ``torch``
and the third-party ``deep_gemm`` package, which is imported lazily inside the
functions (so importing this module never requires deep_gemm).

Only ``should_deepgemm_weight_requant_ue8m0`` is adapted: SGLang's original reads
SGLang-internal ``deep_gemm_wrapper`` flags, so we use vLLM's equivalent signal.
"""

from typing import List, Tuple

import torch


# COPIED FROM SGLang (sglang/srt/utils/common.py)
def ceil_div(x: int, y: int) -> int:
    return (x + y - 1) // y


# COPIED FROM SGLang (sglang/srt/utils/common.py)
def ceil_align(x: int, y: int) -> int:
    return ceil_div(x, y) * y


# COPIED FROM DeepGEMM
def ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))


# COPIED FROM DeepGEMM
def per_block_cast_to_fp8(x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    assert x.dim() == 2
    m, n = x.shape
    x_padded = torch.zeros(
        (ceil_align(m, 128), ceil_align(n, 128)), dtype=x.dtype, device=x.device
    )
    x_padded[:m, :n] = x
    x_view = x_padded.view(-1, 128, x_padded.size(1) // 128, 128)
    x_amax = x_view.abs().float().amax(dim=(1, 3), keepdim=True).clamp(1e-4)
    sf = ceil_to_ue8m0(x_amax / 448.0)
    x_scaled = (x_view * (1.0 / sf)).to(torch.float8_e4m3fn)
    return x_scaled.view_as(x_padded)[:m, :n].contiguous(), sf.view(
        x_view.size(0), x_view.size(2)
    )


def quant_weight_ue8m0(
    weight_dequant: torch.Tensor,
    weight_block_size: List[int],
):
    assert weight_block_size == [128, 128]
    assert (
        weight_dequant.dtype == torch.bfloat16
    ), f"{weight_dequant.dtype=} {weight_dequant.shape=}"

    *batch_dims, n, k = weight_dequant.shape

    weight_dequant_flat = weight_dequant.view((-1, k))
    out_w_flat, out_s_flat = per_block_cast_to_fp8(weight_dequant_flat)

    out_w = out_w_flat.view((*batch_dims, n, k))
    out_s = out_s_flat.view(
        (
            *batch_dims,
            ceil_div(n, weight_block_size[0]),
            ceil_div(k, weight_block_size[1]),
        )
    )

    return out_w, out_s


# NOTE copy and modified from DeepGEMM
def transform_scale_ue8m0(sf, mn, use_torch_impl: bool = False):
    import deep_gemm.utils.layout

    get_mn_major_tma_aligned_packed_ue8m0_tensor = (
        _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl
        if use_torch_impl
        else deep_gemm.utils.layout.get_mn_major_tma_aligned_packed_ue8m0_tensor
    )

    sf = sf.index_select(-2, torch.arange(mn, device=sf.device) // 128)
    sf = get_mn_major_tma_aligned_packed_ue8m0_tensor(sf)
    return sf


# Copied from DeepGEMM tests
def _get_mn_major_tma_aligned_packed_ue8m0_tensor_torch_impl(
    x: torch.Tensor,
) -> torch.Tensor:
    from deep_gemm.utils import align, get_tma_aligned_size

    assert x.dtype == torch.float and x.dim() in (2, 3)

    # First, convert into UE8M0 `uint8_t`
    ue8m0_tensor = (x.view(torch.int) >> 23).to(torch.uint8)

    # Second, make padded packed tensors
    mn, k = x.shape[-2], x.shape[-1]
    remove_dim = False
    if x.dim() == 2:
        x, remove_dim = x.unsqueeze(0), True
    b = x.shape[0]
    aligned_mn = get_tma_aligned_size(mn, 4)
    aligned_k = align(k, 4)
    padded = torch.zeros((b, aligned_mn, aligned_k), device=x.device, dtype=torch.uint8)
    padded[:, :mn, :k] = ue8m0_tensor
    padded = padded.view(-1).view(dtype=torch.int).view(b, aligned_mn, aligned_k // 4)

    # Finally, transpose
    transposed = torch.zeros(
        (b, aligned_k // 4, aligned_mn), device=x.device, dtype=torch.int
    ).mT
    transposed[:, :, :] = padded
    aligned_x = transposed[:, :mn, :]
    return aligned_x.squeeze(0) if remove_dim else aligned_x


def should_deepgemm_weight_requant_ue8m0(weight_block_size) -> bool:
    """Whether to requant fp8 weights into UE8M0 when transferring to vLLM.

    SGLang's original reads its internal ``deep_gemm_wrapper`` flags
    (``ENABLE_JIT_DEEPGEMM`` / ``DEEPGEMM_SCALE_UE8M0``). vime drops the SGLang
    runtime, so we use vLLM's equivalent signal instead.
    """
    if weight_block_size is None:
        return False
    try:
        from vllm.utils.deep_gemm import is_deep_gemm_e8m0_used
    except ImportError:
        return False
    return is_deep_gemm_e8m0_used()


__all__ = [
    "quant_weight_ue8m0",
    "transform_scale_ue8m0",
    "should_deepgemm_weight_requant_ue8m0",
]
