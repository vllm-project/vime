"""Deterministic train/rollout alignment environment.

Centralizes the numerical-alignment environment variables that both train
(Megatron) and rollout (VLLM) actors must share for GLM-5 train/rollout
log-prob alignment. Launchers (and the 6-layer gate test) merge this with
their own connectivity settings (``PYTHONPATH``, ``MASTER_ADDR``, NIC names,
proxy, IBGDA handler), which are cluster-specific and intentionally not here.
"""

from __future__ import annotations


def alignment_env(*, kv_fp8_qat: bool = False) -> dict[str, str]:
    """Return the shared deterministic-alignment env vars.

    ``kv_fp8_qat`` enables the FP8-E4M3 KV-cache QAT path (bf16 KV when False).
    """
    return {
        # Deterministic collectives / matmul.
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        "NCCL_P2P_LEVEL": "NVL",
        "NCCL_ALGO": "^NVLS",
        "CUBLAS_WORKSPACE_CONFIG": ":4096:8",
        "TORCH_COMPILE_DISABLE": "1",
        "NVTE_ALLOW_NONDETERMINISTIC_ALGO": "0",
        "TE_DISABLE_FA3": "TRUE",
        "NVSHMEM_DISABLE_NCCL": "1",
        # DeepGEMM batch-invariant FP8 forward.
        "VLLM_BATCH_INVARIANT": "1",
        # Megatron train side borrows VLLM's aligned kernels.
        "MEGATRON_USE_VLLM_FUSED_RESIDUAL_RMS": "1",
        "MEGATRON_USE_VLLM_FP8_INDEXER": "1",
        "MEGATRON_USE_VLLM_ROUTER_GEMM": "1",
        "MEGATRON_USE_VLLM_ROPE": "1",
        "MEGATRON_USE_VLLM_SPARSE_MLA": "1",
        # DSA KV cache dtype.
        "DSA_KV_FP8_QAT": "1" if kv_fp8_qat else "0",
        "DSA_KV_FP8_QAT_BLOCK_SIZE": "128",
    }
