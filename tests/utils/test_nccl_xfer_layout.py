from __future__ import annotations

import pytest
import torch

from vime.backends.megatron_utils.update_weight.nccl_xfer_layout import analyze_nccl_xfer_layout


@pytest.mark.unit
def test_column_parallel_weight_shards_dim0():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.self_attn.q_proj.weight",
        (4096, 4096),
        torch.bfloat16,
    )

    assert decision.supported
    assert decision.shard_tensor_dim == 0
    assert not decision.replicated


@pytest.mark.unit
def test_row_parallel_weight_shards_dim1():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.self_attn.o_proj.weight",
        (4096, 4096),
        torch.float16,
    )

    assert decision.supported
    assert decision.shard_tensor_dim == 1


@pytest.mark.unit
def test_replicated_1d_weight():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.input_layernorm.weight",
        (4096,),
        torch.float32,
    )

    assert decision.supported
    assert decision.replicated
    assert decision.shard_tensor_dim is None


@pytest.mark.unit
def test_grouped_moe_expert_weight_shards_expert_dim():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.mlp.experts.gate_proj.weight",
        (16, 4096, 4096),
        torch.bfloat16,
    )

    assert decision.supported
    assert decision.shard_tensor_dim == 0


@pytest.mark.unit
def test_ungrouped_moe_expert_weight_is_unsupported():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.mlp.experts.0.gate_proj.weight",
        (4096, 4096),
        torch.bfloat16,
    )

    assert not decision.supported
    assert "rank-3" in decision.reason


@pytest.mark.unit
def test_tensor_rank_greater_than_three_is_unsupported():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.weird_packed.weight",
        (2, 3, 4, 5),
        torch.float16,
    )

    assert not decision.supported
    assert "exceeds NCCL Xfer limit" in decision.reason


@pytest.mark.unit
def test_compressed_tensors_quantization_is_unsupported():
    decision = analyze_nccl_xfer_layout(
        "model.layers.0.self_attn.q_proj.weight",
        (4096, 4096),
        torch.float16,
        quantization_config={"quant_method": "compressed-tensors"},
    )

    assert not decision.supported
    assert "compressed-tensors" in decision.reason
