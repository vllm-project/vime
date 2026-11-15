"""Check the converter against native Megatron DeepSeek-V4 layer names."""

import json
import os
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

pytest.importorskip("megatron.core")
pytest.importorskip("tests.unit_tests.test_utilities")

from megatron.core.process_groups_config import ProcessGroupCollection
from tests.unit_tests.test_utilities import Utils
from tests.unit_tests.transformer.experimental_attention_variant.test_dsv4_hybrid_attention import (
    _build_dsv4_moe_layer,
    _make_config,
)

from vime.backends.megatron_utils.deepseek_v4_weights import (
    _PENDING_ALPHAS,
    convert_deepseek_v4_to_hf,
)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Megatron native layers need CUDA")
def test_four_layer_native_names_match_checkpoint():
    checkpoint = os.environ.get("DSV4_CHECKPOINT")
    if checkpoint is None:
        pytest.skip("Set DSV4_CHECKPOINT to check the local Flash checkpoint")
    pytest.importorskip("transformer_engine")
    index = json.loads((Path(checkpoint) / "model.safetensors.index.json").read_text())["weight_map"]
    Utils.initialize_model_parallel(tensor_model_parallel_size=1, expert_model_parallel_size=1)
    try:
        config = _make_config(
            num_layers=4,
            hidden_size=128,
            num_attention_heads=8,
            v_head_dim=32,
            qk_pos_emb_head_dim=16,
            q_lora_rank=32,
            csa_compress_ratios=[0, 0, 4, 128],
            num_moe_experts=4,
            moe_ffn_hidden_size=256,
            moe_layer_freq=1,
            moe_router_topk=2,
            moe_router_score_function="sqrtsoftplus",
            moe_router_enable_expert_bias=True,
            moe_n_hash_layers=3,
            actual_vocab_size=128,
            activation_func=F.silu,
            gated_linear_unit=True,
            moe_grouped_gemm=True,
            qk_layernorm=True,
            enable_hyper_connections=True,
            mhc_num_residual_streams=4,
        )
        pg = ProcessGroupCollection.use_mpu_process_groups()
        _PENDING_ALPHAS.clear()
        for layer_id in range(4):
            layer = _build_dsv4_moe_layer(config, layer_id + 1, pg)
            exported = []
            for name, tensor in layer.named_parameters():
                mcore_name = f"module.module.decoder.layers.{layer_id}.{name}"
                exported.extend(key for key, _ in convert_deepseek_v4_to_hf(None, mcore_name, tensor))
            for name, tensor in layer.named_buffers():
                if name.endswith(("router.expert_bias", "router.tid2eid")):
                    mcore_name = f"module.module.decoder.layers.{layer_id}.{name}"
                    exported.extend(key for key, _ in convert_deepseek_v4_to_hf(None, mcore_name, tensor))
            assert not _PENDING_ALPHAS
            assert set(exported) <= set(index)
    finally:
        Utils.destroy_model_parallel()
