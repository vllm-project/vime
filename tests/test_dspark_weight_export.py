"""DSpark draft weights pushed to vLLM must match the checkpoint vLLM serves.

vLLM loads the draft from ``--vllm-speculative-config``'s ``model`` at startup
and later receives the trainer's draft through the weight-sync path, which
looks the tensors up by name. ``DSparkModel.load_pretrained_weights`` only
warns on missing or mismatched keys, so a renamed parameter would silently
keep random weights on the trainer and push a tensor vLLM cannot place.

The expected names and shapes below are the safetensors header of
``deepseek-ai/dspark_qwen3_4b_block7`` at revision 3457dff1417cb84927f6098a5fcb7cee85c934b7
(the draft used by examples/dspark), not derived from DSparkModel itself.
"""

import sys
import types

import pytest
import torch
from torch import nn

from vime.backends.megatron_utils.dspark.common import DSparkConfig
from vime.backends.megatron_utils.dspark.modeling import DSparkModel

NUM_GPUS = 0

VOCAB, PADDED_VOCAB, HIDDEN, INTERMEDIATE, HEADS, KV_HEADS, HEAD_DIM = 151936, 152064, 2560, 9728, 32, 8, 128

CHECKPOINT_SHAPES = {
    "confidence_head.proj.bias": (1,),
    "confidence_head.proj.weight": (1, HIDDEN + 256),
    "embed_tokens.weight": (VOCAB, HIDDEN),
    "fc.weight": (HIDDEN, 5 * HIDDEN),
    "hidden_norm.weight": (HIDDEN,),
    "lm_head.weight": (VOCAB, HIDDEN),
    "markov_head.markov_w1.weight": (VOCAB, 256),
    "markov_head.markov_w2.weight": (VOCAB, 256),
    "norm.weight": (HIDDEN,),
    **{
        f"layers.{i}.{name}": shape
        for i in range(5)
        for name, shape in {
            "input_layernorm.weight": (HIDDEN,),
            "post_attention_layernorm.weight": (HIDDEN,),
            "mlp.gate_proj.weight": (INTERMEDIATE, HIDDEN),
            "mlp.up_proj.weight": (INTERMEDIATE, HIDDEN),
            "mlp.down_proj.weight": (HIDDEN, INTERMEDIATE),
            "self_attn.q_proj.weight": (HEADS * HEAD_DIM, HIDDEN),
            "self_attn.k_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
            "self_attn.v_proj.weight": (KV_HEADS * HEAD_DIM, HIDDEN),
            "self_attn.o_proj.weight": (HIDDEN, HEADS * HEAD_DIM),
            "self_attn.q_norm.weight": (HEAD_DIM,),
            "self_attn.k_norm.weight": (HEAD_DIM,),
        }.items()
    },
}


@pytest.fixture
def export():
    """``export_dspark_model_weights`` with Megatron's ``unwrap_model`` stubbed."""
    names = ("megatron", "megatron.core", "megatron.core.utils")
    saved = {name: sys.modules.get(name) for name in names}
    utils = types.ModuleType("megatron.core.utils")
    utils.unwrap_model = lambda model: model
    sys.modules.update({"megatron": types.ModuleType("megatron"), "megatron.core": types.ModuleType("megatron.core")})
    sys.modules["megatron.core.utils"] = utils
    try:
        from vime.backends.megatron_utils.dspark.export import export_dspark_model_weights

        yield export_dspark_model_weights
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module


def _policy_with_qwen3_4b_draft():
    # Megatron pads the vocabulary for TP; export must strip it again.
    config = DSparkConfig(
        hidden_size=HIDDEN,
        vocab_size=PADDED_VOCAB,
        org_vocab_size=VOCAB,
        num_attention_heads=HEADS,
        num_key_value_heads=KV_HEADS,
        head_dim=HEAD_DIM,
        intermediate_size=INTERMEDIATE,
    )
    with torch.device("meta"):
        policy = nn.Module()
        policy.embedding = nn.Module()
        policy.embedding.word_embeddings = nn.Embedding(PADDED_VOCAB, HIDDEN)
        policy.draft_model = DSparkModel(config)
    return policy


@pytest.mark.unit
@pytest.mark.parametrize("use_policy_embedding", [False, True])
def test_exported_draft_matches_served_checkpoint(export, use_policy_embedding):
    exported = export([_policy_with_qwen3_4b_draft()], use_policy_embedding=use_policy_embedding)

    names = [name for name, _ in exported]
    assert len(names) == len(set(names)), "a draft tensor would be pushed twice"
    assert {name: tuple(tensor.shape) for name, tensor in exported} == CHECKPOINT_SHAPES


@pytest.mark.unit
def test_every_checkpoint_tensor_has_a_draft_parameter():
    """``load_pretrained_weights`` only warns on misses, so check names up front."""
    model_shapes = {name: tuple(p.shape) for name, p in _policy_with_qwen3_4b_draft().draft_model.named_parameters()}

    assert set(model_shapes) == set(CHECKPOINT_SHAPES)
    for name, shape in CHECKPOINT_SHAPES.items():
        # Vocabulary-sized tensors are padded on the trainer and stripped on export.
        expected = (PADDED_VOCAB, *shape[1:]) if shape[0] == VOCAB else shape
        assert model_shapes[name] == expected, name


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
