from argparse import Namespace

from vime.backends.megatron_utils.lora_utils import (
    build_peft_lora_config,
    convert_target_modules_to_megatron,
    infer_hf_target_modules,
    normalize_target_modules,
)


def test_normalize_target_modules_expands_all_linear_and_excludes():
    assert normalize_target_modules(["all-linear"], ["down_proj"]) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
    ]


def test_target_module_name_conversion_deduplicates_fused_layers():
    assert convert_target_modules_to_megatron(["q_proj", "k_proj", "v_proj", "o_proj"]) == [
        "linear_qkv",
        "linear_proj",
    ]


def test_canonical_lora_targets_split_projections():
    # CanonicalLoRA rejects the fused names that standard LoRA expects.
    assert convert_target_modules_to_megatron(["all-linear"], "canonical_lora") == [
        "linear_q",
        "linear_k",
        "linear_v",
        "linear_proj",
        "linear_fc1_gate",
        "linear_fc1_up",
        "linear_fc2",
    ]


def test_infer_hf_target_modules_covers_fused_siblings():
    names = [
        "model.layers.0.self_attn.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_B.weight",
        "model.layers.0.self_attn.k_proj.lora_A.weight",
        "model.layers.1.mlp.down_proj.lora_B.weight",
        "model.layers.1.mlp.down_proj.bias",  # non-adapter tensor is ignored
    ]
    assert infer_hf_target_modules(names) == ["down_proj", "k_proj", "q_proj"]


def test_build_peft_lora_config_uses_inferred_target_modules():
    args = Namespace(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
    )

    assert build_peft_lora_config(args, ["q_proj", "k_proj", "v_proj", "down_proj"]) == {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "down_proj"],
    }
