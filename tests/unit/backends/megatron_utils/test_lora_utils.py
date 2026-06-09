from argparse import Namespace

from vime.backends.megatron_utils.lora_utils import (
    build_peft_lora_config,
    convert_target_modules_to_hf,
    convert_target_modules_to_megatron,
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
    assert convert_target_modules_to_hf(["linear_qkv", "linear_proj"]) == [
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
    ]


def test_build_peft_lora_config_uses_hf_target_names():
    args = Namespace(
        lora_rank=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["linear_qkv", "linear_fc2"],
    )

    assert build_peft_lora_config(args) == {
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "r": 8,
        "lora_alpha": 16,
        "lora_dropout": 0.05,
        "bias": "none",
        "target_modules": ["q_proj", "k_proj", "v_proj", "down_proj"],
    }
