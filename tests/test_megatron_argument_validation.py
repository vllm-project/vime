import importlib.util
import sys
import types
from pathlib import Path

import pytest


NUM_GPUS = 0


def load_arguments_module(monkeypatch):
    megatron_mod = types.ModuleType("megatron")
    training_mod = types.ModuleType("megatron.training")
    arguments_mod = types.ModuleType("megatron.training.arguments")
    tokenizer_pkg_mod = types.ModuleType("megatron.training.tokenizer")
    tokenizer_mod = types.ModuleType("megatron.training.tokenizer.tokenizer")
    transformers_mod = types.ModuleType("transformers")

    arguments_mod.parse_args = lambda *args, **kwargs: None
    arguments_mod.validate_args = lambda args: args
    tokenizer_mod._vocab_size_with_padding = lambda vocab_size, _args: vocab_size
    transformers_mod.AutoConfig = types.SimpleNamespace(from_pretrained=lambda *args, **kwargs: None)

    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.training", training_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", arguments_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer", tokenizer_pkg_mod)
    monkeypatch.setitem(sys.modules, "megatron.training.tokenizer.tokenizer", tokenizer_mod)
    monkeypatch.setitem(sys.modules, "transformers", transformers_mod)

    module_path = Path(__file__).resolve().parents[1] / "vime" / "backends" / "megatron_utils" / "arguments.py"
    module_name = "test_megatron_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def load_vime_arguments_module(monkeypatch):
    """Load VIME argument validation without importing the full runtime stack."""
    router_pkg_mod = types.ModuleType("vllm_router")
    router_launch_mod = types.ModuleType("vllm_router.launch_router")
    vllm_arguments_mod = types.ModuleType("vime.backends.vllm_utils.arguments")
    common_mod = types.ModuleType("vime.utils.common")
    logging_utils_mod = types.ModuleType("vime.utils.logging_utils")
    router_launch_mod.RouterArgs = object
    vllm_arguments_mod.vllm_parse_args = lambda *args, **kwargs: None
    vllm_arguments_mod.validate_args = lambda args: args
    common_mod.is_npu = lambda: True
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None
    monkeypatch.setitem(sys.modules, "vllm_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "vllm_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "vime.backends.vllm_utils.arguments", vllm_arguments_mod)
    monkeypatch.setitem(sys.modules, "vime.utils.common", common_mod)
    monkeypatch.setitem(sys.modules, "vime.utils.logging_utils", logging_utils_mod)
    module_path = Path(__file__).resolve().parents[1] / "vime" / "utils" / "arguments.py"
    module_name = "test_vime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def make_vime_validate_args(**overrides):
    values = dict(
        eval_config=None, eval_prompt_data=None, kl_coef=0, use_kl_loss=False,
        ref_load=None, use_opd=False, opd_type=None, opd_teacher_load=None,
        load=None, hf_checkpoint="/tmp/hf", megatron_to_hf_mode="bridge", ref_ckpt_step=None, ckpt_step=None,
        no_load_optim=False, no_load_rng=False, finetune=False, start_rollout_id=None,
        eval_interval=None, save_interval=None, save=None, kl_loss_coef=0,
        advantage_estimator="grpo", normalize_advantages=False, use_rollout_logprobs=False,
        use_tis=False, get_mismatch_metrics=False, custom_tis_function_path=None,
        use_dynamic_batch_size=False, max_tokens_per_gpu=None, log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False, balance_data=False, eps_clip_high=None, eps_clip=0.2,
        eval_reward_key=None, reward_key="reward", dump_details=None,
        save_debug_rollout_data=None, save_debug_train_data=None, load_debug_rollout_data=None,
        rollout_external_engine_addrs=None, debug_train_only=False, actor_num_gpus_per_node=8,
        actor_num_nodes=1, num_gpus_per_node=8, offload=False, offload_train=None,
        offload_rollout=None, debug_rollout_only=False, colocate=False, rollout_num_gpus=8,
        eval_function_path=None, rollout_function_path="custom.rollout", num_steps_per_rollout=None,
        rollout_batch_size=1, n_samples_per_prompt=1, global_batch_size=None,
        grpo_std_normalization=True, over_sampling_batch_size=None, num_epoch=None,
        num_rollout=1, rollout_global_dataset=False, enable_mtp_training=False,
        mtp_num_layers=None, use_rollout_routing_replay=False, use_routing_replay=False,
        custom_config_path=None, eval_max_context_len=None, rollout_max_context_len=None,
        rollout_max_prompt_len=None, train_backend="megatron", release_train=False,
        keep_old_actor=False, only_train_params_name_list=None, freeze_params_name_list=None,
        update_weight_transport="nccl", update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None, update_weight_mode="full", qkv_format="sbhd",
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_args(**overrides):
    values = dict(
        hidden_size=2048,
        num_attention_heads=16,
        num_layers=40,
        ffn_hidden_size=512,
        moe_ffn_hidden_size=512,
        moe_shared_expert_intermediate_size=512,
        moe_layer_freq=[1] * 40,
        untie_embeddings_and_output_weights=True,
        norm_epsilon=1e-6,
        layernorm_epsilon=1e-6,
        rotary_base=10000000,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


def make_qwen3_6_hf_config():
    text_config = types.SimpleNamespace(
        hidden_size=2048,
        num_attention_heads=16,
        num_hidden_layers=40,
        intermediate_size=5632,
        moe_intermediate_size=512,
        shared_expert_intermediate_size=512,
        num_experts=256,
        tie_word_embeddings=False,
        rms_norm_eps=1e-6,
        rope_parameters={"rope_theta": 10000000},
    )
    return types.SimpleNamespace(text_config=text_config)


def make_allgather_cp_args(**overrides):
    values = dict(
        allgather_cp=True,
        context_parallel_size=2,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
def test_hf_validate_all_moe_skips_dense_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    module._hf_validate_args(make_qwen3_6_args(), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_moe_intermediate_size(monkeypatch):
    module = load_arguments_module(monkeypatch)

    with pytest.raises(AssertionError, match="moe_intermediate_size"):
        module._hf_validate_args(make_qwen3_6_args(moe_ffn_hidden_size=256), make_qwen3_6_hf_config())


@pytest.mark.unit
def test_hf_validate_checks_dense_intermediate_size_when_moe_has_dense_layers(monkeypatch):
    module = load_arguments_module(monkeypatch)

    args = make_qwen3_6_args(moe_layer_freq=[0] + [1] * 39)

    with pytest.raises(AssertionError, match="intermediate_size"):
        module._hf_validate_args(args, make_qwen3_6_hf_config())


@pytest.mark.unit
def test_allgather_cp_rejects_non_dsa_cp_models(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args()
    hf_config = types.SimpleNamespace(architectures=["Qwen3ForCausalLM"], model_type="qwen3")

    with pytest.raises(ValueError, match="only supported for DSA attention models"):
        module._validate_allgather_cp_supported(args, hf_config)


@pytest.mark.unit
@pytest.mark.parametrize(
    "hf_config",
    [
        types.SimpleNamespace(architectures=["DeepseekV32ForCausalLM"], model_type="deepseek_v3"),
        types.SimpleNamespace(architectures=["GlmMoeDsaForCausalLM"], model_type="glm"),
    ],
)
def test_allgather_cp_allows_dsa_architectures(monkeypatch, hf_config):
    module = load_arguments_module(monkeypatch)

    module._validate_allgather_cp_supported(make_allgather_cp_args(), hf_config)


@pytest.mark.unit
def test_allgather_cp_ignores_cp_size_one(monkeypatch):
    module = load_arguments_module(monkeypatch)
    args = make_allgather_cp_args(context_parallel_size=1)

    module._validate_allgather_cp_supported(args)


@pytest.mark.unit
def test_update_weight_delta_disk_is_valid(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    module.vime_validate_args(
        make_vime_validate_args(
            update_weight_mode="delta",
            update_weight_transport="disk",
            update_weight_disk_dir="/shared/delta",
            update_weight_local_checkpoint_dir="/local/delta",
        )
    )


@pytest.mark.unit
@pytest.mark.parametrize(
    ("overrides", "error"),
    [
        ({"update_weight_mode": "delta"}, "requires --update-weight-transport=disk"),
        (
            {
                "update_weight_mode": "delta",
                "update_weight_transport": "disk",
                "update_weight_disk_dir": "/shared/delta",
                "colocate": True,
            },
            "not supported with --colocate",
        ),
        (
            {
                "update_weight_mode": "delta",
                "update_weight_transport": "disk",
                "update_weight_disk_dir": "/shared/delta",
            },
            "requires --update-weight-local-checkpoint-dir",
        ),
        ({"update_weight_transport": "disk"}, "supported only with --update-weight-mode=delta"),
    ],
)
def test_update_weight_disk_rejects_invalid_combinations(monkeypatch, overrides, error):
    module = load_vime_arguments_module(monkeypatch)
    with pytest.raises(ValueError, match=error):
        module.vime_validate_args(make_vime_validate_args(**overrides))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
