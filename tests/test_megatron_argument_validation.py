import argparse
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
    router_pkg_mod = types.ModuleType("vllm_router")
    router_launch_mod = types.ModuleType("vllm_router.launch_router")
    vllm_arguments_mod = types.ModuleType("vime.backends.vllm_utils.arguments")
    vllm_external_mod = types.ModuleType("vime.backends.vllm_utils.external")
    logging_utils_mod = types.ModuleType("vime.observability.logging_utils")

    router_launch_mod.RouterArgs = object
    vllm_arguments_mod.vllm_parse_args = lambda *args, **kwargs: None
    vllm_arguments_mod.validate_args = lambda args: args
    vllm_external_mod.apply_external_engine_info_to_args = lambda *args, **kwargs: None
    logging_utils_mod.configure_logger = lambda *args, **kwargs: None

    monkeypatch.setitem(sys.modules, "vllm_router", router_pkg_mod)
    monkeypatch.setitem(sys.modules, "vllm_router.launch_router", router_launch_mod)
    monkeypatch.setitem(sys.modules, "vime.backends.vllm_utils.arguments", vllm_arguments_mod)
    monkeypatch.setitem(sys.modules, "vime.backends.vllm_utils.external", vllm_external_mod)
    monkeypatch.setitem(sys.modules, "vime.observability.logging_utils", logging_utils_mod)

    module_path = Path(__file__).resolve().parents[1] / "vime" / "utils" / "arguments.py"
    module_name = "test_vime_argument_validation_module"
    sys.modules.pop(module_name, None)
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


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
def test_update_weight_disk_dir_required_for_disk_transport(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(update_weight_transport="disk", update_weight_disk_dir=None)

    with pytest.raises(ValueError, match="update-weight-disk-dir"):
        module.vime_validate_args(args)


def make_vime_validate_args(**overrides):
    values = dict(
        eval_config=None,
        eval_prompt_data=None,
        kl_coef=0,
        use_kl_loss=False,
        ref_load=None,
        use_opd=False,
        opd_type=None,
        opd_teacher_load=None,
        load=None,
        hf_checkpoint="/tmp/hf",
        ref_ckpt_step=None,
        ckpt_step=None,
        no_load_optim=False,
        no_load_rng=False,
        finetune=False,
        start_rollout_id=None,
        eval_interval=None,
        save_interval=None,
        save=None,
        kl_loss_coef=0,
        advantage_estimator="grpo",
        normalize_advantages=False,
        use_rollout_logprobs=False,
        use_tis=False,
        get_mismatch_metrics=False,
        custom_tis_function_path=None,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        log_probs_max_tokens_per_gpu=None,
        balance_by_flops=False,
        balance_data=False,
        eps_clip_high=None,
        eps_clip=0.2,
        eval_reward_key=None,
        reward_key="reward",
        dump_details=None,
        save_debug_rollout_data=None,
        save_debug_train_data=None,
        load_debug_rollout_data=None,
        rollout_external_engine_addrs=None,
        debug_train_only=False,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        num_gpus_per_node=8,
        offload=False,
        offload_train=None,
        offload_rollout=None,
        debug_rollout_only=False,
        colocate=False,
        rollout_num_gpus=8,
        eval_function_path=None,
        rollout_function_path="custom.rollout",
        vllm_speculative_config=None,
        num_steps_per_rollout=None,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=None,
        global_batch_size_schedule=None,
        variable_global_batch_size=False,
        grpo_std_normalization=True,
        over_sampling_batch_size=None,
        num_epoch=None,
        num_rollout=1,
        rollout_global_dataset=False,
        enable_mtp_training=False,
        mtp_num_layers=None,
        use_rollout_routing_replay=False,
        use_routing_replay=False,
        custom_config_path=None,
        eval_max_context_len=None,
        rollout_max_context_len=None,
        rollout_max_prompt_len=None,
        train_backend="megatron",
        release_train=False,
        keep_old_actor=False,
        only_train_params_name_list=None,
        freeze_params_name_list=None,
        update_weight_transport="nccl",
        update_weight_disk_dir=None,
        update_weight_local_checkpoint_dir=None,
        update_weight_mode="full",
        rollout_temperature=1.0,
    )
    values.update(overrides)
    return types.SimpleNamespace(**values)


@pytest.mark.unit
@pytest.mark.parametrize("schedule_yaml", ["'3,5'", "[3, 5]"])
def test_custom_config_normalizes_global_batch_schedule(monkeypatch, tmp_path, schedule_yaml):
    module = load_vime_arguments_module(monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text(f"global_batch_size_schedule: {schedule_yaml}\n")
    args = make_vime_validate_args(custom_config_path=str(config), rollout_batch_size=8)

    module.vime_validate_args(args)

    assert args.global_batch_size_schedule == [3, 5]
    assert args.global_batch_size == 3
    assert args.variable_global_batch_size is True


@pytest.mark.unit
def test_custom_config_rejects_invalid_global_batch_schedule(monkeypatch, tmp_path):
    module = load_vime_arguments_module(monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text("global_batch_size_schedule: '3,0,5'\n")
    args = make_vime_validate_args(custom_config_path=str(config), rollout_batch_size=8)

    with pytest.raises(AssertionError, match="must contain positive integers"):
        module.vime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("schedule", ["[3.9, 4.1]", "[true, 7]", "'3,,5'", "'3,5,'", "'{3: left, 5: right}'"])
def test_global_batch_schedule_rejects_non_integers(monkeypatch, tmp_path, schedule):
    module = load_vime_arguments_module(monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text(f"global_batch_size_schedule: {schedule}\n")
    args = make_vime_validate_args(custom_config_path=str(config))

    with pytest.raises(ValueError, match="positive integers"):
        module.vime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides, error",
    [
        ({}, None),
        ({"ckpt_format": "torch"}, "torch_dist"),
        ({"reset_optimizer_states": True}, "reset-optimizer-states"),
    ],
)
def test_custom_config_streaming_is_validated(monkeypatch, tmp_path, overrides, error):
    module = load_vime_arguments_module(monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text("stream_optimizer_state_to_disk: true\n")
    args = make_vime_validate_args(custom_config_path=str(config), bf16=True, offload_train_disk_dir=None, **overrides)
    monkeypatch.setenv("SCRATCH", "/scratch")
    monkeypatch.setenv("VIME_RUN_ID", "yaml-test")

    if error:
        with pytest.raises((ValueError, AssertionError), match=error):
            module.vime_validate_args(args)
    else:
        module.vime_validate_args(args)
        assert args.offload_train_disk_dir == "/scratch/vime_train_offload_yaml-test"


@pytest.mark.unit
def test_custom_config_disables_streaming_before_validation(monkeypatch, tmp_path):
    module = load_vime_arguments_module(monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text("stream_optimizer_state_to_disk: false\n")
    args = make_vime_validate_args(custom_config_path=str(config), stream_optimizer_state_to_disk=True, bf16=False)

    module.vime_validate_args(args)

    assert args.stream_optimizer_state_to_disk is False


@pytest.mark.unit
def test_custom_config_batch_settings_are_derived_after_overrides(monkeypatch, tmp_path):
    module = load_vime_arguments_module(monkeypatch)
    config = tmp_path / "custom.yaml"
    config.write_text("rollout_batch_size: 5\nvariable_global_batch_size: true\n")
    args = make_vime_validate_args(custom_config_path=str(config), rollout_batch_size=8, num_steps_per_rollout=2)

    module.vime_validate_args(args)

    assert args.global_batch_size == 3


@pytest.mark.unit
@pytest.mark.parametrize("samples, steps, batch_size", [(5, 2, 3), (5, 3, 2), (8, 2, 4)])
def test_num_steps_with_variable_batches(monkeypatch, samples, steps, batch_size):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        rollout_batch_size=samples, num_steps_per_rollout=steps, variable_global_batch_size=True
    )

    module.vime_validate_args(args)

    assert args.global_batch_size == batch_size
    assert (samples + batch_size - 1) // batch_size == steps


@pytest.mark.unit
def test_num_steps_with_variable_batches_requires_explicit_schedule_when_needed(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=6, num_steps_per_rollout=4, variable_global_batch_size=True)

    with pytest.raises(ValueError, match="global-batch-size-schedule"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_num_steps_with_fixed_batches_rejects_extra_steps(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=8, num_steps_per_rollout=3)

    with pytest.raises(ValueError, match="global-batch-size-schedule"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_num_steps_with_fixed_batches_keeps_dropped_tail(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=5, num_steps_per_rollout=2)

    module.vime_validate_args(args)

    assert args.global_batch_size == 2


@pytest.mark.unit
def test_explicit_schedule_matches_requested_steps(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=5, num_steps_per_rollout=2, global_batch_size_schedule=[3, 2])

    module.vime_validate_args(args)

    assert args.global_batch_size == 3
    assert args.global_batch_size_schedule == [3, 2]


@pytest.mark.unit
def test_explicit_schedule_rejects_mismatched_requested_steps(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=5, num_steps_per_rollout=2, global_batch_size_schedule=[2, 2, 1])

    with pytest.raises(ValueError, match="schedule length"):
        module.vime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("schedule", [[2, 2], [2, 4]])
@pytest.mark.parametrize("num_steps", [None, 2])
def test_explicit_schedule_must_cover_rollout_groups(monkeypatch, schedule, num_steps):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        rollout_batch_size=5,
        num_steps_per_rollout=num_steps,
        global_batch_size_schedule=schedule,
        rollout_function_path="vime.rollout.vllm_rollout.generate_rollout",
    )

    with pytest.raises(ValueError, match="cover all rollout groups"):
        module.vime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize(
    "overrides",
    [
        {"rollout_function_path": "custom.rollout"},
        {
            "rollout_function_path": "vime.rollout.vllm_rollout.generate_rollout",
            "custom_generate_function_path": "custom.generate",
        },
    ],
)
def test_custom_rollout_schedule_uses_actual_groups(monkeypatch, overrides):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        rollout_batch_size=2, n_samples_per_prompt=2, global_batch_size_schedule=[2], **overrides
    )

    module.vime_validate_args(args)

    assert args.global_batch_size_schedule == [2]


@pytest.mark.unit
def test_custom_rollout_steps_follow_explicit_schedule(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        rollout_batch_size=2,
        num_steps_per_rollout=3,
        global_batch_size_schedule=[1, 1, 1],
    )

    module.vime_validate_args(args)

    assert args.global_batch_size_schedule == [1, 1, 1]


@pytest.mark.unit
def test_nominal_steps_without_schedule_cannot_exceed_nominal_groups(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=2, num_steps_per_rollout=3)

    with pytest.raises(ValueError, match="nominal rollout groups"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_explicit_schedule_counts_samples_per_prompt(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        rollout_batch_size=3, n_samples_per_prompt=2, num_steps_per_rollout=2, global_batch_size_schedule="2,4"
    )

    module.vime_validate_args(args)

    assert args.global_batch_size_schedule == [2, 4]


@pytest.mark.unit
@pytest.mark.parametrize("schedule", [{3: "left", 5: "right"}, {3, 5}, iter([3, 5])])
def test_explicit_schedule_rejects_non_sequences(monkeypatch, schedule):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=8, global_batch_size_schedule=schedule)

    with pytest.raises(ValueError, match="sequence of positive integers"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_critic_streaming_override_fills_default_directory(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = types.SimpleNamespace(
        kl_coef=0.1,
        use_opd=True,
        custom_advantage_function_path="custom.adv",
        untie_embeddings_and_output_weights=False,
        disable_param_buffers_cpu_backup=True,
        stream_optimizer_state_to_disk=False,
        bf16=True,
        fp16=False,
        ckpt_format="torch_dist",
        optimizer="adam",
        use_distributed_optimizer=True,
        optimizer_cpu_offload=False,
        offload_optimizer_states=False,
        async_save=False,
        reset_optimizer_states=False,
        load_main_params_from_ckpt=False,
        offload_train_disk_chunk_mb=64,
        offload_train_disk_dir=None,
        stream_optimizer_state_moment_dtype="bf16",
    )
    monkeypatch.setenv("SCRATCH", "/scratch")
    monkeypatch.setenv("VIME_RUN_ID", "role-test")

    critic_args = module._apply_megatron_role_overrides(args, {"stream_optimizer_state_to_disk": True}, role="critic")

    assert critic_args.stream_optimizer_state_to_disk is True
    assert critic_args.offload_train_disk_dir == "/scratch/vime_train_offload_role-test"


@pytest.mark.unit
@pytest.mark.parametrize(
    "mode",
    [
        "bf16",
        "fp16",
        "fp32",
        "torch",
        "torch_dcp",
        "fsdp_dtensor",
        "reset_optimizer_states",
        "load_main_params_from_ckpt",
    ],
)
def test_nvme_streaming_supported_configurations(monkeypatch, mode):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        stream_optimizer_state_to_disk=True,
        bf16=mode not in ("fp16", "fp32"),
        fp16=mode == "fp16",
        ckpt_format=mode if mode in ("torch", "torch_dcp", "fsdp_dtensor") else "torch_dist",
        offload_train_disk_chunk_mb=64,
        offload_train_disk_dir="/tmp/nvme-test",
        stream_optimizer_state_moment_dtype="fp32",
        reset_optimizer_states=mode == "reset_optimizer_states",
        load_main_params_from_ckpt=mode == "load_main_params_from_ckpt",
        no_load_optim=mode == "load_main_params_from_ckpt",
    )
    if mode == "bf16":
        module.vime_validate_args(args)
    elif mode in ("reset_optimizer_states", "load_main_params_from_ckpt"):
        with pytest.raises(AssertionError, match=mode.replace("_", "-")):
            module.vime_validate_args(args)
    else:
        message = "BF16" if mode in ("fp16", "fp32") else "torch_dist"
        with pytest.raises(ValueError, match=message):
            module.vime_validate_args(args)


@pytest.mark.unit
def test_vime_validate_args_accepts_programmatic_global_batch_schedule(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_batch_size=6, global_batch_size_schedule=[2, 4])

    module.vime_validate_args(args)

    assert args.global_batch_size_schedule == [2, 4]
    assert args.global_batch_size == 2
    assert args.variable_global_batch_size is True


@pytest.mark.unit
def test_vime_validate_args_preserves_explicit_start_rollout_id(monkeypatch):
    """``--start-rollout-id`` is only a fallback when the user did not set it.

    An explicit value is needed when there is no resumable Megatron checkpoint.
    """
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(start_rollout_id=100)

    module.vime_validate_args(args)

    assert args.start_rollout_id == 100


@pytest.mark.unit
def test_vime_validate_args_defaults_start_rollout_id_to_zero(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(start_rollout_id=None)

    module.vime_validate_args(args)

    assert args.start_rollout_id == 0


@pytest.mark.unit
def test_vime_validate_args_derives_dspark_from_speculative_method(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(vllm_speculative_config={"method": "dspark"})

    module.vime_validate_args(args)

    assert args.dspark_enabled is True


@pytest.mark.unit
def test_vime_validate_args_rejects_equal_debug_data_paths(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        save_debug_rollout_data="/tmp/debug_{rollout_id}.pt",
        save_debug_train_data="/tmp/debug_{rollout_id}.pt",
    )

    with pytest.raises(ValueError, match="--save-debug-train-data must not be equal"):
        module.vime_validate_args(args)


@pytest.mark.unit
@pytest.mark.parametrize("temperature", [0.0, -0.1])
def test_vime_validate_args_rejects_non_positive_rollout_temperature(monkeypatch, temperature):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(rollout_temperature=temperature)

    with pytest.raises(ValueError, match="--rollout-temperature must be > 0"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_vime_validate_args_preserves_zero_rollout_gpus_under_colocate(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(colocate=True, rollout_num_gpus=0)

    module.vime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_vime_validate_args_preserves_larger_rollout_gpus_under_colocate(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        colocate=True,
        actor_num_gpus_per_node=8,
        actor_num_nodes=1,
        rollout_num_gpus=12,
    )

    module.vime_validate_args(args)

    assert args.rollout_num_gpus == 12
    assert args.offload_train is True
    assert args.offload_rollout is True


@pytest.mark.unit
def test_vime_validate_args_preserves_zero_rollout_gpus_without_colocate(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(colocate=False, rollout_num_gpus=0)

    module.vime_validate_args(args)

    assert args.rollout_num_gpus == 0
    assert args.actor_num_gpus_per_node == 8
    assert args.actor_num_nodes == 1
    assert args.offload_train is False
    assert args.offload_rollout is False


@pytest.mark.unit
def test_update_weight_delta_disk_is_valid(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/delta",
    )

    module.vime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_disk_transport(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="nccl",
        update_weight_local_checkpoint_dir="/local/delta",
    )

    with pytest.raises(ValueError, match="requires --update-weight-transport=disk"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_rejects_colocate(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
        update_weight_local_checkpoint_dir="/local/delta",
        colocate=True,
    )

    with pytest.raises(ValueError, match="not supported with --colocate"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_update_weight_delta_requires_local_checkpoint_dir(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    args = make_vime_validate_args(
        update_weight_mode="delta",
        update_weight_transport="disk",
        update_weight_disk_dir="/shared/delta",
    )

    with pytest.raises(ValueError, match="requires --update-weight-local-checkpoint-dir"):
        module.vime_validate_args(args)


@pytest.mark.unit
def test_force_fp8_ue8m0_scale_argument(monkeypatch):
    module = load_vime_arguments_module(monkeypatch)
    parser = argparse.ArgumentParser()
    module.get_vime_extra_args_provider()(parser)

    defaults = parser.parse_args(["--rollout-batch-size", "1"])
    configured = parser.parse_args(["--rollout-batch-size", "1", "--force-fp8-ue8m0-scale"])

    assert defaults.force_fp8_ue8m0_scale is False
    assert configured.force_fp8_ue8m0_scale is True


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
