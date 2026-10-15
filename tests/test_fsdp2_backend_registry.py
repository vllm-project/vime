"""CPU tests for training-backend selection and the fsdp2 backend's own flags."""

import sys
from argparse import Namespace
from pathlib import Path

import pytest

_tests_root = Path(__file__).resolve().parent
for path in (_tests_root, _tests_root.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _unit_stubs

_unit_stubs.install_vllm_cli_stubs()

from vime.backends import registry
from vime.backends.fsdp_utils.arguments import validate_fsdp_args

NUM_GPUS = 0

TRAIN_ACTOR_METHODS = (
    "init",
    "train",
    "save_model",
    "update_weights",
    "sleep",
    "wake_up",
    "clear_memory",
    "set_rollout_manager",
    "compute_log_prob",
    "load_other_checkpoint",
    "get_master_addr_and_port",
)


def _pre_parse(monkeypatch, *argv):
    from vime.utils.arguments import _pre_parse_mode

    monkeypatch.setattr(sys, "argv", ["train.py", *argv])
    return _pre_parse_mode()


def test_registered_backends():
    assert registry.train_backend_names()[:2] == ["megatron", "fsdp2"]
    with pytest.raises(ValueError, match="Unknown training backend"):
        registry.get_train_actor_cls("fsdp")


def test_fsdp2_actor_resolves_lazily_and_matches_the_train_group_contract():
    module = "vime.backends.fsdp_utils.actor"
    sys.modules.pop(module, None)
    registry.train_backend_names()
    registry.parse_train_backend_args("fsdp2", [])
    assert module not in sys.modules, "listing backends or parsing flags must not import the actor"

    cls = registry.get_train_actor_cls("fsdp2")
    assert cls.__name__ == "FSDPTrainRayActor"
    missing = [name for name in TRAIN_ACTOR_METHODS if not callable(getattr(cls, name, None))]
    assert not missing, missing


def test_pre_parse_merges_fsdp2_flags(monkeypatch):
    args = _pre_parse(
        monkeypatch,
        "--train-backend",
        "fsdp2",
        "--lr",
        "1e-6",
        "--fsdp-attn-implementation",
        "sdpa",
        "--fsdp-cpu-offload",
        "--no-fsdp-reshard-after-forward",
        "--fsdp-param-dtype",
        "fp32",
    )
    assert args.train_backend == "fsdp2"
    assert args.fsdp_attn_implementation == "sdpa"
    assert args.fsdp_cpu_offload is True
    assert args.fsdp_reshard_after_forward is False
    assert args.fsdp_param_dtype == "fp32"
    assert args.fsdp_gradient_checkpointing is False
    assert not hasattr(args, "lr"), "shared flags are left to the main parser"


def test_megatron_runs_do_not_get_fsdp2_flags(monkeypatch):
    args = _pre_parse(monkeypatch, "--lr", "1e-6")
    assert args.train_backend == "megatron"
    assert not any(name.startswith("fsdp_") for name in vars(args))


def test_unknown_backend_is_rejected(monkeypatch):
    with pytest.raises(SystemExit):
        _pre_parse(monkeypatch, "--train-backend", "fsdp")


def test_train_group_selects_the_registered_actor(monkeypatch):
    import ray

    from vime.ray.actor_group import RayTrainGroup

    selected = []

    class _Stop(Exception):
        pass

    def fake_remote(**options):
        def wrap(actor_impl):
            selected.append(actor_impl)
            raise _Stop

        return wrap

    monkeypatch.setattr(ray, "remote", fake_remote)
    args = Namespace(train_backend="fsdp2", train_env_vars={}, offload_train=False, use_routing_replay=False)
    group = RayTrainGroup(args, num_nodes=1, num_gpus_per_node=1, pg=(object(), [0], [0]))
    with pytest.raises(_Stop):
        group._allocate_gpus_for_actor(group._pg, 1)
    assert selected == [registry.get_train_actor_cls("fsdp2")]


def _valid_args(**overrides):
    values = dict(
        use_critic=False,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        context_parallel_size=1,
        expert_model_parallel_size=1,
        use_opd=False,
        opd_type=None,
        dspark_enabled=False,
        enable_mtp_training=False,
        use_routing_replay=False,
        use_rollout_routing_replay=False,
        update_weight_transport="nccl",
        update_weight_mode="full",
        async_save=False,
    )
    values.update(overrides)
    return Namespace(**values)


def test_validate_accepts_the_supported_configuration():
    validate_fsdp_args(_valid_args())
    validate_fsdp_args(_valid_args(use_opd=True, opd_type="vllm"))


@pytest.mark.parametrize(
    ("overrides", "role", "message"),
    [
        ({}, "critic", "critic"),
        ({"use_critic": True}, "actor", "critic"),
        ({"tensor_model_parallel_size": 2}, "actor", "--tensor-model-parallel-size"),
        ({"context_parallel_size": 2}, "actor", "--context-parallel-size"),
        ({"use_opd": True, "opd_type": "megatron"}, "actor", "--opd-type megatron"),
        ({"dspark_enabled": True}, "actor", "draft/MTP"),
        ({"use_rollout_routing_replay": True}, "actor", "routing replay"),
        ({"update_weight_transport": "disk"}, "actor", "disk/delta"),
        ({"async_save": True}, "actor", "--async-save"),
    ],
)
def test_validate_rejects_unsupported_features(overrides, role, message):
    with pytest.raises(ValueError, match=message):
        validate_fsdp_args(_valid_args(**overrides), role)


def test_validate_rejects_multimodal_and_quantized_checkpoints():
    with pytest.raises(ValueError, match="multimodal"):
        validate_fsdp_args(_valid_args(), hf_config=Namespace(vision_config={}))
    with pytest.raises(ValueError, match="quantized"):
        validate_fsdp_args(_valid_args(), hf_config=Namespace(quantization_config={"quant_method": "fp8"}))


@pytest.mark.skipif(
    not _unit_stubs.real_module_available("megatron.training"), reason="requires the Megatron-LM training package"
)
def test_fsdp2_flags_survive_the_megatron_parser(monkeypatch, tmp_path):
    """Full parse_args for an fsdp2 launch: Megatron's parser and validation, then the backend flags."""
    from transformers import AutoModelForCausalLM, Qwen3Config

    config = Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        rope_theta=1000000,
        rms_norm_eps=1e-6,
        tie_word_embeddings=False,
    )
    AutoModelForCausalLM.from_config(config).save_pretrained(tmp_path)
    model_args = (
        "--swiglu --num-layers 2 --hidden-size 32 --ffn-hidden-size 64 --num-attention-heads 4 "
        "--group-query-attention --num-query-groups 2 --use-rotary-position-embeddings --disable-bias-linear "
        "--normalization RMSNorm --norm-epsilon 1e-6 --rotary-base 1000000 --vocab-size 96 --kv-channels 8 "
        "--qk-layernorm --untie-embeddings-and-output-weights"
    ).split()
    launch = [
        *model_args,
        *("--hf-checkpoint", str(tmp_path), "--ref-load", str(tmp_path), "--debug-train-only"),
        *("--actor-num-nodes", "1", "--actor-num-gpus-per-node", "2", "--num-rollout", "2"),
        *("--rollout-batch-size", "4", "--n-samples-per-prompt", "2", "--global-batch-size", "8"),
        *("--prompt-data", "/dev/null", "--input-key", "prompt", "--advantage-estimator", "grpo", "--lr", "1e-6"),
        *("--train-backend", "fsdp2", "--fsdp-attn-implementation", "sdpa", "--fsdp-param-dtype", "fp32"),
    ]
    monkeypatch.setattr(sys, "argv", ["train.py", *launch])
    from vime.utils.arguments import parse_args

    args = parse_args()
    assert (args.train_backend, args.fsdp_attn_implementation, args.fsdp_param_dtype) == ("fsdp2", "sdpa", "fp32")
    assert args.fsdp_reshard_after_forward is True and args.fsdp_cpu_offload is False
    # Fields the fsdp2 trainer reads directly come from Megatron's parser and validation.
    assert args.load == str(tmp_path), "a fresh start trains from the HF weights in --ref-load"
    assert (args.lr, args.world_size, args.clip_grad) == (1e-6, 2, 1.0)
    assert args.start_weight_decay == args.end_weight_decay == args.weight_decay
    assert args.weight_decay_incr_style == "constant"
    validate_fsdp_args(args)


@pytest.mark.parametrize("offload_train", [True, False])
def test_weight_push_allocates_outside_torch_memory_saver(monkeypatch, offload_train):
    """With --offload-train every allocation is tracked by torch_memory_saver, and
    tracked (VMM) memory cannot be shared as a CUDA IPC handle, so the colocated
    push must run with tracking disabled. Without offload it must not touch it."""
    import types
    from contextlib import contextmanager

    from vime.backends.fsdp_utils import actor as actor_module

    tracking = {"on": True}

    @contextmanager
    def disable():
        tracking["on"] = False
        try:
            yield
        finally:
            tracking["on"] = True

    saver_module = types.SimpleNamespace(torch_memory_saver=types.SimpleNamespace(disable=disable))
    # Without offload the module is unavailable, so importing it would fail the test.
    monkeypatch.setitem(sys.modules, "torch_memory_saver", saver_module if offload_train else None)

    class Remote:
        def __init__(self, value):
            self.value = value

        def remote(self):
            return self.value

    pushes = []
    actor = object.__new__(actor_module.FSDPTrainRayActor)
    actor.args = Namespace(
        debug_train_only=False,
        debug_rollout_only=False,
        rollout_external=False,
        use_fault_tolerance=False,
        offload_train=offload_train,
    )
    actor.rollout_manager = Namespace(
        recover_updatable_engines=Remote(None),
        get_updatable_engines_and_lock=Remote((["engine"], None, 0, [1], [0], None)),
        clear_updatable_num_new_engines=Remote(None),
    )
    actor.weight_updater = Namespace(update_weights=lambda: pushes.append(tracking["on"]))
    monkeypatch.setattr(actor_module.ray, "get", lambda value: value)
    monkeypatch.setattr(actor_module.dist, "get_rank", lambda *args, **kwargs: 0)
    monkeypatch.setattr(actor_module.dist, "barrier", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "get_gloo_group", lambda: None)
    monkeypatch.setattr(actor_module, "clear_memory", lambda *args, **kwargs: None)
    monkeypatch.setattr(actor_module, "print_memory", lambda *args, **kwargs: None)

    actor_module.FSDPTrainRayActor.update_weights(actor)
    assert pushes == [not offload_train]


def test_gpu_alignment_script_replays_megatron_rollout_with_fsdp2(monkeypatch, tmp_path):
    """Check the (GPU-only) alignment test's commands and comparison without running them."""
    import runpy
    import shlex
    import types

    import torch

    from vime.utils import external_utils

    commands = []
    megatron_grad_norm = 0.5

    def execute_train(**kwargs):
        # Stand in for the Megatron run's --ci-save-grad-norm output.
        argv = shlex.split(kwargs["train_args"])
        if "--ci-save-grad-norm" in argv:
            template = argv[argv.index("--ci-save-grad-norm") + 1]
            for step_id in range(2):
                torch.save(megatron_grad_norm, template.format(role="actor", rollout_id=0, step_id=step_id))
        commands.append(kwargs)

    launcher = types.ModuleType("vime.utils.external_utils.command_utils")
    launcher.execute_train = execute_train
    launcher.get_default_wandb_args = lambda _: ""
    monkeypatch.setitem(sys.modules, launcher.__name__, launcher)
    monkeypatch.setattr(external_utils, "command_utils", launcher, raising=False)
    script = runpy.run_path(str(_tests_root / "test_qwen3_0.6B_fsdp2_megatron_align.py"))
    monkeypatch.setitem(script["execute"].__globals__, "_log_prob_diffs", lambda *_: [0.0])
    script["execute"]()

    megatron, fsdp = (shlex.split(command["train_args"]) for command in commands)
    assert [command["megatron_model_type"] for command in commands] == ["qwen3-0.6B"] * 2

    def value(argv, flag):
        return argv[argv.index(flag) + 1]

    assert "--train-backend" not in megatron
    assert value(fsdp, "--train-backend") == "fsdp2"
    assert value(megatron, "--save-debug-rollout-data") == value(fsdp, "--load-debug-rollout-data")
    assert value(megatron, "--ci-save-grad-norm") == value(fsdp, "--ci-load-grad-norm")
    assert value(fsdp, "--ref-load").rstrip("/").endswith("Qwen3-0.6B")
    assert registry.parse_train_backend_args("fsdp2", fsdp).fsdp_attn_implementation == "flash_attention_2"
    assert float(value(megatron, "--entropy-coef")) > 0, "the compared gradient must not depend on rewards alone"

    # A zero Megatron gradient (e.g. all rewards equal) stops the test before the fsdp2 run.
    commands.clear()
    megatron_grad_norm = 0.0
    with pytest.raises(AssertionError, match="vacuous"):
        script["execute"]()
    assert len(commands) == 1

    def dump(path, offset):
        samples = [
            {
                "rollout_position": i,
                "loss_masks": torch.tensor([1, 1, 0]),
                "log_probs": torch.tensor([0.0, offset, 9.0]),
            }
            for i in range(2)
        ]
        torch.save({"samples": samples}, path)

    dump(tmp_path / "megatron.pt", 0.0)
    dump(tmp_path / "fsdp.pt", 0.1)
    real_diffs = runpy.run_path(str(_tests_root / "test_qwen3_0.6B_fsdp2_megatron_align.py"))["_log_prob_diffs"]
    assert real_diffs(str(tmp_path / "megatron.pt"), str(tmp_path / "fsdp.pt")) == pytest.approx([0.05, 0.05])


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
