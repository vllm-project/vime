"""CPU smoke tests of the fsdp2 backend with real ``fully_shard`` over gloo.

Workers shard a tiny randomly initialized Qwen3 with FSDP2 on CPU and drive
the production trainer: log-probs on packed micro-batches, the shared GRPO
loss, optimizer steps, weight export, model switching and checkpoints.

The shared loss and metric helpers read their process groups from
``megatron.core.mpu``. With Megatron installed (the vime image), workers
register the same pure data-parallel layout as the actor (TP = PP = CP = 1,
DP = world) through Megatron's ``initialize_model_parallel``; without it (the
plain CPU image) a stub provides that layout.
"""

from __future__ import annotations

import os
import socket
import sys
import tempfile
import types
from argparse import Namespace
from pathlib import Path

import pytest
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

TESTS_ROOT = Path(__file__).resolve().parent
for path in (TESTS_ROOT, TESTS_ROOT.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import _unit_stubs

NUM_GPUS = 0


def _install_pure_data_parallel_mpu() -> None:
    world = lambda: dist.group.WORLD  # noqa: E731
    mpu = types.ModuleType("megatron.core.mpu")
    mpu.get_context_parallel_world_size = lambda: 1
    mpu.get_context_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_world_size = lambda: 1
    mpu.get_tensor_model_parallel_rank = lambda: 0
    mpu.get_tensor_model_parallel_group = lambda: None  # set per worker in _init_worker
    mpu.get_pipeline_model_parallel_world_size = lambda: 1
    mpu.get_pipeline_model_parallel_rank = lambda: 0
    mpu.is_pipeline_last_stage = lambda ignore_virtual=False: True
    mpu.get_data_parallel_group = lambda with_context_parallel=False: world()
    mpu.get_data_parallel_group_gloo = lambda with_context_parallel=False: world()
    mpu.get_data_parallel_world_size = lambda with_context_parallel=False: dist.get_world_size()
    mpu.get_data_parallel_rank = lambda with_context_parallel=False: dist.get_rank()
    mpu.get_data_parallel_src_rank = lambda with_context_parallel=False: 0
    core = types.ModuleType("megatron.core")
    core.mpu = mpu
    megatron = types.ModuleType("megatron")
    megatron.core = core
    sys.modules.setdefault("megatron", megatron)
    sys.modules.setdefault("megatron.core", core)
    sys.modules.setdefault("megatron.core.mpu", mpu)


def _install_param_meta_stub() -> None:
    try:
        import vllm.distributed.weight_transfer.base  # noqa: F401
    except ImportError:
        from collections import namedtuple

        base = types.ModuleType("vllm.distributed.weight_transfer.base")
        base.ParamMeta = namedtuple("ParamMeta", ["name", "dtype", "shape"])
        sys.modules["vllm.distributed.weight_transfer.base"] = base


if not _unit_stubs.real_module_available("megatron.core"):
    _install_pure_data_parallel_mpu()


def _free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def _tiny_config(tie_word_embeddings: bool):
    from transformers import Qwen3Config

    return Qwen3Config(
        vocab_size=96,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=8,
        max_position_embeddings=128,
        tie_word_embeddings=tie_word_embeddings,
    )


def _save_tiny_model(path: Path, seed: int, tie_word_embeddings: bool) -> None:
    from transformers import AutoModelForCausalLM

    torch.manual_seed(seed)
    model = AutoModelForCausalLM.from_config(_tiny_config(tie_word_embeddings), dtype=torch.float32)
    model.save_pretrained(path)


def _args(tmp: Path, **overrides) -> Namespace:
    values = dict(
        hf_checkpoint=str(tmp / "policy"),
        # Like vime's launcher, trainable weights start from --load (an HF dir here).
        load=str(tmp / "policy"),
        ref_load=str(tmp / "reference"),
        save=str(tmp / "save"),
        ckpt_step=None,
        no_save_optim=False,
        no_load_optim=False,
        no_save_rng=False,
        no_load_rng=False,
        fsdp_attn_implementation="sdpa",
        fsdp_param_dtype="fp32",
        fsdp_reduce_dtype="fp32",
        fsdp_reshard_after_forward=True,
        fsdp_cpu_offload=False,
        fsdp_gradient_checkpointing=False,
        fp16=False,
        optimizer="adam",
        lr=1e-3,
        min_lr=0.0,
        lr_warmup_init=0.0,
        lr_warmup_iters=0,
        lr_warmup_fraction=None,
        lr_decay_iters=None,
        lr_decay_style="constant",
        lr_wsd_decay_iters=None,
        lr_wsd_decay_style=None,
        weight_decay=0.1,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
        clip_grad=1.0,
        num_rollout=4,
        rollout_batch_size=4,
        n_samples_per_prompt=2,
        global_batch_size=4,
        micro_batch_size=2,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        balance_data=False,
        balance_by_flops=False,
        rollout_temperature=0.8,
        rollout_top_p=1.0,
        log_probs_chunk_size=-1,
        use_rollout_entropy=False,
        allgather_cp=False,
        loss_type="policy_loss",
        advantage_estimator="grpo",
        kl_coef=0.0,
        use_kl_loss=False,
        kl_loss_coef=0.0,
        kl_loss_type="low_var_kl",
        use_unbiased_kl=False,
        entropy_coef=0.01,
        eps_clip=0.2,
        eps_clip_high=0.28,
        eps_clip_c=None,
        use_rollout_logprobs=False,
        get_mismatch_metrics=False,
        use_tis=False,
        use_opsm=False,
        use_critic=False,
        use_opd=False,
        normalize_advantages=False,
        custom_advantage_function_path=None,
        custom_pg_loss_reducer_function_path=None,
        custom_tis_function_path=None,
        calculate_per_token_loss=False,
        recompute_loss_function=False,
        save_debug_train_data=None,
        ref_update_interval=None,
        ci_test=False,
        ci_save_grad_norm=None,
        ci_load_grad_norm=None,
        use_wandb=False,
        use_tensorboard=False,
        update_weight_buffer_size=512,
    )
    values.update(overrides)
    return Namespace(**values)


def _samples(num_samples: int, vocab_size: int):
    """Deterministic GRPO batch; some responses end with token id 0."""
    generator = torch.Generator().manual_seed(1234)
    samples = []
    for i in range(num_samples):
        prompt = int(torch.randint(3, 7, (1,), generator=generator))
        response = int(torch.randint(2, 6, (1,), generator=generator))
        tokens = torch.randint(1, vocab_size, (prompt + response,), generator=generator)
        if i % 3 == 0:
            tokens[-1] = 0
        mask = torch.ones(response, dtype=torch.int)
        if i % 4 == 1:
            mask[-1] = 0
        samples.append(
            {
                "tokens": tokens,
                "total_lengths": prompt + response,
                "response_lengths": response,
                "loss_masks": mask,
                "rewards": float(torch.rand((), generator=generator)) - 0.5,
                "rollout_log_probs": -torch.rand(response, generator=generator),
            }
        )
    return samples


def _local_rollout_data(args, samples, rank: int, world: int) -> dict:
    """Build this rank's rollout data with the production DP schedule (one sample per rollout)."""
    from vime.backends.fsdp_utils.trainer import move_rollout_data_to_device
    from vime.utils.dp_schedule import build_dp_schedule

    schedule_args = Namespace(**vars(args), hidden_size=32, num_attention_heads=4, num_query_groups=2)
    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        schedule_args,
        {"dp_size": world, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
        [s["total_lengths"] for s in samples],
        global_batch_size=args.global_batch_size,
        rollout_indices=list(range(len(samples))),
    )
    partition = partitions[rank]
    data = {key: [samples[i][key] for i in partition] for key in samples[0]}
    data["rollout_mask_sums"] = [samples[i]["loss_masks"].sum() for i in partition]
    data.update(
        partition=partition,
        micro_batch_indices=micro_batch_indices[rank],
        num_microbatches=num_microbatches,
        global_batch_sizes=global_batch_sizes,
    )
    return move_rollout_data_to_device(data, torch.device("cpu"))


def _reference_log_probs(path: str, samples, temperature: float) -> list[torch.Tensor]:
    """Unsharded HF model, one sample at a time."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(path, dtype=torch.float32, attn_implementation="sdpa")
    result = []
    with torch.no_grad():
        for sample in samples:
            tokens = sample["tokens"]
            logits = model(input_ids=tokens[None], use_cache=False).logits[0].float() / temperature
            log_probs = logits.log_softmax(-1)[:-1].gather(-1, tokens[1:, None])[:, 0]
            result.append(log_probs[-sample["response_lengths"] :])
    return result


def _build(args, world: int):
    from torch.distributed.device_mesh import init_device_mesh
    from transformers import AutoConfig

    from vime.backends.fsdp_utils.lr_scheduler import get_param_scheduler
    from vime.backends.fsdp_utils.model import build_fsdp_model, build_optimizer, initial_weights_path
    from vime.backends.fsdp_utils.trainer import FSDPTrainer

    mesh = init_device_mesh("cpu", (world,))
    config = AutoConfig.from_pretrained(args.hf_checkpoint)
    model = build_fsdp_model(args, config, initial_weights_path(args), mesh, torch.device("cpu"))
    optimizer = build_optimizer(args, model)
    return FSDPTrainer(args, model, optimizer, get_param_scheduler(args, optimizer), torch.device("cpu"))


def _full_state(model) -> dict[str, torch.Tensor]:
    from torch.distributed.tensor import DTensor

    return {
        name: (param.full_tensor() if isinstance(param, DTensor) else param).detach().clone()
        for name, param in model.named_parameters()
    }


def _init_worker(rank: int, world: int, port: int) -> None:
    os.environ.update(MASTER_ADDR="127.0.0.1", MASTER_PORT=str(port))
    dist.init_process_group("gloo", rank=rank, world_size=world)
    from megatron.core import mpu

    if hasattr(mpu, "initialize_model_parallel"):  # real Megatron; the stub has no initializer
        # Same layout as FSDPTrainRayActor.init registers.
        mpu.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
            create_gloo_process_groups=True,
        )
        assert mpu.get_data_parallel_world_size(with_context_parallel=True) == world
        return
    singletons = [dist.new_group([r]) for r in range(world)]
    # Mutate the installed stub: this module may be imported twice in a worker
    # (as __mp_main__ and by name), and the loss code is bound to the first.
    mpu.get_tensor_model_parallel_group = lambda: singletons[rank]


def _train_worker(rank, world, port, tmp, overrides, with_ref, result_path):
    _init_worker(rank, world, port)
    try:
        tmp = Path(tmp)
        args = _args(tmp, **overrides)
        trainer = _build(args, world)
        samples = _samples(8, 96)
        data = _local_rollout_data(args, samples, rank, world)

        # Packed, sharded log-probs equal the unsharded per-sample reference.
        reference = _reference_log_probs(args.load, [samples[i] for i in data["partition"]], 0.8)
        actor_log_probs = trainer.compute_log_prob(data)["log_probs"]
        for got, expected in zip(actor_log_probs, reference, strict=True):
            torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

        if with_ref:
            from vime.backends.fsdp_utils.model import load_hf_weights_into

            trainer.backup.backup("actor")
            load_hf_weights_into(trainer.model, args.ref_load, "sdpa")
            trainer.backup.backup("ref")
            trainer.switch_model("actor")

        steps = []
        trainer._log_train_step = lambda *log_args: steps.append(log_args)
        trainer.prepare_actor_inputs(data)
        if with_ref:
            ref_expected = _reference_log_probs(args.ref_load, [samples[i] for i in data["partition"]], 0.8)
            for got, expected in zip(data["ref_log_probs"], ref_expected, strict=True):
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)
        trainer.train(rollout_id=0, rollout_data=data)
        if args.save_debug_train_data is not None:
            # A single step reuses the training forward; the dump still gets old-policy log-probs.
            for got, expected in zip(data["log_probs"], reference, strict=True):
                torch.testing.assert_close(got, expected, rtol=1e-5, atol=1e-5)

        state = _full_state(trainer.model)
        if rank == 0:
            torch.save(
                {
                    "state": state,
                    "grad_norms": [log_args[4] for log_args in steps],
                    "losses": [log_args[3] for log_args in steps],
                    "lr": trainer.optimizer.param_groups[0]["lr"],
                },
                result_path,
            )
    finally:
        dist.destroy_process_group()


def _spawn(fn, world, *fn_args):
    mp.spawn(fn, args=(world, _free_port(), *fn_args), nprocs=world, join=True)


@pytest.fixture(scope="module")
def models():
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        for tie in (False, True):
            _save_tiny_model(tmp / f"tie{int(tie)}" / "policy", seed=1, tie_word_embeddings=tie)
            _save_tiny_model(tmp / f"tie{int(tie)}" / "reference", seed=2, tie_word_embeddings=tie)
        yield tmp


@pytest.mark.parametrize(
    ("tie", "overrides", "with_ref"),
    [
        pytest.param(False, {}, False, id="grpo"),
        pytest.param(True, {"calculate_per_token_loss": True}, False, id="tied-per-token"),
        pytest.param(False, {"use_kl_loss": True, "kl_loss_coef": 0.5, "entropy_coef": 0.0}, True, id="kl-ref"),
        pytest.param(False, {"global_batch_size": 8, "save_debug_train_data": "unused"}, False, id="one-step-dump"),
    ],
)
def test_data_parallel_size_does_not_change_the_update(models, tie, overrides, with_ref):
    """dp=1 and dp=2 consume the same global batch and must take the same optimizer steps."""
    tmp = models / f"tie{int(tie)}"
    results = {}
    for world in (1, 2):
        path = tmp / f"result_dp{world}.pt"
        _spawn(_train_worker, world, str(tmp), overrides, with_ref, str(path))
        results[world] = torch.load(path, weights_only=False)

    dp1, dp2 = results[1], results[2]
    num_steps = 8 // overrides.get("global_batch_size", 4)
    assert len(dp1["grad_norms"]) == len(dp2["grad_norms"]) == num_steps
    for norm1, norm2 in zip(dp1["grad_norms"], dp2["grad_norms"], strict=True):
        assert norm1 == pytest.approx(norm2, rel=1e-4)
    for losses1, losses2 in zip(dp1["losses"], dp2["losses"], strict=True):
        assert losses1.keys() == losses2.keys()
        for key in losses1:
            assert losses1[key] == pytest.approx(losses2[key], rel=1e-4, abs=1e-6), key
    if overrides.get("use_kl_loss"):
        assert all("kl_loss" in losses for losses in dp1["losses"])

    from transformers import AutoModelForCausalLM

    initial = AutoModelForCausalLM.from_pretrained(tmp / "policy", dtype=torch.float32).state_dict()
    moved = 0.0
    for name, value in dp1["state"].items():
        torch.testing.assert_close(value, dp2["state"][name], rtol=1e-4, atol=1e-5, msg=name)
        moved = max(moved, (value - initial[name]).abs().max().item())
    assert moved > 1e-4, "the optimizer steps did not change the weights"


def _reference_step_gradient(policy_dir: str, ref_dir: str, samples, *, temperature, entropy_coef, kl_coef):
    """Plain PyTorch: d/dθ of (1/G) Σ_s [-c_ent·mean_t H_t + c_kl·mean_t (log p_t - log p_ref_t)]."""
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(policy_dir, dtype=torch.float32, attn_implementation="sdpa")
    reference = AutoModelForCausalLM.from_pretrained(ref_dir, dtype=torch.float32, attn_implementation="sdpa")
    total = 0.0
    for sample in samples:
        tokens, response = sample["tokens"], sample["response_lengths"]
        target = tokens[1:][-response:]
        log_probs = (model(input_ids=tokens[None], use_cache=False).logits[0] / temperature).log_softmax(-1)
        log_probs = log_probs[:-1][-response:]
        token_log_probs = log_probs.gather(-1, target[:, None])[:, 0]
        entropy = -(log_probs.exp() * log_probs).sum(-1)
        with torch.no_grad():
            ref_log_probs = (reference(input_ids=tokens[None], use_cache=False).logits[0] / temperature).log_softmax(
                -1
            )
            ref_token_log_probs = ref_log_probs[:-1][-response:].gather(-1, target[:, None])[:, 0]
        mask = sample["loss_masks"].float()
        kl = token_log_probs - ref_token_log_probs
        total = total + ((kl_coef * kl - entropy_coef * entropy) * mask).sum() / mask.sum().clamp_min(1)
    (total / len(samples)).backward()
    return {name: param.grad for name, param in model.named_parameters()}


def _gradient_worker(rank, world, port, tmp, overrides, with_ref, result_path):
    _init_worker(rank, world, port)
    try:
        from vime.backends.fsdp_utils.data_packing import iter_step_micro_batches
        from vime.backends.fsdp_utils.model import load_hf_weights_into

        args = _args(Path(tmp), **overrides)
        trainer = _build(args, world)
        samples = _samples(8, 96)
        for sample in samples:
            sample["rewards"] = 0.0  # GRPO advantages are the rewards here: no policy-gradient term
        data = _local_rollout_data(args, samples, rank, world)
        if with_ref:
            trainer.backup.backup("actor")
            load_hf_weights_into(trainer.model, args.ref_load, "sdpa")
            trainer.backup.backup("ref")
            trainer.switch_model("actor")

        grads = {}
        optimizer_step = trainer.optimizer.step

        def capture_and_step():
            for name, param in trainer.model.named_parameters():
                grads[name] = param.grad.full_tensor().clone()
            optimizer_step()

        trainer.optimizer.step = capture_and_step
        trainer.prepare_actor_inputs(data)
        (_, step0), *_ = iter_step_micro_batches(data)
        trainer.train_step(data, step0, data["global_batch_sizes"][0])
        if rank == 0:
            torch.save(grads, result_path)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize(
    ("kl", "reference"),
    [(False, "policy"), (True, "policy"), (True, "reference")],
    ids=["entropy", "kl-ref-equals-actor", "kl-ref-differs"],
)
def test_step_gradient_matches_reference_loss(models, kl, reference):
    """The trainer's gradient (packing, loss scaling, reduce-scatter, reference-model switching)
    equals a plain unsharded computation of the same loss. With k1 the gradient does not
    depend on the reference log-probs, so a differing reference model leaves the expected
    gradient unchanged and only catches a trainer that keeps computing on the reference weights."""
    tmp = models / "tie1"
    overrides = dict(lr=0.0, clip_grad=0.0, entropy_coef=0.01, kl_loss_type="k1", ref_load=str(tmp / reference))
    if kl:
        overrides.update(use_kl_loss=True, kl_loss_coef=1.0)
    path = tmp / f"grads_{int(kl)}_{reference}.pt"
    _spawn(_gradient_worker, 2, str(tmp), overrides, kl, str(path))
    got = torch.load(path, weights_only=False)

    # Static micro-batches of 2 and a global batch of 4: step 0 holds the first 4 rollouts.
    samples = _samples(8, 96)[:4]
    expected = _reference_step_gradient(
        str(tmp / "policy"),
        str(tmp / reference),
        samples,
        temperature=0.8,
        entropy_coef=0.01,
        kl_coef=1.0 if kl else 0.0,
    )
    assert got.keys() == expected.keys()
    for name, grad in expected.items():
        torch.testing.assert_close(got[name], grad, rtol=1e-4, atol=1e-7, msg=name)


def _state_worker(rank, world, port, tmp):
    _init_worker(rank, world, port)
    try:
        from transformers import AutoModelForCausalLM

        from vime.backends.fsdp_utils import checkpoint
        from vime.backends.fsdp_utils.model import load_hf_weights_into
        from vime.backends.fsdp_utils.update_weight import FSDPWeightSource

        tmp = Path(tmp)
        args = _args(tmp)
        trainer = _build(args, world)
        policy = AutoModelForCausalLM.from_pretrained(args.load, dtype=torch.float32)
        samples = _samples(8, 96)
        data = _local_rollout_data(args, samples, rank, world)

        # Weight export: HF names, global shapes, rollout dtype; several buckets.
        _install_param_meta_stub()
        source = FSDPWeightSource(trainer.model, dtype=torch.bfloat16, device=torch.device("cpu"), bucket_bytes=2048)
        expected = dict(policy.named_parameters())
        exported = list(source)
        assert [name for name, _ in exported] == list(expected)
        assert [(m.name, m.shape) for m in source.metadata()] == [(n, tuple(t.shape)) for n, t in exported]
        assert all(m.dtype == torch.bfloat16 for m in source.metadata())
        for name, tensor in exported:
            assert tensor.dtype == torch.bfloat16
            torch.testing.assert_close(tensor, expected[name].detach().to(torch.bfloat16), rtol=0, atol=0, msg=name)

        # Reference weights share the sharded module through host backups.
        trainer.backup.backup("actor")
        load_hf_weights_into(trainer.model, args.ref_load, "sdpa")
        trainer.backup.backup("ref")
        trainer.switch_model("actor")
        local = [samples[i] for i in data["partition"]]
        for tag, path in (("actor", args.load), ("ref", args.ref_load), ("actor", args.load)):
            trainer.switch_model(tag)
            got = trainer.compute_log_prob(data)["log_probs"]
            for value, reference in zip(got, _reference_log_probs(path, local, 0.8), strict=True):
                torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-5)

        # One optimizer step, then a checkpoint round trip into a model built from other weights.
        trainer.prepare_actor_inputs(data)
        trainer._log_train_step = lambda *log_args: None
        trainer.train(rollout_id=0, rollout_data=data)
        checkpoint.save(trainer, rollout_id=3)
        save_root = Path(args.save)
        assert (save_root / checkpoint.TRACKER).read_text() == "3"
        assert (save_root / "iter_0000003" / "model" / ".metadata").exists()
        assert (save_root / "iter_0000003" / "rng" / f"rank_{rank}.pt").exists()

        # --load now holds the checkpoint; the initial weights come from --ref-load.
        resumed = _build(_args(tmp, load=str(save_root)), world)
        assert checkpoint.load(resumed) == 3
        trained, restored = _full_state(trainer.model), _full_state(resumed.model)
        for name, value in trained.items():
            torch.testing.assert_close(restored[name], value, rtol=0, atol=0, msg=name)
        assert resumed.scheduler.num_steps == trainer.scheduler.num_steps == 8
        assert resumed.global_step == trainer.global_step == 2
        for s1, s2 in zip(trainer.optimizer.state.values(), resumed.optimizer.state.values(), strict=True):
            torch.testing.assert_close(s2["exp_avg"].full_tensor(), s1["exp_avg"].full_tensor(), rtol=0, atol=0)

        # Megatron checkpoints are refused with a clear error.
        bogus = tmp / "megatron"
        if rank == 0:
            (bogus / "iter_0000001").mkdir(parents=True, exist_ok=True)
            (bogus / checkpoint.TRACKER).write_text("1")
        dist.barrier()
        with pytest.raises(ValueError, match="not an fsdp2 checkpoint"):
            checkpoint.load(_build(_args(tmp, load=str(bogus)), world))

        # HF export reloads with plain transformers.
        checkpoint.save_hf(trainer, tmp / "hf_export")
        if rank == 0:
            reloaded = AutoModelForCausalLM.from_pretrained(tmp / "hf_export", dtype=torch.float32)
            assert reloaded.config.architectures == ["Qwen3ForCausalLM"]
            for name, value in trained.items():
                torch.testing.assert_close(dict(reloaded.named_parameters())[name], value, msg=name)
    finally:
        dist.destroy_process_group()


@pytest.mark.parametrize("tie", [False, True], ids=["untied", "tied"])
def test_weight_export_model_switch_and_checkpoint(models, tie):
    _spawn(_state_worker, 2, str(models / f"tie{int(tie)}"))


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
