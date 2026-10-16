"""Check Vime's scheduler against the actual DP schedule and Megatron runtime."""

import importlib
import os
import subprocess
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch


def _scheduler_args(**overrides):
    values = dict(
        num_rollout=1,
        rollout_batch_size=1,
        n_samples_per_prompt=1,
        global_batch_size=8,
        variable_global_batch_size=True,
        global_batch_size_schedule=None,
        lr_decay_iters=None,
        lr_wsd_decay_iters=None,
        lr_warmup_fraction=None,
        lr_warmup_iters=0,
        lr_warmup_init=0,
        lr=0.01,
        min_lr=0,
        lr_decay_style="linear",
        start_weight_decay=0,
        end_weight_decay=0.1,
        weight_decay_incr_style="linear",
        use_checkpoint_opt_param_scheduler=False,
        override_opt_param_scheduler=False,
        lr_wsd_decay_style="linear",
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=16,
        micro_batch_size=1,
        balance_data=False,
        balance_by_flops=False,
    )
    values.update(overrides)
    return Namespace(**values)


def _check_variable_batch_scheduler_matches_training_progress(
    num_rollout, rollout_batch_size, n_samples_per_prompt, global_batch_size, schedule
):
    args = _scheduler_args(
        num_rollout=num_rollout,
        rollout_batch_size=rollout_batch_size,
        n_samples_per_prompt=n_samples_per_prompt,
        global_batch_size=global_batch_size,
        global_batch_size_schedule=schedule,
        use_dynamic_batch_size=True,
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=args.lr)
    scheduler = get_optimizer_param_scheduler(args, optimizer)
    sample_count = rollout_batch_size * n_samples_per_prompt
    _, _, _, batch_sizes = build_dp_schedule(
        args,
        dict(dp_size=2, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1),
        [8] * sample_count,
        global_batch_size=global_batch_size,
        rollout_indices=list(range(sample_count)),
    )

    assert args.train_iters == num_rollout * len(batch_sizes)
    for _ in range(num_rollout):
        for batch_size in batch_sizes:
            scheduler.step(increment=batch_size)

    assert scheduler.num_steps == num_rollout * sum(batch_sizes)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.min_lr)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(args.end_weight_decay)


def _check_iteration_budgets_follow_nominal_batch_sizes(
    overrides, expected_decay, expected_warmup, expected_wsd, expected_total
):
    args = _scheduler_args(**overrides)
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=args.lr)

    scheduler = get_optimizer_param_scheduler(args, optimizer)

    assert scheduler.lr_decay_steps == expected_decay
    assert scheduler.lr_warmup_steps == expected_warmup
    assert scheduler.wsd_decay_steps == expected_wsd
    assert scheduler.wd_incr_steps == expected_total


def _check_explicit_schedule_does_not_decay_lr_early():
    args = _scheduler_args(
        rollout_batch_size=2,
        n_samples_per_prompt=3,
        global_batch_size=2,
        global_batch_size_schedule=[2, 4],
        lr_decay_iters=2,
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=args.lr)
    scheduler = get_optimizer_param_scheduler(args, optimizer)

    scheduler.step(increment=2)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.lr * (1 - 2 / 6))
    scheduler.step(increment=4)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.min_lr)


def _check_fixed_batch_drops_tail_per_rollout():
    args = _scheduler_args(
        num_rollout=4,
        rollout_batch_size=5,
        global_batch_size=4,
        variable_global_batch_size=False,
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=args.lr)
    scheduler = get_optimizer_param_scheduler(args, optimizer)
    _, _, _, batch_sizes = build_dp_schedule(
        args,
        dict(dp_size=2, cp_size=1, vpp_size=1, microbatch_group_size_per_vp_stage=1),
        [8] * 5,
        global_batch_size=4,
        rollout_indices=list(range(5)),
    )
    assert batch_sizes == [4]
    assert args.train_iters == 4
    assert scheduler.lr_decay_steps == scheduler.wd_incr_steps == 16
    for _ in range(args.num_rollout):
        scheduler.step(increment=batch_sizes[0])
    assert optimizer.param_groups[0]["lr"] == pytest.approx(args.min_lr)
    assert optimizer.param_groups[0]["weight_decay"] == pytest.approx(args.end_weight_decay)


def _check_variable_batch_initialization():
    from megatron.core.num_microbatches_calculator import ConstantNumMicroBatchesCalculator

    from vime.backends.megatron_utils import initialize

    args = Namespace(
        enable_experimental=False,
        rank=0,
        seed=1,
        data_parallel_random_init=False,
        te_rng_tracker=False,
        inference_rng_tracker=False,
        rampup_batch_size=None,
        global_batch_size=3,
        micro_batch_size=1,
        data_parallel_size=2,
        decrease_batch_size_if_needed=False,
        variable_global_batch_size=True,
        deterministic_mode=False,
        tp_comm_overlap=False,
        custom_megatron_init_path=None,
    )
    with (
        patch.object(initialize, "set_args"),
        patch.object(initialize, "_initialize_distributed"),
        patch.object(initialize, "_set_random_seed"),
        patch.object(initialize, "_build_tokenizer"),
        patch.object(
            initialize,
            "init_num_microbatches_calculator",
            side_effect=lambda rank, rampup, gbs, mbs, dp, decrease: ConstantNumMicroBatchesCalculator(
                gbs, mbs, dp, decrease, rank
            ),
        ) as calculator,
    ):
        initialize.init(args)
    calculator.assert_called_once_with(0, None, 2, 1, 2, False)


def _check_train_logging_tracks_actual_progress():
    model_module = importlib.import_module("vime.backends.megatron_utils.model")
    args = _scheduler_args(num_rollout=3, rollout_batch_size=5, global_batch_size=4)
    args.__dict__.update(
        overlap_grad_reduce=False,
        overlap_param_gather=False,
        reset_optimizer_states=False,
        manual_gc=False,
        enable_mtp_training=False,
        ci_test=True,
        ci_disable_kl_checker=False,
        use_rollout_routing_replay=False,
        ci_save_grad_norm=None,
        ci_load_grad_norm=None,
    )
    optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=args.lr)
    optimizer.scale_loss = lambda loss: loss
    scheduler = get_optimizer_param_scheduler(args, optimizer)
    model = SimpleNamespace(role="actor", train=lambda: None)
    logged = []

    def train_one_step(*call_args, **kwargs):
        if call_args[1] != 3:  # rollout 3 simulates an update skipped for invalid gradients
            scheduler.step(increment=call_args[8])
        return {"kl_loss": 0.0}, 0.0

    with (
        patch.object(model_module, "get_args", return_value=args),
        patch.object(model_module, "get_model_config", return_value=SimpleNamespace()),
        patch.object(model_module, "should_disable_forward_pre_hook", return_value=False),
        patch.object(model_module, "train_one_step", side_effect=train_one_step),
        patch.object(model_module, "_disable_tqdm_for_non_main_rank", return_value=True),
        patch.object(model_module.mpu, "get_data_parallel_rank", return_value=0),
        patch.object(model_module.mpu, "get_tensor_model_parallel_rank", return_value=0),
        patch.object(model_module.mpu, "get_pipeline_model_parallel_rank", return_value=0),
        patch.object(model_module.mpu, "get_pipeline_model_parallel_world_size", return_value=1),
        patch.object(
            model_module.logging_utils, "log", side_effect=lambda args, data, **kw: logged.append(data["train/step"])
        ),
    ):
        for rollout_id, step_sizes in ((0, [4, 1]), (1, [4]), (2, [4, 1]), (3, [2]), (4, [3])):
            model_module.train(rollout_id, [model], optimizer, scheduler, [], [1] * len(step_sizes), step_sizes)

    assert logged == [0, 4, 5, 9, 13, 14, 14]
    resumed_args = _scheduler_args(num_rollout=3, rollout_batch_size=5, global_batch_size=4)
    resumed_optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.ones(1))], lr=resumed_args.lr)
    resumed_scheduler = get_optimizer_param_scheduler(resumed_args, resumed_optimizer)
    resumed_scheduler.load_state_dict(scheduler.state_dict())
    assert resumed_scheduler.num_steps == 17


def test_scheduler_in_isolated_process():
    root = Path(__file__).resolve().parents[2]
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(filter(None, (str(root), env.get("PYTHONPATH"))))
    result = subprocess.run(
        [sys.executable, str(Path(__file__).resolve())],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


if __name__ == "__main__":
    from vime.backends.megatron_utils.model import get_optimizer_param_scheduler
    from vime.utils.dp_schedule import build_dp_schedule

    for case in (
        (1, 4, 2, 16, None),
        (3, 5, 2, 8, None),
        (3, 2, 3, 8, [2, 4]),
        (2, 4, 1, 4, [4]),
        (1, 8, 1, 3, [3, 5]),
    ):
        _check_variable_batch_scheduler_matches_training_progress(*case)
    for case in (
        (
            dict(
                rollout_batch_size=2,
                n_samples_per_prompt=3,
                global_batch_size=2,
                global_batch_size_schedule=[2, 4],
                lr_decay_iters=2,
                lr_warmup_iters=1,
                lr_wsd_decay_iters=1,
            ),
            6,
            2,
            4,
            6,
        ),
        (
            dict(
                num_rollout=2,
                rollout_batch_size=5,
                global_batch_size=4,
                lr_decay_iters=3,
                lr_warmup_iters=1,
                lr_wsd_decay_iters=1,
            ),
            9,
            4,
            4,
            10,
        ),
        (dict(rollout_batch_size=1, global_batch_size=8, lr_decay_iters=1, lr_wsd_decay_iters=1), 1, 0, 1, 1),
        (
            dict(
                num_rollout=2,
                rollout_batch_size=4,
                global_batch_size=4,
                variable_global_batch_size=False,
                lr_decay_iters=2,
                lr_warmup_iters=1,
                lr_wsd_decay_iters=1,
            ),
            8,
            4,
            4,
            8,
        ),
    ):
        _check_iteration_budgets_follow_nominal_batch_sizes(*case)
    _check_explicit_schedule_does_not_decay_lr_early()
    _check_fixed_batch_drops_tail_per_rollout()
    _check_variable_batch_initialization()
    _check_train_logging_tracks_actual_progress()
