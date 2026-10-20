"""Check the restored training cursor before the checkpoint test takes a step."""

import json
import os
from pathlib import Path

import torch.distributed as dist


def before_train_step(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    if dist.get_rank() != 0 or step_id != 0:
        return
    phase = os.environ["VIME_CKPT_TEST_PHASE"]
    expected = (2,) if phase == "load" else (0, 1)
    assert rollout_id in expected, (phase, rollout_id)
    assert opt_param_scheduler.num_steps == rollout_id * args.global_batch_size
    if phase == "load":
        assert not args.finetune and not args.no_load_optim, "Must restore the optimizer, not just the model"
    report = {
        "rollout_id": rollout_id,
        "scheduler_num_steps": opt_param_scheduler.num_steps,
        "no_load_optim": args.no_load_optim,
        "finetune": args.finetune,
        "optimizer_cpu_offload": args.optimizer_cpu_offload,
    }
    path = Path(os.environ["VIME_CKPT_TEST_DIR"])
    path.mkdir(parents=True, exist_ok=True)
    (path / f"before_{rollout_id}.json").write_text(json.dumps(report))
