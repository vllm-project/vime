# Copyright (c) 2022, NVIDIA CORPORATION. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Learning-rate and weight-decay schedule for the FSDP2 backend.

Adapted from slime's ``slime/backends/fsdp_utils/lr_scheduler.py``
(https://github.com/THUDM/slime at 0104a9e9, Apache-2.0), itself derived from
Megatron-LM's ``OptimizerParamScheduler``. Like the Megatron backend, progress
is counted in samples: ``step(increment)`` receives the step's global batch
size and every boundary is ``iters * global_batch_size``. The same flags then
yield the same per-step learning rate on both backends.
"""

import logging
import math

import torch

logger = logging.getLogger(__name__)


class FSDPParamScheduler:
    """Sets ``lr`` and ``weight_decay`` of each param group from samples consumed."""

    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        *,
        init_lr: float,
        max_lr: float,
        min_lr: float,
        lr_warmup_steps: float,
        lr_decay_steps: int,
        lr_decay_style: str,
        start_wd: float,
        end_wd: float,
        wd_incr_steps: int,
        wd_incr_style: str,
        wsd_decay_steps: int | None = None,
        lr_wsd_decay_style: str | None = None,
    ) -> None:
        self.optimizer = optimizer
        self.init_lr = init_lr
        self.max_lr = float(max_lr)
        self.min_lr = min_lr
        assert self.min_lr >= 0.0
        assert self.max_lr >= self.min_lr
        assert self.init_lr <= self.max_lr

        self.lr_warmup_steps = lr_warmup_steps
        self.lr_decay_steps = lr_decay_steps
        assert self.lr_decay_steps > 0
        assert self.lr_warmup_steps < self.lr_decay_steps
        self.lr_decay_style = lr_decay_style
        self.wsd_decay_steps = wsd_decay_steps
        self.lr_wsd_decay_style = lr_wsd_decay_style
        if self.lr_decay_style == "WSD":
            assert self.wsd_decay_steps is not None

        self.start_wd = start_wd
        self.end_wd = end_wd
        assert 0.0 <= self.start_wd <= self.end_wd
        self.wd_incr_steps = wd_incr_steps
        self.wd_incr_style = wd_incr_style

        self.num_steps = 0
        self.step(0)
        logger.info(f"> learning rate decay style: {self.lr_decay_style}")

    def get_wd(self) -> float:
        if self.num_steps > self.wd_incr_steps:
            return self.end_wd
        if self.wd_incr_style == "constant":
            assert self.start_wd == self.end_wd
            return self.end_wd
        incr_ratio = float(self.num_steps) / float(self.wd_incr_steps)
        assert 0.0 <= incr_ratio <= 1.0
        delta_wd = self.end_wd - self.start_wd
        if self.wd_incr_style == "linear":
            coeff = incr_ratio
        elif self.wd_incr_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * (1 - incr_ratio)) + 1.0)
        else:
            raise ValueError(f"{self.wd_incr_style} weight decay increment style is not supported.")
        return self.start_wd + coeff * delta_wd

    def get_lr(self, param_group: dict) -> float:
        max_lr = param_group.get("max_lr", self.max_lr)
        min_lr = param_group.get("min_lr", self.min_lr)

        # Use linear warmup for the initial part.
        if self.lr_warmup_steps > 0 and self.num_steps <= self.lr_warmup_steps:
            return self.init_lr + ((max_lr - self.init_lr) * float(self.num_steps) / float(self.lr_warmup_steps))

        # If the learning rate is constant, just return the initial value.
        if self.lr_decay_style == "constant":
            return max_lr

        # For any steps larger than `self.lr_decay_steps`, use `min_lr`.
        if self.num_steps > self.lr_decay_steps:
            return min_lr

        # If we are done with the warmup period, use the decay style.
        if self.lr_decay_style == "inverse-square-root":
            warmup_steps = max(self.lr_warmup_steps, 1)
            num_steps = max(self.num_steps, 1)
            lr = max_lr * warmup_steps**0.5 / (num_steps**0.5)
            return max(min_lr, lr)

        num_steps_ = self.num_steps - self.lr_warmup_steps
        decay_steps_ = self.lr_decay_steps - self.lr_warmup_steps
        decay_ratio = float(num_steps_) / float(decay_steps_)
        assert 0.0 <= decay_ratio <= 1.0
        delta_lr = max_lr - min_lr

        if self.lr_decay_style == "linear":
            coeff = 1.0 - decay_ratio
        elif self.lr_decay_style == "cosine":
            coeff = 0.5 * (math.cos(math.pi * decay_ratio) + 1.0)
        elif self.lr_decay_style == "WSD":
            wsd_anneal_start_ = self.lr_decay_steps - self.wsd_decay_steps
            if self.num_steps <= wsd_anneal_start_:
                coeff = 1.0
            else:
                wsd_steps = self.num_steps - wsd_anneal_start_
                wsd_decay_ratio = float(wsd_steps) / float(self.wsd_decay_steps)
                if self.lr_wsd_decay_style == "linear":
                    coeff = 1.0 - wsd_decay_ratio
                elif self.lr_wsd_decay_style == "cosine":
                    coeff = 0.5 * (math.cos(math.pi * wsd_decay_ratio) + 1.0)
                elif self.lr_wsd_decay_style == "exponential":
                    coeff = (2.0 * math.pow(0.5, wsd_decay_ratio)) - 1.0
                elif self.lr_wsd_decay_style == "minus_sqrt":
                    coeff = 1.0 - math.sqrt(wsd_decay_ratio)
                else:
                    raise ValueError(f"{self.lr_wsd_decay_style} WSD decay style is not supported.")
        else:
            raise ValueError(f"{self.lr_decay_style} decay style is not supported.")
        return min_lr + coeff * delta_lr

    def step(self, increment: int) -> None:
        """Advance by ``increment`` samples and apply the new values to every param group."""
        self.num_steps += increment
        new_wd = self.get_wd()
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = self.get_lr(param_group) * param_group.get("lr_mult", 1.0)
            param_group["weight_decay"] = new_wd * param_group.get("wd_mult", 1.0)

    def state_dict(self) -> dict:
        return {"num_steps": self.num_steps}

    def load_state_dict(self, state_dict: dict) -> None:
        self.num_steps = state_dict["num_steps"]
        self.step(0)


def get_param_scheduler(args, optimizer: torch.optim.Optimizer) -> FSDPParamScheduler:
    """Build the schedule from the shared flags, sized exactly like the Megatron backend's."""
    args.train_iters = args.num_rollout * args.rollout_batch_size * args.n_samples_per_prompt // args.global_batch_size
    if args.lr_decay_iters is None:
        args.lr_decay_iters = args.train_iters
    lr_decay_steps = args.lr_decay_iters * args.global_batch_size
    wd_incr_steps = args.train_iters * args.global_batch_size
    wsd_decay_steps = None
    if args.lr_wsd_decay_iters is not None:
        wsd_decay_steps = args.lr_wsd_decay_iters * args.global_batch_size
    if args.lr_warmup_fraction is not None:
        lr_warmup_steps = args.lr_warmup_fraction * lr_decay_steps
    else:
        lr_warmup_steps = args.lr_warmup_iters * args.global_batch_size

    return FSDPParamScheduler(
        optimizer,
        init_lr=args.lr_warmup_init,
        max_lr=args.lr,
        min_lr=args.min_lr,
        lr_warmup_steps=lr_warmup_steps,
        lr_decay_steps=lr_decay_steps,
        lr_decay_style=args.lr_decay_style,
        start_wd=args.start_weight_decay,
        end_wd=args.end_weight_decay,
        wd_incr_steps=wd_incr_steps,
        wd_incr_style=args.weight_decay_incr_style,
        wsd_decay_steps=wsd_decay_steps,
        lr_wsd_decay_style=args.lr_wsd_decay_style,
    )
