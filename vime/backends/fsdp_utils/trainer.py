"""Training core of the FSDP2 backend, independent of Ray.

Log-probabilities, advantages, losses and step metrics come from the same
functions the Megatron backend uses (``megatron_utils.loss`` and
``train_metric_utils``); this module only drives the HF model through vime's
micro-batch schedule and applies Megatron's gradient scaling, so both
backends optimize the same objective for the same batch.
"""

from __future__ import annotations

import logging
import math

import torch
import torch.distributed as dist

from vime.observability import logging_utils
from vime.observability.timer import timer

from .data_packing import iter_step_micro_batches, pack_micro_batch, restore_sample_order
from .model import ShardBackup, total_grad_norm

logger = logging.getLogger(__name__)

LOG_PROB_KEYS = ["tokens", "loss_masks", "total_lengths", "response_lengths"]
TRAIN_KEYS = [
    "tokens",
    "total_lengths",
    "response_lengths",
    "loss_masks",
    "log_probs",
    "ref_log_probs",
    "values",
    "advantages",
    "returns",
    "rollout_log_probs",
    "rollout_top_p_log_probs",
    "teacher_log_probs",
    "rollout_mask_sums",
]


def _with_rollout_top_p_token_keys(args, keys: list[str]) -> list[str]:
    from vime.backends.megatron_utils.loss import ROLLOUT_TOP_P_TOKEN_KEYS

    return [*keys, *ROLLOUT_TOP_P_TOKEN_KEYS] if args.rollout_top_p != 1.0 else keys


def move_rollout_data_to_device(rollout_data: dict, device: torch.device) -> dict:
    """Place the per-sample tensors used by the forward/loss on ``device``."""
    rollout_data["tokens"] = [torch.as_tensor(t).to(device=device, dtype=torch.long) for t in rollout_data["tokens"]]
    rollout_data["loss_masks"] = [
        torch.as_tensor(t).to(device=device, dtype=torch.int) for t in rollout_data["loss_masks"]
    ]
    if "rollout_mask_sums" in rollout_data:
        rollout_data["rollout_mask_sums"] = torch.as_tensor(rollout_data["rollout_mask_sums"]).to(
            device=device, dtype=torch.float32
        )
    for key in ("rollout_log_probs", "teacher_log_probs"):
        if key in rollout_data:
            rollout_data[key] = [
                torch.as_tensor(value).to(device=device, dtype=torch.float32) for value in rollout_data[key]
            ]
    return rollout_data


class FSDPTrainer:
    """Holds the sharded model, optimizer and schedule, and runs one rollout's work."""

    def __init__(self, args, model, optimizer, scheduler, device: torch.device) -> None:
        self.args = args
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = torch.device(device)
        self.backup = ShardBackup(model)
        self.global_step = 0

    def switch_model(self, tag: str) -> None:
        if tag not in self.backup.tags:
            raise ValueError(f"Cannot switch to unknown model tag: {tag}")
        self.backup.restore(tag)

    def forward_logits(self, packed) -> torch.Tensor:
        # use_cache=False is required: with a KV cache HF skips the per-sample
        # masks it derives from restarting position_ids.
        output = self.model(input_ids=packed.input_ids, position_ids=packed.position_ids, use_cache=False)
        return output.logits.float()

    @torch.no_grad()
    def compute_log_prob(
        self, rollout_data: dict, store_prefix: str = "", use_rollout_top_p_replay: bool = True
    ) -> dict[str, list[torch.Tensor]]:
        """Per-sample response log-probs (and entropy if requested), in local sample order."""
        from vime.backends.megatron_utils.loss import get_log_probs_and_entropy, get_rollout_top_p_logprob_kwargs

        keys = _with_rollout_top_p_token_keys(self.args, LOG_PROB_KEYS) if use_rollout_top_p_replay else LOG_PROB_KEYS
        outputs: dict[str, list[torch.Tensor]] = {}
        order: list[int] = []
        self.model.eval()
        try:
            with timer(f"{store_prefix}log_probs"):
                for indices in rollout_data["micro_batch_indices"]:
                    packed = pack_micro_batch(rollout_data, indices, keys, device=self.device)
                    top_p_kwargs = (
                        get_rollout_top_p_logprob_kwargs(self.args, packed.batch) if use_rollout_top_p_replay else {}
                    )
                    _, result = get_log_probs_and_entropy(
                        self.forward_logits(packed),
                        args=self.args,
                        unconcat_tokens=packed.batch["unconcat_tokens"],
                        total_lengths=packed.batch["total_lengths"],
                        response_lengths=packed.batch["response_lengths"],
                        with_entropy=self.args.use_rollout_entropy,
                        **top_p_kwargs,
                    )
                    for key, values in result.items():
                        outputs.setdefault(key, []).extend(values)
                    order.extend(indices)
        finally:
            self.model.train()
        return {f"{store_prefix}{key}": restore_sample_order(values, order) for key, values in outputs.items()}

    def prepare_actor_inputs(self, rollout_data: dict) -> None:
        """Add reference/old log-probs, advantages and returns to ``rollout_data`` in place."""
        from vime.backends.megatron_utils.loss import compute_advantages_and_returns

        args = self.args
        if "ref" in self.backup.tags:
            self.switch_model("ref")
            rollout_data.update(
                self.compute_log_prob(rollout_data, store_prefix="ref_", use_rollout_top_p_replay=False)
            )
            self.switch_model("actor")

        # Same reuse rule as the Megatron actor: with a single optimizer step the
        # training forward reproduces the old-policy log-probs exactly.
        can_reuse_log_probs_in_loss = (
            len(rollout_data["num_microbatches"]) == 1
            and args.loss_type == "policy_loss"
            and args.kl_coef == 0
            and not args.use_rollout_logprobs
            and not args.get_mismatch_metrics
            and not args.use_critic
            and not args.use_opd
            and args.advantage_estimator != "gspo"
        )
        if (not args.use_rollout_logprobs or args.get_mismatch_metrics) and not can_reuse_log_probs_in_loss:
            rollout_data.update(self.compute_log_prob(rollout_data))
        compute_advantages_and_returns(args, rollout_data)

    def train(self, rollout_id: int, rollout_data: dict) -> None:
        """Run every optimizer step of one rollout, then refresh the model backups."""
        from vime.backends.megatron_utils.loss import drain_captured_log_probs, enable_log_prob_capture

        steps = list(iter_step_micro_batches(rollout_data))
        global_batch_sizes = rollout_data["global_batch_sizes"]
        assert len(steps) == len(global_batch_sizes)
        # Without a separate log-prob pass, the train debug dump takes the
        # training forward's log-probs, as in the Megatron actor.
        capture_log_probs = self.args.save_debug_train_data is not None and "log_probs" not in rollout_data
        if capture_log_probs:
            enable_log_prob_capture()
        for step_id, step_micro_batches in steps:
            loss_dict, grad_norm = self.train_step(rollout_data, step_micro_batches, global_batch_sizes[step_id])
            self._log_train_step(rollout_id, step_id, len(steps), loss_dict, grad_norm, global_batch_sizes[step_id])
        if capture_log_probs:
            captured = drain_captured_log_probs()
            if captured:
                rollout_data["log_probs"] = [captured[position] for position in rollout_data["partition"]]

        if "ref" in self.backup.tags:
            self.backup.backup("actor")
            if self.args.ref_update_interval is not None and (rollout_id + 1) % self.args.ref_update_interval == 0:
                if dist.get_rank() == 0:
                    logger.info(f"Updating ref model at rollout_id {rollout_id}")
                self.backup.backup("ref")

    def train_step(
        self, rollout_data: dict, step_micro_batches: list[list[int]], step_global_batch_size: int
    ) -> tuple[dict[str, float], float]:
        from megatron.core import mpu

        from vime.backends.megatron_utils.loss import loss_function
        from vime.observability.train_metric_utils import reduce_train_step_metrics

        args = self.args
        keys = TRAIN_KEYS + (["partition"] if args.save_debug_train_data is not None else [])
        keys = _with_rollout_top_p_token_keys(args, keys)
        num_microbatches = len(step_micro_batches)
        dp_group = mpu.get_data_parallel_group(with_context_parallel=True)

        # loss_function scales a per-sample-mean loss by num_microbatches /
        # step_global_batch_size * dp_size and relies on Megatron's schedule to
        # divide by num_microbatches; FSDP's gradient reduce-scatter then averages
        # over ranks like Megatron's DDP. Per-token losses are instead divided by
        # the step's token count across ranks (Megatron's finalize_model_grads).
        token_scale = None
        if args.calculate_per_token_loss:
            local_tokens = sum(
                torch.clamp_min(rollout_data["loss_masks"][i].sum(), 1).item()
                for indices in step_micro_batches
                for i in indices
            )
            total_tokens = torch.tensor([local_tokens], dtype=torch.float64, device=self.device)
            dist.all_reduce(total_tokens, group=dp_group)
            token_scale = dist.get_world_size(dp_group) / total_tokens.item()

        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        losses_reduced = []
        for indices in step_micro_batches:
            packed = pack_micro_batch(rollout_data, indices, keys, device=self.device)
            loss, _, log = loss_function(
                args, packed.batch, num_microbatches, step_global_batch_size, self.forward_logits(packed)
            )
            loss = loss * token_scale if token_scale is not None else loss / num_microbatches
            loss.backward()
            losses_reduced.append(log)

        grad_norm = total_grad_norm(self.model.parameters(), args.clip_grad)
        if math.isfinite(grad_norm):
            self.optimizer.step()
            self.scheduler.step(increment=step_global_batch_size)
        else:
            logger.warning(f"Skipping optimizer step {self.global_step}: grad norm is {grad_norm}")
        self.optimizer.zero_grad(set_to_none=True)
        self.global_step += 1

        loss_dict = reduce_train_step_metrics(
            losses_reduced,
            calculate_per_token_loss=args.calculate_per_token_loss,
            step_global_batch_size=step_global_batch_size,
            cp_size=1,
            dp_with_cp_group=dp_group,
        )
        return loss_dict, grad_norm

    def _log_train_step(self, rollout_id, step_id, num_steps, loss_dict, grad_norm, global_batch_size) -> None:
        """Report one optimizer step with the Megatron backend's keys and CI checks."""
        if dist.get_rank() != 0:
            return
        args = self.args
        accumulated_step_id = rollout_id * num_steps + step_id
        log_dict = {f"train/{key}": value for key, value in loss_dict.items()}
        log_dict["train/grad_norm"] = grad_norm
        for group_id, group in enumerate(self.optimizer.param_groups):
            log_dict[f"train/lr-pg_{group_id}"] = self.scheduler.get_lr(group)
        log_dict["train/global_batch_size"] = global_batch_size
        log_dict["train/step"] = accumulated_step_id
        logging_utils.log(args, log_dict, step_key="train/step")

        if args.ci_test and "train/train_rollout_logprob_abs_diff" in log_dict:
            threshold = args.ci_train_rollout_logprob_abs_diff_threshold
            assert log_dict["train/train_rollout_logprob_abs_diff"] <= threshold, f"{threshold=} {log_dict=}"
        if args.ci_test and not args.ci_disable_kl_checker:
            if step_id == 0 and "train/ppo_kl" in log_dict and "train/pg_clipfrac" in log_dict:
                assert log_dict["train/ppo_kl"] < 1e-8, f"{log_dict=}"
            if accumulated_step_id == 0 and args.rollout_top_p == 1.0 and "train/kl_loss" in log_dict:
                assert log_dict["train/kl_loss"] < 1e-8, f"{log_dict=}"
        logger.info(f"step {accumulated_step_id}: {log_dict}")

        if args.ci_save_grad_norm is not None:
            path = args.ci_save_grad_norm.format(role="actor", rollout_id=rollout_id, step_id=step_id)
            torch.save(grad_norm, path)
        elif args.ci_load_grad_norm is not None:
            path = args.ci_load_grad_norm.format(role="actor", rollout_id=rollout_id, step_id=step_id)
            expected_grad_norm = torch.load(path)
            assert math.isclose(
                grad_norm, expected_grad_norm, rel_tol=0.01, abs_tol=0.01
            ), f"grad norm mismatch: {grad_norm} != {expected_grad_norm}"
