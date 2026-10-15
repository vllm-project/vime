"""CPU tests for the fsdp2 backend's micro-batch packing and LR/weight-decay schedule."""

import math
import sys
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from vime.backends.fsdp_utils.data_packing import iter_step_micro_batches, pack_micro_batch, restore_sample_order
from vime.backends.fsdp_utils.lr_scheduler import get_param_scheduler
from vime.backends.fsdp_utils.model import build_optimizer
from vime.utils.dp_schedule import build_dp_schedule

NUM_GPUS = 0


def _rollout_data():
    # Sample 1 ends in token id 0, which must survive packing and unpacking.
    tokens = [torch.tensor([5, 6, 7, 8]), torch.tensor([9, 1, 0]), torch.tensor([3, 4, 2, 2, 7, 1])]
    return {
        "tokens": tokens,
        "total_lengths": [4, 3, 6],
        "response_lengths": [2, 2, 3],
        "loss_masks": [torch.ones(2, dtype=torch.int), torch.tensor([1, 0], dtype=torch.int), torch.ones(3)],
        "rollout_log_probs": [torch.randn(2), torch.randn(2), torch.randn(3)],
    }


@pytest.mark.parametrize("pad_to_multiple_of", [1, 8])
def test_pack_round_trip_keeps_trailing_zero_tokens(pad_to_multiple_of):
    data = _rollout_data()
    indices = [2, 0, 1]
    keys = ["tokens", "total_lengths", "response_lengths", "loss_masks", "rollout_log_probs", "values"]
    packed = pack_micro_batch(data, indices, keys, pad_to_multiple_of=pad_to_multiple_of)

    total = sum(len(data["tokens"][i]) for i in indices)
    assert packed.input_ids.shape == packed.position_ids.shape == (1, total + packed.num_pad_tokens)
    assert (total + packed.num_pad_tokens) % pad_to_multiple_of == 0
    assert packed.num_pad_tokens == (-total) % pad_to_multiple_of
    expected_positions = torch.cat([torch.arange(len(data["tokens"][i])) for i in indices])
    torch.testing.assert_close(packed.position_ids[0, :total], expected_positions)
    if packed.num_pad_tokens:
        # The padding tail restarts its positions so it never attends to the samples.
        torch.testing.assert_close(packed.position_ids[0, total:], torch.arange(packed.num_pad_tokens))

    bounds = packed.cu_seqlens.tolist()
    assert bounds[-1] == packed.input_ids.numel()
    if packed.num_pad_tokens:
        bounds = bounds[:-1]
    segments = [packed.input_ids[0, start:end] for start, end in zip(bounds[:-1], bounds[1:], strict=True)]
    for segment, index in zip(segments, indices, strict=True):
        torch.testing.assert_close(segment, data["tokens"][index])
    assert segments[-1][-1].item() == 0

    # Per-sample fields follow the micro-batch order, missing keys stay None.
    assert packed.batch["total_lengths"] == [6, 4, 3]
    assert packed.batch["response_lengths"] == [3, 2, 2]
    assert packed.batch["values"] is None
    for got, index in zip(packed.batch["unconcat_tokens"], indices, strict=True):
        torch.testing.assert_close(got, data["tokens"][index])
    for got, index in zip(packed.batch["rollout_log_probs"], indices, strict=True):
        assert got is data["rollout_log_probs"][index]


def test_restore_sample_order_inverts_the_schedule():
    order = [2, 0, 3, 1]
    values = [f"sample{i}" for i in order]
    assert restore_sample_order(values, order) == ["sample0", "sample1", "sample2", "sample3"]
    with pytest.raises(AssertionError):
        restore_sample_order(["a", "b"], [0, 0])


def _schedule_args(**overrides):
    values = dict(
        micro_batch_size=2,
        use_dynamic_batch_size=False,
        max_tokens_per_gpu=None,
        balance_data=False,
        balance_by_flops=False,
        hidden_size=16,
        num_attention_heads=2,
        num_query_groups=2,
        vocab_size=32,
        ffn_hidden_size=64,
        num_experts=None,
        num_layers=2,
        kv_channels=8,
    )
    values.update(overrides)
    return SimpleNamespace(**values)


@pytest.mark.parametrize(
    "overrides",
    [{}, {"use_dynamic_batch_size": True, "max_tokens_per_gpu": 24}],
    ids=["static", "dynamic"],
)
def test_shared_schedule_is_consumed_once_per_sample(overrides):
    dp_size = 2
    lengths = [5, 9, 3, 12, 7, 4, 8, 6, 10, 2, 11, 5, 4, 9, 3, 6]
    partitions, micro_batch_indices, num_microbatches, global_batch_sizes = build_dp_schedule(
        _schedule_args(**overrides),
        {"dp_size": dp_size, "cp_size": 1, "vpp_size": 1, "microbatch_group_size_per_vp_stage": 1},
        lengths,
        global_batch_size=8,
        rollout_indices=list(range(len(lengths))),
    )
    for rank in range(dp_size):
        rank_data = {
            "micro_batch_indices": micro_batch_indices[rank],
            "num_microbatches": num_microbatches,
            "tokens": [torch.arange(lengths[i]) for i in partitions[rank]],
        }
        steps = list(iter_step_micro_batches(rank_data))
        assert [step_id for step_id, _ in steps] == list(range(len(global_batch_sizes)))
        assert [len(mbs) for _, mbs in steps] == num_microbatches
        order = [i for _, mbs in steps for indices in mbs for i in indices]
        assert sorted(order) == list(range(len(partitions[rank])))
        for _, mbs in steps:
            for indices in mbs:
                packed = pack_micro_batch(rank_data, indices, ["tokens"])
                assert packed.input_ids.numel() == sum(lengths[partitions[rank][i]] for i in indices)


def _megatron_lr(num_samples, *, init_lr, max_lr, min_lr, warmup, decay, style):
    """Reference: Megatron-LM OptimizerParamScheduler.get_lr in sample units."""
    if warmup > 0 and num_samples <= warmup:
        return init_lr + (max_lr - init_lr) * num_samples / warmup
    if style == "constant":
        return max_lr
    if num_samples > decay:
        return min_lr
    ratio = (num_samples - warmup) / (decay - warmup)
    coeff = 1.0 - ratio if style == "linear" else 0.5 * (math.cos(math.pi * ratio) + 1.0)
    return min_lr + coeff * (max_lr - min_lr)


def _schedule_namespace(**overrides):
    values = dict(
        num_rollout=4,
        rollout_batch_size=8,
        n_samples_per_prompt=4,
        global_batch_size=16,
        lr=1e-5,
        min_lr=1e-6,
        lr_warmup_init=0.0,
        lr_decay_iters=None,
        lr_warmup_iters=2,
        lr_warmup_fraction=None,
        lr_wsd_decay_iters=None,
        lr_wsd_decay_style=None,
        lr_decay_style="cosine",
        weight_decay=0.1,
        start_weight_decay=0.1,
        end_weight_decay=0.1,
        weight_decay_incr_style="constant",
        optimizer="adam",
        adam_beta1=0.9,
        adam_beta2=0.98,
        adam_eps=1e-8,
    )
    values.update(overrides)
    return Namespace(**values)


@pytest.mark.parametrize("style", ["cosine", "linear", "constant"])
def test_lr_schedule_counts_samples_like_megatron(style):
    args = _schedule_namespace(lr_decay_style=style)
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(args, model)
    scheduler = get_param_scheduler(args, optimizer)
    assert args.train_iters == 4 * 8 * 4 // 16
    gbs = args.global_batch_size
    expected = dict(init_lr=0.0, max_lr=1e-5, min_lr=1e-6, warmup=2 * gbs, decay=args.train_iters * gbs, style=style)

    seen = []
    for step in range(args.train_iters + 2):
        seen.append(optimizer.param_groups[0]["lr"])
        assert seen[-1] == pytest.approx(_megatron_lr(step * gbs, **expected))
        scheduler.step(increment=gbs)
    if style != "constant":
        assert seen[1] < seen[2] and seen[-1] == pytest.approx(1e-6)

    restored = get_param_scheduler(_schedule_namespace(lr_decay_style=style), build_optimizer(args, model))
    restored.load_state_dict(scheduler.state_dict())
    assert restored.optimizer.param_groups[0]["lr"] == pytest.approx(optimizer.param_groups[0]["lr"])


def test_optimizer_groups_match_megatron_weight_decay_rule():
    args = _schedule_namespace(lr_decay_style="constant")
    model = torch.nn.Sequential(torch.nn.Linear(4, 4), torch.nn.LayerNorm(4))
    optimizer = build_optimizer(args, model)
    get_param_scheduler(args, optimizer)
    decay, no_decay = optimizer.param_groups
    assert [p.shape for p in decay["params"]] == [torch.Size([4, 4])]
    assert len(no_decay["params"]) == 3  # linear bias, norm weight, norm bias
    assert decay["weight_decay"] == pytest.approx(0.1)
    assert no_decay["weight_decay"] == 0.0


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
