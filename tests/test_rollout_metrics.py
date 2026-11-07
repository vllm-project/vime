import base64
from argparse import Namespace

import numpy as np
import pytest
import torch

from vime.observability.rollout_metrics import _compute_spec_metrics, _compute_top_p_kept_vocab_metrics
from vime.observability.spec_tail_metrics import compute_spec_length_tier_metrics, compute_tail_wall_metrics
from vime.utils.misc import decode_int32_meta_array
from vime.utils.types import Sample

NUM_GPUS = 0


def _make_args():
    return Namespace(vllm_speculative_config=None, num_layers=2, moe_router_topk=2)


@pytest.mark.unit
def test_spec_metrics_use_vllm_speculative_config():
    sample = Sample()
    sample.spec_info.spec_accept_token_num = 6
    sample.spec_info.spec_draft_token_num = 8
    sample.spec_info.spec_verify_ct = 2
    args = Namespace(vllm_speculative_config={"method": "mtp", "num_speculative_tokens": 4})
    metrics = _compute_spec_metrics(args, [sample])
    assert metrics["spec_accept_rate"] == sample.spec_info.spec_accept_rate
    assert metrics["spec_accept_length"] == sample.spec_info.spec_accept_length
    assert _compute_spec_metrics(_make_args(), [sample]) == {}


@pytest.mark.unit
def test_top_p_kept_vocab_metric_uses_loss_mask():
    samples = [
        Sample(
            response_length=4,
            loss_mask=torch.tensor([1, 0, 1, 0], dtype=torch.int32),
            rollout_top_p_token_offsets=torch.tensor([0, 3, 8, 10, 20], dtype=torch.int32),
        ),
        Sample(
            response_length=2,
            loss_mask=None,
            rollout_top_p_token_offsets=torch.tensor([0, 4, 9], dtype=torch.int32),
        ),
    ]

    metrics = _compute_top_p_kept_vocab_metrics(samples)

    assert metrics["top_p_kept_vocab_per_token"] == pytest.approx(3.5)


@pytest.mark.unit
def test_top_p_kept_vocab_metric_skips_removed_samples():
    samples = [
        Sample(
            response_length=3,
            loss_mask=[1, 1, 1],
            remove_sample=True,
            rollout_top_p_token_offsets=torch.tensor([0, 2, 4, 6], dtype=torch.int32),
        )
    ]

    assert _compute_top_p_kept_vocab_metrics(samples) == {}


def _b64_int32(values: list[int]) -> str:
    return base64.b64encode(np.array(values, dtype=np.int32).tobytes()).decode("ascii")


@pytest.mark.unit
def test_decode_int32_meta_array_decodes_base64_to_tensor():
    decoded = decode_int32_meta_array({"routed_experts": _b64_int32([1, 2, 3])}, "routed_experts")

    assert torch.is_tensor(decoded)
    assert decoded.dtype == torch.int32
    torch.testing.assert_close(decoded, torch.tensor([1, 2, 3], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_merges_top_p_tensors():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.tokens == [0, 1, 10, 20]
    assert sample.response_length == 3
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_can_skip_terminal_status_for_streaming_chunks():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.3],
        rollout_top_p_token_ids=torch.tensor([1], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 1], dtype=torch.int32),
    )

    sample.append_response_tokens(
        _make_args(),
        tokens=[10, 20],
        log_probs=[-0.1, -0.2],
        trainable=True,
        meta_info={
            "top_p_token_ids": _b64_int32([10, 11, 20]),
            "top_p_token_offsets": _b64_int32([0, 2, 3]),
            "finish_reason": {"type": "stop"},
        },
        update_terminal_info=False,
    )

    assert sample.status is Sample.Status.PENDING
    assert sample.loss_mask == [1, 1, 1]
    assert sample.rollout_log_probs == [-0.3, -0.1, -0.2]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([1, 10, 11, 20], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 1, 3, 4], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_decodes_routed_experts():
    sample = Sample(tokens=[101, 102, 103])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts.shape == (2, 2, 2)
    torch.testing.assert_close(
        sample.rollout_routed_experts,
        torch.tensor([[[0, 1], [2, 3]], [[4, 5], [6, 7]]], dtype=torch.int32),
    )


@pytest.mark.unit
def test_append_response_tokens_ignores_split_pd_routed_experts():
    sample = Sample(tokens=[101, 102, 103, 104])

    sample.append_response_tokens(
        _make_args(),
        tokens=[],
        trainable=True,
        meta_info={
            "pd_prefill_routed_experts": _b64_int32([0, 1, 2, 3, 4, 5, 6, 7]),
            "pd_decode_routed_experts": _b64_int32([8, 9, 10, 11]),
            "finish_reason": {"type": "stop"},
        },
    )

    assert sample.rollout_routed_experts is None


@pytest.mark.unit
def test_append_response_tokens_rejects_mismatched_routed_experts_shape():
    sample = Sample(tokens=[101, 102, 103])

    with pytest.raises(ValueError, match="routed_experts element count"):
        sample.append_response_tokens(
            _make_args(),
            tokens=[],
            trainable=True,
            meta_info={
                "routed_experts": _b64_int32([0, 1, 2, 3]),
                "finish_reason": {"type": "stop"},
            },
        )


@pytest.mark.unit
def test_append_response_tokens_pads_top_p_for_non_trainable_tokens():
    sample = Sample(
        tokens=[0, 1],
        response_length=1,
        loss_mask=[1],
        rollout_log_probs=[-0.1],
        rollout_top_p_token_ids=torch.tensor([10, 11], dtype=torch.int32),
        rollout_top_p_token_offsets=torch.tensor([0, 2], dtype=torch.int32),
    )

    sample.append_response_tokens(tokens=[200, 201, 202], trainable=False)

    assert sample.tokens == [0, 1, 200, 201, 202]
    assert sample.response_length == 4
    assert sample.loss_mask == [1, 0, 0, 0]
    assert sample.rollout_log_probs == [-0.1, 0.0, 0.0, 0.0]
    torch.testing.assert_close(sample.rollout_top_p_token_ids, torch.tensor([10, 11], dtype=torch.int32))
    torch.testing.assert_close(sample.rollout_top_p_token_offsets, torch.tensor([0, 2, 2, 2, 2], dtype=torch.int32))


@pytest.mark.unit
def test_append_response_tokens_requires_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="trainable response tokens require rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], trainable=True)


@pytest.mark.unit
def test_append_response_tokens_rejects_non_trainable_log_probs():
    sample = Sample()

    with pytest.raises(ValueError, match="non-trainable response tokens should not pass rollout log probabilities"):
        sample.append_response_tokens(tokens=[10], log_probs=[-0.1], trainable=False)


def _spec_sample(completion_tokens, verify_steps):
    sample = Sample()
    sample.spec_info.completion_token_num = completion_tokens
    sample.spec_info.spec_verify_ct = verify_steps
    return sample


@pytest.mark.unit
def test_spec_length_tiers_are_token_weighted_and_skip_unverified_samples():
    # 99 short samples accept 2 tokens per verify step; one long sample accepts 4.
    samples = [_spec_sample(10, 5) for _ in range(99)] + [_spec_sample(1000, 250)]
    # Aborted before decoding: no verify steps, must not drag any tier towards 0.
    samples.append(_spec_sample(0, 0))

    metrics = compute_spec_length_tier_metrics(samples)

    assert metrics["spec_accept_length/token_weighted"] == pytest.approx((99 * 10 + 1000) / (99 * 5 + 250))
    assert metrics["spec_accept_length/len_p0_p50"] == pytest.approx(2.0)
    assert metrics["spec_accept_length/len_p99_p100"] == pytest.approx(4.0)
    assert compute_spec_length_tier_metrics([_spec_sample(0, 0)]) == {}


@pytest.mark.unit
def test_spec_metrics_include_length_tiers_only_with_speculation():
    sample = _spec_sample(12, 3)
    args = Namespace(vllm_speculative_config={"method": "dspark", "num_speculative_tokens": 4})
    assert _compute_spec_metrics(args, [sample])["spec_accept_length/token_weighted"] == pytest.approx(4.0)
    assert _compute_spec_metrics(_make_args(), [sample]) == {}


def _traced_sample(*spans):
    sample = Sample()
    sample.trace = {"events": []}
    for start, end in spans:
        sample.trace["events"].append({"type": "span_start", "name": "vllm_generate", "ts": start})
        sample.trace["events"].append({"type": "span_end", "name": "vllm_generate", "ts": end})
    # Other spans (reward, tool calls) after the last generate do not count.
    sample.trace["events"].append({"type": "span_end", "name": "reward", "ts": 10_000.0})
    return sample


@pytest.mark.unit
def test_tail_wall_share_measures_time_after_ninety_percent_finished():
    samples = [_traced_sample((100.0, 100.0 + i)) for i in range(1, 10)]
    # Multi-turn straggler: its last generate ends at t=200.
    samples.append(_traced_sample((100.0, 105.0), (150.0, 200.0)))
    samples.append(Sample())  # untraced samples are ignored

    metrics = compute_tail_wall_metrics(samples)

    # 90% (9 of 10) finished by t=109; the last one finished at t=200.
    assert metrics["tail/generate_wall_time"] == pytest.approx(100.0)
    assert metrics["tail/last_10pct_wall_share"] == pytest.approx((200.0 - 109.0) / 100.0)


@pytest.mark.unit
def test_tail_wall_share_ignores_spans_from_an_earlier_partial_rollout_step():
    samples = [_traced_sample((100.0, 100.0 + i)) for i in range(1, 11)]
    # Aborted in the previous step at t=10..40 and resumed in this one.
    samples.append(_traced_sample((10.0, 40.0), (100.0, 150.0)))

    metrics = compute_tail_wall_metrics(samples, rollout_time=55.0)

    # Without the window, start would be t=10 and the wall time 140 s.
    assert metrics["tail/generate_wall_time"] == pytest.approx(50.0)
    assert metrics["tail/last_10pct_wall_share"] == pytest.approx((150.0 - 110.0) / 50.0)


@pytest.mark.unit
def test_tail_wall_share_needs_traced_samples():
    assert compute_tail_wall_metrics([Sample(), Sample()]) == {}
    assert compute_tail_wall_metrics([_traced_sample((0.0, 1.0))]) == {}


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
