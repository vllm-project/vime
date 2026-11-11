import copy
import math

import pytest
import torch

from vime.backends.megatron_utils.dspark.attention import DSparkParallelAttention, DSparkRotaryEmbedding


NUM_GPUS = 0


def _reference_rotary(position_ids, head_dim, rotary_base):
    inv_freq = 1.0 / (
        rotary_base ** (torch.arange(0, head_dim, 2, device=position_ids.device, dtype=torch.float32) / head_dim)
    )
    angles = position_ids.float().unsqueeze(-1) * inv_freq
    angles = torch.cat([angles, angles], dim=-1).unsqueeze(1)
    return angles.cos(), angles.sin()


def test_dspark_rotary_embedding_keeps_long_position_ids_out_of_output_dtype():
    rotary = DSparkRotaryEmbedding(head_dim=8, rotary_base=10000.0)
    position_ids = torch.tensor([[0, 1, 17, 128]], dtype=torch.long)

    cos, sin = rotary(position_ids)

    inv_freq = 1.0 / (10000.0 ** (torch.arange(0, 8, 2, dtype=torch.float32) / 8))
    freqs = torch.einsum("i,bj->bji", inv_freq, position_ids.float())
    emb = torch.cat([freqs, freqs], dim=-1)
    expected_cos = emb.cos().unsqueeze(1)
    expected_sin = emb.sin().unsqueeze(1)

    assert cos.dtype == torch.float32
    assert sin.dtype == torch.float32
    torch.testing.assert_close(cos, expected_cos, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(sin, expected_sin, rtol=1e-5, atol=1e-6)
    assert torch.count_nonzero(sin).item() > 0
    assert math.isclose(float(cos.abs().max()), 1.0, rel_tol=0, abs_tol=1e-7)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
@pytest.mark.parametrize("head_dim,rotary_base", [(64, 10000.0), (128, 500000.0)])
@pytest.mark.parametrize("roundtrip", [False, True])
def test_rotary_remains_fp32_after_parent_dtype_conversion(dtype, head_dim, rotary_base, roundtrip):
    parent = torch.nn.Module()
    parent.rotary = DSparkRotaryEmbedding(head_dim, rotary_base)
    parent.to(dtype=dtype)
    if roundtrip:
        parent.float()  # Widening an already-rounded buffer cannot recover it.
    position_ids = torch.tensor([[0, 1, 128, 1024, 8192, 32768], [32769, 4096, 17, 2, 1, 0]], dtype=torch.long)

    actual = parent.rotary(position_ids)
    expected = _reference_rotary(position_ids, head_dim, rotary_base)

    assert not parent.state_dict()  # No persistent rotary checkpoint keys.
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("autocast_dtype", [torch.float16, torch.bfloat16])
def test_rotary_autocast_preserves_fp32_phases(device, autocast_dtype):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is required for the CUDA autocast regression")
    rotary = DSparkRotaryEmbedding(head_dim=64).to(device=device)
    positions = torch.tensor([[0, 128, 8192, 32768]], dtype=torch.long, device=device)
    expected = _reference_rotary(positions, 64, 10000.0)
    with torch.autocast(device_type=device, dtype=autocast_dtype):
        actual = rotary(positions)
    for value, reference in zip(actual, expected, strict=True):
        assert value.dtype == torch.float32
        torch.testing.assert_close(value, reference, rtol=1e-5, atol=1e-6)


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_attention_matches_fresh_fp32_rotary_forward_and_backward(dtype):
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(31)
        attention = DSparkParallelAttention(
            hidden_size=256, num_attention_heads=4, num_key_value_heads=2, head_dim=64
        ).to(dtype=dtype)
        reference = copy.deepcopy(attention)
        reference.rotary_emb = DSparkRotaryEmbedding(head_dim=64)
        hidden = torch.randn(2, 3, 256, dtype=dtype, requires_grad=True)
        context = torch.randn(2, 5, 256, dtype=dtype, requires_grad=True)
    reference_hidden = hidden.detach().clone().requires_grad_()
    reference_context = context.detach().clone().requires_grad_()
    positions = torch.tensor([[0, 17, 128, 4096, 8192, 16384, 32768, 32769]], dtype=torch.long).expand(2, -1)
    mask = torch.ones((2, 1, 3, 8), dtype=torch.bool)

    actual = attention(hidden, context, positions, mask)
    expected = reference(reference_hidden, reference_context, positions, mask)
    assert actual.dtype == dtype
    torch.testing.assert_close(actual.float(), expected.float(), rtol=1e-5, atol=1e-6)
    actual.float().square().mean().backward()
    expected.float().square().mean().backward()
    for actual_input, expected_input in ((hidden, reference_hidden), (context, reference_context)):
        assert actual_input.grad is not None and expected_input.grad is not None
        torch.testing.assert_close(actual_input.grad.float(), expected_input.grad.float(), rtol=1e-5, atol=1e-6)
    for actual_param, expected_param in zip(attention.parameters(), reference.parameters(), strict=True):
        assert actual_param.grad is not None and expected_param.grad is not None
        torch.testing.assert_close(actual_param.grad.float(), expected_param.grad.float(), rtol=1e-5, atol=1e-6)


def test_dspark_attention_casts_rotary_values_to_query_dtype():
    attention = DSparkParallelAttention(
        hidden_size=256,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=64,
    ).to(dtype=torch.bfloat16)
    hidden_states = torch.randn(2, 3, 256, dtype=torch.bfloat16)
    target_hidden_states = torch.randn(2, 5, 256, dtype=torch.bfloat16)
    position_ids = torch.arange(8, dtype=torch.long).unsqueeze(0).expand(2, -1)
    attention_mask = torch.ones((2, 1, 3, 8), dtype=torch.bool)

    output = attention(
        hidden_states=hidden_states,
        target_hidden_states=target_hidden_states,
        position_ids=position_ids,
        attention_mask=attention_mask,
    )

    assert output.dtype == torch.bfloat16
    assert output.shape == hidden_states.shape
    assert torch.isfinite(output).all()
