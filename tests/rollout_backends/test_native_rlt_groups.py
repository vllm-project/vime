"""Check the native adapter's prompt grouping before reward normalization."""

import copy
from argparse import Namespace
from types import SimpleNamespace

import pytest

from vime.rollout import vllm_rlt_rollout
from vime.utils.reward_normalization import normalize_rewards
from vime.utils.types import Sample


def setup_rollout(monkeypatch, corrupt=None):
    groups = [
        [Sample(group_index=group, index=group * 2 + i, prompt=str(group), label="2") for i in range(2)]
        for group in range(2)
    ]

    def generate(samples, rollout_id):
        generated = copy.deepcopy(samples)
        for sample in generated:
            sample.tokens.append(2 if sample.index % 2 else 3)
            sample.response_length = 1
        if corrupt == "reorder":
            generated[0], generated[1] = generated[1], generated[0]
        elif corrupt == "regroup":
            generated[2].group_index = 0
        elif corrupt == "missing":
            generated.pop()
        return generated

    tokenizer = SimpleNamespace(
        encode=lambda prompt, **kwargs: [int(prompt) + 1],
        decode=lambda tokens, **kwargs: r"\boxed{" + str(tokens[-1]) + "}",
    )
    monkeypatch.setattr(vllm_rlt_rollout, "_tokenizer", lambda checkpoint: tokenizer)
    monkeypatch.setattr(vllm_rlt_rollout.ray, "get", lambda value: value)
    args = Namespace(
        rollout_batch_size=2,
        n_samples_per_prompt=2,
        hf_checkpoint="fixture",
        apply_chat_template=False,
        rm_type="math",
        custom_rm_path=None,
        rlt_engine=SimpleNamespace(generate=SimpleNamespace(remote=generate)),
        advantage_estimator="grpo",
        rewards_normalization=True,
        grpo_std_normalization=True,
    )
    return args, groups


def test_native_groups_and_actual_math_rewards(monkeypatch):
    args, groups = setup_rollout(monkeypatch)
    output = vllm_rlt_rollout.generate_rollout(args, 0, SimpleNamespace(get_samples=lambda count: groups))
    assert [[sample.index for sample in group] for group in output.samples] == [[0, 1], [2, 3]]
    assert [[sample.reward for sample in group] for group in output.samples] == [[0, 1], [0, 1]]
    rewards = normalize_rewards(args, [sample.reward for group in output.samples for sample in group])
    assert rewards[0] == rewards[2] < 0 and rewards[1] == rewards[3] > 0
    assert rewards[0] == -rewards[1]


@pytest.mark.parametrize("corrupt", ["reorder", "regroup", "missing"])
def test_native_rejects_corrupt_groups_before_rewards(monkeypatch, corrupt):
    args, groups = setup_rollout(monkeypatch, corrupt)
    monkeypatch.setattr(vllm_rlt_rollout, "async_rm", lambda *args: pytest.fail("Reward must not run"))
    with pytest.raises(ValueError, match="prompt group and sample index"):
        vllm_rlt_rollout.generate_rollout(args, 0, SimpleNamespace(get_samples=lambda count: groups))


def test_native_rejects_incomplete_input_group(monkeypatch):
    args, groups = setup_rollout(monkeypatch)
    groups[1].pop()
    with pytest.raises(ValueError, match="complete prompt groups"):
        vllm_rlt_rollout.generate_rollout(args, 0, SimpleNamespace(get_samples=lambda count: groups))
