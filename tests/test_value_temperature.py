import asyncio
import sys
import types
from argparse import Namespace

import pytest
import torch


NUM_GPUS = 0


def test_opd_teacher_uses_rollout_temperature(monkeypatch):
    from vime.rollout import on_policy_distillation

    captured = {}

    async def fake_post(url, payload):
        captured.update(url=url, payload=payload)
        return {"prompt_logprobs": []}

    monkeypatch.setattr(on_policy_distillation, "post", fake_post)
    args = Namespace(
        rm_url="http://teacher:8000/inference/v1/generate",
        rollout_temperature=0.7,
    )
    sample = Namespace(tokens=[1, 2], multimodal_inputs=None)

    asyncio.run(on_policy_distillation.reward_func(args, sample))

    assert captured["payload"]["sampling_params"]["temperature"] == 0.7


def test_get_values_does_not_apply_rollout_temperature(monkeypatch):
    previous_loss = sys.modules.pop("vime.backends.megatron_utils.loss", None)
    previous_cp_utils = sys.modules.pop("vime.backends.megatron_utils.cp_utils", None)

    mpu_stub = types.SimpleNamespace(
        get_context_parallel_world_size=lambda: 1,
        get_context_parallel_rank=lambda: 0,
    )
    megatron_mod = types.ModuleType("megatron")
    core_mod = types.ModuleType("megatron.core")
    core_mod.mpu = mpu_stub
    monkeypatch.setitem(sys.modules, "megatron", megatron_mod)
    monkeypatch.setitem(sys.modules, "megatron.core", core_mod)

    try:
        from vime.backends.megatron_utils.loss import get_values

        args = Namespace(rollout_temperature=0.5, allgather_cp=False)
        logits = torch.tensor([[[1.0], [2.0], [3.0], [4.0]]], dtype=torch.float32)
        tokens = [torch.tensor([10, 11, 12, 13], dtype=torch.long)]

        _, result = get_values(
            logits,
            args=args,
            unconcat_tokens=tokens,
            total_lengths=[4],
            response_lengths=[2],
        )

        torch.testing.assert_close(result["values"][0], torch.tensor([2.0, 3.0]))
    finally:
        if previous_loss is None:
            sys.modules.pop("vime.backends.megatron_utils.loss", None)
        else:
            sys.modules["vime.backends.megatron_utils.loss"] = previous_loss
        if previous_cp_utils is None:
            sys.modules.pop("vime.backends.megatron_utils.cp_utils", None)
        else:
            sys.modules["vime.backends.megatron_utils.cp_utils"] = previous_cp_utils


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
