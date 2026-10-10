"""GLM-5.2 deterministic train/rollout alignment gate."""

import math
import re

NUM_GPUS = 8


def _assert_alignment_result(output: str, rollout_path: str, threshold: float) -> None:
    import torch

    values = [float(value) for value in re.findall(r"'train/train_rollout_logprob_abs_diff':\s*([^,}\s]+)", output)]
    assert values, "Training did not report the train/rollout alignment metric"
    assert all(math.isfinite(value) and 0 <= value <= threshold for value in values), values
    samples = torch.load(rollout_path, map_location="cpu", weights_only=False)["samples"]
    assert len(samples) == 8, f"Expected 8 rollout samples, got {len(samples)}"
    for sample in samples:
        length = sample["response_length"]
        assert length > 0, "Alignment requires generated response tokens"
        log_probs = sample["rollout_log_probs"]
        assert len(log_probs) == length and all(math.isfinite(value) for value in log_probs)
        mask = sample.get("loss_mask")
        assert mask is None or (len(mask) == length and sum(mask) > 0), "Response is entirely masked"
        assert sample["weight_versions"] and set(map(str, sample["weight_versions"])) == {
            "1"
        }, f"Rollout did not use the initial synchronized weights: {sample['weight_versions']}"


def run_gate(*, layerwise_zero: bool = False, rollout_max_response_len: int = 4096) -> None:
    del layerwise_zero, rollout_max_response_len
    # vLLM sparse MLA does not support batch-invariant inference.
    raise RuntimeError("GLM-5.2 deterministic alignment is temporarily unsupported with vLLM")


def test_glm52_6layer_deterministic_train_rollout_alignment():
    run_gate()
