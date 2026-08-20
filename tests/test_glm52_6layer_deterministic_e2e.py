"""GLM-5.2 deterministic train/rollout alignment gate."""

NUM_GPUS = 8


def run_gate(*, layerwise_zero: bool = False, rollout_max_response_len: int = 4096) -> None:
    del layerwise_zero, rollout_max_response_len
    # vLLM 0.27.1 sparse MLA does not support batch-invariant inference.
    raise RuntimeError("GLM-5.2 deterministic alignment is temporarily unsupported with vLLM 0.27.1")


def test_glm52_6layer_deterministic_train_rollout_alignment():
    run_gate()
