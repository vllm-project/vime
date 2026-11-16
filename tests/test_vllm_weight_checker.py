from types import SimpleNamespace

import pytest

from vime.backends.vllm_utils.vllm_engine import VLLMEngine


def test_vllm_weight_check_uses_server_checksum_reset_and_compare():
    calls = []

    def request(endpoint, payload):
        calls.append((endpoint, payload))
        return {"checksums": {"rank0.weight": "abc"}, "match": True}

    engine = SimpleNamespace(node_rank=0, _make_request=request)
    for action in ("snapshot", "reset_tensors", "compare"):
        VLLMEngine.check_weights(engine, action)

    assert calls == [
        ("weight_checker", {"action": "checksum"}),
        ("weight_checker", {"action": "reset"}),
        ("weight_checker", {"action": "compare", "baseline": {"rank0.weight": "abc"}}),
    ]


def test_vllm_weight_check_rejects_mismatch():
    def request(endpoint, payload):
        return {"checksums": {}, "match": False, "mismatches": ["rank0.weight"]}

    engine = SimpleNamespace(node_rank=0, _make_request=request)
    VLLMEngine.check_weights(engine, "snapshot")
    with pytest.raises(AssertionError, match="rank0.weight"):
        VLLMEngine.check_weights(engine, "compare")
