"""Exercise the adapter against real tiny CPU engines; GPU E2E is separate."""

import asyncio
from argparse import Namespace
from dataclasses import asdict
from uuid import uuid4

import pytest
import torch
from safetensors.torch import save_file

pytest.importorskip("vllm_rlt")
from vllm_rlt import LLM, CacheConfig
from vllm_rlt.models.ouro import OuroConfig, OuroForCausalLM

from vime.backends.vllm_rlt_utils.engine import NativeEngine
from vime.utils.types import Sample


@pytest.mark.parametrize("backend", ["vllm", "vllm-rlt"])
def test_http_client_concurrency_matches_backend(backend, monkeypatch):
    from vime.utils import http_utils

    args = Namespace(rollout_num_gpus=2, rollout_num_gpus_per_engine=1, use_distributed_post=False)
    if backend == "vllm-rlt":
        args.rlt_max_num_seqs = 3
    else:
        args.vllm_server_concurrency = 3
    monkeypatch.setattr(http_utils, "_http_client", None)
    monkeypatch.setattr(http_utils, "_client_concurrency", 0)
    if backend == "vllm-rlt":
        http_utils.init_http_client(args, backend=backend)
    else:
        http_utils.init_http_client(args)
    assert http_utils._client_concurrency == 6
    asyncio.run(http_utils._http_client.aclose())


@pytest.fixture
def engine():
    torch.manual_seed(42)
    model = OuroForCausalLM(
        OuroConfig(
            vocab_size=11,
            hidden_size=8,
            intermediate_size=16,
            num_hidden_layers=1,
            num_attention_heads=2,
            num_key_value_heads=1,
            head_dim=4,
            max_position_embeddings=32,
            eos_token_id=None,
        )
    )
    adapter = NativeEngine()
    adapter.llm = LLM(model, cache_config=CacheConfig(num_blocks=8))
    adapter.args = Namespace(
        rlt_depth=4,
        seed=42,
        rollout_temperature=0.8,
        rollout_max_response_len=2,
        rlt_model_family="ouro",
        rlt_model_revision="tiny-cpu",
        rlt_engine_revision="test-public-contract",
    )
    adapter.epoch = uuid4().hex
    adapter.paused, adapter.ready = True, False
    adapter.committed_digest = None
    yield adapter
    adapter.close()


def publish(engine, path, version, *, omit_head=False):
    path.mkdir(exist_ok=True)
    state = {name: tensor.detach().clone() for name, tensor in engine.llm.engine.model.named_parameters()}
    if omit_head:
        state.pop("lm_head.weight")
    save_file(state, str(path / "model.safetensors"))
    return engine.update_weights_from_disk(str(path), str(version))


def test_publication_cohort_and_serialization(engine, tmp_path):
    with pytest.raises(RuntimeError, match="Publish"):
        engine.generate([Sample(tokens=[1, 2], index=0)], 0)
    path = tmp_path / "weights"
    assert publish(engine, path, 1) == 1
    assert engine.update_weights_from_disk(str(path), "1") == 1
    engine.continue_generation()
    samples = engine.generate([Sample(tokens=[1, 2], index=i, group_index=0) for i in range(2)], 0)
    assert len({sample.recurrent_trace.runtime_epoch for sample in samples}) == 1
    for sample in samples:
        assert sample.weight_versions == ["1"]
        assert sample.status == Sample.Status.TRUNCATED
        assert sample.recurrent_trace.prefill_depth == 4
        assert sample.recurrent_trace.decode_depths == [4, 4]
        assert len(sample.rollout_log_probs) == sample.response_length == 2
        assert all(probability <= 0 for probability in sample.rollout_log_probs)
        restored = Sample.from_dict(sample.to_dict())
        assert asdict(restored.recurrent_trace) == asdict(sample.recurrent_trace)
    engine.pause_generation()
    engine.flush_cache()
    state = {name: tensor.detach().clone() for name, tensor in engine.llm.engine.model.named_parameters()}
    state["lm_head.weight"][0, 0] += 0.25
    save_file(state, str(path / "model.safetensors"))
    with pytest.raises(ValueError, match="Conflicting"):
        engine.update_weights_from_disk(str(path), "1")


def test_incomplete_publication_cannot_resume_or_reuse_old_manifest(engine, tmp_path):
    complete, incomplete = tmp_path / "complete", tmp_path / "incomplete"
    assert publish(engine, complete, 1) == 1
    with pytest.raises(ValueError, match="every physical"):
        publish(engine, incomplete, 2, omit_head=True)
    assert engine.get_weight_version() == 1
    with pytest.raises(RuntimeError, match="No complete"):
        engine.continue_generation()
    with pytest.raises(ValueError, match="Conflicting"):
        engine.update_weights_from_disk(str(complete), "1")
    assert publish(engine, complete, 2) == 2
    engine.continue_generation()
    assert engine.generate([Sample(tokens=[1, 2], index=0)], 1)[0].weight_versions == ["2"]
