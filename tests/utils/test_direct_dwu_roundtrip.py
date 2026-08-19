"""The comprehensive CPU test suite for VIME direct DWU.

It drives the REAL sender (CheckpointDeltaSource) into the REAL receiver
(VimeDeltaNCCLWeightTransferEngine): manifests travel as
``chunk.update_info()`` and weights as ``chunk.wire_tensors()`` in sender
order, and the receiver drains them through its own broadcast sequence into a
fake model, so schema, framing, lifecycle, and value semantics are all
checked against each other in one place. Sender-side refusals and
receiver-side protocol rejections live here too, against the same round-trip
harness, instead of in per-side stub suites.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from dataclasses import dataclass
from pathlib import Path

import pytest
import torch

from vime.backends.megatron_utils.update_weight.checkpoint_delta import CheckpointDeltaSource

NUM_GPUS = 0


@dataclass
class _Patch:
    name: str
    shape: tuple[int, ...]
    dtype: torch.dtype
    values: torch.Tensor
    indices: torch.Tensor | None = None


@pytest.fixture
def receiver_module():
    module_names = [
        "vllm",
        "vllm.distributed",
        "vllm.distributed.weight_transfer",
        "vllm.distributed.weight_transfer.base",
        "vllm.distributed.weight_transfer.nccl_engine",
        "vllm.model_executor",
        "vllm.model_executor.model_loader",
        "vllm.model_executor.model_loader.reload",
    ]
    saved = {name: sys.modules.get(name) for name in module_names}

    class Factory:
        registry = {}

        @classmethod
        def register_engine(cls, name, engine_cls):
            cls.registry[name] = engine_cls

    class WeightTransferUpdateInfo:
        pass

    class NCCLWeightTransferEngine:
        def __init__(self, config, vllm_config, device, model):
            self.config = config
            self.vllm_config = vllm_config
            self.parallel_config = vllm_config.parallel_config
            self.model_config = vllm_config.model_config
            self.device = device
            self.model = model
            self.model_update_group = None

        def update_weights(self, update_info):
            typed = self.update_info_cls(**update_info)
            self.receive_weights(typed)
            if torch.accelerator.is_available():
                torch.accelerator.synchronize()

        def shutdown(self):
            self.model_update_group = None

    vllm = types.ModuleType("vllm")
    vllm.__path__ = []
    distributed = types.ModuleType("vllm.distributed")
    distributed.__path__ = []
    weight_transfer = types.ModuleType("vllm.distributed.weight_transfer")
    weight_transfer.__path__ = []
    weight_transfer.WeightTransferEngineFactory = Factory
    base = types.ModuleType("vllm.distributed.weight_transfer.base")
    base.WeightTransferUpdateInfo = WeightTransferUpdateInfo
    nccl = types.ModuleType("vllm.distributed.weight_transfer.nccl_engine")
    nccl.NCCLWeightTransferEngine = NCCLWeightTransferEngine

    model_executor = types.ModuleType("vllm.model_executor")
    model_executor.__path__ = []
    model_loader = types.ModuleType("vllm.model_executor.model_loader")
    model_loader.__path__ = []
    reload_module = types.ModuleType("vllm.model_executor.model_loader.reload")
    reload_module.events = []
    reload_module.initialize_layerwise_reload = lambda model: reload_module.events.append("initialize")
    reload_module.finalize_layerwise_reload = lambda model, config: reload_module.events.append("finalize")

    sys.modules.update(
        {
            "vllm": vllm,
            "vllm.distributed": distributed,
            "vllm.distributed.weight_transfer": weight_transfer,
            "vllm.distributed.weight_transfer.base": base,
            "vllm.distributed.weight_transfer.nccl_engine": nccl,
            "vllm.model_executor": model_executor,
            "vllm.model_executor.model_loader": model_loader,
            "vllm.model_executor.model_loader.reload": reload_module,
        }
    )

    module_path = Path(__file__).resolve().parents[2] / "vime" / "backends" / "vllm_utils" / "checkpoint_delta.py"
    module_name = "test_vime_direct_dwu_roundtrip_module"
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    module._test_factory = Factory
    module._test_reload = reload_module

    try:
        yield module
    finally:
        sys.modules.pop(module_name, None)
        for name, original in saved.items():
            if original is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = original


class _WireGroup:
    """Fake NCCL group fed with the sender's wire tensors in sender order.

    Assumes what production relies on: vLLM's ``trainer_send_weights(...,
    packed=False)`` decomposes into one broadcast per wire tensor in iteration
    order, which the receiver drains with one matching broadcast each."""

    def __init__(self):
        self.payloads: list[torch.Tensor] = []

    def send_chunk(self, chunk) -> None:
        self.payloads.extend(tensor for _, tensor in chunk.wire_tensors())

    def broadcast(self, destination, *, src, stream):
        assert src == 0
        assert stream == "current-stream"
        if destination.numel() == 0:
            return
        payload = self.payloads.pop(0)
        assert payload.dtype == destination.dtype
        assert payload.numel() == destination.numel()
        destination.copy_(payload)


def _apply_patches(model, patches, *, max_chunk_bytes, validate_unique_indices):
    """Faithful single-copy stand-in for vLLM's checkpoint patch API on an
    unsharded model: dense patches replace the runtime tensor, sparse patches
    scatter absolute values into flat checkpoint positions."""
    assert max_chunk_bytes > 0
    assert validate_unique_indices is False
    applied = set()
    for patch in patches:
        destination = model.runtime[patch.name]
        assert tuple(destination.shape) == tuple(patch.shape)
        values = patch.values.to(patch.dtype)
        if patch.indices is None:
            assert values.numel() == destination.numel()
            destination.copy_(values.reshape(patch.shape))
        else:
            assert patch.indices.dtype == torch.int32
            assert not torch.isnan(values).any()
            destination.reshape(-1)[patch.indices.to(torch.long)] = values
        applied.add(patch.name)
        model.load_calls.append(patch.name)
    return applied


def _make_engine(module, monkeypatch, model):
    monkeypatch.setattr(module, "_checkpoint_patch_api", lambda: (_Patch, _apply_patches))
    monkeypatch.setattr(torch.cuda, "current_stream", lambda: "current-stream")
    engine = object.__new__(module.VimeDeltaNCCLWeightTransferEngine)
    engine.config = types.SimpleNamespace()
    engine.vllm_config = types.SimpleNamespace(
        parallel_config=types.SimpleNamespace(),
        model_config=types.SimpleNamespace(dtype=torch.bfloat16),
    )
    engine.parallel_config = engine.vllm_config.parallel_config
    engine.model_config = engine.vllm_config.model_config
    engine.device = torch.device("cpu")
    engine.model = model
    engine.model_update_group = _WireGroup()
    engine._committed_version = 0
    engine._session_base_version = None
    engine._session_target_version = None
    engine._session_encoding = None
    engine._next_sequence_no = 0
    engine._reload_initialized = False
    engine._final_received = False
    engine._update_failed = False
    return engine


def _bf16(values) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.bfloat16)


def _encode_session(source, buckets, *, base_version, target_version):
    """Run one trainer update and return its data chunks plus final manifest."""
    source.begin_update(base_version=base_version, target_version=target_version)
    chunks = [chunk for bucket in buckets for chunk in source.encode_chunk(bucket)]
    return chunks, source.finish_update()


def _run_session(source, engine, buckets, *, base_version, target_version):
    """Push one full trainer update through the receiver, sender-framed."""
    chunks, final = _encode_session(source, buckets, base_version=base_version, target_version=target_version)
    engine.start_weight_update()
    for chunk in [*chunks, final]:
        engine.model_update_group.send_chunk(chunk)
        engine.update_weights(chunk.update_info())
    engine.finish_weight_update()
    source.commit()
    assert engine.model_update_group.payloads == []
    return chunks


@pytest.mark.unit
def test_seed_sparse_and_noop_sessions_apply_exact_values(receiver_module, monkeypatch):
    model = types.SimpleNamespace(
        runtime={
            "model.a": torch.zeros(2, 2, dtype=torch.bfloat16),
            "model.b": torch.zeros(3, dtype=torch.bfloat16),
        },
        load_calls=[],
    )
    engine = _make_engine(receiver_module, monkeypatch, model)
    source = CheckpointDeltaSource()
    reload_events = receiver_module._test_reload.events

    # Dense seed across two exporter buckets: the rollout starts from
    # different weights and must converge.
    seed = [
        [("model.a", _bf16([[1, 2], [3, 4]]))],
        [("model.b", _bf16([0.0, 20, 30]))],
    ]
    chunks = _run_session(source, engine, seed, base_version=0, target_version=1)
    assert [chunk.encoding for chunk in chunks] == ["dense", "dense"]
    assert engine._committed_version == 1
    assert reload_events == ["initialize", "finalize"]
    for bucket in seed:
        for name, expected in bucket:
            assert torch.equal(model.runtime[name], expected)

    # Sparse update: absolute values at changed positions, including a bitwise
    # 0.0 -> -0.0 flip that a value compare would miss.
    step_two = [
        [("model.a", _bf16([[1, 9], [3, -4]]))],
        [("model.b", _bf16([-0.0, 20, 30]))],
    ]
    dense_calls = len(model.load_calls)
    chunks = _run_session(source, engine, step_two, base_version=1, target_version=2)
    assert [chunk.encoding for chunk in chunks] == ["indices", "indices"]
    assert engine._committed_version == 2
    assert reload_events == ["initialize", "finalize"]  # sparse sessions skip reload
    for bucket in step_two:
        for name, expected in bucket:
            assert torch.equal(model.runtime[name], expected)
    assert torch.signbit(model.runtime["model.b"][0])
    assert len(model.load_calls) > dense_calls
    assert 0 < source.changed_elements < source.total_elements
    assert source.wire_bytes > 0

    # No-op update: nothing changed, only the final manifest travels, the
    # version still advances, and no weights are loaded.
    load_calls = len(model.load_calls)
    chunks = _run_session(source, engine, step_two, base_version=2, target_version=3)
    assert chunks == []
    assert engine._committed_version == 3
    assert len(model.load_calls) == load_calls
    for bucket in step_two:
        for name, expected in bucket:
            assert torch.equal(model.runtime[name], expected)


@pytest.mark.unit
def test_abandoned_session_poisons_receiver_until_restart(receiver_module, monkeypatch):
    model = types.SimpleNamespace(
        runtime={"model.a": torch.zeros(2, dtype=torch.bfloat16)},
        load_calls=[],
    )
    engine = _make_engine(receiver_module, monkeypatch, model)
    source = CheckpointDeltaSource()
    _run_session(source, engine, [[("model.a", _bf16([1, 2]))]], base_version=0, target_version=1)

    # The trainer dies after shipping data but before the final manifest.
    source.begin_update(base_version=1, target_version=2)
    chunks = source.encode_chunk([("model.a", _bf16([1, 5]))])
    assert chunks
    engine.start_weight_update()
    for chunk in chunks:
        engine.model_update_group.send_chunk(chunk)
        engine.update_weights(chunk.update_info())
    source.abort()

    with pytest.raises(RuntimeError, match="without a final manifest"):
        engine.finish_weight_update()
    with pytest.raises(RuntimeError, match="previous direct DWU session failed"):
        engine.start_weight_update()

    # The aborted source keeps its committed baseline: the next update diffs
    # against version 1, not the half-shipped version 2 state.
    source.begin_update(base_version=1, target_version=2)
    retry = source.encode_chunk([("model.a", _bf16([1, 5]))])
    assert len(retry) == 1
    source.abort()


@pytest.mark.unit
def test_source_refuses_invalid_updates():
    nan = float("nan")
    source = CheckpointDeltaSource()
    with pytest.raises(ValueError, match="must start at version 0"):
        source.begin_update(base_version=1, target_version=2)

    source.begin_update(base_version=0, target_version=1)
    source.encode_chunk([("model.a", _bf16([1, nan]))])
    source.encode_chunk([("model.b", _bf16([3, 4]))])
    source.finish_update()
    source.commit()

    # Version guards against the committed snapshot.
    with pytest.raises(RuntimeError, match="snapshot=1, update=0"):
        source.begin_update(base_version=0, target_version=1)
    with pytest.raises(ValueError, match=r"base_version \+ 1"):
        source.begin_update(base_version=1, target_version=3)

    # Inventory drift within a bucket fails instead of silently diffing.
    source.begin_update(base_version=1, target_version=2)
    with pytest.raises(RuntimeError, match="tensor count changed"):
        source.encode_chunk([("model.a", _bf16([1, nan])), ("model.x", _bf16([9.0]))])
    source.abort()

    # A NaN that survives training bit-identically is not a change; a weight
    # flipping TO NaN is refused at the source, naming the tensor, because NaN
    # is the patch API's unchanged-value sentinel.
    source.begin_update(base_version=1, target_version=2)
    assert source.encode_chunk([("model.a", _bf16([1, nan]))]) == []
    with pytest.raises(ValueError, match=r"model\.b: training produced NaN"):
        source.encode_chunk([("model.b", _bf16([3, nan]))])
    source.abort()


@pytest.mark.unit
def test_receiver_rejects_protocol_drift(receiver_module, monkeypatch):
    """Every malformed session poisons the worker instead of corrupting it."""

    def fresh():
        model = types.SimpleNamespace(
            runtime={"model.a": torch.zeros(2, dtype=torch.bfloat16)},
            load_calls=[],
        )
        return _make_engine(receiver_module, monkeypatch, model)

    source = CheckpointDeltaSource()
    chunks, final = _encode_session(source, [[("model.a", _bf16([1, 2]))]], base_version=0, target_version=1)
    data_info = chunks[0].update_info()

    def deliver(engine, chunk, **overrides):
        engine.model_update_group.send_chunk(chunk)
        engine.update_weights({**chunk.update_info(), **overrides})

    # Wrong base version: the wire payload is still drained first so all TP
    # ranks complete the same collectives, then the worker fails stop.
    engine = fresh()
    engine._committed_version = 4
    engine.start_weight_update()
    with pytest.raises(RuntimeError, match="base version mismatch"):
        deliver(engine, chunks[0])
    assert engine.model_update_group.payloads == []
    assert engine._update_failed is True

    # Out-of-order sequence number.
    engine = fresh()
    engine.start_weight_update()
    with pytest.raises(ValueError, match="sequence mismatch"):
        deliver(engine, chunks[0], sequence_no=5)

    # Non-BF16 wire dtype.
    engine = fresh()
    engine.start_weight_update()
    with pytest.raises(ValueError, match="must be BF16"):
        deliver(engine, chunks[0], value_dtype_name="float16")

    # A dense session whose only manifest is the final one shipped no weights.
    engine = fresh()
    engine.start_weight_update()
    with pytest.raises(ValueError, match="did not carry any weights"):
        engine.update_weights({**final.update_info(), "sequence_no": 0})

    # Data arriving after the final manifest.
    engine = fresh()
    engine.start_weight_update()
    deliver(engine, chunks[0])
    engine.update_weights(final.update_info())
    engine.model_update_group.send_chunk(chunks[0])
    with pytest.raises(ValueError, match="after the final manifest"):
        engine.update_weights({**data_info, "sequence_no": 2})

