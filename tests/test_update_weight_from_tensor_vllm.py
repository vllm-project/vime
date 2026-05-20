"""
Unit tests for UpdateVLLMWeightFromTensor.

Design notes
------------
* All tests run without a GPU, CUDA, or any of the heavy training framework
  packages (megatron, vllm, ray, etc.).  Heavy dependencies are injected into
  ``sys.modules`` as lightweight stubs before the module under test is loaded.
* The class under test is instantiated via ``_make_instance()`` which directly
  sets the instance attributes that ``__init__`` would normally compute,
  bypassing ``HfWeightIteratorBase.create()`` and other GPU-requiring calls.
* Remote Ray calls are intercepted by ``RecordingRemoteMethod`` / ``RecordingEngine``
  objects modelled after ``test_update_weight_from_distributed.py``.
* ``IPCWeightTransferEngine.trainer_send_weights`` is replaced by a callable
  stored on ``RecordingIPCEngine`` so we can assert it was called with the right
  arguments without touching any CUDA primitives.
"""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from argparse import Namespace
from dataclasses import dataclass, field
from unittest.mock import MagicMock, call, patch

import pytest
import torch


# ─────────────────────────────────────────────────────────────────────────────
# Stub modules injected before importing the module under test
# ─────────────────────────────────────────────────────────────────────────────

def _install_stubs():
    """
    Inject lightweight stubs for every GPU/training framework package the
    module under test imports at the top level.  This must be called before
    the module is imported by the pytest fixture.
    """
    # ---- megatron ----
    mpu_stub = MagicMock()
    mpu_stub.get_data_parallel_rank.return_value = 0
    mpu_stub.get_tensor_model_parallel_rank.return_value = 0
    mpu_stub.get_pipeline_model_parallel_rank.return_value = 0

    megatron = types.ModuleType("megatron")
    megatron_core = types.ModuleType("megatron.core")
    megatron_core.mpu = mpu_stub
    sys.modules.setdefault("megatron", megatron)
    sys.modules.setdefault("megatron.core", megatron_core)

    # ---- ray ----
    ray_stub = MagicMock()
    ray_stub.get = lambda refs: refs  # pass-through: returns refs unchanged
    ray_stub.actor = types.ModuleType("ray.actor")
    ray_stub.actor.ActorHandle = object

    ray_mod = types.ModuleType("ray")
    ray_mod.get = ray_stub.get
    ray_mod.actor = ray_stub.actor
    sys.modules.setdefault("ray", ray_mod)
    sys.modules.setdefault("ray.actor", ray_mod.actor)

    # ---- torch.distributed (keep real torch, stub dist) ----
    dist_stub = MagicMock()
    dist_stub.get_rank.return_value = 0
    dist_stub.barrier = MagicMock()

    # Patch torch.distributed at the attribute level (don't replace the module)
    import torch.distributed as _dist
    for attr in ("get_rank", "barrier"):
        setattr(_dist, attr, getattr(dist_stub, attr))

    # ---- slime.utils.distributed_utils ----
    slime_utils = types.ModuleType("slime.utils.distributed_utils")
    slime_utils.get_gloo_group = MagicMock(return_value="gloo_group")
    slime_pkg = types.ModuleType("slime")
    slime_utils_pkg = types.ModuleType("slime.utils")
    sys.modules.setdefault("slime", slime_pkg)
    sys.modules.setdefault("slime.utils", slime_utils_pkg)
    sys.modules.setdefault("slime.utils.distributed_utils", slime_utils)

    # ---- slime.backends.megatron_utils.update_weight sub-modules ----
    # HfWeightIteratorBase stub returned from .create()
    hf_iter_stub = MagicMock()
    hf_iter_stub.get_hf_weight_chunks.return_value = iter([])

    hf_base_cls = MagicMock()
    hf_base_cls.create.return_value = hf_iter_stub

    hf_iter_base_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.hf_weight_iterator_base"
    )
    hf_iter_base_mod.HfWeightIteratorBase = hf_base_cls

    upw_dist_mod = types.ModuleType(
        "slime.backends.megatron_utils.update_weight.update_weight_from_distributed"
    )
    upw_dist_mod.connect_rollout_engines_from_distributed = MagicMock(return_value="groups")
    upw_dist_mod.disconnect_rollout_engines_from_distributed = MagicMock()
    upw_dist_mod.post_process_weights = MagicMock()
    upw_dist_mod.update_weights_from_distributed = MagicMock(return_value=[])

    backends = types.ModuleType("slime.backends")
    backends_mega = types.ModuleType("slime.backends.megatron_utils")
    backends_mega_upw = types.ModuleType("slime.backends.megatron_utils.update_weight")

    for key, mod in [
        ("slime.backends", backends),
        ("slime.backends.megatron_utils", backends_mega),
        ("slime.backends.megatron_utils.update_weight", backends_mega_upw),
        ("slime.backends.megatron_utils.update_weight.hf_weight_iterator_base", hf_iter_base_mod),
        ("slime.backends.megatron_utils.update_weight.update_weight_from_distributed", upw_dist_mod),
    ]:
        sys.modules.setdefault(key, mod)

    # ---- vllm.distributed.weight_transfer.ipc_engine ----
    ipc_mod = types.ModuleType("vllm.distributed.weight_transfer.ipc_engine")
    ipc_mod.IPCTrainerSendWeightsArgs = MagicMock
    ipc_mod.IPCWeightTransferEngine = MagicMock()

    vllm_pkg = types.ModuleType("vllm")
    vllm_dist = types.ModuleType("vllm.distributed")
    vllm_dist_wt = types.ModuleType("vllm.distributed.weight_transfer")

    for key, mod in [
        ("vllm", vllm_pkg),
        ("vllm.distributed", vllm_dist),
        ("vllm.distributed.weight_transfer", vllm_dist_wt),
        ("vllm.distributed.weight_transfer.ipc_engine", ipc_mod),
    ]:
        sys.modules.setdefault(key, mod)

    return hf_iter_stub, hf_base_cls, upw_dist_mod, ipc_mod


# Install stubs once at collection time so that the importlib fixture works.
_HF_ITER_STUB, _HF_BASE_CLS, _UPW_DIST_MOD, _IPC_MOD = _install_stubs()


# ─────────────────────────────────────────────────────────────────────────────
# Fixtures
# ─────────────────────────────────────────────────────────────────────────────

MODULE_PATH = "slime.backends.megatron_utils.update_weight.update_weight_from_tensor_vllm"


@pytest.fixture(scope="module")
def upw_vllm():
    """Import the module under test (stubs already installed above)."""
    # Remove any cached version so the fixture gets a fresh load.
    sys.modules.pop(MODULE_PATH, None)
    return importlib.import_module(MODULE_PATH)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers / recording stubs
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class _RemoteCall:
    args: tuple
    kwargs: dict


class RecordingRemoteMethod:
    """Records every .remote(*args, **kwargs) call and returns a fixed ref."""

    def __init__(self, return_value: object = "ref"):
        self._return_value = return_value
        self.calls: list[_RemoteCall] = []

    def remote(self, *args, **kwargs):
        self.calls.append(_RemoteCall(args=args, kwargs=kwargs))
        return self._return_value


@dataclass
class RecordingVLLMEngine:
    """Mimics a colocated vLLM rollout engine actor."""

    pause_generation: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    flush_cache: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    init_weight_transfer_engine: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    start_weight_update: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    finish_weight_update: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    continue_generation: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())


@dataclass
class RecordingDistributedEngine:
    """Mimics a distributed (non-colocated) rollout engine actor."""

    pause_generation: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    flush_cache: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())
    continue_generation: RecordingRemoteMethod = field(default_factory=lambda: RecordingRemoteMethod())


def _default_args(
    *,
    actor_num_nodes: int = 1,
    actor_num_gpus_per_node: int = 4,
    rollout_num_gpus_per_engine: int = 2,
) -> Namespace:
    return Namespace(
        actor_num_nodes=actor_num_nodes,
        actor_num_gpus_per_node=actor_num_gpus_per_node,
        rollout_num_gpus_per_engine=rollout_num_gpus_per_engine,
        megatron_to_hf_mode="raw",
        update_weight_buffer_size=1 << 30,
    )


def _make_instance(upw_vllm, args=None):
    """
    Create an ``UpdateVLLMWeightFromTensor`` without touching any GPU/megatron
    code by bypassing ``__init__`` and setting attributes manually.
    """
    if args is None:
        args = _default_args()

    obj = object.__new__(upw_vllm.UpdateVLLMWeightFromTensor)
    obj.args = args
    obj.model = []
    obj.weights_getter = lambda: {}
    obj.model_name = "test_model"
    obj.quantization_config = None
    obj.weight_version = 0
    obj._hf_weight_iterator = _HF_ITER_STUB
    obj._colocated_engines = []
    obj._distributed_engines = []
    obj._model_update_groups = None
    obj._is_distributed_src_rank = False
    obj._group_name = "slime"
    obj._ipc_initialized = False
    return obj


# ─────────────────────────────────────────────────────────────────────────────
# Signature / structural tests
# ─────────────────────────────────────────────────────────────────────────────

@pytest.mark.unit
def test_class_exists(upw_vllm):
    assert hasattr(upw_vllm, "UpdateVLLMWeightFromTensor")


@pytest.mark.unit
def test_init_signature(upw_vllm):
    sig = inspect.signature(upw_vllm.UpdateVLLMWeightFromTensor.__init__)
    params = sig.parameters
    for expected in ("args", "model", "weights_getter", "model_name", "quantization_config"):
        assert expected in params, f"Expected __init__ param '{expected}' not found"


@pytest.mark.unit
def test_update_weights_signature(upw_vllm):
    sig = inspect.signature(upw_vllm.UpdateVLLMWeightFromTensor.update_weights)
    # Only 'self' — no extra required params
    params = [p for p in sig.parameters if p != "self"]
    assert params == []


@pytest.mark.unit
def test_connect_rollout_engines_signature(upw_vllm):
    sig = inspect.signature(upw_vllm.UpdateVLLMWeightFromTensor.connect_rollout_engines)
    for expected in ("rollout_engines", "rollout_engine_lock", "engine_gpu_counts", "engine_gpu_offsets"):
        assert expected in sig.parameters


# ─────────────────────────────────────────────────────────────────────────────
# connect_rollout_engines: colocated / distributed split
# ─────────────────────────────────────────────────────────────────────────────

def _connect(obj, engines, *, counts, offsets):
    """Helper: call connect_rollout_engines with explicit counts/offsets."""
    lock = MagicMock()
    obj.connect_rollout_engines(engines, lock, engine_gpu_counts=counts, engine_gpu_offsets=offsets)


@pytest.mark.unit
def test_all_colocated_when_all_fit(upw_vllm):
    """Two engines with 2 GPUs each, 4 actor GPUs → both colocated."""
    obj = _make_instance(upw_vllm, _default_args(actor_num_gpus_per_node=4))
    engines = [RecordingVLLMEngine(), RecordingVLLMEngine()]
    _connect(obj, engines, counts=[2, 2], offsets=[0, 2])

    assert obj._colocated_engines == engines
    assert obj._distributed_engines == []


@pytest.mark.unit
def test_all_distributed_when_none_fit(upw_vllm):
    """Engine starts at GPU offset 4 but only 4 actor GPUs → fully distributed."""
    obj = _make_instance(upw_vllm, _default_args(actor_num_gpus_per_node=4))
    engines = [RecordingVLLMEngine(), RecordingVLLMEngine()]
    _connect(obj, engines, counts=[2, 2], offsets=[4, 6])

    assert obj._colocated_engines == []
    assert obj._distributed_engines == engines


@pytest.mark.unit
def test_split_colocated_and_distributed(upw_vllm):
    """First engine fits (offset 0–1), second does not (offset 4–5) → split."""
    obj = _make_instance(upw_vllm, _default_args(actor_num_gpus_per_node=4))
    e0, e1 = RecordingVLLMEngine(), RecordingVLLMEngine()
    _connect(obj, [e0, e1], counts=[2, 2], offsets=[0, 4])

    assert obj._colocated_engines == [e0]
    assert obj._distributed_engines == [e1]


@pytest.mark.unit
def test_default_offsets_dense_packing(upw_vllm):
    """
    When engine_gpu_offsets is None the class must infer dense packing.
    With 4 actor GPUs and rollout_num_gpus_per_engine=2, two engines fit.
    """
    obj = _make_instance(upw_vllm, _default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=2))
    engines = [RecordingVLLMEngine(), RecordingVLLMEngine()]
    lock = MagicMock()
    obj.connect_rollout_engines(engines, lock)  # no explicit offsets/counts

    assert obj._colocated_engines == engines
    assert obj._distributed_engines == []


@pytest.mark.unit
def test_default_counts_from_args(upw_vllm):
    """
    When engine_gpu_counts is None the class reads rollout_num_gpus_per_engine.
    4 actor GPUs, engine needs 4 GPUs (whole node) → 1 exactly fits → colocated.
    """
    obj = _make_instance(upw_vllm, _default_args(actor_num_gpus_per_node=4, rollout_num_gpus_per_engine=4))
    engine = RecordingVLLMEngine()
    lock = MagicMock()
    obj.connect_rollout_engines([engine], lock)

    assert obj._colocated_engines == [engine]
    assert obj._distributed_engines == []


@pytest.mark.unit
def test_nccl_groups_created_for_distributed(upw_vllm):
    """
    connect_rollout_engines must call connect_rollout_engines_from_distributed
    exactly once when there are distributed engines and the rank is src rank.
    """
    obj = _make_instance(upw_vllm)
    obj._is_distributed_src_rank = True  # pretend we are the src rank

    dist_engine = RecordingVLLMEngine()
    _UPW_DIST_MOD.connect_rollout_engines_from_distributed.reset_mock()

    with patch.object(sys.modules["megatron.core"].mpu, "get_data_parallel_rank", return_value=0), \
         patch.object(sys.modules["megatron.core"].mpu, "get_tensor_model_parallel_rank", return_value=0), \
         patch.object(sys.modules["megatron.core"].mpu, "get_pipeline_model_parallel_rank", return_value=0):
        _connect(obj, [dist_engine], counts=[2], offsets=[4])

    _UPW_DIST_MOD.connect_rollout_engines_from_distributed.assert_called_once()


@pytest.mark.unit
def test_no_nccl_for_colocated_only(upw_vllm):
    """No NCCL bridge should be created when all engines are colocated."""
    obj = _make_instance(upw_vllm, _default_args(actor_num_gpus_per_node=4))
    _UPW_DIST_MOD.connect_rollout_engines_from_distributed.reset_mock()

    _connect(obj, [RecordingVLLMEngine(), RecordingVLLMEngine()], counts=[2, 2], offsets=[0, 2])

    _UPW_DIST_MOD.connect_rollout_engines_from_distributed.assert_not_called()


# ─────────────────────────────────────────────────────────────────────────────
# update_weights: lifecycle tests
# ─────────────────────────────────────────────────────────────────────────────

def _run_update_weights(obj, hf_chunks=None):
    """
    Drive ``update_weights`` with stubs for every external call.

    Returns the ``IPCWeightTransferEngine`` mock so callers can assert on it.
    """
    if hf_chunks is None:
        hf_chunks = [_real_tensors(2)]

    ipc_engine_cls = MagicMock()
    ipc_trainer_args_cls = MagicMock(side_effect=lambda **kw: kw)

    # Replace the iterator stub to yield our chunks
    obj._hf_weight_iterator = MagicMock()
    obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter(hf_chunks)

    ipc_mod_patch = {
        "IPCWeightTransferEngine": ipc_engine_cls,
        "IPCTrainerSendWeightsArgs": ipc_trainer_args_cls,
    }

    with patch.dict("sys.modules", {
        "vllm.distributed.weight_transfer.ipc_engine": types.SimpleNamespace(**ipc_mod_patch),
    }):
        # Stub dist.get_rank to return 0 (rank-0 path exercises rank-0 branches)
        with patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.barrier"):
            obj.update_weights()

    return ipc_engine_cls


def _real_tensors(n: int = 2):
    return [(f"layer.{i}.weight", torch.zeros(2, 2)) for i in range(n)]


@pytest.mark.unit
def test_update_weights_calls_pause_and_flush(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]

    _run_update_weights(obj)

    assert len(engine.pause_generation.calls) == 1
    assert len(engine.flush_cache.calls) == 1


@pytest.mark.unit
def test_update_weights_calls_continue_generation(upw_vllm):
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]

    _run_update_weights(obj)

    assert len(engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_ipc_init_called_on_first_update_only(upw_vllm):
    """init_weight_transfer_engine must be called exactly once, not on repeats."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]

    # First call → should initialize
    _run_update_weights(obj)
    assert len(engine.init_weight_transfer_engine.calls) == 1

    # Second call → should NOT re-initialize (flag already set)
    _run_update_weights(obj)
    assert len(engine.init_weight_transfer_engine.calls) == 1  # still 1


@pytest.mark.unit
def test_ipc_initialized_flag_set_after_first_update(upw_vllm):
    obj = _make_instance(upw_vllm)
    obj._colocated_engines = [RecordingVLLMEngine()]
    assert obj._ipc_initialized is False

    _run_update_weights(obj)

    assert obj._ipc_initialized is True


@pytest.mark.unit
def test_start_and_finish_weight_update_called(upw_vllm):
    """vLLM lifecycle: start_weight_update before, finish_weight_update after."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]

    _run_update_weights(obj)

    assert len(engine.start_weight_update.calls) == 1
    assert len(engine.finish_weight_update.calls) == 1

    # start must use is_checkpoint_format=True
    kw = engine.start_weight_update.calls[0].kwargs
    assert kw.get("is_checkpoint_format") is True


@pytest.mark.unit
def test_start_before_finish_order(upw_vllm):
    """
    Verify start_weight_update is called before finish_weight_update by
    recording a shared ordered call log.
    """
    obj = _make_instance(upw_vllm)
    order: list[str] = []

    class OrderedEngine:
        class pause_generation:
            @staticmethod
            def remote():
                return "ref"

        class flush_cache:
            @staticmethod
            def remote():
                return "ref"

        class init_weight_transfer_engine:
            @staticmethod
            def remote(x):
                return "ref"

        class start_weight_update:
            @staticmethod
            def remote(**kw):
                order.append("start")
                return "ref"

        class finish_weight_update:
            @staticmethod
            def remote():
                order.append("finish")
                return "ref"

        class continue_generation:
            @staticmethod
            def remote():
                return "ref"

    obj._colocated_engines = [OrderedEngine()]
    _run_update_weights(obj)

    assert order == ["start", "finish"]


@pytest.mark.unit
def test_trainer_send_weights_called_per_chunk(upw_vllm):
    """trainer_send_weights must be called once per HF weight chunk."""
    obj = _make_instance(upw_vllm)
    obj._colocated_engines = [RecordingVLLMEngine()]
    chunks = [_real_tensors(2), _real_tensors(2), _real_tensors(2)]

    ipc_engine = _run_update_weights(obj, hf_chunks=chunks)

    assert ipc_engine.trainer_send_weights.call_count == len(chunks)


@pytest.mark.unit
def test_trainer_send_weights_uses_ray_mode(upw_vllm):
    """trainer_send_weights must be called with send_mode='ray' args."""
    obj = _make_instance(upw_vllm)
    engine = RecordingVLLMEngine()
    obj._colocated_engines = [engine]

    captured_args: list[dict] = []

    ipc_engine_cls = MagicMock()
    ipc_engine_cls.trainer_send_weights.side_effect = lambda **kw: captured_args.append(kw)

    # We need to capture the IPCTrainerSendWeightsArgs construction call
    send_modes: list[str] = []

    def fake_args_cls(**kw):
        send_modes.append(kw.get("send_mode"))
        return kw

    with patch.dict("sys.modules", {
        "vllm.distributed.weight_transfer.ipc_engine": types.SimpleNamespace(
            IPCWeightTransferEngine=ipc_engine_cls,
            IPCTrainerSendWeightsArgs=fake_args_cls,
        ),
    }):
        with patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.barrier"):
            obj._hf_weight_iterator = MagicMock()
            obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter([_real_tensors(1)])
            obj.update_weights()

    assert send_modes == ["ray"]


@pytest.mark.unit
def test_trainer_send_weights_passes_engine_list(upw_vllm):
    """The llm_handle passed to IPCTrainerSendWeightsArgs must be the engine list."""
    obj = _make_instance(upw_vllm)
    engine_a = RecordingVLLMEngine()
    engine_b = RecordingVLLMEngine()
    obj._colocated_engines = [engine_a, engine_b]

    captured_llm_handles: list = []

    def fake_args_cls(**kw):
        captured_llm_handles.append(kw.get("llm_handle"))
        return kw

    ipc_engine_cls = MagicMock()
    with patch.dict("sys.modules", {
        "vllm.distributed.weight_transfer.ipc_engine": types.SimpleNamespace(
            IPCWeightTransferEngine=ipc_engine_cls,
            IPCTrainerSendWeightsArgs=fake_args_cls,
        ),
    }):
        with patch("torch.distributed.get_rank", return_value=0), \
             patch("torch.distributed.barrier"):
            obj._hf_weight_iterator = MagicMock()
            obj._hf_weight_iterator.get_hf_weight_chunks.return_value = iter([_real_tensors(1)])
            obj.update_weights()

    assert captured_llm_handles == [[engine_a, engine_b]]


@pytest.mark.unit
def test_no_ipc_calls_when_no_colocated_engines(upw_vllm):
    """With only distributed engines, IPC path must not be triggered."""
    obj = _make_instance(upw_vllm)
    obj._colocated_engines = []
    obj._distributed_engines = [RecordingVLLMEngine()]

    ipc_engine = _run_update_weights(obj)

    ipc_engine.trainer_send_weights.assert_not_called()


@pytest.mark.unit
def test_distributed_engines_receive_nccl_update(upw_vllm):
    """
    When there are distributed engines and this is the src rank,
    ``update_weights_from_distributed`` must be called once per chunk.
    """
    obj = _make_instance(upw_vllm)
    obj._colocated_engines = []
    obj._distributed_engines = [RecordingVLLMEngine()]
    obj._is_distributed_src_rank = True
    obj._model_update_groups = "some_groups"
    obj.weight_version = 5

    _UPW_DIST_MOD.update_weights_from_distributed.reset_mock()
    _UPW_DIST_MOD.update_weights_from_distributed.return_value = []

    chunks = [_real_tensors(2), _real_tensors(2)]
    _run_update_weights(obj, hf_chunks=chunks)

    assert _UPW_DIST_MOD.update_weights_from_distributed.call_count == len(chunks)


@pytest.mark.unit
def test_weight_version_increments_on_each_call(upw_vllm):
    obj = _make_instance(upw_vllm)
    obj._colocated_engines = [RecordingVLLMEngine()]

    assert obj.weight_version == 0
    _run_update_weights(obj)
    assert obj.weight_version == 1
    _run_update_weights(obj)
    assert obj.weight_version == 2


@pytest.mark.unit
def test_multiple_colocated_engines_all_get_lifecycle_calls(upw_vllm):
    """All three colocated engines must receive start/finish calls."""
    obj = _make_instance(upw_vllm)
    engines = [RecordingVLLMEngine() for _ in range(3)]
    obj._colocated_engines = engines

    _run_update_weights(obj)

    for e in engines:
        assert len(e.start_weight_update.calls) == 1
        assert len(e.finish_weight_update.calls) == 1
        assert len(e.pause_generation.calls) == 1
        assert len(e.continue_generation.calls) == 1


@pytest.mark.unit
def test_mixed_colocated_and_distributed_engines(upw_vllm):
    """
    With both colocated and distributed engines:
    - colocated engines get IPC lifecycle (start/finish)
    - distributed engines get NCCL via update_weights_from_distributed
    - both get pause/flush/continue
    """
    obj = _make_instance(upw_vllm)
    col_engine = RecordingVLLMEngine()
    dist_engine = RecordingVLLMEngine()
    obj._colocated_engines = [col_engine]
    obj._distributed_engines = [dist_engine]
    obj._is_distributed_src_rank = True
    obj._model_update_groups = "groups"

    _UPW_DIST_MOD.update_weights_from_distributed.reset_mock()
    _UPW_DIST_MOD.update_weights_from_distributed.return_value = []

    ipc_engine = _run_update_weights(obj, hf_chunks=[_real_tensors(2)])

    # Colocated: IPC lifecycle
    assert len(col_engine.start_weight_update.calls) == 1
    assert len(col_engine.finish_weight_update.calls) == 1
    ipc_engine.trainer_send_weights.assert_called_once()

    # Distributed: NCCL
    _UPW_DIST_MOD.update_weights_from_distributed.assert_called_once()

    # Both: pause and continue
    assert len(col_engine.pause_generation.calls) == 1
    assert len(dist_engine.pause_generation.calls) == 1
    assert len(col_engine.continue_generation.calls) == 1
    assert len(dist_engine.continue_generation.calls) == 1


@pytest.mark.unit
def test_quantization_post_process_called(upw_vllm):
    """When quantization_config has compressed-tensors, post_process_weights is called."""
    obj = _make_instance(upw_vllm)
    obj._colocated_engines = [RecordingVLLMEngine()]
    obj.quantization_config = {"quant_method": "compressed-tensors"}

    _UPW_DIST_MOD.post_process_weights.reset_mock()

    _run_update_weights(obj)

    # Should be called twice: once with restore_weights_before_load=True, once False
    calls = _UPW_DIST_MOD.post_process_weights.call_args_list
    assert len(calls) == 2
    assert calls[0].kwargs["restore_weights_before_load"] is True
    assert calls[1].kwargs["restore_weights_before_load"] is False


@pytest.mark.unit
def test_source_uses_ipc_not_gloo_gather(upw_vllm):
    """
    The vLLM IPC implementation must NOT contain sglang-style Gloo gather code
    (no FlattenedTensorBucket / MultiprocessingSerializer / gather_object pattern).
    """
    src = inspect.getsource(upw_vllm.UpdateVLLMWeightFromTensor)
    assert "FlattenedTensorBucket" not in src
    assert "MultiprocessingSerializer" not in src
    assert "ipc_gather_group" not in src


@pytest.mark.unit
def test_source_uses_vllm_ipc_engine(upw_vllm):
    """The implementation must import and use IPCWeightTransferEngine."""
    src = inspect.getsource(upw_vllm.UpdateVLLMWeightFromTensor)
    assert "IPCWeightTransferEngine" in src
    assert "trainer_send_weights" in src


@pytest.mark.unit
def test_source_uses_ray_send_mode(upw_vllm):
    """send_mode='ray' must be used (not 'http') for colocated actor communication."""
    src = inspect.getsource(upw_vllm.UpdateVLLMWeightFromTensor.update_weights)
    assert '"ray"' in src or "'ray'" in src
