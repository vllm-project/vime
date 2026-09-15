import sys
import types
from argparse import Namespace
from dataclasses import dataclass
from enum import Enum
from types import SimpleNamespace

import pytest
import torch

ray_module = types.ModuleType("ray")
ray_module.get = lambda refs: refs
ray_actor_module = types.ModuleType("ray.actor")
ray_actor_module.ActorHandle = object
sys.modules.setdefault("ray", ray_module)
sys.modules.setdefault("ray.actor", ray_actor_module)

megatron_module = types.ModuleType("megatron")
megatron_core_module = types.ModuleType("megatron.core")
megatron_core_module.mpu = types.SimpleNamespace()
megatron_module.core = megatron_core_module
sys.modules.setdefault("megatron", megatron_module)
sys.modules.setdefault("megatron.core", megatron_core_module)

distributed_module = types.ModuleType("vime.utils.distributed_utils")
distributed_module.get_gloo_group = lambda: object()
sys.modules.setdefault("vime.utils.distributed_utils", distributed_module)

iterator_module = types.ModuleType("vime.backends.megatron_utils.update_weight.hf_weight_iterator_direct")


class HfWeightIteratorDirect:
    def __init__(self, **_kwargs):
        pass

    def get_hf_weight_chunks(self, _weights, progress_desc, should_convert_chunk):
        assert progress_desc == "Stage ModelExpress weights"
        chunks = [[("weight", object())]]
        return iter(chunk for index, chunk in enumerate(chunks) if should_convert_chunk(index))


iterator_module.HfWeightIteratorDirect = HfWeightIteratorDirect
sys.modules.setdefault(iterator_module.__name__, iterator_module)


class WeightPayloadFormat(Enum):
    FULL_TENSOR = "FULL_TENSOR"
    FULL_HF_CHECKPOINT = "FULL_HF_CHECKPOINT"
    XOR_DELTA = "XOR_DELTA"


class WeightVersionState(Enum):
    STAGING = "STAGING"
    READY = "READY"


class ObjectStorageType(Enum):
    S3 = "S3"


class ObjectStorageConfig(SimpleNamespace):
    pass


class ModelExpressTrainerConfig(SimpleNamespace):
    pass


@dataclass(frozen=True)
class ObjectStorageSource:
    storage_type: ObjectStorageType
    uri: str


@dataclass(frozen=True)
class WeightVersionRef:
    version_id: str


modelexpress_rl = types.ModuleType("modelexpress_rl")
modelexpress_rl.ModelExpressControlClient = object
modelexpress_rl.ModelExpressTrainerClient = object
modelexpress_rl.ModelExpressTrainerConfig = ModelExpressTrainerConfig
modelexpress_rl.ObjectStorageConfig = ObjectStorageConfig
modelexpress_rl.ObjectStorageSource = ObjectStorageSource
modelexpress_rl.ObjectStorageType = ObjectStorageType
modelexpress_rl.TrainerStagingMode = SimpleNamespace(WRITE_TO_STORAGE="WRITE_TO_STORAGE")
modelexpress_rl.WeightPayloadFormat = WeightPayloadFormat
modelexpress_rl.WeightVersionRef = WeightVersionRef
modelexpress_rl.WeightVersionState = WeightVersionState
sys.modules.setdefault("modelexpress_rl", modelexpress_rl)

from vime.backends.megatron_utils.update_weight import update_weight_from_modelexpress as mx_module
from vime.backends.megatron_utils.update_weight.update_weight_from_modelexpress import UpdateWeightFromModelExpress

pytestmark = pytest.mark.unit


class RemoteMethod:
    def __init__(self, fn):
        self._fn = fn

    def remote(self, *args, **kwargs):
        return self._fn(*args, **kwargs)


class FakeControl:
    def __init__(self):
        self.created = []
        self._next_version = 1
        self.create_error = None
        self.state_updates = []
        self.state_error = None

    def create_weight_version(self, **kwargs):
        self.created.append(kwargs)
        if self.create_error is not None:
            raise RuntimeError(self.create_error)
        version_id = kwargs.get("uid")
        if version_id is None:
            version_id = f"opaque-{self._next_version}"
            self._next_version += 1
        return SimpleNamespace(version_id=version_id)

    def update_weight_version_state(self, version_id, state):
        self.state_updates.append((version_id, state))
        if self.state_error is not None:
            error, self.state_error = self.state_error, None
            raise RuntimeError(error)
        return SimpleNamespace(version_id=version_id, state=state)


class FakeStaged:
    def __init__(self, trainer, version_id, buckets):
        self._trainer = trainer
        self._version_id = version_id
        self._buckets = buckets

    def publish(self):
        self._trainer.publishes.append((self._version_id, self._buckets))


class FakeTrainer:
    def __init__(self):
        self.model_name = "policy"
        self.server_url = "dns:///mx:50051"
        self.baselines = []
        self.stages = []
        self.publishes = []
        self.metrics = {
            "changed_bytes": 25,
            "total_bytes": 100,
            "wire_bytes": 123,
            "stage_delta_time": 7.0,
            "publish_object_storage_time": 8.0,
        }

    def prepare_delta_base(self, *, hf_tensor_iter):
        self.baselines.append(list(hf_tensor_iter))

    def stage_shard(self, *, version, hf_tensor_iter):
        buckets = list(hf_tensor_iter)
        self.stages.append((version.version_id, buckets))
        return FakeStaged(self, version.version_id, buckets)

    def pop_metrics(self):
        metrics, self.metrics = self.metrics, {}
        return metrics


class FakeEngine:
    def __init__(self, events, update_error=None, init_error=None):
        self.events = events
        self.update_error = update_error
        self.init_error = init_error
        self.active = False
        self.init_weight_transfer_engine = RemoteMethod(self._init)
        self.pause_generation = RemoteMethod(lambda: self._event("pause"))
        self.flush_cache = RemoteMethod(lambda: self._event("flush"))
        self.start_weight_update = RemoteMethod(self._start)
        self.update_weights = RemoteMethod(self._update)
        self.finish_weight_update = RemoteMethod(self._finish)
        self.continue_generation = RemoteMethod(lambda: self._event("continue"))

    def _event(self, name):
        self.events.append((name, None))
        return {"ok": True}

    def _init(self, payload):
        self.events.append(("init", payload))
        if self.init_error:
            raise RuntimeError(self.init_error)
        return {"ok": True}

    def _start(self):
        if self.active:
            raise RuntimeError("already active")
        self.active = True
        return self._event("start")

    def _update(self, update_info):
        version_id = update_info["version_id"]
        self._event(f"update:{version_id}")
        if self.update_error:
            self.active = False
            raise RuntimeError(self.update_error)
        return {"ok": True}

    def _finish(self):
        if not self.active:
            raise RuntimeError("not active")
        self.active = False
        return self._event("finish")


def args(**config_overrides):
    modelexpress_config = {
        "model_name": "policy",
        "server_url": "dns:///mx:50051",
        "initial_base_version_id": "base-uid",
        "seed_checkpoint_path": "/models/seed",
        "refit_checkpoint_dir": "/mxdelta/refit",
        "s3_uri_prefix": "s3://weights/run/policy",
        "s3_endpoint_url": "http://minio:9000",
        "s3_region_name": "us-west-2",
        "rpc_timeout_seconds": 321.0,
        "max_transfer_attempts": 4,
    }
    modelexpress_config.update(config_overrides)
    values = dict(
        modelexpress_config=modelexpress_config,
    )
    out = Namespace(**values)
    vars(out)["hf_checkpoint"] = "/models/not-used-by-modelexpress"
    return out


@pytest.fixture(autouse=True)
def patch_runtime(monkeypatch):
    monkeypatch.setattr(mx_module.ray, "get", lambda refs: refs)
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: 0)
    monkeypatch.setattr(mx_module.dist, "get_world_size", lambda: 1)
    monkeypatch.setattr(mx_module.dist, "barrier", lambda group=None: None)
    monkeypatch.setattr(
        mx_module.dist,
        "all_reduce",
        lambda value, op=None, group=None: None,
    )
    monkeypatch.setattr(
        mx_module.dist,
        "broadcast_object_list",
        lambda values, src, group=None: None,
    )
    monkeypatch.setattr(mx_module, "get_gloo_group", lambda: object())


def updater(monkeypatch, control, trainer, **config_overrides):
    monkeypatch.setattr(
        mx_module,
        "ModelExpressControlClient",
        SimpleNamespace(connect=lambda **_kwargs: control),
    )
    monkeypatch.setattr(
        mx_module,
        "ModelExpressTrainerClient",
        SimpleNamespace(initialize=lambda _config: trainer),
    )
    instance = UpdateWeightFromModelExpress(
        args(**config_overrides),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
    )
    return instance


def test_vime_builds_generic_trainer_object_storage_config_and_normalizes_nulls(monkeypatch):
    captured = {}

    class Config(SimpleNamespace):
        pass

    class TrainerClient:
        @staticmethod
        def initialize(config):
            captured["config"] = config
            trainer = FakeTrainer()
            trainer.model_name = config.model_name
            trainer.server_url = config.server_url
            return trainer

    monkeypatch.setattr(mx_module, "ModelExpressTrainerClient", TrainerClient)
    monkeypatch.setattr(mx_module, "ModelExpressTrainerConfig", Config)
    monkeypatch.setattr(
        mx_module,
        "ModelExpressControlClient",
        SimpleNamespace(connect=lambda **_kwargs: FakeControl()),
    )

    UpdateWeightFromModelExpress(
        args(),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
    )

    config = captured["config"]
    assert config.object_storage.storage_type is ObjectStorageType.S3
    assert config.object_storage.uri_prefix == "s3://weights/run/policy"
    assert config.object_storage.seed_checkpoint_path == "/models/seed"
    assert not hasattr(config.object_storage, "process_group")
    assert config.process_group is not None

    UpdateWeightFromModelExpress(
        args(
            s3_uri_prefix=None,
            initial_base_version_id=None,
            seed_checkpoint_path=None,
        ),
        model=[],
        weights_getter=lambda: {},
        model_name="qwen3",
        quantization_config=None,
    )

    object_storage = captured["config"].object_storage
    assert object_storage.uri_prefix == ""
    assert object_storage.initial_base_version_id == ""
    assert object_storage.seed_checkpoint_path == ""


@pytest.mark.parametrize(("rank", "expected"), [(0, [0, 2]), (1, [1, 3])])
def test_vime_assigns_each_hf_bucket_to_one_trainer_rank(monkeypatch, rank, expected):
    instance = updater(monkeypatch, FakeControl(), FakeTrainer())

    class Iterator:
        def get_hf_weight_chunks(self, _weights, progress_desc, should_convert_chunk):
            assert progress_desc == "Stage ModelExpress weights"
            return iter([("weight", index)] for index in range(4) if should_convert_chunk(index))

    instance._weight_iterator = Iterator()
    monkeypatch.setattr(mx_module.dist, "get_rank", lambda: rank)
    monkeypatch.setattr(mx_module.dist, "get_world_size", lambda: 2)

    assert [chunk[0][1] for chunk in instance._iter_hf_buckets()] == expected


def test_vime_initializes_vllm_and_publishes_version_owned_s3_delta(monkeypatch):
    control = FakeControl()
    trainer = FakeTrainer()
    instance = updater(monkeypatch, control, trainer)
    events = []
    instance.connect_rollout_engines([FakeEngine(events)], object())
    base_version = {
        "uid": "base-uid",
        "model_name": "policy",
        "idempotency_key": "vime:s3://weights/run/policy/v0/model.safetensors.index.json",
        "payload_format": WeightPayloadFormat.FULL_TENSOR,
        "object_storage": ObjectStorageSource(
            storage_type=ObjectStorageType.S3,
            uri="s3://weights/run/policy/v0/model.safetensors.index.json",
        ),
        "state": WeightVersionState.READY,
    }
    assert control.created == [base_version]
    assert not trainer.baselines

    instance.update_weights()
    clock = iter([0.0, 5.0])
    reductions = []
    monkeypatch.setattr(mx_module, "perf_counter", lambda: next(clock))

    def reduce_metrics(value, op=None, group=None):
        reductions.append((value.tolist(), op))
        if value.dtype == torch.int64:
            value.copy_(torch.tensor([50, 200, 246], dtype=value.dtype))
        else:
            value.copy_(torch.tensor([17.0, 18.0, 15.0], dtype=value.dtype))

    monkeypatch.setattr(mx_module.dist, "all_reduce", reduce_metrics)
    instance.update_weights()

    assert trainer.baselines
    assert trainer.stages[0][0] == "opaque-1"
    assert trainer.publishes[0][0] == "opaque-1"
    assert control.created == [
        base_version,
        {
            "model_name": "policy",
            "idempotency_key": "vime:s3://weights/run/policy/v1/model.safetensors.index.json",
            "payload_format": WeightPayloadFormat.XOR_DELTA,
            "base_version_id": "base-uid",
            "object_storage": ObjectStorageSource(
                storage_type=ObjectStorageType.S3,
                uri="s3://weights/run/policy/v1/model.safetensors.index.json",
            ),
            "state": WeightVersionState.STAGING,
        },
    ]
    assert control.state_updates == [("opaque-1", WeightVersionState.READY)]
    assert events[0] == (
        "init",
        {
            "init_info": {
                "model_name": "policy",
                "server_url": "dns:///mx:50051",
                "initial_base_version_id": "base-uid",
                "seed_checkpoint_path": "/models/seed",
                "refit_checkpoint_dir": "/mxdelta/refit",
                "object_storage_type": "S3",
                "object_storage_endpoint_url": "http://minio:9000",
                "object_storage_region_name": "us-west-2",
                "registration_ttl_seconds": None,
                "lease_ttl_seconds": None,
                "max_transfer_attempts": 4,
                "rpc_timeout_seconds": 321.0,
            }
        },
    )
    assert [event for event, _payload in events[1:]] == [
        "pause",
        "flush",
        "start",
        "update:opaque-1",
        "finish",
        "continue",
    ]
    assert instance.weight_version == 1
    assert instance._current_version_id == "opaque-1"
    assert instance.pop_metrics() == {
        "perf/update_weights_density": 0.25,
        "perf/update_weights_wire_bytes": 246,
        "perf/mx_stage_delta_time": 17.0,
        "perf/mx_publish_object_storage_time": 18.0,
        "perf/mx_update_engine_weights_time": 15.0,
    }
    assert reductions == [
        ([25, 100, 123], torch.distributed.ReduceOp.SUM),
        (
            [7.0, 8.0, 5.0],
            torch.distributed.ReduceOp.MAX,
        ),
    ]


def test_vime_publishes_periodic_full_hf_checkpoints(monkeypatch):
    control = FakeControl()
    instance = updater(
        monkeypatch,
        control,
        FakeTrainer(),
        full_hf_checkpoint_interval=2,
    )
    instance.connect_rollout_engines([FakeEngine([])], object())

    instance.update_weights()
    for _ in range(3):
        instance.update_weights()

    assert [created["payload_format"] for created in control.created] == [
        WeightPayloadFormat.FULL_TENSOR,
        WeightPayloadFormat.XOR_DELTA,
        WeightPayloadFormat.FULL_HF_CHECKPOINT,
        WeightPayloadFormat.XOR_DELTA,
    ]
    assert control.created[1]["base_version_id"] == "base-uid"
    assert "base_version_id" not in control.created[2]
    assert control.created[3]["base_version_id"] == "opaque-2"


@pytest.mark.parametrize("interval", [0, -1, True, 1.5, "2"])
def test_vime_rejects_invalid_full_hf_checkpoint_intervals(monkeypatch, interval):
    with pytest.raises(ValueError, match="full_hf_checkpoint_interval must be a positive integer"):
        updater(
            monkeypatch,
            FakeControl(),
            FakeTrainer(),
            full_hf_checkpoint_interval=interval,
        )


def test_reconnecting_the_same_vllm_cohort_is_a_noop(monkeypatch):
    instance = updater(monkeypatch, FakeControl(), FakeTrainer())
    events = []
    engine = FakeEngine(events)

    instance.connect_rollout_engines([engine], object())
    instance.connect_rollout_engines([engine], object())

    assert [event for event, _payload in events if event == "init"] == ["init"]


def test_failed_vllm_initialization_is_retried_for_the_same_cohort(monkeypatch):
    instance = updater(monkeypatch, FakeControl(), FakeTrainer())
    events = []
    engine = FakeEngine(events, init_error="init failed")

    with pytest.raises(RuntimeError, match="init failed"):
        instance.connect_rollout_engines([engine], object())

    assert instance.rollout_engines is None
    engine.init_error = None
    instance.connect_rollout_engines([engine], object())

    assert instance.rollout_engines == (engine,)
    assert [event for event, _payload in events if event == "init"] == ["init", "init"]


def test_failed_baseline_registration_is_retried_before_vllm_init(monkeypatch):
    control = FakeControl()
    control.create_error = "catalog unavailable"
    instance = updater(monkeypatch, control, FakeTrainer())
    events = []
    engine = FakeEngine(events)

    with pytest.raises(RuntimeError, match="catalog unavailable"):
        instance.connect_rollout_engines([engine], object())

    assert events == []
    assert instance.rollout_engines is None
    control.create_error = None
    instance.connect_rollout_engines([engine], object())

    assert len(control.created) == 2
    assert control.created[1] == control.created[0]
    assert [event for event, _payload in events] == ["init"]


def test_full_checkpoint_ready_failure_does_not_advance_version(monkeypatch):
    control = FakeControl()
    control.state_error = "ready failed"
    trainer = FakeTrainer()
    instance = updater(
        monkeypatch,
        control,
        trainer,
        full_hf_checkpoint_interval=1,
    )
    engine = FakeEngine([])
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()

    with pytest.raises(RuntimeError, match="ready failed"):
        instance.update_weights()

    assert instance.weight_version == 0
    assert instance._current_version_id == "base-uid"
    assert len(control.created) == 2
    assert len(trainer.publishes) == 1
    assert control.created[1]["payload_format"] is WeightPayloadFormat.FULL_HF_CHECKPOINT
    assert "base_version_id" not in control.created[1]


def test_failed_vllm_update_does_not_resume_or_advance_version(monkeypatch):
    control = FakeControl()
    trainer = FakeTrainer()
    instance = updater(monkeypatch, control, trainer)
    events = []
    engine = FakeEngine(events, update_error="install failed")
    instance.connect_rollout_engines([engine], object())
    instance.update_weights()

    with pytest.raises(RuntimeError, match="install failed"):
        instance.update_weights()

    assert instance.weight_version == 0
    assert instance._current_version_id == "base-uid"
    assert len(control.created) == 2
    assert len(trainer.publishes) == 1
    assert not any(event == "continue" for event, _payload in events[1:])


def test_update_engine_weights_runs_in_bulk_phases(monkeypatch):
    instance = updater(monkeypatch, FakeControl(), FakeTrainer())
    events = []
    first = FakeEngine(events)
    second = FakeEngine(events)
    instance.connect_rollout_engines([first, second], object())
    events.clear()

    instance._update_engine_weights("opaque-target")

    assert [event for event, _payload in events] == [
        "pause",
        "pause",
        "flush",
        "flush",
        "start",
        "start",
        "update:opaque-target",
        "update:opaque-target",
        "finish",
        "finish",
        "continue",
        "continue",
    ]
    assert instance._update_engine_weights_time >= 0
