"""Initial weight-check orchestration with local RPC fixtures and the real manager."""

import copy
import runpy
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import _unit_stubs
import pytest

_unit_stubs.install_rollout_optional_stubs()

from vime.ray import rollout

NUM_GPUS = 0
pytestmark = pytest.mark.unit
KEY = "dp0:pp0:pcp0:tp0:model.weight"


class Reply:
    def __init__(self, value=None, error=None, ready=True):
        self.value = value
        self.error = error
        self.ready = ready


class Engine:
    def __init__(self, name, events):
        self.name = name
        self.events = events
        self._actor_id = SimpleNamespace(hex=lambda: name)
        self.values = {KEY: "a" * 64}
        self.worker_ids = {"dp0:pp0:pcp0:tp0": "1" * 32}
        self.info = {"url": f"http://{name}:8000", "generation": 1, "rank_prefixes": ["dp0:pp0:pcp0:tp0"]}
        self.paused = False
        self.errors = {}
        self.responses = {}
        self.timeout = None
        self.pause_generation = SimpleNamespace(remote=lambda: self.request("pause"))
        self.continue_generation = SimpleNamespace(remote=lambda: self.request("resume"))
        self.get_weight_check_info = SimpleNamespace(remote=lambda: self.request("info"))
        self.check_weights = SimpleNamespace(remote=lambda action, baseline=None: self.request(action, baseline))

    def request(self, action, baseline=None):
        self.events.append((self.name, action))
        if action in self.errors:
            return Reply(error=self.errors[action])
        if action == self.timeout:
            return Reply(ready=False)
        if action in self.responses:
            return Reply(self.responses[action])
        if action == "pause":
            self.paused = True
            value = None
        elif action == "resume":
            self.paused = False
            value = None
        elif action == "info":
            assert self.paused
            value = copy.deepcopy(self.info)
        elif action == "checksum":
            value = {"checksums": dict(self.values), "worker_ids": dict(self.worker_ids)}
        elif action == "reset":
            self.values = {key: "0" * 64 for key in self.values}
            value = {"status": "reset"}
        else:
            mismatches = sorted(
                key for key in self.values.keys() | baseline.keys() if self.values.get(key) != baseline.get(key)
            )
            value = {"match": not mismatches, "mismatches": mismatches, "worker_ids": dict(self.worker_ids)}
        return Reply(value)


@pytest.fixture
def manager(monkeypatch):
    def get(reply, timeout=None):
        if isinstance(reply, Reply):
            if reply.error is not None:
                raise reply.error
            return reply.value
        return reply

    monkeypatch.setattr(rollout.ray, "get", get)
    monkeypatch.setattr(
        rollout.ray,
        "wait",
        lambda refs, **kwargs: ([ref for ref in refs if ref.ready], [ref for ref in refs if not ref.ready]),
    )
    cls = rollout.RolloutManager.__ray_metadata__.modified_class
    result = object.__new__(cls)
    result.events = []
    result.args = SimpleNamespace(debug_train_only=False, debug_rollout_only=False, weight_checker_timeout=17.0)
    result._weight_check_state = "idle"
    result._weight_check_records = []
    result._recovery_admission_was_paused = None
    result.producer = SimpleNamespace(paused=False)

    def pause(*, drain=False):
        previous, result.producer.paused = result.producer.paused, True
        return previous

    result.producer.pause = pause
    result.producer.resume = lambda: setattr(result.producer, "paused", False)
    result.data_source = SimpleNamespace(consumers={"fully_async": result.producer})
    result.health_monitoring_pause = lambda: result.events.append(("health", "pause"))
    result.health_monitoring_resume = lambda: result.events.append(("health", "resume"))
    result.engines = [Engine("actor0", result.events), Engine("actor1", result.events)]
    result.frozen = Engine("ref", result.events)
    result.servers = {
        "ref": SimpleNamespace(update_weights=False, engines=[result.frozen]),
        "actor": SimpleNamespace(update_weights=True, engines=result.engines),
    }
    return result


def restore(manager):
    for engine in manager.engines:
        engine.values = {KEY: "a" * 64}


@pytest.mark.parametrize("initially_paused", [False, True])
def test_initial_check_resumes_after_every_engine_compares(manager, initially_paused):
    manager.producer.paused = initially_paused
    assert manager.prepare_weight_check()
    assert all(engine.paused for engine in manager.engines)
    assert manager.producer.paused
    assert manager.frozen.values == {KEY: "a" * 64}
    assert manager.prepare_weight_check()  # A lost preparation reply keeps the original baseline.
    assert manager.events.count(("actor0", "reset")) == 1
    restore(manager)
    manager.finish_weight_check()
    manager.finish_weight_check()
    assert manager._weight_check_state == "verified"
    assert all(engine.paused for engine in manager.engines)
    manager.training_ready()
    assert manager._weight_check_state == "ready"
    assert manager._weight_check_records == []
    assert manager.producer.paused is initially_paused
    assert not any(engine.paused for engine in manager.engines)
    assert manager.events.index(("actor1", "compare")) < manager.events.index(("actor0", "resume"))
    assert not any(name == "ref" for name, _ in manager.events)


@pytest.mark.parametrize("failure", ["invalid_checksum", "rpc_error", "missing_rank", "timeout"])
def test_all_baselines_are_required_before_reset(manager, failure):
    engine = manager.engines[1]
    if failure == "invalid_checksum":
        engine.responses["checksum"] = {"checksums": {}}
    elif failure == "rpc_error":
        engine.errors["checksum"] = RuntimeError("checksum unavailable")
    elif failure == "missing_rank":
        engine.info["rank_prefixes"].append("dp1:pp0:pcp0:tp0")
    else:
        engine.timeout = "checksum"
    with pytest.raises(RuntimeError, match="actor1"):
        manager.prepare_weight_check()
    assert manager._weight_check_state == "failed"
    assert all((engine.name, "checksum") in manager.events for engine in manager.engines)
    assert not any(action == "reset" for _, action in manager.events)
    assert manager.producer.paused
    with pytest.raises(RuntimeError, match="state"):
        manager.training_ready()


@pytest.mark.parametrize(
    "response",
    [
        None,
        [],
        {},
        {"match": 1, "mismatches": []},
        {"match": False, "mismatches": []},
        {"match": True, "mismatches": [KEY]},
    ],
)
def test_malformed_compare_never_enters_ready(manager, response):
    manager.prepare_weight_check()
    restore(manager)
    manager.engines[1].responses["compare"] = response
    with pytest.raises(RuntimeError, match="actor1.*invalid compare"):
        manager.finish_weight_check()
    assert manager._weight_check_state == "failed"
    assert not any(action == "resume" for _, action in manager.events)


def test_match_false_names_every_failing_engine_and_tensor(manager):
    manager.prepare_weight_check()
    with pytest.raises(RuntimeError) as error:
        manager.finish_weight_check()
    assert "actor0" in str(error.value) and "actor1" in str(error.value)
    assert KEY in str(error.value)
    original = manager._weight_check_records[0]["baseline"].copy()
    with pytest.raises(RuntimeError, match="new training attachment"):
        manager.prepare_weight_check()
    assert manager._weight_check_records[0]["baseline"] == original
    assert manager.producer.paused


@pytest.mark.parametrize("replacement", ["actor", "generation", "endpoint", "layout"])
def test_engine_changes_invalidate_baseline(manager, replacement):
    manager.prepare_weight_check()
    restore(manager)
    if replacement == "actor":
        manager.engines[0] = Engine("replacement", manager.events)
    elif replacement == "generation":
        manager.engines[0].info["generation"] += 1
    elif replacement == "endpoint":
        manager.engines[0].info["url"] = "http://replacement:8000"
    else:
        manager.engines[0].info["rank_prefixes"].append("dp1:pp0:pcp0:tp0")
    with pytest.raises(RuntimeError, match="changed"):
        manager.finish_weight_check()
    assert not any(action == "compare" for _, action in manager.events)


def test_external_worker_restart_invalidates_equal_weights(manager):
    manager.prepare_weight_check()
    restore(manager)
    # Rebooting the backend can reload the HF baseline without reinitializing
    # the adapter. URL, Ray actor and layout all remain unchanged.
    manager.engines[1].worker_ids["dp0:pp0:pcp0:tp0"] = "2" * 32
    with pytest.raises(RuntimeError, match="actor1.*worker instances changed"):
        manager.finish_weight_check()
    assert manager._weight_check_state == "failed"
    assert manager.producer.paused
    assert not any(action == "resume" for _, action in manager.events)


@pytest.mark.parametrize("phase", ["checksum", "compare"])
def test_missing_worker_identity_fails_closed(manager, phase):
    if phase == "compare":
        manager.prepare_weight_check()
        restore(manager)
    manager.engines[1].responses[phase] = (
        {"checksums": {KEY: "a" * 64}} if phase == "checksum" else {"match": True, "mismatches": []}
    )
    operation = manager.prepare_weight_check if phase == "checksum" else manager.finish_weight_check
    with pytest.raises(RuntimeError, match="actor1.*worker_ids"):
        operation()
    assert manager.producer.paused
    assert not any(action == ("reset" if phase == "checksum" else "resume") for _, action in manager.events)


@pytest.mark.parametrize("mode", ["reused", "debug_train_only", "debug_rollout_only"])
def test_reused_and_debug_serving_skip_destructive_check(manager, mode):
    if mode != "reused":
        setattr(manager.args, mode, True)
    assert manager.prepare_weight_check(reused=mode == "reused") is False
    assert manager._weight_check_state == "skipped"
    assert manager.events == []


def test_partial_reset_failure_keeps_all_engines_paused(manager):
    manager.engines[1].errors["reset"] = RuntimeError("reset failed")
    with pytest.raises(RuntimeError, match="reset failed"):
        manager.prepare_weight_check()
    assert all(engine.paused for engine in manager.engines)
    assert manager._weight_check_records[0]["baseline"] == {KEY: "a" * 64}
    with pytest.raises(RuntimeError, match="state"):
        manager.training_ready()


@pytest.mark.parametrize("resume_error", [False, True])
def test_ready_failure_keeps_producer_paused(manager, resume_error):
    manager.prepare_weight_check()
    restore(manager)
    manager.finish_weight_check()
    if resume_error:
        manager.engines[1].errors["resume"] = RuntimeError("resume failed")
        with pytest.raises(RuntimeError, match="resume failed"):
            manager.training_ready()
        assert manager.producer.paused
        assert manager._weight_check_state == "failed"
    else:
        manager.detach_training()
        assert manager._weight_check_state == "failed"


@pytest.mark.parametrize("offload", [False, True])
@pytest.mark.parametrize("compare_fails", [False, True])
def test_driver_compares_before_kv_and_training_ready(monkeypatch, offload, compare_fails):
    monkeypatch.setitem(sys.modules, "vime.utils.arguments", SimpleNamespace(parse_args=None))
    train = runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))["train"]
    events = []
    actor, endpoint = Mock(), Mock()
    actor.update_weights.side_effect = lambda **kwargs: events.append(("update", kwargs))
    endpoint.onload_weights.remote.side_effect = lambda: events.append(("weights", None))
    endpoint.onload_kv.remote.side_effect = lambda: events.append(("kv", None))
    endpoint.training_ready.remote.side_effect = lambda **kwargs: events.append(("ready", None))

    def compare():
        events.append(("compare", None))
        if compare_fails:
            raise RuntimeError("comparison failed")

    endpoint.finish_weight_check.remote.side_effect = compare
    monkeypatch.setitem(train.__globals__, "create_training_models", lambda *args: (actor, None))
    monkeypatch.setattr(rollout.ray, "get", lambda value: value)
    args = SimpleNamespace(
        release_train=False, offload_rollout=offload, start_rollout_id=0, num_rollout=0, eval_interval=None
    )
    if compare_fails:
        with pytest.raises(RuntimeError, match="comparison failed"):
            train(args, {}, endpoint, 1, None, weight_check_prepared=True)
        endpoint.training_ready.remote.assert_not_called()
        endpoint.onload_kv.remote.assert_not_called()
    else:
        train(args, {}, endpoint, 1, None, weight_check_prepared=True)
        expected = [("update", {"resume_generation": False}), ("compare", None)]
        if offload:
            expected = [("weights", None), *expected, ("kv", None)]
        assert events == [*expected, ("ready", None)]


@pytest.mark.parametrize("kv_fails", [False, True])
def test_unchecked_offload_restores_kv_before_engines_and_producers_resume(manager, monkeypatch, kv_fails):
    monkeypatch.setitem(sys.modules, "vime.utils.arguments", SimpleNamespace(parse_args=None))
    train = runpy.run_path(str(Path(__file__).resolve().parents[1] / "train.py"))["train"]
    actor, endpoint = Mock(), Mock()
    manager._weight_check_state = "skipped"
    manager._recovery_admission_was_paused = False
    manager.get_updatable_engines_and_lock = lambda: (manager.engines, None)
    manager.resume_rollout_admission = lambda was_paused: manager.events.append(("producer", "resume"))

    def sync(*, resume_generation):
        assert not resume_generation
        manager.events.append(("actor", "update"))

    def wake_kv():
        manager.events.append(("kv", "wake"))
        if kv_fails:
            raise RuntimeError("wake failed")

    actor.update_weights.side_effect = sync
    endpoint.onload_kv.remote.side_effect = wake_kv
    endpoint.training_ready.remote.side_effect = manager.training_ready
    monkeypatch.setitem(train.__globals__, "create_training_models", lambda *args: (actor, None))
    args = SimpleNamespace(
        release_train=False, offload_rollout=True, start_rollout_id=0, num_rollout=0, eval_interval=None
    )
    expected = [("actor", "update"), ("kv", "wake")]
    if kv_fails:
        with pytest.raises(RuntimeError, match="wake failed"):
            train(args, {}, endpoint, 1, None)
    else:
        train(args, {}, endpoint, 1, None)
        expected.extend([("actor0", "resume"), ("actor1", "resume"), ("producer", "resume")])
    assert manager.events == expected


@pytest.mark.parametrize("resume_generation", [False, True])
def test_full_disk_reload_honors_resume_generation(monkeypatch, tmp_path, resume_generation):
    from vime.ray import actor_group

    group = object.__new__(actor_group.RayTrainGroup)
    engine, manager = Mock(), Mock()
    manager.get_updatable_engines_and_lock.remote.return_value = ([engine],)
    group._rollout_manager = manager
    group.args = SimpleNamespace(
        offload_rollout=False,
        update_weight_local_checkpoint_dir=None,
        flush_cache_interval=1,
        ci_test=False,
        update_weight_disk_keep_files=True,
    )
    monkeypatch.setattr(actor_group.ray, "get", lambda value: value)
    group._reload_rollout_weights_from_disk(tmp_path, "1", resume_generation=resume_generation)
    engine.update_weights_from_disk.remote.assert_called_once()
    assert engine.continue_generation.remote.call_count == int(resume_generation)


@pytest.mark.parametrize("corrupt", [False, True])
@pytest.mark.parametrize("restart", [False, True])
def test_cpu_http_checksum_reset_restore_and_compare(manager, corrupt, restart):
    """Exercise the real HTTP adapter against CPU tensors and a protocol fixture."""
    import hashlib
    import json
    from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
    from threading import Thread

    import torch

    _unit_stubs.install_vllm_cli_stubs()
    from vime.backends.vllm_utils.vllm_engine import VLLMEngine

    model = torch.nn.Linear(2, 2, bias=False)
    model.register_buffer("cache", torch.tensor(7.0))
    original = model.weight.detach().clone()
    state = SimpleNamespace(paused=False, worker_id="1" * 32)

    def checksums():
        return {
            f"dp0:pp0:pcp0:tp0:model.{name}": hashlib.sha256(
                parameter.detach().contiguous().view(-1).view(torch.uint8).numpy()
            ).hexdigest()
            for name, parameter in model.named_parameters()
        }

    class Handler(BaseHTTPRequestHandler):
        def respond(self, body):
            data = json.dumps(body).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            body = (
                {"is_paused": state.paused} if self.path == "/is_paused" else {"vllm_config": {"parallel_config": {}}}
            )
            self.respond(body)

        def do_POST(self):
            path = self.path.split("?")[0]
            payload = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            if path in ("/pause", "/resume"):
                state.paused = path == "/pause"
                body = {"status": "paused" if state.paused else "resumed"}
            elif payload["action"] == "checksum":
                assert payload["include_worker_ids"] is True
                body = {"checksums": checksums(), "worker_ids": {"dp0:pp0:pcp0:tp0": state.worker_id}}
            elif payload["action"] == "reset":
                with torch.no_grad():
                    model.weight.zero_()
                body = {"status": "reset"}
            else:
                assert payload["include_worker_ids"] is True
                current, baseline = checksums(), payload["baseline"]
                mismatches = sorted(
                    key for key in current.keys() | baseline.keys() if current.get(key) != baseline.get(key)
                )
                body = {
                    "match": not mismatches,
                    "mismatches": mismatches,
                    "worker_ids": {"dp0:pp0:pcp0:tp0": state.worker_id},
                }
            self.respond(body)

        def log_message(self, *args):
            pass

    def rpc(method):
        def call(*args, **kwargs):
            try:
                return Reply(method(*args, **kwargs))
            except Exception as error:
                return Reply(error=error)

        return SimpleNamespace(remote=call)

    with ThreadingHTTPServer(("127.0.0.1", 0), Handler) as server:
        thread = Thread(target=server.serve_forever, daemon=True)
        thread.start()
        try:
            client = VLLMEngine(SimpleNamespace(weight_checker_timeout=5.0), rank=0)
            client.node_rank, client.server_host, client.server_port = 0, "127.0.0.1", server.server_port
            engine = Engine("http", manager.events)
            for name in ("pause_generation", "continue_generation", "get_weight_check_info", "check_weights"):
                setattr(engine, name, rpc(getattr(client, name)))
            manager.engines = [engine]
            manager.servers["actor"].engines = manager.engines
            assert manager.prepare_weight_check()
            assert state.paused and not model.weight.count_nonzero()
            assert model.cache.item() == 7.0
            with torch.no_grad():
                model.weight.copy_(original)
                if corrupt:
                    model.weight[0, 0].add_(1)
            if restart:
                state.worker_id = "2" * 32
            if corrupt or restart:
                error = "worker instances changed" if restart else "dp0:pp0:pcp0:tp0:model.weight"
                with pytest.raises(RuntimeError, match=error):
                    manager.finish_weight_check()
                assert state.paused and manager.producer.paused
            else:
                manager.finish_weight_check()
                manager.training_ready()
                assert not state.paused and not manager.producer.paused
        finally:
            server.shutdown()
            thread.join(timeout=5)


@pytest.mark.parametrize("transport", ["nccl", "disk"])
@pytest.mark.parametrize("resume_generation", [False, True])
def test_train_group_propagates_initial_resume_control(monkeypatch, tmp_path, transport, resume_generation):
    from vime.ray import actor_group

    group = object.__new__(actor_group.RayTrainGroup)
    group.role = "actor"
    group.args = SimpleNamespace(
        update_weight_mode="full",
        update_weight_transport=transport,
        update_weight_disk_dir=str(tmp_path),
        release_train=False,
    )
    group._actor_handlers = [Mock(), Mock()]
    group._disk_weight_version = 0
    group._reload_rollout_weights_from_disk = Mock()
    monkeypatch.setattr(actor_group.ray, "get", lambda value: value)
    group.update_weights(resume_generation=resume_generation)
    for actor in group._actor_handlers:
        actor.update_weights.remote.assert_called_once_with(resume_generation=resume_generation)
    if transport == "disk":
        group._reload_rollout_weights_from_disk.assert_called_once_with(
            tmp_path / "weight_v000001", "1", resume_generation=resume_generation
        )
    else:
        group._reload_rollout_weights_from_disk.assert_not_called()
