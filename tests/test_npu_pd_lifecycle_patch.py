"""Exercise shipped PD patch logic on CPU, without applying vendor patches."""

import ast
import dataclasses
import logging
import os
import queue
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

NUM_GPUS = 0
PATCHES = Path(__file__).resolve().parents[1] / "docker/npu_patch"


def new_side(patch, filename):
    section = patch.split(f"diff --git a/{filename} b/{filename}\n", 1)[1].split("\ndiff --git ", 1)[0]
    return "\n".join(
        line[1:] if line else ""
        for line in section.splitlines()
        if not line or (line.startswith((" ", "+")) and not line.startswith("+++"))
    )


def method(source, name):
    lines = source.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(f"    def {name}("))
    end = next(
        (i for i in range(start + 1, len(lines)) if lines[i].strip() and len(lines[i]) - len(lines[i].lstrip()) <= 4),
        len(lines),
    )
    return "\n".join(lines[start:end])


@pytest.fixture
def global_te():
    source = new_side(
        (PATCHES / "vllm-ascend.patch").read_text(),
        "vllm_ascend/distributed/kv_transfer/utils/mooncake_transfer_engine.py",
    )
    namespace = {}
    exec(compile(ast.parse(source), "<shipped-mooncake-te>", "exec"), namespace)
    te = namespace["GlobalTE"]()
    active = {}
    events = []

    def register(ptr, size):
        assert ptr not in active
        active[ptr] = size
        events.append(("register", ptr))
        return 0

    def unregister(ptr):
        assert ptr in active
        active.pop(ptr)
        events.append(("unregister", ptr))
        return 0

    te.transfer_engine = SimpleNamespace(register_memory=register, unregister_memory=unregister)
    return te, active, events


def test_registered_physical_pages_are_released_and_reregistered_after_remap(global_te):
    te, active, events = global_te
    te.register_buffer([100, 200], [16, 32])
    te.register_buffer([100, 200], [16, 32])
    assert len(events) == 2  # repeated registration must not pin twice
    for _ in range(2):
        te.unregister_buffer()
        assert active == {}  # fake CaMem unmap is only legal without pins
        assert not te.is_register_buffer
        events.append(("unmap-and-remap", None))
        te.reregister_buffer()
        assert active == {100: 16, 200: 32}
        assert te.is_register_buffer


def test_failed_deregistration_propagates_before_memory_can_be_unmapped(global_te):
    te, active, events = global_te
    te.register_buffer([100], [16])
    te.transfer_engine.unregister_memory = lambda ptr: 7
    with pytest.raises(RuntimeError, match="deregistration.*ptr=100"):
        te.unregister_buffer()
    assert active == {100: 16}
    assert events == [("register", 100)]


def test_failed_reregistration_cannot_report_a_healthy_wake(global_te):
    te, active, _ = global_te
    te.register_buffer([100], [16])
    te.unregister_buffer()
    te.transfer_engine.register_memory = lambda ptr, size: 9
    with pytest.raises(RuntimeError, match="registration.*ptr=100"):
        te.reregister_buffer()
    assert not te.is_register_buffer and not active


@pytest.mark.parametrize("tags, should_register", [(None, True), (["kv_cache"], True), (["weights"], False)])
def test_weight_only_wake_does_not_register_unmapped_kv_pages(tags, should_register):
    source = new_side((PATCHES / "vllm-ascend.patch").read_text(), "vllm_ascend/worker/worker.py")
    calls = []
    connector = SimpleNamespace(
        prepare_sleep=lambda: calls.append("sleep"), resume_after_wake=lambda: calls.append("wake")
    )
    namespace = {"has_kv_transfer_group": lambda: True, "get_kv_transfer_group": lambda: connector}
    code = (
        "class Worker:\n"
        + method(source, "_prepare_kv_transfer_sleep")
        + "\n"
        + method(source, "_resume_kv_transfer_after_wake")
    )
    exec(compile(ast.parse(code), "<shipped-worker-sleep-hooks>", "exec"), namespace)
    worker = namespace["Worker"]()
    worker._prepare_kv_transfer_sleep()
    worker._resume_kv_transfer_after_wake(tags)
    assert calls == (["sleep", "wake"] if should_register else ["sleep"])
    assert "self._prepare_kv_transfer_sleep()" in source
    assert "self._resume_kv_transfer_after_wake(tags)" in source


@pytest.mark.parametrize("pending", [False, True])
def test_pull_queue_must_drain_before_deregistration(pending):
    source = new_side(
        (PATCHES / "vllm-ascend.patch").read_text(),
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py",
    )
    calls = []
    base = type("Base", (), {"prepare_sleep": lambda self: calls.append("unregister")})
    namespace = {"Base": base, "get_transfer_timeout_value": lambda: 0}
    exec(
        compile(ast.parse("class Pull(Base):\n" + method(source, "prepare_sleep")), "<shipped-pull-sleep>", "exec"),
        namespace,
    )
    worker = namespace["Pull"]()
    worker._recving_thread = SimpleNamespace(request_queue=queue.Queue())
    if pending:
        worker._recving_thread.request_queue.put("in-flight read")
        with pytest.raises(TimeoutError, match="did not drain"):
            worker.prepare_sleep()
        assert calls == []
    else:
        worker.prepare_sleep()
        assert calls == ["unregister"]


def test_pull_drain_caps_large_hccl_derived_timeout():
    source = new_side(
        (PATCHES / "vllm-ascend.patch").read_text(),
        "vllm_ascend/distributed/kv_transfer/kv_p2p/mooncake/pull_worker.py",
    )
    namespace = {"Base": object, "get_transfer_timeout_value": lambda: 33064}
    code = "class Pull(Base):\n" + method(source, "prepare_sleep")
    exec(compile(ast.parse(code), "<shipped-pull-bounded-drain>", "exec"), namespace)

    class Condition:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def wait_for(self, predicate, timeout):
            assert timeout == 300
            return False

    worker = namespace["Pull"]()
    pending = SimpleNamespace(all_tasks_done=Condition(), unfinished_tasks=1)
    worker._recving_thread = SimpleNamespace(request_queue=pending)
    with pytest.raises(TimeoutError):
        worker.prepare_sleep()


def test_scheduler_consumes_producer_ack_even_after_empty_worker_output():
    source = new_side((PATCHES / "vllm.patch").read_text(), "vllm/v1/core/sched/scheduler.py")
    block = source.split("        # KV Connector: update state for finished KV Transfers.\n", 1)[1]
    block = block.split("\n\n", 1)[0]
    # The worker discarded its empty no-forward output, while an independent
    # scheduler-side ACK must release producer blocks and let pause finish.
    requests = {"finished-prefill": "retained-blocks"}
    acks = queue.SimpleQueue()

    def update(output):
        while not acks.empty():
            requests.pop(acks.get_nowait())

    scheduler = SimpleNamespace(connector=object(), _update_from_kv_xfer_finished=update)
    namespace = {"KVConnectorOutput": SimpleNamespace}
    exec(
        compile(
            "def tick(self, kv_connector_output):\n" + textwrap.indent(textwrap.dedent(block), "    "),
            "<shipped-scheduler-kv-poll>",
            "exec",
        ),
        namespace,
    )
    namespace["tick"](scheduler, None)
    assert requests
    acks.put("finished-prefill")
    namespace["tick"](scheduler, None)
    assert not requests


@pytest.fixture
def camem_allocator():
    source = new_side((PATCHES / "vllm-ascend.patch").read_text(), "vllm_ascend/device_allocator/camem.py")
    tree = ast.parse(source)
    # Execute the shipped Python state transitions with a fake device API.
    # Imports/torch allocation are excluded; native transitions are compiled
    # and exercised independently in test_npu_camem_native_patch.py.
    methods = {"sleep", "wake_up", "python_free_callback"}
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    allocation = next(node for node in classes if node.name == "AllocationData")
    allocator = next(node for node in classes if node.name == "CaMemAllocator")
    allocator.body = [node for node in allocator.body if isinstance(node, ast.FunctionDef) and node.name in methods]
    calls = []

    class Tensor:
        def data_ptr(self):
            return 1000

        def numel(self):
            return 64

        def element_size(self):
            return 1

    namespace = {
        "dataclasses": dataclasses,
        "HandleType": tuple,
        "os": os,
        "logger": logging.getLogger(__name__),
        "torch": SimpleNamespace(
            Tensor=Tensor,
            uint8=object(),
            empty=lambda *a, **kw: Tensor(),
            npu=SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None),
        ),
        "gc": SimpleNamespace(collect=lambda: None),
        "memcpy": lambda *args: calls.append(("copy", args[-1])),
        "create_and_map": lambda handle: calls.append(("map", handle[2])),
        "unmap_and_release": lambda handle: calls.append(("unmap", handle[2])),
    }
    exec(compile(ast.Module(body=[allocation, allocator], type_ignores=[]), "<shipped-camem>", "exec"), namespace)
    cls = namespace["CaMemAllocator"]
    cls.default_tag = "default"
    cls.sleep_persistent_tag = "sleep_persistent"
    instance = cls()
    data_cls = namespace["AllocationData"]
    instance.pointer_to_data = {
        ptr: data_cls((0, 64, ptr, ptr + 1), tag)
        for ptr, tag in ((100, "weights"), (200, "kv_cache"), (300, "sleep_persistent"))
    }
    return instance, calls, namespace


def test_python_camem_selective_wake_then_sleep_does_not_double_unmap(camem_allocator):
    allocator, calls, _ = camem_allocator
    allocator.sleep(offload_tags=("weights",))
    allocator.wake_up(["weights"])
    allocator.wake_up(["weights"])
    assert calls == [("copy", 2), ("unmap", 100), ("unmap", 200), ("map", 100), ("copy", 1)]
    allocator.sleep(offload_tags=())
    assert calls[-1] == ("unmap", 100)
    allocator.wake_up(["kv_cache"])
    assert calls[-1] == ("map", 200)
    assert allocator.pointer_to_data[300].mapped  # persistent allocations stay mapped


def test_python_camem_failed_transition_keeps_retry_state(camem_allocator):
    allocator, calls, namespace = camem_allocator

    def fail(handle):
        raise RuntimeError("driver failure")

    namespace["unmap_and_release"] = fail
    with pytest.raises(RuntimeError, match="driver failure"):
        allocator.sleep(offload_tags=())
    assert allocator.pointer_to_data[100].mapped
    namespace["unmap_and_release"] = lambda handle: None
    allocator.sleep(offload_tags=())
    namespace["create_and_map"] = fail
    with pytest.raises(RuntimeError, match="driver failure"):
        allocator.wake_up(["weights"])
    assert not allocator.pointer_to_data[100].mapped


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
