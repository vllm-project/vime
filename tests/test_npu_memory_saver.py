"""NPU allocator regression: run preload checks in a fresh process."""

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path


def _receive_ipc(queue, result):
    import torch_npu  # noqa: F401

    tensor = queue.get()
    values = (tensor.min().item(), tensor.max().item(), tensor.numel())
    del tensor
    result.send(values)
    result.close()


def _worker():
    import signal
    import threading

    assert signal.getsignal(signal.SIGINT) is not None, "preload must preserve Python signal initialization"

    import torch
    import torch_npu  # noqa: F401
    from torch_memory_saver import torch_memory_saver as tms

    torch.npu.set_device(0)
    tms._ensure_initialized()
    cdll = tms._impl._binary_wrapper.cdll

    def config():
        return (
            cdll.tms_get_current_tag().decode(),
            cdll.tms_get_interesting_region(),
            cdll.tms_get_enable_cpu_backup(),
        )

    expected = ("default", True, True)
    assert config() == expected
    weights = torch.full((4 * 1024 * 1024,), 17, dtype=torch.int32, device="npu")
    address = weights.data_ptr()
    thread_tensors = []

    def allocate_from_another_thread():
        torch.npu.set_device(0)
        thread_tensors.append(torch.full_like(weights, 83))

    with tms.region(tag="grad_buffer", enable_cpu_backup=False):
        grads = torch.full_like(weights, 31)
        thread = threading.Thread(target=allocate_from_another_thread)
        thread.start()
        thread.join()
        assert config() == ("grad_buffer", True, False)
    assert config() == expected
    assert len(thread_tensors) == 1

    try:
        with tms.region(tag="exception", enable_cpu_backup=False):
            raise ValueError("expected region failure")
    except ValueError:
        pass
    assert config() == expected

    cycles = []
    for step in range(2):
        torch.npu.synchronize()
        free_before = torch.npu.mem_get_info()[0]
        tms.pause()
        free_paused = torch.npu.mem_get_info()[0]
        assert free_paused > free_before
        try:
            with tms.disable():
                temporary = torch.full_like(weights, 59)
                assert temporary.min().item() == 59
                del temporary
                if step == 1:
                    raise ValueError("expected export failure")
        except ValueError:
            pass
        assert config() == expected
        tms.resume()
        assert weights.data_ptr() == address
        assert weights.min().item() == weights.max().item() == 17
        assert thread_tensors[0].min().item() == thread_tensors[0].max().item() == 83
        # Non-backed contents must be reconstructed before being read.
        grads.zero_()
        assert grads.max().item() == 0
        torch.npu.empty_cache()
        cycles.append({"released_bytes": free_paused - free_before, "data_restored": True})

    # Adam creates its moments during the first step, outside explicit regions.
    cpu_param = torch.nn.Parameter(torch.tensor([1.0, 2.0]))
    npu_param = torch.nn.Parameter(cpu_param.detach().to("npu"))
    cpu_optimizer = torch.optim.Adam([cpu_param], lr=0.01, foreach=False)
    npu_optimizer = torch.optim.Adam([npu_param], lr=0.01, foreach=False)
    for _ in range(2):
        for parameter, optimizer in [(cpu_param, cpu_optimizer), (npu_param, npu_optimizer)]:
            optimizer.zero_grad(set_to_none=True)
            parameter.square().sum().backward()
            optimizer.step()
        torch.testing.assert_close(npu_param.cpu(), cpu_param)
        tms.pause()
        tms.resume()
        for name in ["exp_avg", "exp_avg_sq"]:
            torch.testing.assert_close(
                npu_optimizer.state[npu_param][name].cpu(), cpu_optimizer.state[cpu_param][name]
            )
        torch.testing.assert_close(npu_param.cpu(), cpu_param)

    # Serving must not inherit the trainer's saver or its paused allocations.
    context = torch.multiprocessing.get_context("spawn")
    queue = context.SimpleQueue()  # Serialize on this thread, inside disable().
    receiver_result, child_result = context.Pipe(duplex=False)
    receiver = context.Process(target=_receive_ipc, args=(queue, child_result))
    tms.pause()
    try:
        with tms.disable():
            payload = torch.full((1024,), 73, device="npu")
            names = ["LD_PRELOAD", "TMS_HOOK_MODE", "TMS_INIT_ENABLE", "TMS_INIT_ENABLE_CPU_BACKUP"]
            inherited = {name: os.environ.pop(name) for name in names if name in os.environ}
            try:
                receiver.start()
            finally:
                os.environ.update(inherited)
            child_result.close()
            queue.put(payload)
            assert receiver_result.poll(60), "IPC receiver did not respond"
            assert receiver_result.recv() == (73, 73, 1024)
            del payload
        receiver.join(30)
        assert receiver.exitcode == 0
    finally:
        if receiver.is_alive():
            receiver.terminate()
            receiver.join(10)
        receiver_result.close()
        queue.close()
        tms.resume()

    print(
        json.dumps(
            {
                "cycles": cycles,
                "thread_isolated": True,
                "exception_state_restored": True,
                "adam_restored": True,
                "ipc_values_verified": True,
            }
        )
    )


def _run():
    spec = importlib.util.find_spec("torch_memory_saver")
    assert spec is not None and spec.origin is not None, "NPU TMS must be installed"
    package_root = Path(spec.origin).parent.parent
    libraries = list(package_root.glob("torch_memory_saver_hook_mode_preload.*.so"))
    assert len(libraries) == 1, libraries
    env = {
        **os.environ,
        "LD_PRELOAD": str(libraries[0]),
        "TMS_HOOK_MODE": "preload",
        "TMS_INIT_ENABLE": "1",
        "TMS_INIT_ENABLE_CPU_BACKUP": "1",
        "PYTORCH_NPU_ALLOC_CONF": "expandable_segments:False",
    }
    # Also require eager symbol binding to work without loading CANN at startup.
    subprocess.run(
        [sys.executable, "-c", "import signal; assert signal.getsignal(signal.SIGINT) is not None"],
        env={**env, "LD_BIND_NOW": "1"},
        check=True,
        timeout=30,
    )
    subprocess.run([sys.executable, __file__, "--worker"], env=env, check=True, timeout=180)


def test_npu_memory_saver():
    import pytest

    if not list(Path("/dev").glob("davinci[0-9]*")):
        pytest.skip("requires an NPU")
    _run()


if __name__ == "__main__":
    if "--worker" in sys.argv:
        _worker()
    else:
        _run()
