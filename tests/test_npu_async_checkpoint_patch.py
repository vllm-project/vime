"""Exercise the shipped persistent worker patch with real spawn and CPU IO.

The device bridge is a stub; NPU checkpoint roundtrip still requires CI. Read
the pinned vendor blob locally so these tests never fetch sources or apply a
patch to the developer's Megatron worktree.
"""

import os
import subprocess
import sys
from pathlib import Path

import pytest

NUM_GPUS = 0
PIN = "4b4acac9a1d28ea6829c8d4f566d75698a21249d"
MODULE = "megatron/core/dist_checkpointing/strategies/async_utils.py"
REPO = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def worker_source(tmp_path_factory):
    vendor = os.environ.get("VIME_TEST_MEGATRON_SOURCE", "/root/Megatron-LM")
    source = subprocess.run(["git", "-C", vendor, "show", f"{PIN}:{MODULE}"], capture_output=True, text=True)
    if source.returncode:
        pytest.skip(f"Pinned Megatron {PIN[:8]} Git blob required; set VIME_TEST_MEGATRON_SOURCE")
    root = tmp_path_factory.mktemp("persistent-worker")
    module = root / MODULE
    module.parent.mkdir(parents=True)
    module.write_text(source.stdout)
    subprocess.run(
        ["git", "apply", f"--include={MODULE}", str(REPO / "docker/npu_patch/megatron.patch")],
        cwd=root,
        capture_output=True,
        text=True,
        check=True,
    )
    for directory in [module.parent, *module.parents[1:4]]:
        (directory / "__init__.py").touch()
    (root / "megatron/core/_rank_utils.py").write_text("def safe_get_rank(): return 0\n")
    (root / "megatron/core/utils.py").write_text(
        "def log_single_rank(logger, level, message): logger.log(level, message)\n"
    )
    (root / "megatron/core/dist_checkpointing/utils.py").write_text(
        "from contextlib import nullcontext\ndef debug_time(*a): return nullcontext()\n"
    )
    bridge = root / "torch_npu/contrib"
    bridge.mkdir(parents=True)
    (bridge / "__init__.py").touch()
    (bridge.parent / "__init__.py").write_text(
        "import os, torch\nfrom pathlib import Path\nfrom types import SimpleNamespace\n"
        "def select(index):\n"
        " assert getattr(torch, '_test_npu_bridge_loaded', False)\n"
        " with open(os.environ['TEST_DEVICE_LOG'], 'a') as f: f.write(str(index) + '\\n')\n"
        "torch.npu = SimpleNamespace(device_count=lambda: int(os.environ.get('TEST_DEVICE_COUNT', '2')), "
        "current_device=lambda: 1, set_device=select)\n"
    )
    (bridge / "transfer_to_npu.py").write_text(
        "import torch\ntorch._test_npu_bridge_loaded = True\ntorch.cuda.current_device = lambda: 'cpu'\n"
    )
    (root / "worker_fixtures.py").write_text(
        "import os, time, torch\nfrom pathlib import Path\n"
        "def write(path, values, delay):\n"
        " time.sleep(delay)\n"
        " torch.save(values, path)\n"
        "def crash(*args): os._exit(23)\n"
        "def preload(): return torch.arange(5)\n"
        "def rank_main(rank, rendezvous, root):\n"
        " import torch_npu.contrib.transfer_to_npu\n"
        " from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest, PersistentAsyncCaller\n"
        " torch.distributed.init_process_group('gloo', init_method=rendezvous, rank=rank, world_size=2)\n"
        " caller = PersistentAsyncCaller()\n"
        " try:\n"
        "  request = AsyncRequest(crash if rank == 0 else write, (str(Path(root)/'peer.pt'), torch.ones(2), 0.2), [])\n"
        "  caller.schedule_async_call(request)\n"
        "  try: caller.is_current_async_call_done(blocking=True)\n"
        "  except RuntimeError as error:\n"
        "   assert 'checkpoint worker' in str(error)\n"
        "   (Path(root)/('rank-' + str(rank))).write_text('failure propagated')\n"
        "  else: raise AssertionError('distributed worker failure was lost')\n"
        " finally:\n"
        "  caller.close(abort=True)\n"
        "  torch.distributed.destroy_process_group()\n"
    )
    driver = root / "driver.py"
    driver.write_text(
        "import sys, os, time, torch\nfrom pathlib import Path\n"
        "from megatron.core.dist_checkpointing.strategies.async_utils import AsyncRequest, PersistentAsyncCaller\n"
        "import worker_fixtures as fixture\n"
        "def main():\n"
        " import torch_npu.contrib.transfer_to_npu\n"
        " scenario, root = sys.argv[1], Path(sys.argv[2])\n"
        " if scenario == 'distributed':\n"
        "  torch.multiprocessing.spawn(fixture.rank_main, args=((root/'rendezvous').as_uri(), str(root)), nprocs=2)\n"
        "  assert all((root/('rank-' + str(rank))).read_text() == 'failure propagated' for rank in range(2))\n"
        "  return\n"
        " torch.distributed.init_process_group('gloo', init_method=(root/'rendezvous').as_uri(), rank=0, world_size=1)\n"
        " caller = PersistentAsyncCaller()\n"
        " try:\n"
        "  if scenario == 'close-dead':\n"
        "   caller.process = caller._get_process(0)\n"
        "   caller.process.kill(); caller.process.join(timeout=5)\n"
        "   try: caller.close()\n"
        "   except RuntimeError as error: assert 'checkpoint worker' in str(error)\n"
        "   else: raise AssertionError('close silently accepted a dead worker')\n"
        "   return\n"
        "  if scenario == 'healthy':\n"
        "   for index in range(2):\n"
        "    path = root/(str(index) + '.pt')\n"
        "    preload = fixture.preload if index else None\n"
        "    request = AsyncRequest(fixture.write, (str(path), None if preload else torch.arange(5), 0), [], preload_fn=preload, call_idx=index)\n"
        "    caller.schedule_async_call(request)\n"
        "    assert caller.is_current_async_call_done(blocking=True, no_dist=True)\n"
        "    torch.testing.assert_close(torch.load(path, weights_only=True), torch.arange(5))\n"
        "   caller.close()\n"
        "   assert Path(os.environ['TEST_DEVICE_LOG']).read_text().strip() == '1'\n"
        "  else:\n"
        "   preload = fixture.crash if scenario == 'preload-dead' else None\n"
        "   request = AsyncRequest(fixture.crash, ('unused', None, 0), [], preload_fn=preload)\n"
        "   try:\n"
        "    caller.schedule_async_call(request)\n"
        "    caller.is_current_async_call_done(blocking=True, no_dist=True)\n"
        "   except RuntimeError as error: assert 'checkpoint worker' in str(error)\n"
        "   else: raise AssertionError('worker death did not fail the caller')\n"
        " finally:\n"
        "  caller.close(abort=True)\n"
        "  torch.distributed.destroy_process_group()\n"
        "if __name__ == '__main__': main()\n"
    )
    return root, driver


@pytest.mark.parametrize(
    "scenario", ["healthy", "no-device", "preload-dead", "write-dead", "close-dead", "distributed"]
)
def test_persistent_spawn_worker_initialization_io_and_failure_propagation(worker_source, tmp_path, scenario):
    root, driver = worker_source
    env = {
        **os.environ,
        "PYTHONPATH": str(root),
        "VIME_PLATFORM": "npu",
        "TEST_DEVICE_COUNT": "0" if scenario == "no-device" else "2",
        "TEST_DEVICE_LOG": str(tmp_path / "devices"),
        "OMP_NUM_THREADS": "1",
        "TORCH_DEVICE_BACKEND_AUTOLOAD": "0",
    }
    result = subprocess.run(
        [sys.executable, str(driver), scenario, str(tmp_path)],
        env=env,
        capture_output=True,
        text=True,
        timeout=45,
    )
    assert result.returncode == 0, result.stdout + result.stderr
