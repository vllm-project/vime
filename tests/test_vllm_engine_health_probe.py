"""Unit tests for the vLLM server startup health probe.

Covers ``vime.backends.vllm_utils.vllm_engine._wait_server_healthy``:

* every ``/health`` probe carries a timeout, so a server that accepts the
  connection but never responds cannot stall the startup loop forever
  (vllm-project/vime#461);
* a dead server process is still detected promptly via ``is_process_alive``;
* transient connection errors are retried until the probe succeeds.
"""

import os
import sys
import types
import unittest
from unittest import mock

import requests


def _stub_modules():
    """Build stub modules so the engine module imports on CPU.

    Returned as a dict for ``mock.patch.dict(sys.modules, ...)`` so the
    stubs only exist while importing the module under test and never leak
    into the global ``sys.modules`` for sibling tests.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    stubs = {}

    def make_pkg(name, path):
        mod = types.ModuleType(name)
        mod.__path__ = [path]
        stubs[name] = mod

    def make_mod(name, **attrs):
        mod = types.ModuleType(name)
        for key, value in attrs.items():
            setattr(mod, key, value)
        stubs[name] = mod

    vime_dir = os.path.join(repo_root, "vime")
    make_pkg("vllm", [])
    make_pkg("vllm.utils", [])
    make_mod("vllm.utils.system_utils", kill_process_tree=lambda *a, **k: None)
    make_mod("cloudpickle")
    make_pkg("vime", vime_dir)
    make_pkg("vime.backends", os.path.join(vime_dir, "backends"))
    make_pkg("vime.backends.vllm_utils", os.path.join(vime_dir, "backends", "vllm_utils"))
    make_mod("vime.backends.vllm_utils.external", get_server_info=lambda *a, **k: {})
    make_pkg("vime.ray", os.path.join(vime_dir, "ray"))
    make_mod("vime.ray.ray_actor", RayActor=type("RayActor", (), {}))
    make_pkg("vime.utils", os.path.join(vime_dir, "utils"))
    make_mod(
        "vime.utils.http_utils",
        _wrap_ipv6=lambda host: host,
        get_host_info=lambda *a, **k: {},
    )
    return stubs


with mock.patch.dict(sys.modules, _stub_modules()):
    from vime.backends.vllm_utils.vllm_engine import _wait_server_healthy  # noqa: E402


def _ok_response():
    resp = mock.Mock()
    resp.status_code = 200
    return resp


class WaitServerHealthyTest(unittest.TestCase):
    def test_probe_carries_default_timeout(self):
        with mock.patch("requests.get", return_value=_ok_response()) as get:
            _wait_server_healthy("http://127.0.0.1:8000", is_process_alive=lambda: True)
        get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=5.0)

    def test_custom_probe_timeout_is_used(self):
        with mock.patch("requests.get", return_value=_ok_response()) as get:
            _wait_server_healthy(
                "http://127.0.0.1:8000",
                is_process_alive=lambda: True,
                probe_timeout=1.5,
            )
        get.assert_called_once_with("http://127.0.0.1:8000/health", timeout=1.5)

    def test_unresponsive_server_does_not_block_liveness_check(self):
        # The probe raises ConnectTimeout on every attempt while the server
        # process dies: the loop must surface the dead process instead of
        # hanging inside requests.get (vllm-project/vime#461).
        alive = [True, False]
        with (
            mock.patch("requests.get", side_effect=requests.exceptions.ConnectTimeout("timed out")),
            mock.patch("time.sleep"),
        ):
            with self.assertRaisesRegex(Exception, "Server process terminated unexpectedly"):
                _wait_server_healthy("http://127.0.0.1:8000", is_process_alive=lambda: alive.pop(0))

    def test_transient_errors_are_retried_until_success(self):
        get = mock.Mock(
            side_effect=[
                requests.exceptions.ConnectionError("refused"),
                requests.exceptions.ConnectionError("refused"),
                _ok_response(),
            ]
        )
        with mock.patch("requests.get", get), mock.patch("time.sleep"):
            _wait_server_healthy("http://127.0.0.1:8000", is_process_alive=lambda: True)
        self.assertEqual(get.call_count, 3)


if __name__ == "__main__":
    unittest.main()
