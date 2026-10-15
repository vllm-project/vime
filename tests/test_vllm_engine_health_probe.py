"""CPU regressions for startup health probes, retry budgets, and scoped imports."""

import os
import runpy
import socketserver
import sys
import threading
import time
import types
import unittest
from contextlib import contextmanager
from unittest import mock

import requests
from urllib3.util import Timeout

NUM_GPUS = 0


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
    make_mod("vime.backends.vllm_utils.server_control", unregister_worker=lambda *a, **k: None)
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
    from vime.backends.vllm_utils.vllm_engine import _wait_server_healthy


def _response(status=200):
    response = mock.MagicMock(status_code=status)
    response.__enter__.return_value = response
    return response


class _Clock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


class WaitServerHealthyTest(unittest.TestCase):
    def setUp(self):
        self.clock = _Clock()
        # Patch only this imported module's clock, not the process-wide time module.
        clock_patch = mock.patch.dict(_wait_server_healthy.__globals__, time=self.clock)
        clock_patch.start()
        self.addCleanup(clock_patch.stop)

    def assertProbe(self, call, timeout):
        self.assertEqual(call.args, ("http://127.0.0.1:8000/health_generate",))
        self.assertTrue(call.kwargs["stream"])
        self.assertFalse(call.kwargs["allow_redirects"])
        self.assertIsInstance(call.kwargs["timeout"], Timeout)
        self.assertEqual(call.kwargs["timeout"].total, timeout)

    def test_probe_carries_default_timeout(self):
        response = _response()
        alive = mock.Mock(return_value=True)
        with mock.patch("requests.get", return_value=response) as get:
            _wait_server_healthy("http://127.0.0.1:8000", alive)
        self.assertProbe(get.call_args, 5.0)
        response.__exit__.assert_called_once()
        alive.assert_not_called()
        self.assertEqual(self.clock.sleeps, [])

    def test_custom_probe_timeout_is_used(self):
        with mock.patch("requests.get", return_value=_response()) as get:
            _wait_server_healthy("http://127.0.0.1:8000", lambda: True, probe_timeout=1.5)
        self.assertProbe(get.call_args, 1.5)

    def test_unresponsive_server_does_not_block_liveness_check(self):
        for error in (requests.exceptions.ConnectTimeout, requests.exceptions.ReadTimeout):
            with self.subTest(error=error):
                alive = mock.Mock(side_effect=[True, False])
                with mock.patch("requests.get", side_effect=error("timed out")) as get:
                    with self.assertRaisesRegex(Exception, "Server process terminated unexpectedly"):
                        _wait_server_healthy("http://127.0.0.1:8000", alive)
                self.assertEqual(alive.call_count, 2)
                for call in get.call_args_list:
                    self.assertProbe(call, 5.0)

    def test_transient_connection_errors_are_retried_until_success(self):
        with mock.patch(
            "requests.get",
            side_effect=[requests.exceptions.ConnectionError("refused"), _response()],
        ) as get:
            _wait_server_healthy("http://127.0.0.1:8000", lambda: True)
        self.assertEqual(get.call_count, 2)
        self.assertEqual(self.clock.sleeps, [2])

    def test_non_ready_response_is_closed_before_retry(self):
        not_ready, ready = _response(503), _response()
        with mock.patch("requests.get", side_effect=[not_ready, ready]):
            _wait_server_healthy("http://127.0.0.1:8000", lambda: True)
        not_ready.__exit__.assert_called_once()
        ready.__exit__.assert_called_once()
        self.assertEqual(self.clock.sleeps, [2])

    def test_slow_load_has_no_overall_deadline_by_default(self):
        def get(*args, **kwargs):
            self.clock.advance(kwargs["timeout"].total)
            if self.clock.now < 30:
                raise requests.exceptions.ReadTimeout("still loading")
            return _response()

        with mock.patch("requests.get", side_effect=get) as request:
            _wait_server_healthy("http://127.0.0.1:8000", lambda: True)
        self.assertGreater(self.clock.now, 5.0)
        self.assertGreater(request.call_count, 1)
        for call in request.call_args_list:
            self.assertProbe(call, 5.0)

    def test_probe_timeout_shrinks_with_remaining_wait_budget(self):
        not_ready = _response(503)
        timeouts = []

        def get(*args, **kwargs):
            timeouts.append(kwargs["timeout"].total)
            if len(timeouts) == 1:
                self.clock.advance(1)
                return not_ready
            self.clock.advance(kwargs["timeout"].total)
            raise requests.exceptions.ReadTimeout("timed out")

        with mock.patch("requests.get", side_effect=get):
            with self.assertRaisesRegex(TimeoutError, "within 5.0 seconds"):
                _wait_server_healthy("http://127.0.0.1:8000", lambda: True, wait_timeout=5.0)
        self.assertEqual(timeouts, [5.0, 2.0])
        self.assertEqual(self.clock.sleeps, [2])
        self.assertEqual(self.clock.now, 5.0)
        not_ready.__exit__.assert_called_once()

    def test_retry_sleep_uses_only_remaining_budget(self):
        def get(*args, **kwargs):
            self.clock.advance(1)
            raise requests.exceptions.ConnectionError("refused")

        with mock.patch("requests.get", side_effect=get) as request:
            with self.assertRaises(TimeoutError):
                _wait_server_healthy("http://127.0.0.1:8000", lambda: True, wait_timeout=1.5)
        self.assertProbe(request.call_args, 1.5)
        self.assertEqual(request.call_count, 1)
        self.assertEqual(self.clock.sleeps, [0.5])
        self.assertEqual(self.clock.now, 1.5)

    def test_exhausted_budget_does_not_send_a_probe(self):
        with mock.patch("requests.get") as get:
            with self.assertRaises(TimeoutError):
                _wait_server_healthy("http://127.0.0.1:8000", lambda: True, wait_timeout=0)
        get.assert_not_called()
        self.assertEqual(self.clock.sleeps, [])

    def test_late_success_does_not_extend_wait_budget(self):
        response = _response()

        def get(*args, **kwargs):
            self.clock.advance(3)
            return response

        with mock.patch("requests.get", side_effect=get):
            with self.assertRaises(TimeoutError):
                _wait_server_healthy("http://127.0.0.1:8000", lambda: True, wait_timeout=2.0)
        response.__exit__.assert_called_once()
        self.assertEqual(self.clock.sleeps, [])

    def test_process_check_time_counts_towards_budget(self):
        def alive():
            self.clock.advance(2)
            return True

        with mock.patch("requests.get", side_effect=requests.exceptions.ConnectionError("refused")):
            with self.assertRaises(TimeoutError):
                _wait_server_healthy("http://127.0.0.1:8000", alive, wait_timeout=1.0)
        self.assertEqual(self.clock.sleeps, [])

    def test_invalid_timeouts_are_rejected_before_requesting(self):
        cases = [{"probe_timeout": value} for value in (0, -1, float("inf"), float("nan"))] + [
            {"wait_timeout": value} for value in (-1, float("inf"), float("nan"))
        ]
        for kwargs in cases:
            with self.subTest(kwargs=kwargs), mock.patch("requests.get") as get:
                with self.assertRaises(ValueError):
                    _wait_server_healthy("http://127.0.0.1:8000", lambda: True, **kwargs)
                get.assert_not_called()


class ImportIsolationTest(unittest.TestCase):
    def assertImportRestoresCache(self):
        names = [*_stub_modules(), "vime.backends.vllm_utils.vllm_engine"]
        missing = object()
        before = {name: sys.modules.get(name, missing) for name in names}
        loaded = runpy.run_path(__file__, run_name="health_probe_isolation_check")
        self.assertTrue(callable(loaded["_wait_server_healthy"]))
        for name, original in before.items():
            with self.subTest(module=name):
                self.assertIs(sys.modules.get(name, missing), original)

    def test_import_does_not_leave_new_stubs_in_module_cache(self):
        with mock.patch.dict(sys.modules):
            for name in [*_stub_modules(), "vime.backends.vllm_utils.vllm_engine"]:
                sys.modules.pop(name, None)
            self.assertImportRestoresCache()

    def test_import_preserves_preexisting_modules(self):
        sentinels = {name: types.ModuleType(name) for name in _stub_modules()}
        with mock.patch.dict(sys.modules, sentinels):
            self.assertImportRestoresCache()


@contextmanager
def _loopback_server(*, send_headers=False):
    received, release = threading.Event(), threading.Event()

    class Handler(socketserver.BaseRequestHandler):
        def handle(self):
            self.request.settimeout(2)
            self.request.recv(4096)
            if send_headers:
                self.request.sendall(b"HTTP/1.1 200 OK\r\nContent-Length: 100\r\n\r\n")
            received.set()
            release.wait(3)

    with socketserver.TCPServer(("127.0.0.1", 0), Handler) as server:
        thread = threading.Thread(target=server.handle_request, daemon=True)
        thread.start()
        try:
            with mock.patch.dict(os.environ, {"NO_PROXY": "127.0.0.1", "no_proxy": "127.0.0.1"}):
                yield f"http://127.0.0.1:{server.server_address[1]}", received
        finally:
            release.set()
            thread.join(4)
            if thread.is_alive():
                raise AssertionError("loopback server thread did not stop")


class NetworkHealthProbeTest(unittest.TestCase):
    def test_real_silent_server_reaches_process_liveness_check(self):
        alive = mock.Mock(return_value=False)
        with _loopback_server() as (url, received):
            with self.assertRaisesRegex(Exception, "Server process terminated unexpectedly"):
                _wait_server_healthy(url, alive, probe_timeout=0.1)
            self.assertTrue(received.wait(1))
        alive.assert_called_once()

    def test_real_silent_server_is_bounded_by_total_wait_budget(self):
        with _loopback_server() as (url, received):
            started = time.monotonic()
            with self.assertRaises(TimeoutError):
                _wait_server_healthy(url, lambda: True, probe_timeout=5.0, wait_timeout=0.2)
            elapsed = time.monotonic() - started
            self.assertTrue(received.wait(1))
        self.assertLess(elapsed, 2.0)

    def test_ready_headers_do_not_wait_for_a_stalled_response_body(self):
        with _loopback_server(send_headers=True) as (url, received):
            _wait_server_healthy(url, lambda: False, probe_timeout=0.1, wait_timeout=1.0)
            self.assertTrue(received.wait(1))


if __name__ == "__main__":
    unittest.main()
