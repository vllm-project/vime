"""Unit tests for the Modal sandbox backend (vime.agent.sandbox.ModalSandbox)
and the coding_agent_rl make_sandbox() factory.

No network and no real ``modal`` dependency: a fully faked ``modal`` module is
injected into ``sys.modules`` so ``import modal`` inside ModalSandbox picks it
up. The fakes record every SDK call (App.lookup / Image.from_registry /
Secret.from_dict / Sandbox.create / exec / stdin / wait / terminate) so we can
assert the exact wiring: image-from-registry, runuser user emulation, env
whitelist forwarding, stdin file streaming, chown-after-write, output capping,
timeout contract, and always-terminate cleanup.

Run: python -m pytest tests/test_modal_sandbox.py -v   (no pytest-asyncio needed)
"""

from __future__ import annotations

import asyncio
import functools
import sys
import types
from pathlib import Path

import pytest

# Make the dev workspace importable (vime/ and examples/ live one level up).
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


# ---------------------------------------------------------------------------
# async test plumbing (no pytest-asyncio)
# ---------------------------------------------------------------------------
def aiotest(afn):
    @functools.wraps(afn)
    def wrapper(*args, **kwargs):
        return asyncio.run(afn(*args, **kwargs))

    return wrapper


# ---------------------------------------------------------------------------
# Fake modal SDK
# ---------------------------------------------------------------------------
class _Aio:
    """Wrap an async fn so ``obj.aio(*a, **k)`` returns its coroutine, matching
    Modal's synchronicity surface (e.g. ``await sb.exec.aio(...)``)."""

    def __init__(self, afn):
        self._afn = afn

    def aio(self, *a, **k):
        return self._afn(*a, **k)


class FakeStream:
    def __init__(self, getter):
        self._getter = getter

    @property
    def read(self):
        getter = self._getter

        async def _read():
            return getter()

        return _Aio(_read)


class FakeStdin:
    def __init__(self, sb):
        self._sb = sb

    def write(self, data):
        self._sb.stdin_writes.append(data)

    def write_eof(self):
        self._sb.stdin_eof = True

    @property
    def drain(self):
        async def _drain():
            return None

        return _Aio(_drain)


class FakeProc:
    def __init__(self, sb):
        self._sb = sb
        self.stdin = FakeStdin(sb)
        self.stdout = FakeStream(lambda: sb.next_stdout)
        self.stderr = FakeStream(lambda: sb.next_stderr)

    @property
    def wait(self):
        sb = self._sb

        async def _wait():
            if sb.raise_timeout:
                raise sys.modules["modal"].exception.SandboxTimeoutError("simulated timeout")
            return sb.next_rc

        return _Aio(_wait)


class FakeSandbox:
    def __init__(self, **create_kwargs):
        self.create_kwargs = create_kwargs
        self.object_id = "sb-fake-0001"
        self.exec_calls = []  # list of (argv_tuple, kwargs_dict)
        self.stdin_writes = []
        self.stdin_eof = False
        self.next_stdout = "OUT"
        self.next_stderr = "ERR"
        self.next_rc = 0
        self.raise_timeout = False
        self.terminated = False

    @property
    def exec(self):
        sb = self

        async def _exec(*argv, **kwargs):
            sb.exec_calls.append((argv, kwargs))
            return FakeProc(sb)

        return _Aio(_exec)

    @property
    def terminate(self):
        sb = self

        async def _terminate():
            sb.terminated = True

        return _Aio(_terminate)


def _make_fake_modal():
    mod = types.ModuleType("modal")

    exc = types.ModuleType("modal.exception")

    class SandboxTimeoutError(Exception):
        pass

    class NotFoundError(Exception):
        pass

    exc.SandboxTimeoutError = SandboxTimeoutError
    exc.NotFoundError = NotFoundError
    mod.exception = exc

    # App.lookup.aio(name, *, create_if_missing=...)
    class _App:
        def __init__(self, name, create_if_missing):
            self.name = name
            self.create_if_missing = create_if_missing

    class AppNS:
        lookups = []

    async def _app_lookup(name, *, create_if_missing=False, **_):
        a = _App(name, create_if_missing)
        AppNS.lookups.append(a)
        return a

    AppNS.lookup = _Aio(_app_lookup)
    mod.App = AppNS

    # Image.from_registry(tag, secret=None)  -- synchronous
    class _Image:
        def __init__(self, tag, secret):
            self.tag = tag
            self.secret = secret

    class ImageNS:
        calls = []

    def _from_registry(tag, secret=None, **_):
        img = _Image(tag, secret)
        ImageNS.calls.append((tag, secret))
        return img

    ImageNS.from_registry = staticmethod(_from_registry)
    mod.Image = ImageNS

    # Secret.from_dict(d)  -- synchronous
    class _Secret:
        def __init__(self, d):
            self.d = d

    class SecretNS:
        created = []

    def _from_dict(d):
        SecretNS.created.append(d)
        return _Secret(d)

    SecretNS.from_dict = staticmethod(_from_dict)
    mod.Secret = SecretNS

    # Sandbox.create.aio(**kwargs)
    class SandboxNS:
        created = []

    async def _create(**kwargs):
        sb = FakeSandbox(**kwargs)
        SandboxNS.created.append(sb)
        return sb

    SandboxNS.create = _Aio(_create)
    mod.Sandbox = SandboxNS

    return mod


@pytest.fixture(autouse=True)
def fake_modal(monkeypatch):
    """Install a fresh fake ``modal`` and isolate all backend env knobs."""
    for var in (
        "VIME_AGENT_SANDBOX_BACKEND",
        "VIME_AGENT_SANDBOX_MODAL_APP",
        "SWE_SANDBOX_MODAL_APP",
        "VIME_AGENT_SANDBOX_MODAL_CPU",
        "SWE_SANDBOX_MODAL_CPU",
        "VIME_AGENT_SANDBOX_MODAL_MEMORY_MB",
        "SWE_SANDBOX_MODAL_MEMORY_MB",
        "VIME_AGENT_SANDBOX_MODAL_BLOCK_NETWORK",
        "SWE_SANDBOX_MODAL_BLOCK_NETWORK",
        "VIME_AGENT_SANDBOX_MODAL_MAX_OUTPUT_BYTES",
        "SWE_SANDBOX_MODAL_MAX_OUTPUT_BYTES",
        "VIME_AGENT_SANDBOX_LIFETIME_SEC",
        "SWE_SANDBOX_LIFETIME_SEC",
        "DOCKER_USERNAME",
        "DOCKER_PASSWORD",
    ):
        monkeypatch.delenv(var, raising=False)
    mod = _make_fake_modal()
    monkeypatch.setitem(sys.modules, "modal", mod)
    monkeypatch.setitem(sys.modules, "modal.exception", mod.exception)
    return mod


def _import_sandbox_module():
    import vime.agent.sandbox as s

    return s


async def _entered(image="ghcr.io/x/img:latest", **kwargs):
    s = _import_sandbox_module()
    ms = s.ModalSandbox(image, **kwargs)
    await ms.__aenter__()
    return ms


# ---------------------------------------------------------------------------
# Protocol + construction
# ---------------------------------------------------------------------------
def test_protocol_conformance():
    s = _import_sandbox_module()
    ms = s.ModalSandbox("img:latest")
    assert isinstance(ms, s.Sandbox)  # runtime_checkable Protocol
    assert ms.sandbox_id == ""


def test_init_defaults():
    s = _import_sandbox_module()
    ms = s.ModalSandbox("img:latest")
    assert ms.timeout == 3600
    assert ms.app_name == "vime-coding-agent-sandbox"
    assert ms.cpu == 2.0
    assert ms.memory_mb == 8192
    assert ms.block_network is False
    assert ms.max_output_bytes == 0


def test_init_env_overrides(monkeypatch):
    monkeypatch.setenv("VIME_AGENT_SANDBOX_MODAL_APP", "my-app")
    monkeypatch.setenv("VIME_AGENT_SANDBOX_MODAL_CPU", "4")
    monkeypatch.setenv("VIME_AGENT_SANDBOX_MODAL_MEMORY_MB", "16384")
    monkeypatch.setenv("VIME_AGENT_SANDBOX_MODAL_BLOCK_NETWORK", "true")
    monkeypatch.setenv("VIME_AGENT_SANDBOX_MODAL_MAX_OUTPUT_BYTES", "1024")
    monkeypatch.setenv("VIME_AGENT_SANDBOX_LIFETIME_SEC", "900")
    s = _import_sandbox_module()
    ms = s.ModalSandbox("img:latest")
    assert ms.app_name == "my-app"
    assert ms.cpu == 4.0
    assert ms.memory_mb == 16384
    assert ms.block_network is True
    assert ms.max_output_bytes == 1024
    assert ms.timeout == 900


def test_init_kwarg_overrides_win_over_env(monkeypatch):
    monkeypatch.setenv("VIME_AGENT_SANDBOX_MODAL_CPU", "4")
    s = _import_sandbox_module()
    ms = s.ModalSandbox("img:latest", cpu=8.0, memory_mb=2048, timeout=120, block_network=True, app_name="kw")
    assert ms.cpu == 8.0
    assert ms.memory_mb == 2048
    assert ms.timeout == 120
    assert ms.block_network is True
    assert ms.app_name == "kw"


def test_swe_fallback_env(monkeypatch):
    monkeypatch.setenv("SWE_SANDBOX_MODAL_CPU", "3")
    monkeypatch.setenv("SWE_SANDBOX_LIFETIME_SEC", "1234")
    s = _import_sandbox_module()
    ms = s.ModalSandbox("img:latest")
    assert ms.cpu == 3.0
    assert ms.timeout == 1234


# ---------------------------------------------------------------------------
# __aenter__ / image / app / create
# ---------------------------------------------------------------------------
@aiotest
async def test_aenter_creates_and_sets_id(fake_modal):
    ms = await _entered(image="ghcr.io/epoch-research/swe-bench.eval.x86_64.foo:latest", cpu=2.0, memory_mb=8192)
    # App.lookup(create_if_missing=True)
    assert len(fake_modal.App.lookups) == 1
    assert fake_modal.App.lookups[0].create_if_missing is True
    # Image.from_registry(tag)
    assert fake_modal.Image.calls[-1][0] == "ghcr.io/epoch-research/swe-bench.eval.x86_64.foo:latest"
    # Sandbox.create kwargs
    sb = fake_modal.Sandbox.created[-1]
    assert sb.create_kwargs["cpu"] == 2.0
    assert sb.create_kwargs["memory"] == 8192
    assert sb.create_kwargs["timeout"] == 3600
    assert sb.create_kwargs["block_network"] is False
    assert sb.create_kwargs["image"] is not None
    assert sb.create_kwargs["app"] is not None
    # sandbox_id reflects object_id
    assert ms.sandbox_id == "sb-fake-0001"


@aiotest
async def test_aenter_public_image_no_secret(fake_modal):
    await _entered()
    assert fake_modal.Secret.created == []  # no DOCKER_* -> no registry secret
    assert fake_modal.Image.calls[-1][1] is None  # secret arg None


@aiotest
async def test_aenter_private_registry_secret(fake_modal, monkeypatch):
    monkeypatch.setenv("DOCKER_USERNAME", "alice")
    monkeypatch.setenv("DOCKER_PASSWORD", "s3cr3t")
    await _entered()
    assert fake_modal.Secret.created, "expected a registry secret to be built"
    d = fake_modal.Secret.created[-1]
    assert d == {"REGISTRY_USERNAME": "alice", "REGISTRY_PASSWORD": "s3cr3t"}
    assert fake_modal.Image.calls[-1][1] is not None  # secret passed to from_registry


# ---------------------------------------------------------------------------
# exec: user emulation + env + return + check + cap + timeout
# ---------------------------------------------------------------------------
@aiotest
async def test_exec_root_argv_and_return(fake_modal):
    ms = await _entered()
    ms._sb.next_stdout, ms._sb.next_stderr, ms._sb.next_rc = "hello", "warn", 0
    rc, out, err = await ms.exec("echo hi")
    assert (rc, out, err) == (0, "hello", "warn")
    argv, kwargs = ms._sb.exec_calls[-1]
    assert argv == ("bash", "-lc", "echo hi")
    assert kwargs["env"] is None
    assert kwargs["timeout"] == 120


@aiotest
async def test_exec_root_with_env_passed(fake_modal):
    ms = await _entered()
    env = {"ANTHROPIC_BASE_URL": "http://x", "ANTHROPIC_AUTH_TOKEN": "tok"}
    await ms.exec("claude --version", env=env, timeout=30)
    argv, kwargs = ms._sb.exec_calls[-1]
    assert argv == ("bash", "-lc", "claude --version")
    assert kwargs["env"] == env
    assert kwargs["timeout"] == 30


@aiotest
async def test_exec_nonroot_wraps_with_runuser(fake_modal):
    ms = await _entered()
    await ms.exec("git diff", user="agent")
    argv, kwargs = ms._sb.exec_calls[-1]
    assert argv == ("runuser", "-u", "agent", "--", "bash", "-lc", "git diff")
    assert kwargs["env"] is None


@aiotest
async def test_exec_nonroot_with_env_whitelists_keys(fake_modal):
    ms = await _entered()
    env = {"A": "1", "B": "2"}
    await ms.exec("run", user="agent", env=env)
    argv, kwargs = ms._sb.exec_calls[-1]
    assert argv[0:3] == ("runuser", "-u", "agent")
    assert argv[3] == "--whitelist-environment=A,B"
    assert argv[4:] == ("--", "bash", "-lc", "run")
    assert kwargs["env"] == env


@aiotest
async def test_exec_check_raises_on_nonzero(fake_modal):
    ms = await _entered()
    ms._sb.next_rc, ms._sb.next_stderr = 2, "boom"
    with pytest.raises(RuntimeError) as ei:
        await ms.exec("false", check=True)
    assert "exit=2" in str(ei.value)


@aiotest
async def test_exec_check_ok_on_zero(fake_modal):
    ms = await _entered()
    ms._sb.next_rc = 0
    rc, _, _ = await ms.exec("true", check=True)
    assert rc == 0


@aiotest
async def test_exec_no_check_returns_nonzero(fake_modal):
    ms = await _entered()
    ms._sb.next_rc = 7
    rc, _, _ = await ms.exec("maybe-fails", check=False)
    assert rc == 7


@aiotest
async def test_exec_output_capping(fake_modal):
    ms = await _entered(max_output_bytes=10)
    ms._sb.next_stdout = "x" * 50
    _, out, _ = await ms.exec("spew")
    assert out.startswith("x" * 10)
    assert "truncated" in out


@aiotest
async def test_exec_no_cap_by_default(fake_modal):
    ms = await _entered()  # max_output_bytes == 0 -> unlimited (mirror E2B)
    ms._sb.next_stdout = "y" * 5000
    _, out, _ = await ms.exec("spew")
    assert out == "y" * 5000


@aiotest
async def test_exec_timeout_noncheck_returns_sentinel(fake_modal):
    ms = await _entered()
    ms._sb.raise_timeout = True
    rc, out, err = await ms.exec("sleep 999", timeout=1, check=False)
    assert rc == -1
    assert out == ""
    assert "timeout" in err


@aiotest
async def test_exec_timeout_check_raises(fake_modal):
    ms = await _entered()
    ms._sb.raise_timeout = True
    with pytest.raises(RuntimeError) as ei:
        await ms.exec("sleep 999", timeout=1, check=True)
    assert "timed out" in str(ei.value)


# Modal 1.4.2 does NOT raise on a per-exec timeout; wait() returns rc == -1
# (verified on real Modal). These cover that real path, not just the raise path.
@aiotest
async def test_exec_rc_minus1_noncheck_returns_timeout(fake_modal):
    ms = await _entered()
    ms._sb.next_rc = -1
    ms._sb.next_stdout = ""
    ms._sb.next_stderr = ""  # real Modal timeout yields empty streams + rc=-1
    rc, out, err = await ms.exec("sleep 999", timeout=1, check=False)
    assert rc == -1
    assert "timeout" in err or "killed" in err


@aiotest
async def test_exec_rc_minus1_check_raises_as_timeout(fake_modal):
    ms = await _entered()
    ms._sb.next_rc = -1
    with pytest.raises(RuntimeError) as ei:
        await ms.exec("sleep 999", timeout=1, check=True)
    msg = str(ei.value)
    assert "timed out" in msg or "killed" in msg


# ---------------------------------------------------------------------------
# write_file: str / bytes / host Path, mkdir+cat, eof, chown
# ---------------------------------------------------------------------------
@aiotest
async def test_write_file_str_root_no_chown(fake_modal):
    ms = await _entered()
    await ms.write_file("/workspace/PROBLEM_STATEMENT.md", "hello world")
    # exactly one exec (the create); no chown because user == root
    assert len(ms._sb.exec_calls) == 1
    argv, kwargs = ms._sb.exec_calls[0]
    assert argv[0:2] == ("bash", "-lc")
    assert "mkdir -p" in argv[2] and "cat >" in argv[2]
    assert "/workspace/PROBLEM_STATEMENT.md" in argv[2]
    assert kwargs["text"] is False  # binary stdin
    assert b"".join(ms._sb.stdin_writes) == b"hello world"
    assert ms._sb.stdin_eof is True


@aiotest
async def test_write_file_bytes(fake_modal):
    ms = await _entered()
    await ms.write_file("/tmp/blob.bin", b"\x00\x01\x02\xff")
    assert b"".join(ms._sb.stdin_writes) == b"\x00\x01\x02\xff"
    assert ms._sb.stdin_eof is True


@aiotest
async def test_write_file_host_path_streams_chunks(fake_modal, tmp_path):
    # 5 MiB host file -> multiple 2 MiB stdin chunks streamed verbatim.
    payload = bytes(range(256)) * (5 * 1024 * 1024 // 256)
    host = tmp_path / "node22.tar"
    host.write_bytes(payload)
    ms = await _entered()
    await ms.write_file("/tmp/node22.tar", host)
    assert len(ms._sb.stdin_writes) >= 2  # streamed in chunks, not one shot
    assert b"".join(ms._sb.stdin_writes) == payload
    assert ms._sb.stdin_eof is True


@aiotest
async def test_write_file_chowns_for_nonroot_user(fake_modal):
    ms = await _entered()
    await ms.write_file("/home/agent/.cagent_run.sh", "#!/bin/bash\n", user="agent")
    # create exec + chown exec
    assert len(ms._sb.exec_calls) == 2
    chown_argv, _ = ms._sb.exec_calls[1]
    assert chown_argv == ("bash", "-lc", "chown agent:agent /home/agent/.cagent_run.sh")


@aiotest
async def test_write_file_nonzero_raises_oserror(fake_modal):
    ms = await _entered()
    ms._sb.next_rc = 1  # the cat command "fails"
    with pytest.raises(OSError):
        await ms.write_file("/root/denied", "x")


# ---------------------------------------------------------------------------
# read_file
# ---------------------------------------------------------------------------
@aiotest
async def test_read_file_success(fake_modal):
    ms = await _entered()
    ms._sb.next_rc, ms._sb.next_stdout = 0, '{"tests": []}'
    out = await ms.read_file("/workspace/swepro_eval/result.json", user="agent")
    assert out == '{"tests": []}'
    argv, _ = ms._sb.exec_calls[-1]
    # read runs as the requested user via runuser, cat-ing the path
    assert argv[0:3] == ("runuser", "-u", "agent")
    assert "cat --" in argv[-1]


@aiotest
async def test_read_file_failure_returns_empty(fake_modal):
    ms = await _entered()
    ms._sb.next_rc = 1  # cat fails (missing file)
    out = await ms.read_file("/nope")
    assert out == ""


# ---------------------------------------------------------------------------
# __aexit__ cleanup (always terminate)
# ---------------------------------------------------------------------------
@aiotest
async def test_aexit_terminates(fake_modal):
    ms = await _entered()
    sb = ms._sb
    await ms.__aexit__(None, None, None)
    assert sb.terminated is True
    assert ms._sb is None


@aiotest
async def test_aexit_safe_without_sandbox(fake_modal):
    s = _import_sandbox_module()
    ms = s.ModalSandbox("img:latest")  # never entered
    await ms.__aexit__(None, None, None)  # must not raise


@aiotest
async def test_async_context_manager_roundtrip(fake_modal):
    s = _import_sandbox_module()
    async with s.ModalSandbox("img:latest") as ms:
        assert ms.sandbox_id == "sb-fake-0001"
        sb = ms._sb
    assert sb.terminated is True


# ---------------------------------------------------------------------------
# make_sandbox() factory
# ---------------------------------------------------------------------------
def _import_factory():
    import examples.coding_agent_rl.sandbox as cas

    return cas


def test_factory_default_is_e2b():
    cas = _import_factory()
    s = _import_sandbox_module()
    assert isinstance(cas.make_sandbox("img:latest"), s.E2BSandbox)


def test_factory_explicit_e2b(monkeypatch):
    monkeypatch.setenv("VIME_AGENT_SANDBOX_BACKEND", "e2b")
    cas = _import_factory()
    s = _import_sandbox_module()
    assert isinstance(cas.make_sandbox("img:latest"), s.E2BSandbox)


def test_factory_modal(monkeypatch):
    monkeypatch.setenv("VIME_AGENT_SANDBOX_BACKEND", "modal")
    cas = _import_factory()
    s = _import_sandbox_module()
    assert isinstance(cas.make_sandbox("img:latest"), s.ModalSandbox)


def test_factory_modal_case_insensitive(monkeypatch):
    monkeypatch.setenv("VIME_AGENT_SANDBOX_BACKEND", "  MODAL ")
    cas = _import_factory()
    s = _import_sandbox_module()
    assert isinstance(cas.make_sandbox("img:latest"), s.ModalSandbox)


def test_factory_unknown_raises(monkeypatch):
    monkeypatch.setenv("VIME_AGENT_SANDBOX_BACKEND", "podman")
    cas = _import_factory()
    with pytest.raises(ValueError):
        cas.make_sandbox("img:latest")
