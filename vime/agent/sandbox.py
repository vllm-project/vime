"""Sandbox backends for agent rollouts.

The public sandbox contract is intentionally small: async context management,
command execution, and file read/write. Agent examples can build task-specific
setup, runner, and evaluator logic on top of this without depending directly on
one sandbox provider.
"""

from __future__ import annotations

import asyncio
import io
import json
import logging
import os
import shlex
from pathlib import Path
from typing import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


ExecResult = tuple[int, str, str]
FileContent = str | bytes | Path


@runtime_checkable
class Sandbox(Protocol):
    """Minimal async sandbox interface used by agent rollouts.

    ``write_file`` accepts either in-memory content (``str``/``bytes``) or a
    host ``Path`` to stream into the sandbox.
    """

    sandbox_id: str

    async def __aenter__(self) -> Sandbox: ...

    async def __aexit__(self, exc_type, exc, tb) -> None: ...

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult: ...

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None: ...

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str: ...


def _getenv(*names: str, default: str = "") -> str:
    for name in names:
        value = os.environ.get(name)
        if value is not None and value.strip():
            return value
    return default


class E2BSandbox:
    """Async context manager around e2b.AsyncSandbox."""

    metadata_file_env = ("VIME_AGENT_SANDBOX_METADATA_FILE", "SWE_SANDBOX_METADATA_FILE")
    metadata_json_env = ("VIME_AGENT_SANDBOX_METADATA_JSON", "SWE_SANDBOX_METADATA_JSON")
    image_metadata_key_env = ("VIME_AGENT_SANDBOX_IMAGE_METADATA_KEY", "SWE_SANDBOX_IMAGE_METADATA_KEY")
    lifetime_sec_env = ("VIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    rpc_retries_env = ("VIME_AGENT_SANDBOX_RPC_RETRIES", "SWE_RPC_RETRIES")

    default_lifetime_sec = 3600
    default_rpc_retries = 3
    # With retries=3 the sleep budget is 3s, which handles common E2B h2 reset
    # / SSL / pool-timeout flaps without stalling rollout steps for too long.
    rpc_backoff_base_sec = 1.0

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        metadata: dict[str, str] | None = None,
        image_metadata_key: str | None = None,
        rpc_retries: int | None = None,
    ) -> None:
        self.image = image
        self.timeout = timeout if timeout is not None else self._lifetime_sec_from_env()
        self.metadata = dict(metadata) if metadata is not None else self._metadata_from_env()
        self.image_metadata_key = image_metadata_key or self._image_metadata_key_from_env()
        self.rpc_retries = rpc_retries if rpc_retries is not None else self._rpc_retries_from_env()
        self._sb = None
        self.sandbox_id = ""

    @classmethod
    def _metadata_from_env(cls) -> dict[str, str]:
        """Read E2B routing metadata from file or JSON environment values."""
        file_path = _getenv(*cls.metadata_file_env)
        raw = ""
        if file_path:
            try:
                raw = Path(file_path).read_text()
            except OSError as e:
                logger.warning("[agent.sandbox] metadata file %s unreadable: %s", file_path, e)
                raw = ""
        if not raw:
            raw = _getenv(*cls.metadata_json_env)
        if not raw:
            return {}
        try:
            md = json.loads(raw)
        except json.JSONDecodeError as e:
            logger.warning("[agent.sandbox] metadata not valid JSON, ignoring: %s", e)
            return {}
        if not isinstance(md, dict):
            logger.warning("[agent.sandbox] metadata must be a JSON object, got %s", type(md).__name__)
            return {}
        return {str(k): str(v) for k, v in md.items()}

    @classmethod
    def _image_metadata_key_from_env(cls) -> str | None:
        return _getenv(*cls.image_metadata_key_env) or None

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @classmethod
    def _rpc_retries_from_env(cls) -> int:
        return int(_getenv(*cls.rpc_retries_env, default=str(cls.default_rpc_retries)))

    @staticmethod
    def _is_transient_rpc_error(e: BaseException) -> bool:
        """True if e is a transient E2B client-side failure safe to retry."""
        name = type(e).__name__
        if name in {
            "ProtocolError",
            "LocalProtocolError",
            "WriteError",
            "ReadError",
            "ConnectError",
            "ConnectTimeout",
            "ReadTimeout",
            "WriteTimeout",
            "PoolTimeout",
            "RemoteProtocolError",
            "SSLError",
        }:
            return True
        msg = str(e)
        if name == "SandboxException":
            if "does not exist" in msg or "STOPPED state" in msg:
                return False
            return True
        return False

    async def _rpc_retry(self, op_name: str, coro_factory):
        """Run coro_factory() with retries for transient E2B RPC failures."""
        last_err = None
        for attempt in range(self.rpc_retries):
            try:
                return await coro_factory()
            except Exception as e:
                if not self._is_transient_rpc_error(e):
                    raise
                last_err = e
                if attempt + 1 < self.rpc_retries:
                    backoff = self.rpc_backoff_base_sec * (2**attempt)
                    logger.debug(
                        "[agent.sandbox] %s transient %s, retry %d/%d in %.1fs: %s",
                        op_name,
                        type(e).__name__,
                        attempt + 1,
                        self.rpc_retries,
                        backoff,
                        str(e)[:120],
                    )
                    await asyncio.sleep(backoff)
        assert last_err is not None
        raise last_err

    async def __aenter__(self) -> E2BSandbox:
        if self.image_metadata_key is None:
            raise RuntimeError(
                "VIME_AGENT_SANDBOX_IMAGE_METADATA_KEY is not set. Export it "
                "to the metadata key your E2B gateway uses for image routing. "
                "The legacy SWE_SANDBOX_IMAGE_METADATA_KEY name is also "
                "accepted for coding-agent examples."
            )
        from e2b import AsyncSandbox  # type: ignore

        md = dict(self.metadata)
        md.setdefault(self.image_metadata_key, self.image)
        self._sb = await AsyncSandbox.create(timeout=self.timeout, metadata=md)
        self.sandbox_id = self._sb.sandbox_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        try:
            if self._sb is not None:
                await self._sb.kill()
        except Exception as e:
            logger.warning("[agent.sandbox] kill %s failed: %s", self.sandbox_id[:8], e)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        from e2b.sandbox.commands.command_handle import CommandExitException

        try:
            res = await self._rpc_retry(
                f"exec({cmd[:60]!r})",
                lambda: self._sb.commands.run(
                    cmd,
                    user=user,
                    envs=env,
                    timeout=timeout,
                    on_stdout=lambda s: None,
                    on_stderr=lambda s: None,
                ),
            )
            return res.exit_code, res.stdout or "", res.stderr or ""
        except CommandExitException as e:
            if check:
                raise RuntimeError(
                    f"e2b exec failed (exit={e.exit_code}): {cmd[:120]}\n{(e.stderr or '')[:400]}"
                ) from None
            return e.exit_code, e.stdout or "", e.stderr or ""

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        if isinstance(content, Path):
            host_path = content

            async def _do_path():
                with open(host_path, "rb") as fp:
                    await self._sb.files.write(
                        sandbox_path,
                        fp,
                        user=user,
                        gzip=False,
                        use_octet_stream=True,
                        request_timeout=600,
                    )

            await self._rpc_retry(f"write_file({sandbox_path} <- {host_path.name})", _do_path)
            return

        if isinstance(content, bytes):

            async def _do_bytes():
                await self._sb.files.write(
                    sandbox_path,
                    io.BytesIO(content),
                    user=user,
                    gzip=False,
                    use_octet_stream=True,
                    request_timeout=600,
                )

            await self._rpc_retry(f"write_file({sandbox_path}, bytes={len(content)})", _do_bytes)
            return

        await self._rpc_retry(
            f"write_file({sandbox_path})",
            lambda: self._sb.files.write(sandbox_path, content, user=user),
        )

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        try:
            return await self._rpc_retry(
                f"read_file({sandbox_path})",
                lambda: self._sb.files.read(sandbox_path, user=user),
            )
        except Exception:
            return ""


# ---------------------------------------------------------------------------
# Modal backend
# ---------------------------------------------------------------------------
# A second concrete Sandbox provider (alongside E2BSandbox) backed by
# ``modal.Sandbox``. Selected at runtime by the coding-agent example's
# ``make_sandbox()`` factory via ``VIME_AGENT_SANDBOX_BACKEND=modal``; the E2B
# path is untouched when this backend is not selected.
#
# Reverse-network note: coding_agent_rl runs Claude Code *inside* the sandbox,
# which dials back to the in-process Anthropic adapter on the training head
# (``ANTHROPIC_BASE_URL=adapter_url``). Modal sandboxes have outbound internet
# when ``block_network=False`` (the default here), so they can reach the head as
# long as the adapter URL is publicly routable. On a private cluster, expose the
# adapter's ``SHIM_PORT`` through a tunnel (e.g. cloudflared) and point
# ``SLIME_HEAD_HOST`` / ``ADAPTER_URL_OVERRIDE`` at it. (Verified end-to-end:
# a Modal sandbox successfully dialed back through a cloudflared tunnel.)
#
# Gap map vs E2B:
#   * Modal exec has no ``user=`` -> emulate with ``runuser -u <user>``.
#   * ``write_file`` streams str/bytes/host-Path over command stdin, then chowns.
#   * Image comes from a registry tag (``Image.from_registry``), optionally with
#     ``REGISTRY_USERNAME``/``REGISTRY_PASSWORD`` from DOCKER_* env for private
#     registries.
#   * Cleanup always reaches ``terminate`` (a leaked Modal sandbox counts against
#     the account's concurrent-sandbox cap until its wall-clock timeout).

_modal_app_env = ("VIME_AGENT_SANDBOX_MODAL_APP", "SWE_SANDBOX_MODAL_APP")
_modal_cpu_env = ("VIME_AGENT_SANDBOX_MODAL_CPU", "SWE_SANDBOX_MODAL_CPU")
_modal_memory_mb_env = ("VIME_AGENT_SANDBOX_MODAL_MEMORY_MB", "SWE_SANDBOX_MODAL_MEMORY_MB")
_modal_block_network_env = ("VIME_AGENT_SANDBOX_MODAL_BLOCK_NETWORK", "SWE_SANDBOX_MODAL_BLOCK_NETWORK")
_modal_max_output_bytes_env = ("VIME_AGENT_SANDBOX_MODAL_MAX_OUTPUT_BYTES", "SWE_SANDBOX_MODAL_MAX_OUTPUT_BYTES")

_MODAL_WRITE_CHUNK = 2 * 1024 * 1024  # 2 MiB stdin chunks for host-Path uploads


def _truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


class ModalSandbox:
    """Async context manager around ``modal.Sandbox``.

    Mirrors :class:`E2BSandbox`'s public surface so the coding-agent example can
    swap backends without touching its bootstrap / runner / evaluator code: same
    ``__aenter__`` / ``__aexit__``, ``exec``, ``write_file``, ``read_file``, and
    ``sandbox_id``. ``modal`` is imported lazily so importing this module never
    requires the dependency unless the Modal backend is actually used.
    """

    # Shared lifetime env with E2BSandbox so run.sh sets one knob for both.
    lifetime_sec_env = ("VIME_AGENT_SANDBOX_LIFETIME_SEC", "SWE_SANDBOX_LIFETIME_SEC")
    default_lifetime_sec = 3600
    default_app_name = "vime-coding-agent-sandbox"
    default_cpu = 2.0
    default_memory_mb = 8192

    def __init__(
        self,
        image: str,
        *,
        timeout: int | None = None,
        app_name: str | None = None,
        cpu: float | None = None,
        memory_mb: int | None = None,
        block_network: bool | None = None,
        max_output_bytes: int | None = None,
    ) -> None:
        self.image = image
        self.timeout = int(timeout) if timeout is not None else self._lifetime_sec_from_env()
        self.app_name = app_name or _getenv(*_modal_app_env, default=self.default_app_name)
        self.cpu = float(cpu) if cpu is not None else float(_getenv(*_modal_cpu_env, default=str(self.default_cpu)))
        self.memory_mb = (
            int(memory_mb)
            if memory_mb is not None
            else int(_getenv(*_modal_memory_mb_env, default=str(self.default_memory_mb)))
        )
        if block_network is not None:
            self.block_network = bool(block_network)
        else:
            self.block_network = _truthy(_getenv(*_modal_block_network_env, default="0"))
        self.max_output_bytes = (
            int(max_output_bytes)
            if max_output_bytes is not None
            else int(_getenv(*_modal_max_output_bytes_env, default="0"))
        )
        self._sb = None
        self._app = None
        self.sandbox_id = ""

    @classmethod
    def _lifetime_sec_from_env(cls) -> int:
        return int(_getenv(*cls.lifetime_sec_env, default=str(cls.default_lifetime_sec)))

    @staticmethod
    def _registry_secret(modal):
        """Build a private-registry Secret from DOCKER_* env, or None for public."""
        user = os.environ.get("DOCKER_USERNAME")
        pw = os.environ.get("DOCKER_PASSWORD")
        if user and pw:
            return modal.Secret.from_dict({"REGISTRY_USERNAME": user, "REGISTRY_PASSWORD": pw})
        return None

    async def __aenter__(self) -> ModalSandbox:
        import modal

        self._app = await modal.App.lookup.aio(self.app_name, create_if_missing=True)
        secret = self._registry_secret(modal)
        image = (
            modal.Image.from_registry(self.image, secret=secret)
            if secret is not None
            else modal.Image.from_registry(self.image)
        )
        self._sb = await modal.Sandbox.create.aio(
            image=image,
            app=self._app,
            cpu=self.cpu,
            memory=self.memory_mb,
            timeout=self.timeout,
            block_network=self.block_network,
        )
        self.sandbox_id = self._sb.object_id
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        # Always reach terminate: a leaked sandbox counts against Modal's
        # concurrent-sandbox cap until its wall-clock timeout fires.
        if self._sb is not None:
            try:
                await self._sb.terminate.aio()
            except Exception as e:
                logger.warning("[agent.sandbox] modal terminate %s failed: %s", (self.sandbox_id or "")[:12], e)
            finally:
                self._sb = None

    @staticmethod
    def _build_argv(cmd: str, user: str, env: dict[str, str] | None) -> list[str]:
        """argv for running ``cmd`` as ``user``. Modal has no user= on exec, so
        non-root uses ``runuser``; env keys are whitelisted through so the target
        user's shell still sees them (matches the E2B ``envs=`` semantics)."""
        if user == "root":
            return ["bash", "-lc", cmd]
        argv = ["runuser", "-u", user]
        if env:
            argv.append("--whitelist-environment=" + ",".join(env.keys()))
        argv += ["--", "bash", "-lc", cmd]
        return argv

    def _cap(self, text: str | None) -> str:
        text = text or ""
        if self.max_output_bytes and len(text) > self.max_output_bytes:
            return text[: self.max_output_bytes] + f"\n<truncated; output exceeded {self.max_output_bytes} bytes>\n"
        return text

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
    ) -> ExecResult:
        import modal

        argv = self._build_argv(cmd, user, env)
        proc = await self._sb.exec.aio(*argv, timeout=timeout, env=env or None)
        try:
            # Gather reads with wait so a chatty stream can't backpressure-stall
            # the other before the process is reaped (matches Modal's own idiom).
            out, err, rc = await asyncio.gather(proc.stdout.read.aio(), proc.stderr.read.aio(), proc.wait.aio())
        except modal.exception.SandboxTimeoutError:
            # Defensive: most Modal paths signal a per-exec timeout via rc == -1
            # (handled below, no exception), but keep this for any that raise.
            if check:
                raise RuntimeError(f"modal exec timed out after {timeout}s: {cmd[:120]}") from None
            return -1, "", f"<modal exec timeout after {timeout}s>"
        rc = int(rc) if rc is not None else 0
        out, err = self._cap(out), self._cap(err)
        # Modal returns rc == -1 (no exception) when a per-exec timeout fires or
        # the process is killed. Surface it as a clear timeout rather than a
        # generic non-zero exit.
        if rc == -1:
            if check:
                raise RuntimeError(f"modal exec timed out / was killed after {timeout}s: {cmd[:120]}")
            return -1, out, err or f"<modal exec timeout/killed after {timeout}s>"
        if check and rc != 0:
            raise RuntimeError(f"modal exec failed (exit={rc}): {cmd[:120]}\n{err[:400]}")
        return rc, out, err

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        directory = os.path.dirname(sandbox_path) or "."
        create_cmd = f"mkdir -p {shlex.quote(directory)} && cat > {shlex.quote(sandbox_path)}"
        # text=False -> binary stdin so host Paths / bytes stream verbatim.
        proc = await self._sb.exec.aio("bash", "-lc", create_cmd, timeout=600, text=False)
        try:
            if isinstance(content, Path):
                with open(content, "rb") as fp:
                    while True:
                        chunk = fp.read(_MODAL_WRITE_CHUNK)
                        if not chunk:
                            break
                        proc.stdin.write(chunk)
                        await proc.stdin.drain.aio()
            else:
                data = content.encode() if isinstance(content, str) else content
                proc.stdin.write(data)
                await proc.stdin.drain.aio()
            proc.stdin.write_eof()
            await proc.stdin.drain.aio()
            # Drain stderr alongside wait so the write can't stall on a full
            # stderr pipe, and so the failure message is preserved.
            err_raw, rc = await asyncio.gather(proc.stderr.read.aio(), proc.wait.aio())
        except Exception as e:
            raise OSError(f"modal write_file({sandbox_path}) failed: {e!r}") from e
        if rc != 0:
            err_msg = (
                err_raw.decode("utf-8", "replace") if isinstance(err_raw, (bytes, bytearray)) else str(err_raw or "")
            )
            raise OSError(f"modal write_file({sandbox_path}) exit={rc}: {err_msg[:300]}")
        if user != "root":
            crc, _, cerr = await self.exec(
                f"chown {shlex.quote(user)}:{shlex.quote(user)} {shlex.quote(sandbox_path)}",
                user="root",
                check=False,
            )
            if crc != 0:
                logger.warning("[agent.sandbox] modal chown %s -> %s failed: %s", sandbox_path, user, cerr[:200])

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        rc, out, _ = await self.exec(f"cat -- {shlex.quote(sandbox_path)}", user=user, timeout=120, check=False)
        return out if rc == 0 else ""
