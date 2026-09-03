"""Remote Docker sandbox — drop-in replacement for E2BSandbox.

Connects to a remote Docker daemon over SSH (no TCP port needed).
Implements the same interface as vime.agent.sandbox.Sandbox so
sandbox.py needs zero changes other than the import.

Usage in sandbox.py — replace:
    from vime.agent.sandbox import E2BSandbox, Sandbox
with:
    from .docker_sandbox import DockerSandbox as E2BSandbox, Sandbox

Environment variables:
    DOCKER_SANDBOX_HOST   SSH target, e.g. root@192.168.13.188  (REQUIRED)
    DOCKER_SANDBOX_MEM    container memory limit, default 8g
    DOCKER_SANDBOX_CPUS   container CPU quota,   default 4
    DOCKER_CONTAINER_TIMEOUT  default exec timeout seconds, default 120
"""

from __future__ import annotations

import asyncio
import io
import logging
import os
import tarfile
import tempfile
from pathlib import Path
from typing import Any

import docker
import docker.errors

logger = logging.getLogger(__name__)

_DOCKER_HOST = os.environ.get("DOCKER_SANDBOX_HOST", "root@192.168.13.188")
_MEM_LIMIT    = os.environ.get("DOCKER_SANDBOX_MEM",  "8g")
_CPUS         = float(os.environ.get("DOCKER_SANDBOX_CPUS", "4"))
_DEFAULT_TIMEOUT = int(os.environ.get("DOCKER_CONTAINER_TIMEOUT", "120"))

# Module-level shared client — one SSH connection, reused across sandboxes.
# _client: docker.DockerClient | None = None
# _client_lock = asyncio.Lock()


def _make_client() -> docker.DockerClient:
    """每个调用新建一个 DockerClient，避免多进程/多线程 SSH channel 竞争。"""
    return docker.DockerClient(
        base_url=f"ssh://{_DOCKER_HOST}",
        max_pool_size=1,
    )
# ---------------------------------------------------------------------------
# Minimal Sandbox base (mirrors vime.agent.sandbox.Sandbox interface)
# ---------------------------------------------------------------------------
class Sandbox:
    """Abstract interface — matches vime.agent.sandbox.Sandbox."""

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        check: bool = True,
        timeout: int = _DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        raise NotImplementedError

    async def write_file(
        self,
        sandbox_path: str,
        content_or_host_path: str | bytes | Path,
        user: str = "root",
    ) -> None:
        raise NotImplementedError

    async def read_file(self, sandbox_path: str, user: str = "root") -> str:
        raise NotImplementedError


# ---------------------------------------------------------------------------
# DockerSandbox
# ---------------------------------------------------------------------------
class DockerSandbox(Sandbox):
    def __init__(self, image: str) -> None:
        self.image = image
        self._container = None
        self._client = None  # 实例级，不共享

    async def __aenter__(self) -> "DockerSandbox":
        loop = asyncio.get_event_loop()
        docker_tarball_dir = os.environ.get("DOCKER_TARBALL_DIR")
        def _start():
            self._client = _make_client()  # 每个沙箱独立连接
            vime_head_host = os.environ.get("VIME_HEAD_HOST", "")
            no_proxy=os.environ.get("no_proxy", f"127.0.0.1,localhost,{vime_head_host}")
            NO_PROXY=os.environ.get("NO_PROXY", f"127.0.0.1,localhost,{vime_head_host}")
            http_proxy=os.environ.get("http_proxy", "")
            https_proxy=os.environ.get("https_proxy", "")
            return self._client.containers.run(
                self.image,
                command="sleep infinity",
                detach=True,
                mem_limit=_MEM_LIMIT,
                nano_cpus=int(_CPUS * 1e9),
                network_mode="bridge",
                cap_add=["SYS_PTRACE"],
                remove=False,
                volumes={
                    docker_tarball_dir: {
                        "bind": docker_tarball_dir, 
                        "mode": "rw"
                    }
                },
                environment={
                    "http_proxy": http_proxy,
                    "https_proxy": https_proxy,
                    "no_proxy": no_proxy,
                    "NO_PROXY": NO_PROXY,
                },
            )

        self._container = await loop.run_in_executor(None, _start)
        await self.exec("mkdir -p /workspace", user="root", check=False)
        return self

    async def __aexit__(self, *args):
        if self._container is None:
            return
        loop = asyncio.get_event_loop()
        container = self._container
        client = self._client

        def _stop():
            try:
                container.remove(force=True)
            except Exception:
                pass
            try:
                client.close()
            except Exception:
                pass

        await loop.run_in_executor(None, _stop)
        self._container = None
        self._client = None

    # ------------------------------------------------------------------
    # exec
    # ------------------------------------------------------------------
    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        check: bool = True,
        timeout: int = _DEFAULT_TIMEOUT,
        env: dict[str, str] | None = None,
    ) -> tuple[int, str, str]:
        loop = asyncio.get_event_loop()

        def _run() -> tuple[int, str, str]:
            result = self._container.exec_run(
                ["bash", "-c", cmd],
                user=user,
                environment=env or {},
                demux=True,
                tty=False,
            )
            exit_code = result.exit_code
            stdout_b, stderr_b = result.output or (b"", b"")
            stdout = (stdout_b or b"").decode("utf-8", errors="replace")
            stderr = (stderr_b or b"").decode("utf-8", errors="replace")
            return exit_code, stdout, stderr

        try:
            exit_code, stdout, stderr = await asyncio.wait_for(
                loop.run_in_executor(None, _run),
                timeout=timeout,
            )
        except asyncio.TimeoutError:
            logger.warning("[docker_sandbox] exec timeout (%ds): %s", timeout, cmd[:120])
            if check:
                raise
            return -1, "", f"timeout after {timeout}s"

        if check and exit_code != 0:
            raise RuntimeError(
                f"[docker_sandbox] exec failed (exit {exit_code})\n"
                f"cmd: {cmd[:200]}\nstderr: {stderr[:500]}"
            )
        return exit_code, stdout, stderr

    # ------------------------------------------------------------------
    # write_file
    # ------------------------------------------------------------------
    async def write_file(
        self,
        sandbox_path: str,
        content_or_host_path: str | bytes | Path,
        user: str = "root",
    ) -> None:
        loop = asyncio.get_event_loop()
        sandbox_path = str(sandbox_path)

        # Resolve content bytes
        p_str = str(content_or_host_path)
        if isinstance(content_or_host_path, Path):
            data = content_or_host_path.read_bytes()
        elif isinstance(content_or_host_path, bytes):
            data = content_or_host_path
        else:
            data = p_str.encode("utf-8")

        filename = os.path.basename(sandbox_path)
        dirpath  = os.path.dirname(sandbox_path) or "/"

        # Ensure parent directory exists
        await self.exec(f"mkdir -p {dirpath}", user="root", check=False)

        # Pack into tar and send via put_archive
        def _put() -> None:
            buf = io.BytesIO()
            with tarfile.open(fileobj=buf, mode="w") as tf:
                info = tarfile.TarInfo(name=filename)
                info.size = len(data)
                tf.addfile(info, io.BytesIO(data))
            buf.seek(0)
            self._container.put_archive(dirpath, buf)

        await loop.run_in_executor(None, _put)

        # Fix ownership if needed
        if user and user != "root":
            await self.exec(f"chown {user}:{user} {sandbox_path}", user="root", check=False)

    # ------------------------------------------------------------------
    # read_file
    # ------------------------------------------------------------------
    async def read_file(self, sandbox_path: str, user: str = "root") -> str:
        loop = asyncio.get_event_loop()

        def _get() -> str:
            bits, _ = self._container.get_archive(sandbox_path)
            buf = io.BytesIO(b"".join(bits))
            with tarfile.open(fileobj=buf) as tf:
                member = tf.getmembers()[0]
                f = tf.extractfile(member)
                return f.read().decode("utf-8", errors="replace") if f else ""

        return await loop.run_in_executor(None, _get)


# ---------------------------------------------------------------------------
# Quick smoke-test (run directly: python docker_sandbox.py)
# ---------------------------------------------------------------------------
async def _smoke_test() -> None:
    image = os.environ.get("DOCKER_SANDBOX_TEST_IMAGE", "ubuntu:22.04")
    print(f"[smoke] connecting to {_DOCKER_HOST}, image={image}")
    async with DockerSandbox(image) as sb:
        # exec
        ec, out, err = await sb.exec("echo hello && uname -m", user="root")
        print(f"[smoke] exec exit={ec} stdout={out.strip()!r}")
        assert ec == 0 and "hello" in out

        # write_file (string content)
        await sb.write_file("/tmp/test.txt", "hello docker\n", user="root")

        # read_file
        content = await sb.read_file("/tmp/test.txt")
        print(f"[smoke] read_file: {content!r}")
        assert "hello docker" in content

        # write_file (host path)
        with tempfile.NamedTemporaryFile(suffix=".txt", delete=False) as f:
            f.write(b"from host\n")
            host_path = f.name
        await sb.write_file("/tmp/from_host.txt", Path(host_path), user="root")
        content2 = await sb.read_file("/tmp/from_host.txt")
        print(f"[smoke] host-path write: {content2!r}")
        assert "from host" in content2

    print("[smoke] ALL PASSED")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(_smoke_test())