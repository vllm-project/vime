from __future__ import annotations

import asyncio
import json
import os
import secrets
import shlex
import time
from pathlib import Path

from vime.agent.sandbox import ExecResult, FileContent


class LocalDockerSandbox:
    def __init__(self, image: str, **_kwargs) -> None:
        self.image = image
        self.sandbox_id = f"vime-agent-{secrets.token_hex(6)}"

    async def __aenter__(self):
        await self._run(
            "docker",
            "run",
            "--detach",
            "--rm",
            "--network",
            "host",
            "--name",
            self.sandbox_id,
            self.image,
            "sleep",
            "infinity",
            check=True,
        )
        self._trace("sandbox_start", image=self.image)
        return self

    async def __aexit__(self, _exc_type, _exc, _tb) -> None:
        self._trace("sandbox_stop")
        await self._run("docker", "rm", "--force", self.sandbox_id)

    async def exec(
        self,
        cmd: str,
        *,
        user: str = "root",
        env: dict[str, str] | None = None,
        timeout: int = 120,
        check: bool = False,
        idempotent: bool = True,
    ) -> ExecResult:
        del idempotent
        argv = ["docker", "exec", "--user", user]
        for key, value in (env or {}).items():
            argv.extend(("--env", f"{key}={value}"))
        argv.extend((self.sandbox_id, "bash", "-lc", cmd))
        result = await asyncio.wait_for(self._run(*argv, check=check), timeout=timeout)
        self._trace(
            "exec",
            user=user,
            cmd=cmd,
            returncode=result[0],
            stdout=result[1],
            stderr=result[2],
        )
        return result

    async def write_file(self, sandbox_path: str, content: FileContent, *, user: str = "root") -> None:
        data = content.read_bytes() if isinstance(content, Path) else content.encode() if isinstance(content, str) else content
        process = await asyncio.create_subprocess_exec(
            "docker",
            "exec",
            "--interactive",
            "--user",
            "root",
            self.sandbox_id,
            "bash",
            "-lc",
            f"cat > {shlex.quote(sandbox_path)}",
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await process.communicate(data)
        if process.returncode != 0:
            raise RuntimeError(f"file upload failed ({process.returncode}): {stderr.decode(errors='replace')}")
        if user != "root":
            await self.exec(f"chown {shlex.quote(user)} {shlex.quote(sandbox_path)}", check=True)
        self._trace("write_file", user=user, path=sandbox_path, size=len(data))

    async def read_file(self, sandbox_path: str, *, user: str = "root") -> str:
        _, stdout, _ = await self.exec(f"cat {sandbox_path}", user=user, check=True)
        return stdout

    def _trace(self, event: str, **payload) -> None:
        trace_dir = os.environ.get("VIME_LOCAL_SANDBOX_TRACE_DIR")
        if not trace_dir:
            return
        path = Path(trace_dir)
        path.mkdir(parents=True, exist_ok=True)
        record = {"time": time.time(), "sandbox_id": self.sandbox_id, "event": event, **payload}
        with (path / "sandbox-events.jsonl").open("a") as output:
            output.write(json.dumps(record, default=str) + "\n")

    @staticmethod
    async def _run(*argv: str, check: bool = False) -> ExecResult:
        process = await asyncio.create_subprocess_exec(
            *argv,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await process.communicate()
        result = process.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
        if check and process.returncode != 0:
            raise RuntimeError(f"command failed ({process.returncode}): {' '.join(argv)}\n{result[2]}")
        return result
