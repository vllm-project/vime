"""Start mooncake_master for local smoke tests."""

from __future__ import annotations

import shutil
import signal
import socket
import subprocess
import time


def ensure_mooncake_master(host: str = "127.0.0.1", rpc_port: int = 50051, metadata_port: int = 18080) -> None:
    if _port_open(host, rpc_port):
        return
    mooncake_master = shutil.which("mooncake_master")
    if mooncake_master is None:
        raise RuntimeError("mooncake_master not found; pip install mooncake-transfer-engine")
    proc = subprocess.Popen(
        [
            mooncake_master,
            "--enable_http_metadata_server=true",
            f"--http_metadata_server_host={host}",
            f"--http_metadata_server_port={metadata_port}",
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    deadline = time.time() + 15.0
    while time.time() < deadline:
        if proc.poll() is not None:
            raise RuntimeError("mooncake_master exited early")
        if _port_open(host, rpc_port) and _port_open(host, metadata_port):
            return
        time.sleep(0.2)
    proc.send_signal(signal.SIGTERM)
    raise RuntimeError("mooncake_master did not become ready")


def _port_open(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.5)
        return sock.connect_ex((host, port)) == 0
