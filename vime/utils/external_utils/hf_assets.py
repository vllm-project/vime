"""Download public test models with a bounded fallback for mirror/Xet failures."""

import json
import os
import shlex
import struct
import subprocess
import time
from pathlib import Path


def validate_model_files(directory: str | Path) -> None:
    directory = Path(directory)
    if not (directory / "config.json").is_file():
        raise ValueError(f"Downloaded model is missing config.json: {directory}")
    index = directory / "model.safetensors.index.json"
    if index.is_file():
        weight_map = json.loads(index.read_text())["weight_map"]
        if not weight_map:
            raise ValueError(f"Empty model weight index: {index}")
        filenames = set(weight_map.values())
    else:
        filenames = {"model.safetensors"}
    for filename in filenames:
        path = directory / filename
        if not path.is_file() or path.stat().st_size < 8:
            raise ValueError(f"Downloaded model shard is missing or incomplete: {path}")
        with path.open("rb") as file:
            header_size = struct.unpack("<Q", file.read(8))[0]
            if header_size > min(path.stat().st_size - 8, 100_000_000):
                raise ValueError(f"Invalid safetensors header: {path}")
            header = json.loads(file.read(header_size))
        offsets = [item["data_offsets"] for name, item in header.items() if name != "__metadata__"]
        if not offsets or max(end for _, end in offsets) + 8 + header_size != path.stat().st_size:
            raise ValueError(f"Downloaded model shard has an incomplete payload: {path}")


def download_public_model(repo: str, directory: str | Path, *, revision: str | None = None) -> None:
    command = ["hf", "download", repo, "--local-dir", str(directory)]
    if revision is not None:
        command += ["--revision", revision]
    original = os.environ.copy()
    fallback = {
        **original,
        "HF_ENDPOINT": "https://huggingface.co",
        "HF_HUB_DISABLE_XET": "1",
        "HF_HUB_DISABLE_IMPLICIT_TOKEN": "1",
    }
    # Each CLI process reads Hub settings afresh. Reuse completed files and the
    # CLI's per-file locks; never wipe the persistent asset cache on a retry.
    for attempt, env in enumerate((original, fallback, fallback), start=1):
        endpoint = env.get("HF_ENDPOINT", "https://huggingface.co")
        print(f"Public model download {attempt}/3: endpoint={endpoint}; {shlex.join(command)}", flush=True)
        try:
            subprocess.run(command, env=env, check=True)
        except subprocess.CalledProcessError:
            if attempt == 3:
                raise
            time.sleep(2 * attempt)
        else:
            validate_model_files(directory)
            return
