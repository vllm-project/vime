from __future__ import annotations

import glob
import json
import os
import struct
import zlib

import numpy as np


NUM_WORKERS = min(32, os.cpu_count() or 8)


def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """Encode changed byte positions and their replacement values."""
    positions = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate(
        [np.array([positions.size], "<u4").view(np.uint8), positions.view(np.uint8), new[changed_mask]]
    )


class _Adler32:
    def __init__(self) -> None:
        self._value = 1

    def update(self, data) -> None:
        self._value = zlib.adler32(data, self._value)

    def hexdigest(self) -> str:
        return f"{self._value:08x}"


def _new_hasher(algorithm: str):
    if algorithm == "xxh3-128":
        import xxhash

        return xxhash.xxh3_128()
    if algorithm == "blake3":
        import blake3

        return blake3.blake3()
    if algorithm == "adler32":
        return _Adler32()
    raise KeyError(f"Unknown checksum algorithm {algorithm!r}")


def checksum(algorithm: str, buffer) -> str:
    hasher = _new_hasher(algorithm)
    hasher.update(buffer)
    return hasher.hexdigest()


def _tensor_locations(checkpoint_dir: str) -> dict[str, tuple[str, int, int]]:
    locations: dict[str, tuple[str, int, int]] = {}
    for path in glob.glob(os.path.join(checkpoint_dir, "*.safetensors")):
        with open(path, "rb") as tensor_file:
            (header_len,) = struct.unpack("<Q", tensor_file.read(8))
            header = json.loads(tensor_file.read(header_len))
        for name, info in header.items():
            if name == "__metadata__":
                continue
            begin, end = info["data_offsets"]
            locations[name] = (path, 8 + header_len + begin, end - begin)
    return locations


def make_tensor_reader(checkpoint_dir: str):
    """Return a direct byte reader for tensors in a safetensors checkpoint."""
    locations = _tensor_locations(checkpoint_dir)

    def read(name: str) -> np.ndarray:
        path, offset, nbytes = locations[name]
        with open(path, "rb") as tensor_file:
            tensor_file.seek(offset)
            return np.frombuffer(tensor_file.read(nbytes), dtype=np.uint8)

    return read
