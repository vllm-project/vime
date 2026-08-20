from __future__ import annotations

import numpy as np
import safetensors.numpy

from vime.utils.disk_delta import checksum, make_tensor_reader, overwrite_encode


def test_overwrite_encode_contains_changed_positions_and_values():
    old = np.array([1, 2, 3, 4], dtype=np.uint8)
    new = np.array([1, 9, 3, 8], dtype=np.uint8)

    encoded = overwrite_encode(new, new != old)

    count = int.from_bytes(encoded[:4].tobytes(), "little")
    positions = np.frombuffer(encoded[4 : 4 + count * 4].tobytes(), dtype="<u4")
    assert count == 2
    np.testing.assert_array_equal(positions, [1, 3])
    np.testing.assert_array_equal(encoded[4 + count * 4 :], [9, 8])


def test_checksum_algorithms_are_deterministic():
    payload = np.arange(32, dtype=np.uint8)
    for algorithm in ("adler32", "blake3", "xxh3-128"):
        assert checksum(algorithm, payload) == checksum(algorithm, payload)


def test_tensor_reader_reads_safetensor_bytes(tmp_path):
    weight = np.arange(12, dtype=np.float32).reshape(3, 4)
    safetensors.numpy.save_file({"weight": weight}, tmp_path / "model.safetensors")

    actual = make_tensor_reader(str(tmp_path))("weight")

    np.testing.assert_array_equal(actual, weight.view(np.uint8).reshape(-1))
