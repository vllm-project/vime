import json
import struct

import pytest

from vime.utils.disk_delta import make_tensor_reader


def test_missing_safetensors(tmp_path):
    with pytest.raises(FileNotFoundError, match=str(tmp_path)):
        make_tensor_reader(str(tmp_path))


@pytest.mark.parametrize(
    "content",
    [b"short", struct.pack("<Q", 100) + b"{}", struct.pack("<Q", 1) + b"{", struct.pack("<Q", 2) + b"[]"],
)
def test_invalid_header_names_file(tmp_path, content):
    path = tmp_path / "bad.safetensors"
    path.write_bytes(content)
    with pytest.raises(RuntimeError, match="bad.safetensors"):
        make_tensor_reader(str(tmp_path))


@pytest.mark.parametrize("offsets", [None, [0], [0, 1, 2], [-1, 1], [2, 1], [0, 5], [False, 1], [0, "1"]])
def test_invalid_offsets_name_tensor_and_file(tmp_path, offsets):
    path = tmp_path / "bad.safetensors"
    header = json.dumps({"weight": {"data_offsets": offsets}}).encode()
    path.write_bytes(struct.pack("<Q", len(header)) + header + b"1234")
    with pytest.raises(RuntimeError, match="tensor 'weight'.*bad.safetensors"):
        make_tensor_reader(str(tmp_path))


def test_reader_detects_truncation_after_indexing(tmp_path):
    path = tmp_path / "weight.safetensors"
    header = json.dumps({"weight": {"data_offsets": [0, 4]}}).encode()
    content = struct.pack("<Q", len(header)) + header + b"1234"
    path.write_bytes(content)
    reader = make_tensor_reader(str(tmp_path))
    assert reader("weight").tobytes() == b"1234"
    path.write_bytes(content[:-1])
    with pytest.raises(RuntimeError, match="Truncated tensor 'weight'.*weight.safetensors"):
        reader("weight")
