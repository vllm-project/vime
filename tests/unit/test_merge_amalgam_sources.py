"""Unit tests for the offline multi-source merge (vime_plugins.data.merge_amalgam_sources)."""

from __future__ import annotations

import json

import pytest

from vime_plugins.data import merge_amalgam_sources as mod


def _write_jsonl(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")


def test_merge_attaches_metadata_and_keeps_label(tmp_path):
    math = tmp_path / "math.jsonl"
    _write_jsonl(math, [{"text": "1+1?", "label": "2"}, {"text": "2+2?", "label": "4"}])
    mix = {"math": {"path": str(math), "weight": 1, "rm_type": "math"}}

    rows = mod.merge_sources(mix, input_key="text", label_key="label", metadata_key="metadata", seed=1, shuffle=False)

    assert len(rows) == 2
    for r in rows:
        assert r["metadata"]["data_source"] == "math"
        assert r["metadata"]["rm_type"] == "math"
        assert r["metadata"]["id"].startswith("math:")
        assert "label" in r


def test_integer_weight_replicates(tmp_path):
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, [{"text": "a"}, {"text": "b"}])
    mix = {"s": {"path": str(src), "weight": 3}}

    rows = mod.merge_sources(mix, label_key=None, seed=1, shuffle=False)
    assert len(rows) == 6  # 2 rows * weight 3


def test_existing_metadata_id_preserved(tmp_path):
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, [{"text": "a", "metadata": {"id": "orig-1", "reward_model": [{"evaluator_name": "x"}]}}])
    mix = {"s": {"path": str(src), "weight": 1}}

    rows = mod.merge_sources(mix, label_key=None, seed=1, shuffle=False)
    assert rows[0]["metadata"]["id"] == "orig-1"
    assert rows[0]["metadata"]["reward_model"] == [{"evaluator_name": "x"}]


def test_fractional_weight_probabilistic(tmp_path):
    src = tmp_path / "s.jsonl"
    _write_jsonl(src, [{"text": str(i)} for i in range(100)])
    mix = {"s": {"path": str(src), "weight": 1.5}}

    rows = mod.merge_sources(mix, label_key=None, seed=1, shuffle=False)
    # 100 (integer part) + ~50 (fractional 0.5 of 100); allow wide tolerance for RNG.
    assert 130 <= len(rows) <= 170


def test_multiple_sources_merged(tmp_path):
    a = tmp_path / "a.jsonl"
    b = tmp_path / "b.jsonl"
    _write_jsonl(a, [{"text": "a1"}])
    _write_jsonl(b, [{"text": "b1"}, {"text": "b2"}])
    mix = {"a": {"path": str(a), "weight": 1, "rm_type": "math"}, "b": {"path": str(b), "weight": 1, "rm_type": "f1"}}

    rows = mod.merge_sources(mix, label_key=None, seed=1, shuffle=False)
    sources = {r["metadata"]["data_source"] for r in rows}
    rm_types = {r["metadata"]["rm_type"] for r in rows}
    assert sources == {"a", "b"}
    assert rm_types == {"math", "f1"}
    assert len(rows) == 3


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
