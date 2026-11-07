import copy
import json
from dataclasses import asdict, replace

import pytest

from vime.utils.draft_feature_contract import (
    DraftFeatureManifest,
    DraftSequence,
    build_token_map,
    manifest_from_dict,
    normalize_weight_versions,
)

pytestmark = pytest.mark.unit


def manifest():
    sequences = (
        DraftSequence(7, 3, 2, (10, 11, 12, 13), (0, 1, 1, 0), ("v2",)),
        DraftSequence(4, 3, 2, (20, 21, 22), (1, 0, 0), ()),
    )
    return DraftFeatureManifest(
        1,
        "job/2/0",
        "job",
        2,
        "v2",
        "v2",
        "actor",
        "config",
        "tokenizer",
        "lm_head_input",
        "float32",
        build_token_map(sequences, 3),
        "head.safetensors",
        "batch.safetensors",
        36,
        True,
        3,
    )


def test_json_round_trip_preserves_nested_types_and_token_coordinates():
    original = manifest()
    serialized = json.dumps(asdict(original), sort_keys=True)
    decoded = manifest_from_dict(json.loads(serialized))
    assert decoded == original
    assert json.dumps(asdict(decoded), sort_keys=True) == serialized
    assert [
        (t.sample_id, t.token_position, t.packed_position, t.target_token_id)
        for t in decoded.token_map.selected_tokens
    ] == [(7, 1, 1, 12), (7, 2, 2, 13), (4, 0, 4, 21)]


@pytest.mark.parametrize("value", [None, [], ()])
def test_unknown_versions_are_not_invented(value):
    assert normalize_weight_versions(value) == ()


def test_all_policy_versions_are_preserved():
    assert normalize_weight_versions(["v2", "v3", "v2"]) == ("v2", "v3", "v2")


@pytest.mark.parametrize("value", ["12", 12, [12], ["v2", None], [""]])
def test_invalid_versions_are_rejected(value):
    with pytest.raises(ValueError, match="weight_versions"):
        normalize_weight_versions(value)


@pytest.mark.parametrize(
    "field,value",
    [
        ("schema_version", 2),
        ("schema_version", True),
        ("round_id", True),
        ("head_source_version", "v3"),
        ("ready", False),
        ("ready", 1),
        ("dtype", "float64"),
        ("byte_count", 35),
        ("byte_count", True),
        ("hidden_size", 0),
        ("tp", True),
        ("dp", 2),
        ("capture_point", "decoder"),
        ("payload_ref", "../batch.safetensors"),
    ],
)
def test_tampered_manifest_is_rejected(field, value):
    payload = asdict(manifest())
    payload[field] = value
    with pytest.raises(ValueError):
        manifest_from_dict(payload)


@pytest.mark.parametrize("field,value", [("packed_position", 99), ("target_token_id", 10), ("sample_id", True)])
def test_tampered_causal_target_is_rejected(field, value):
    payload = json.loads(json.dumps(asdict(manifest())))
    payload["token_map"]["selected_tokens"][0][field] = value
    with pytest.raises(ValueError):
        manifest_from_dict(payload)


def test_offsets_and_duplicate_samples_are_rejected():
    payload = json.loads(json.dumps(asdict(manifest())))
    payload["token_map"]["sequence_offsets"][1] = 3
    with pytest.raises(ValueError, match="offset"):
        manifest_from_dict(payload)
    payload = json.loads(json.dumps(asdict(manifest())))
    payload["token_map"]["sequences"].append(copy.deepcopy(payload["token_map"]["sequences"][0]))
    with pytest.raises(ValueError):
        manifest_from_dict(payload)


def test_empty_mask_cannot_publish_ready_features():
    sequence = replace(manifest().token_map.sequences[0], loss_mask=(0, 0, 0, 0))
    with pytest.raises(ValueError, match="no selected"):
        build_token_map((sequence,), 10)


def test_bool_tokens_and_masks_are_not_integer_indices():
    sequence = manifest().token_map.sequences[0]
    with pytest.raises(ValueError):
        replace(sequence, token_ids=(True, 11, 12, 13))
    with pytest.raises(ValueError):
        replace(sequence, loss_mask=(0, True, 1, 0))
