"""Offline multi-source data merge (the vime-native replacement for AmalgamDataset).

vime loads a single prompt file via ``--prompt-data`` and has no seam to swap the data-source
class, so multi-source mixing is done offline: this CLI reads a "mix" spec describing several
source files (each with a weight and optional per-source metadata) and writes ONE merged JSONL
that vime consumes directly. Per-example metadata is preserved, so:

- ``--custom-rm-path vime_plugins.rm.ai21_evaluators.ai21_reward`` picks up the per-row
  ai21-evaluators VerifiableTask spec from ``metadata`` (reward_model, aggregation_config, id, ...).
- the built-in ``async_rm`` dispatch picks up ``metadata["rm_type"]`` for per-source reward routing.
- ``vime_plugins.filters.snoozing`` uses ``metadata["id"]`` (or ``--snooze-id-key``) as the
  stable prompt id.

This re-homes ai21-verl's AmalgamDataset weighting (verl/ai21/datasets/amalgam.py ``__iter__``):
integer weight N => row replicated N times; fractional part p => row included with probability p.

Mix spec (JSON), e.g.::

    {
      "math":   {"path": "math.jsonl",   "weight": 2,   "rm_type": "math"},
      "coding": {"path": "coding.jsonl", "weight": 1.5, "rm_type": "ai21_evaluators"}
    }

Per-source keys: ``path`` (required), ``weight`` (default 1.0), plus any other keys (e.g.
``rm_type``, ``jlm_model_name``) that are merged into each row's metadata column.

Usage::

    python -m vime_plugins.data.merge_amalgam_sources \
        --mix mix.json --out merged.jsonl \
        --input-key text --label-key label --metadata-key metadata --seed 1
"""

import argparse
import json
import os
import random


def _read_jsonl(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _weighted_rows(rows: list[dict], weight: float, rng: random.Random) -> list[dict]:
    """Replicate by the integer part of ``weight``; include once more with prob = fractional part."""
    n_int = int(weight)
    out = rows * n_int
    frac = weight - n_int
    if frac > 0:
        out += [row for row in rows if rng.random() < frac]
    return out


def merge_sources(
    mix: dict,
    *,
    input_key: str = "text",
    label_key: str | None = "label",
    metadata_key: str = "metadata",
    id_key: str = "id",
    seed: int = 1,
    shuffle: bool = True,
) -> list[dict]:
    """Merge the sources described by ``mix`` into a single list of output rows."""
    rng = random.Random(seed)
    merged: list[dict] = []

    for source_name, source_cfg in mix.items():
        path = source_cfg["path"]
        weight = float(source_cfg.get("weight", 1.0))
        # any per-source key other than path/weight becomes per-row metadata (e.g. rm_type).
        extra_meta = {k: v for k, v in source_cfg.items() if k not in ("path", "weight")}

        source_rows: list[dict] = []
        for i, row in enumerate(_read_jsonl(path)):
            metadata = dict(row.get(metadata_key) or {})
            metadata.setdefault("data_source", source_name)
            # stable per-prompt id for snoozing/curriculum if the row didn't carry one.
            if id_key not in metadata:
                metadata[id_key] = f"{source_name}:{i}"
            for k, v in extra_meta.items():
                metadata.setdefault(k, v)

            out_row = {input_key: row[input_key], metadata_key: metadata}
            if label_key is not None and label_key in row:
                out_row[label_key] = row[label_key]
            source_rows.append(out_row)

        merged.extend(_weighted_rows(source_rows, weight, rng))

    if shuffle:
        rng.shuffle(merged)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge multiple prompt sources into one JSONL for vime.")
    parser.add_argument("--mix", required=True, help="Path to the mix-spec JSON file.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--input-key", default="text", help="Prompt column (matches vime --input-key).")
    parser.add_argument("--label-key", default="label", help="Label column (matches vime --label-key); '' to omit.")
    parser.add_argument("--metadata-key", default="metadata", help="Metadata column (matches vime --metadata-key).")
    parser.add_argument("--id-key", default="id", help="Per-prompt id key inside metadata (for snoozing).")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    with open(args.mix) as f:
        mix = json.load(f)

    merged = merge_sources(
        mix,
        input_key=args.input_key,
        label_key=args.label_key or None,
        metadata_key=args.metadata_key,
        id_key=args.id_key,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for row in merged:
            f.write(json.dumps(row) + "\n")

    print(f"Wrote {len(merged)} rows from {len(mix)} sources to {args.out}")


if __name__ == "__main__":
    main()
