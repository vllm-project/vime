"""Convert an ai21-verl AmalgamDataset mix spec into a single vime-ready JSONL.

ai21-verl trained from a "mix json" describing several sources (each a weight + a data file/dir
in the ai21-evaluators VerifiableTask format). vime loads one ``--prompt-data`` file, so this
flattens the mix into one JSONL whose ``metadata`` column carries exactly what
``vime_plugins.rm.ai21_evaluators.ai21_reward`` and ``vime_plugins.filters.snoozing`` read.

It reproduces the per-row derivations from ai21-verl's SingleSourceDataset._read_files_and_tokenize:
- ``evaluations`` (or ``reward_metrics``) -> ``metadata.reward_model``
- synthesized stable ``metadata.id`` = ``ai21/<source>_<row-index>`` (+ ``_<row id>`` if present)
- ``metadata.aggregation_config`` from the source override, else the row's, else {"expression": "mean"}
- ``metadata.extra_info`` from the row's ``metadata`` column, else {"split", "index"}
- per-source ``chat_template_kwargs.force_thinking`` -> ``metadata.force_thinking``
- ``tools`` / ``jlm_model_name`` carried through when present
- integer/fractional source ``weight`` -> row replication (shared with merge_amalgam_sources)

Mix spec (JSON), per ai21-verl::

    {
      "math_short":  {"path": "math.jsonl",  "weight": 1.0,
                      "aggregation_config": {"expression": "mean"},
                      "chat_template_kwargs": {"force_thinking": false}},
      "coding_short": {"path": "coding/",    "weight": 2.0}
    }

Remote paths are read via fsspec. ``mammoth://`` is just the GCS bucket ``gs://ai21-mammoth-storage``,
so the simplest route is to map ``mammoth://`` -> ``gs://ai21-mammoth-storage/`` and let ``gcsfs``
(in requirements_ai21.txt) resolve it — no custom protocol backend needed, only GCS auth. The
``--path-prefix-map`` is applied to ``--mix`` and to the source paths inside it.

Usage (recommended — mammoth via GCS)::

    python -m vime_plugins.data.amalgam_to_jsonl \
        --mix mammoth://users/asafk/verl-data-configs/regression_short_train.json \
        --out /root/regression_short_train.jsonl \
        --input-key prompt --metadata-key metadata \
        --path-prefix-map mammoth://=gs://ai21-mammoth-storage/

If a real ``mammoth://`` fsspec backend is installed instead, pass it via ``--register-import
<module>`` and drop the prefix map. For already-downloaded data, map to a local dir, e.g.
``--path-prefix-map mammoth://users/asafk/=/root/data/``.
"""

import argparse
import json
import os
import random

from vime_plugins.data.merge_amalgam_sources import _weighted_rows

DEFAULT_AGGREGATION_CONFIG = {"expression": "mean"}
_PROMPT_FALLBACK_KEYS = ("text", "messages", "prompt")


def is_remote(path: str) -> bool:
    return "://" in path and not path.startswith("file://")


def register_protocol_imports(modules: list[str]) -> None:
    """Import modules that register custom fsspec protocols (e.g. ``mammoth://``) as a side effect.

    Some backends register via ``fsspec.register_implementation`` at import time rather than via
    an entry point, so the protocol is unknown until the providing package is imported. ai21-verl
    worked because an AI21 package was already imported in-process; this converter imports only
    fsspec, so pass the registering module(s) via ``--register-import``.
    """
    import importlib

    for module in modules or []:
        importlib.import_module(module)


def _fsspec():
    try:
        import fsspec
    except ImportError as e:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "Reading remote paths (e.g. mammoth://) needs fsspec and the protocol's backend. "
            "Install requirements_ai21.txt, or download the files locally and use --path-prefix-map."
        ) from e
    return fsspec


def _resolve_path(path: str, prefix_map: dict[str, str]) -> str:
    for prefix, local in prefix_map.items():
        if path.startswith(prefix):
            return local + path[len(prefix) :]
    return path


def _load_json_any(path: str):
    if is_remote(path):
        with _fsspec().open(path, "r") as f:
            return json.load(f)
    with open(path) as f:
        return json.load(f)


def _iter_jsonl_text(handle) -> list[dict]:
    rows = []
    for line in handle:
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _read_remote_rows(path: str) -> list[dict]:
    fsspec = _fsspec()
    # A single file (…/x.jsonl|.json) vs. a directory of *.jsonl shards.
    if path.endswith((".jsonl", ".json")):
        files = [path]
    else:
        fs, _ = fsspec.core.url_to_fs(path)
        files = sorted(fs.glob(path.rstrip("/") + "/*.jsonl"))
        if not files:
            raise FileNotFoundError(f"No .jsonl files under {path}")
    rows: list[dict] = []
    for fp in files:
        with fsspec.open(fp, "r") as f:
            rows.extend(_iter_jsonl_text(f))
    return rows


def _read_rows(path: str) -> list[dict]:
    if is_remote(path):
        return _read_remote_rows(path)
    files = [path]
    if os.path.isdir(path):
        files = sorted(os.path.join(path, f) for f in os.listdir(path) if f.endswith(".jsonl"))
    rows: list[dict] = []
    for fp in files:
        with open(fp) as f:
            rows.extend(_iter_jsonl_text(f))
    return rows


def _row_to_record(row, source_name, index, source_cfg, input_key, metadata_key) -> dict:
    evaluations = row.get("evaluations")
    if evaluations is None:
        evaluations = row.get("reward_metrics")
    if evaluations is None:
        raise ValueError(
            f"{source_name}[{index}]: missing 'evaluations'/'reward_metrics' (the ai21-evaluators "
            "EvaluationEntry list). This source is not in the expected VerifiableTask format."
        )

    rid = f"ai21/{source_name}_{index}"
    if row.get("id") is not None:
        rid = f"{rid}_{row['id']}"

    aggregation = source_cfg.get("aggregation_config") or row.get("aggregation_config") or DEFAULT_AGGREGATION_CONFIG
    extra_info = row["metadata"] if row.get("metadata") is not None else {"split": row.get("split"), "index": index}
    force_thinking = bool(source_cfg.get("chat_template_kwargs", {}).get("force_thinking", False))

    prompt = row.get(input_key)
    if prompt is None:
        for key in _PROMPT_FALLBACK_KEYS:
            if key in row:
                prompt = row[key]
                break
    if prompt is None:
        raise ValueError(f"{source_name}[{index}]: no prompt column ({input_key} or one of {_PROMPT_FALLBACK_KEYS}).")

    metadata = {
        "id": rid,
        "reward_model": evaluations,
        "aggregation_config": aggregation,
        "extra_info": extra_info,
        "data_source": f"ai21/{source_name}",
        "force_thinking": force_thinking,
        # Keep the raw chat messages so the ai21_evaluators reward can build VerifiableTask.messages.
        # When the run uses --apply-chat-template, sample.prompt becomes a rendered string, so the
        # plugin's metadata["messages"] fallback (sample.prompt) would otherwise hand it a str.
        # Normalize a plain-string prompt the same way vime's _build_messages does (wrap as a
        # single user turn) so VerifiableTask.messages is always a list of message dicts.
        "messages": prompt if isinstance(prompt, list) else [{"role": "user", "content": prompt}],
    }
    if row.get("jlm_model_name") is not None:
        metadata["jlm_model_name"] = row["jlm_model_name"]
    if row.get("tools"):
        metadata["tools"] = row["tools"]

    return {input_key: prompt, metadata_key: metadata}


def convert_mix(
    mix: dict,
    *,
    input_key: str = "prompt",
    metadata_key: str = "metadata",
    prefix_map: dict[str, str] | None = None,
    seed: int = 1,
    shuffle: bool = True,
) -> list[dict]:
    prefix_map = prefix_map or {}
    rng = random.Random(seed)
    merged: list[dict] = []

    for source_name, source_cfg in mix.items():
        path = _resolve_path(source_cfg["path"], prefix_map)
        weight = float(source_cfg.get("weight", 1.0))
        rows = _read_rows(path)
        records = [
            _row_to_record(row, source_name, i, source_cfg, input_key, metadata_key) for i, row in enumerate(rows)
        ]
        merged.extend(_weighted_rows(records, weight, rng))

    if shuffle:
        rng.shuffle(merged)
    return merged


def _parse_prefix_map(items: list[str]) -> dict[str, str]:
    out = {}
    for item in items or []:
        if "=" not in item:
            raise ValueError(f"--path-prefix-map expects '<prefix>=<local>', got '{item}'")
        prefix, local = item.split("=", 1)
        out[prefix] = local
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert an ai21-verl Amalgam mix spec into a vime JSONL.")
    parser.add_argument("--mix", required=True, help="Path to the Amalgam mix-spec JSON.")
    parser.add_argument("--out", required=True, help="Output JSONL path.")
    parser.add_argument("--input-key", default="prompt", help="Prompt column to write (matches vime --input-key).")
    parser.add_argument("--metadata-key", default="metadata", help="Metadata column (matches vime --metadata-key).")
    parser.add_argument(
        "--path-prefix-map",
        action="append",
        default=[],
        help="Optional: rewrite a remote prefix to a local dir for already-downloaded data, "
        "e.g. mammoth://users/asafk/=/root/data/ (repeatable).",
    )
    parser.add_argument(
        "--register-import",
        action="append",
        default=[],
        help="Import this module before any remote IO so it can register its fsspec protocol "
        "(e.g. the package that provides mammoth://). Repeatable.",
    )
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no-shuffle", action="store_true")
    args = parser.parse_args()

    register_protocol_imports(args.register_import)
    prefix_map = _parse_prefix_map(args.path_prefix_map)

    # Apply the prefix map to --mix too (not just the source paths inside it), so one
    # --path-prefix-map mammoth://=gs://ai21-mammoth-storage/ covers the mix and its sources.
    mix = _load_json_any(_resolve_path(args.mix, prefix_map))

    records = convert_mix(
        mix,
        input_key=args.input_key,
        metadata_key=args.metadata_key,
        prefix_map=prefix_map,
        seed=args.seed,
        shuffle=not args.no_shuffle,
    )

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")

    print(f"Wrote {len(records)} rows from {len(mix)} sources to {args.out}")


if __name__ == "__main__":
    main()
