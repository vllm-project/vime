#!/usr/bin/env python3
"""On/off benchmark for speculative decoding in RL rollout.

Runs the DSpark example (examples/dspark; another example with the same
SPEC_MODE contract via ``plan --example``, e.g. examples/mtp) in three
speculative modes and two placements, with the same dataset, seed and number of steps, and compares them:

- ``off``: no speculative decoding
- ``frozen``: the pretrained DSpark draft is served but never trained or synced
- ``online``: the draft is trained during RL and synced every step

``plan --drafts NAME=JSON`` adds an arm per placement that serves another
pretrained vLLM draft (for example DFlash2 or EAGLE3) without any DSpark
training: the example runs with ``SPEC_MODE=off`` and the JSON is passed to
train.py as ``--vllm-speculative-config``.

All arms train the policy with GRPO (``DSPARK_FREEZE_POLICY=0``). In the
``online`` arm the draft reads detached policy hidden states and logits, so its
loss does not reach the policy parameters, and its gradients are clipped by
their own norm rather than the policy's. No arm writes checkpoints
(``NO_SAVE=1``). Metrics come from each arm's TensorBoard log; an arm that ends
without TensorBoard events counts as failed.

    python tools/bench_spec_rollout.py plan --out /data/bench --num-rollout 20
    python tools/bench_spec_rollout.py run --out /data/bench          # or --dry-run
    python tools/bench_spec_rollout.py summarize --out /data/bench

``plan`` writes ``plan.json``; ``run`` executes the arms one after another
(each arm owns the whole node, as the example scripts start and stop Ray);
``summarize`` writes ``summary.json`` and ``summary.md``.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import subprocess
import sys
import time
from pathlib import Path

MODES = ("off", "frozen", "online")
DEFAULT_EXAMPLE = "examples/dspark/run-qwen3-4B-dspark-{layout}.sh"
LAYOUTS = ("colocate", "non-colocate")
REPO_ROOT = Path(__file__).resolve().parents[1]

# Per rollout step (TensorBoard step = rollout step).
STEP_TAGS = {
    "rollout_time": ["perf/rollout_time"],
    "step_time": ["perf/step_time"],
    "update_weights_time": ["perf/update_weights_time"],
    "update_draft_weights_time": ["perf/update_draft_weights_time"],
    "spec_accept_rate": ["rollout/spec_accept_rate"],
    "spec_accept_length": ["rollout/spec_accept_length/token_weighted", "rollout/spec_accept_length"],
    "tail_wall_share": ["perf/tail/last_10pct_wall_share"],
    "reward": ["rollout/raw_reward", "rollout/rewards"],
    "logprob_abs_diff": ["train/train_rollout_logprob_abs_diff"],
}


def parse_draft(text: str) -> tuple[str, dict]:
    """``NAME=JSON`` for ``plan --drafts``: a new arm name and a vLLM speculative config with a ``model``."""
    name, _, config = text.partition("=")
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]*", name) or name in MODES:
        raise argparse.ArgumentTypeError(f"draft arm name must be lowercase and not one of {MODES}: {text!r}")
    spec = json.loads(config)
    if not isinstance(spec, dict) or "model" not in spec:
        raise argparse.ArgumentTypeError(f"draft config must be a JSON object with a model: {text!r}")
    if re.search(r"\s", json.dumps(spec, separators=(",", ":"))):
        raise argparse.ArgumentTypeError(
            f"draft config must not contain whitespace (EXTRA_ARGS is word-split): {text!r}"
        )
    return name, spec


def plan(args) -> dict:
    out = Path(args.out).resolve()
    # (arm mode, SPEC_MODE for the example, extra train.py arguments)
    variants = [(mode, mode, []) for mode in args.modes] + [
        (name, "off", [f"--vllm-speculative-config {json.dumps(spec, separators=(',', ':'))}"])
        for name, spec in getattr(args, "drafts", [])
    ]
    arms = []
    for layout in args.layouts:
        for mode, spec_mode, spec_args in variants:
            name = f"{layout}-{mode}"
            arm_dir = out / name
            extra = [
                f"--seed {args.seed}",
                f"--rollout-seed {args.seed}",
                "--use-tensorboard",
            ]
            arms.append(
                {
                    "name": name,
                    "layout": layout,
                    "mode": mode,
                    "script": str(REPO_ROOT / args.example.format(layout=layout)),
                    "dir": str(arm_dir),
                    "env": {
                        "SPEC_MODE": spec_mode,
                        "DSPARK_FREEZE_POLICY": "0",
                        "NUM_ROLLOUT": str(args.num_rollout),
                        "NUM_GPUS": str(args.num_gpus),
                        # No evaluation (train.py would otherwise evaluate before step 0) and no checkpoints.
                        "EVAL_INTERVAL": "off",
                        "NO_SAVE": "1",
                        "TENSORBOARD_DIR": str(arm_dir / "tensorboard"),
                        "EXTRA_ARGS": " ".join(extra + spec_args + args.extra_args),
                        **dict(kv.split("=", 1) for kv in args.env),
                    },
                }
            )
    result = {"created": time.strftime("%Y-%m-%dT%H:%M:%S%z"), "seed": args.seed, "arms": arms}
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(json.dumps(result, indent=2))
    return result


def run(args) -> int:
    out = Path(args.out).resolve()
    arms = json.loads((out / "plan.json").read_text())["arms"]
    if args.arms:
        arms = [a for a in arms if a["name"] in args.arms]
    failures = 0
    for arm in arms:
        env_desc = " ".join(f"{k}={json.dumps(v)}" for k, v in arm["env"].items())
        print(f"[{arm['name']}] {env_desc} bash {arm['script']}", flush=True)
        if args.dry_run:
            continue
        arm_dir = Path(arm["dir"])
        arm_dir.mkdir(parents=True, exist_ok=True)
        start = time.time()
        with open(arm_dir / "run.log", "w") as log:
            code = subprocess.call(
                ["bash", arm["script"]], env={**os.environ, **arm["env"]}, stdout=log, stderr=subprocess.STDOUT
            )
        events = len(list((arm_dir / "tensorboard").rglob("events.out.tfevents.*")))
        record = {
            "exit_code": code,
            "tensorboard_event_files": events,
            "start": start,
            "wall_time": time.time() - start,
        }
        (arm_dir / "run.json").write_text(json.dumps(record, indent=2))
        print(f"[{arm['name']}] exit {code}, {events} event files, {record['wall_time']:.0f}s", flush=True)
        failures += code != 0 or events == 0
    return 1 if failures else 0


def read_tensorboard(directory: Path) -> dict[str, dict[int, float]]:
    """``{tag: {step: value}}`` from every event file below ``directory``."""
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    scalars: dict[str, dict[int, float]] = {}
    for path in sorted(directory.rglob("events.out.tfevents.*")):
        acc = EventAccumulator(str(path), size_guidance={"scalars": 0})
        acc.Reload()
        for tag in acc.Tags()["scalars"]:
            for event in acc.Scalars(tag):
                scalars.setdefault(tag, {})[event.step] = event.value
    return scalars


def summarize_arm(scalars: dict[str, dict[int, float]], warmup: int) -> dict:
    """Means over steps after ``warmup``, plus first/last-third acceptance."""
    summary: dict = {}
    for name, candidates in STEP_TAGS.items():
        tag = next((t for t in candidates if t in scalars), None)
        if tag is None:
            continue
        series = [value for step, value in sorted(scalars[tag].items())]
        measured = series[warmup:] if len(series) > warmup else series
        finite = [v for v in measured if math.isfinite(v)]
        if not finite:
            continue
        summary[name] = sum(finite) / len(finite)
        summary[f"{name}_steps"] = len(finite)
        if name == "spec_accept_rate" and len(finite) >= 3:
            third = len(finite) // 3
            summary["spec_accept_rate_first_third"] = sum(finite[:third]) / third
            summary["spec_accept_rate_last_third"] = sum(finite[-third:]) / third
    return summary


def compare(summaries: dict[str, dict], arms: list[dict]) -> list[dict]:
    """One row per arm; speedups are against the ``off`` arm of the same layout."""
    rows = []
    for arm in arms:
        s = summaries.get(arm["name"], {})
        base = summaries.get(f"{arm['layout']}-off", {})
        row = {"arm": arm["name"], "layout": arm["layout"], "mode": arm["mode"], **s}
        for metric in ("rollout_time", "step_time"):
            if s.get(metric) and base.get(metric):
                row[f"{metric}_speedup_vs_off"] = base[metric] / s[metric]
        rows.append(row)
    return rows


COLUMNS = (
    ("arm", "arm", "{}"),
    ("rollout_time", "rollout s", "{:.1f}"),
    ("rollout_time_speedup_vs_off", "rollout ×off", "{:.2f}"),
    ("step_time", "step s", "{:.1f}"),
    ("step_time_speedup_vs_off", "step ×off", "{:.2f}"),
    ("spec_accept_rate", "accept rate", "{:.3f}"),
    ("spec_accept_rate_first_third", "accept first ⅓", "{:.3f}"),
    ("spec_accept_rate_last_third", "accept last ⅓", "{:.3f}"),
    ("spec_accept_length", "accept len", "{:.2f}"),
    ("tail_wall_share", "tail share", "{:.2f}"),
    ("update_draft_weights_time", "draft sync s", "{:.2f}"),
    ("logprob_abs_diff", "|logp diff|", "{:.4f}"),
    ("reward", "reward", "{:.3f}"),
)


def render(rows: list[dict], plan_data: dict, warmup: int) -> str:
    out = [
        "# Speculative rollout benchmark",
        "",
        f"Seed {plan_data['seed']}; means over rollout steps after the first {warmup}. "
        "Speedups divide the `off` arm's time by the arm's time (same placement). "
        "All arms train the policy; in `online` the draft loss is detached from it and "
        "the draft's gradients are clipped by their own norm.",
        "",
        "| " + " | ".join(header for _, header, _ in COLUMNS) + " |",
        "| " + " | ".join("---" if key == "arm" else "---:" for key, _, _ in COLUMNS) + " |",
    ]
    for row in rows:
        cells = [fmt.format(row[key]) if key in row else "—" for key, _, fmt in COLUMNS]
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out) + "\n"


def summarize(args) -> list[dict]:
    out = Path(args.out).resolve()
    plan_data = json.loads((out / "plan.json").read_text())
    summaries = {}
    for arm in plan_data["arms"]:
        tb_dir = Path(arm["dir"]) / "tensorboard"
        if tb_dir.exists():
            summaries[arm["name"]] = summarize_arm(read_tensorboard(tb_dir), args.warmup)
    rows = compare(summaries, plan_data["arms"])
    (out / "summary.json").write_text(json.dumps(rows, indent=2))
    report = render(rows, plan_data, args.warmup)
    (out / "summary.md").write_text(report)
    print(report)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("plan", help="write plan.json")
    p.add_argument("--out", required=True)
    p.add_argument("--num-rollout", type=int, default=20)
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--num-gpus", type=int, default=8, help="GPUs on the node (non-colocate splits them in half)")
    p.add_argument(
        "--modes", nargs="*", choices=MODES, default=list(MODES), help="built-in arms; pass none to run only --drafts"
    )
    p.add_argument(
        "--example",
        default=DEFAULT_EXAMPLE,
        help="example script relative to the repo, {layout} is filled in; it must accept SPEC_MODE off/frozen/online",
    )
    p.add_argument("--layouts", nargs="+", choices=LAYOUTS, default=list(LAYOUTS))
    p.add_argument("--env", nargs="*", default=[], help="extra KEY=VALUE for every arm, e.g. MODEL_DIR=/data/Qwen3-4B")
    p.add_argument("--extra-args", nargs="*", default=[], help="extra train.py arguments for every arm")
    p.add_argument(
        "--drafts",
        nargs="*",
        type=parse_draft,
        default=[],
        help='extra arms NAME=JSON serving another pretrained vLLM draft, e.g. dflash2=\'{"method":"dflash","model":"/m","num_speculative_tokens":8}\'',
    )

    r = sub.add_parser("run", help="run the arms in plan.json one by one")
    r.add_argument("--out", required=True)
    r.add_argument("--arms", nargs="*", help="only these arms")
    r.add_argument("--dry-run", action="store_true")

    s = sub.add_parser("summarize", help="compare arms from their TensorBoard logs")
    s.add_argument("--out", required=True)
    s.add_argument("--warmup", type=int, default=1, help="rollout steps to leave out of the means")

    args = parser.parse_args()
    if args.command == "plan":
        result = plan(args)
        print(f"{len(result['arms'])} arms -> {Path(args.out).resolve() / 'plan.json'}")
    elif args.command == "run":
        sys.exit(run(args))
    else:
        summarize(args)


if __name__ == "__main__":
    main()
