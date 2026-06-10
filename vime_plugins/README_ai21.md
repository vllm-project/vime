# AI21 add-ons for vime

Custom AI21 code migrated from `ai21-verl`, re-homed onto vime's extension seams. Everything here
is **additive** — new modules under `vime_plugins/` wired by dotted path — so no upstream vime
file is modified and fork merges stay clean.

Install the extra deps (on top of core vime):

```bash
pip install -e .
pip install -r requirements_ai21.txt    # needs the AI21 private index; see docker/Dockerfile.ai21
```

## 1. Reward — AI21 evaluators

`vime_plugins/rm/ai21_evaluators.py` wraps the `ai21-evaluators` library (ported pure logic in
`ai21_evaluators_lib.py`). Wire:

```bash
--custom-rm-path vime_plugins.rm.ai21_evaluators.ai21_reward
```

Each prompt carries its evaluator spec in `metadata` (`reward_model`, `aggregation_config`, `id`,
optional `force_thinking`, `jlm_model_name`). Config via args or env:
`AI21_EVALUATORS_TIMEOUT` (60.0), `AI21_EVALUATORS_CONFIG_FILE`, `AI21_CLEAN_THINKING_TRACE`.
Returns a float, or a dict (when `--reward-key` is set) with `score`/`status`/`do_exclude`.

## 2. Curriculum + multi-source data

**Curriculum filter** — `vime_plugins/filters/snoozing.py` is a superset of the built-in
`check_reward_nonzero_std`: it drops zero-variance groups and *snoozes* easy (consistently solved)
prompts for N future encounters.

```bash
--dynamic-sampling-filter-path vime_plugins.filters.snoozing.snoozing_filter
```

Config: `AI21_SNOOZE_NUM_TIMES` (0 = snoozing off → identical to `check_reward_nonzero_std`),
`AI21_SNOOZE_MEAN_SCORE_THRESHOLD` (1.0), `AI21_SNOOZE_ID_KEY` (`id`).

**Multi-source data** — vime loads one `--prompt-data` file and has no data-source-swap seam, so
merge offline (replaces AmalgamDataset's weighted interleaving):

```bash
python -m vime_plugins.data.merge_amalgam_sources --mix mix.json --out merged.jsonl \
    --input-key text --label-key label --metadata-key metadata
# then train with:  --prompt-data merged.jsonl --metadata-key metadata
```

`mix.json`: `{"<source>": {"path": "...", "weight": 2, "rm_type": "math"}, ...}` — per-source keys
other than `path`/`weight` are merged into each row's `metadata` (so per-source `rm_type` and the
evaluator spec reach the reward function and snoozing id).

## 3. GCS checkpoint sync

`vime_plugins/checkpoint/gcs_sync.py` (ported from ai21-verl). Run a sidecar next to training —
zero core changes; `gsutil rsync` is incremental:

```bash
python -m vime_plugins.checkpoint.gcs_sync watch \
    --local-dir $SAVE --gcs-dest gs://bucket/run-id --poll-interval 60
```

`sync_checkpoint_dir()` is also callable from a future `--checkpoint-sync-path` hook if vime adds
one. ESI/preemption early-save is deferred.

## 4. Multi-turn agent loop

vime already supports multi-turn/agentic rollouts. `vime_plugins/rollout/multi_turn_agent.py` is a
scaffold that owns the vime contract (token accounting, loss masking, status, budgets); plug your
own per-turn logic in via `agent_step`:

```bash
--custom-generate-function-path vime_plugins.rollout.multi_turn_agent.generate
--agent-step-path my_pkg.my_agent.agent_step          # or env AGENT_STEP_PATH
```

`agent_step(args, sample, assistant_text, turn_index) -> {"done", "observation", "reward"}`.
Config: `AGENT_MAX_TURNS` (8). Model tokens get `loss_mask=1`, injected observations `loss_mask=0`.

## Not migrated (already in vime / verl-specific / obsolete)

Training-side NUMA binding (`vime/ray/train_actor.py`), MoE aux/z-loss (Megatron-native),
`triton_device_fix`, the verl `AI21GRPOTrainer` / Hydra overlays / `AI21*Config` worker dataclasses,
and the Maestro agent loop (being replaced — see §4).

## Tests

CPU unit tests under `tests/unit/` (run in CI's Docker image, which has torch + the private deps):

```bash
python -m pytest tests/unit/rollout/test_ai21_evaluators_rm.py \
                 tests/unit/rollout/test_snoozing_filter.py \
                 tests/unit/rollout/test_multi_turn_agent.py \
                 tests/unit/test_merge_amalgam_sources.py \
                 tests/unit/test_gcs_sync.py
```
