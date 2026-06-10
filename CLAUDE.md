# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

Vime is an LLM post-training (RL scaling) framework derived from [slime](https://github.com/THUDM/slime). It pairs **Megatron-LM** (training backend) with **vLLM + [vllm-router](https://github.com/vllm-project/router)** (rollout/inference backend), orchestrated by **Ray**. Training and rollout run as separate Ray actor groups that exchange data through a "data buffer" and synchronize weights after each training step.

## Commands

```bash
# Editable install (CI uses --no-deps because the base image already has torch/megatron/vllm)
pip install -e .

# Lint / format — this is the CI gate (ruff, isort, black, autoflake); always run before committing
pre-commit install
pre-commit run --all-files --show-diff-on-failure --color=always

# CPU unit tests (pytest)
python -m pytest tests/unit tests/utils
python -m pytest tests/test_rm_math.py          # a single pytest file
pytest -m unit                                  # select by marker (unit/integration/system/...)
```

**Two test conventions live in `tests/`:**
- **pytest files** (`tests/unit/`, `tests/utils/`, `tests/test_rm_*.py`, `tests/plugin_contracts/`, validation tests) — CPU-only, no GPUs. Each also has a `if __name__ == "__main__"` shim so CI can run it as `python tests/<file>.py`.
- **GPU end-to-end scripts** (`tests/test_<model>_*.py`) — NOT pytest. Each defines `prepare()` (downloads model/dataset) and `execute()` (launches a real training run via `vime.utils.external_utils.command_utils`). Run with `python tests/ci/gpu_lock_exec.py --count <N> -- python tests/<file>.py`. These require GPUs and the CI Docker image (`inferactinc/public:vime-latest`) and are gated behind `run-ci-*` PR labels.

When adding an e2e test, set a module-level `NUM_GPUS = <n>` — CI's changed-test detector greps for it.

## Entry points & the training loop

- **`train.py`** — synchronous loop. Per `rollout_id`: `rollout_manager.generate()` → `actor_model.async_train()` → `actor_model.update_weights()` (push new weights to rollout) → periodic eval/save.
- **`train_async.py`** — overlaps generation of step *N+1* with training of step *N*; weights pushed every `--update-weights-interval`. Requires `--colocate` to be off.
- Both call `vime/ray/placement_group.py` to build everything, then drive the same actor/rollout handles. Read these two files first — they are the clearest map of the whole system.

## Architecture (the parts that span files)

`vime/ray/` is the orchestration layer:
- **`placement_group.py`** — allocates one Ray placement group across all GPUs, then partitions bundles between training (offset 0) and rollout (offset = actor GPUs). `--colocate` puts both on the same GPUs (forces offload). Builds `RayTrainGroup` (actor + optional critic) and the `RolloutManager`.
- **`rollout.py`** (`RolloutManager`) — owns the vLLM `ServerGroup`s (a group = homogeneous engines; PD-disaggregation uses separate prefill/decode groups) and the router. Exposes `.generate()`, `.eval()`, `.save()/.load()`, sleep/wake offload (`onload_weights`/`onload_kv`/`offload`).
- **`actor_group.py` / `train_actor.py`** — the Megatron training actors; `.async_train()`, `.update_weights()`, `.save_model()`.

`vime/backends/` is the backend code:
- **`megatron_utils/`** — Megatron actor/critic, loss, checkpointing, and `megatron_to_hf/` (per-model-family weight converters: qwen, deepseekv3, glm4, llama, gpt_oss, …). `update_weight/` implements pushing trained weights into the running vLLM engines (distributed or tensor-based, optionally via the HF bridge).
- **`vllm_utils/`** — vLLM engine wrapper, config, and `arguments.py` (the `--vllm-*` / `--router-*` arg surface).

`vime/rollout/` is where data is generated and scored:
- Rollout functions (`vllm_rollout.py` is the default `generate_rollout`; also `fully_async_rollout.py`, `vllm_streaming_rollout.py`, `sft_rollout.py`, `on_policy_distillation.py`). Selected via `--rollout-function-path`; the contract is `RolloutFnTrainOutput` / `RolloutFnEvalOutput` in `base_types.py`.
- **`rm_hub/`** — reward functions (math/dapo, deepscaler, gpqa, f1, ifbench). `async_rm()` dispatches to a built-in `--rm-type`, a remote `--rm-url`, or a `--custom-rm-path`.
- **`filter_hub/`** — dynamic-sampling filters applied during/after rollout.

`vime_plugins/` holds optional, swappable implementations (extra `models/`, megatron-bridge integrations, `rollout_buffer/`) loaded by path rather than hardcoded.

## Arguments (three layers)

Args are parsed in `vime/utils/arguments.py`, which layers three sources:
1. **Megatron args** — passed straight through (e.g. `--tensor-model-parallel-size`, `--use-dynamic-batch-size`).
2. **vLLM/router args** — `--vllm-*` for engines, `--router-*` for vllm-router's native options, `--vllm-router-*` for telling Vime where the router lives. See `vime/backends/vllm_utils/arguments.py`.
3. **Vime framework args** — cluster sizing (`--actor-num-nodes`, `--rollout-num-gpus`, `--rollout-num-gpus-per-engine` = TP size per engine, `--colocate`), data, RL algorithm (`--advantage-estimator grpo`, `--use-critic`), offload, eval. See `vime/utils/arguments.py`.

## Extension points

These are the supported "add a thing" seams — each has a Skill (invoke via the matching `/add-*` skill) describing the wiring:
- **Reward function** → `--custom-rm-path` (`add-reward-function`)
- **Rollout function** → `--rollout-function-path` (`add-rollout-function`)
- **Dynamic/filter hook** → `filter_hub` (`add-dynamic-filter`)
- **Eval dataset config** → `--eval-config` / `--eval-prompt-data` (`add-eval-dataset-config`)
- **Tests & CI** → (`add-tests-and-ci`)

Functions are wired by dotted path and resolved at runtime via `vime.utils.misc.load_function`, so new code generally does not require touching the core loop — add the function and pass its path.

## Conventions

- Python ≥ 3.10. Line length **119** (black/isort); ruff `line-length` is 320 but only `E/F/B/UP` rules are enforced (E402 and E501 ignored).
- First-party packages: `vime`, `vime_plugins`. Keep changes minimal and pattern-matching — the project explicitly prioritizes a lightweight, stable core over new abstractions (see `CONTRIBUTING.md`).
- GPU (CUDA) and NPU (Ascend) support live on **separate branches** rather than being abstracted together.
- New features should start as a GitHub Issue before a large PR.
- Upstream slime docs (`docs/`, DeepWiki) remain the reference for shared Megatron/customization concepts under the old `slime` naming.
