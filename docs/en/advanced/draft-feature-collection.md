# Collect target features for an external draft

`--draft-feature-mode collect-only` exports selected inputs to the actor's final, bias-free LM head during an existing log-probability forward. It does not train or publish a draft model. The default `off` path creates no export directory or hook.

The first version supports TP=PP=CP=DP=1, no VPP, micro-batch=1 and the actor model. Disable `--keep-old-actor` and `--use-rollout-logprobs`, and use `--update-weights-interval 1` so the version counter cannot lag actor updates. A batch that reuses the training forward instead of doing a separate actor log-probability forward is rejected before exporting; collection never inserts a compensating forward. For synchronous GRPO, use more than one optimizer step per rollout so the ordinary actor path already computes log probabilities separately.

Add these options to a working training recipe:

```bash
--draft-feature-mode collect-only \
--draft-feature-output-dir /local/experiments/draft-features \
--draft-feature-run-id run-001 \
--draft-feature-max-tokens 4096 \
--draft-feature-max-batches 8 \
--draft-feature-max-bytes 1073741824
```

Each round writes `run-001/round-N/head.safetensors`, selected `batch-XXXX.safetensors` and a corresponding JSON manifest. The head is copied once per round; features own their storage. The manifest is renamed into place after the tensors finish writing. Consumers must ignore temporary files and read only complete manifests. An existing round directory is an error; use a new run ID after a restart.

Budgets apply **per round** to tensor payloads: head bytes once plus selected tokens × hidden size × actual feature element size. JSON, serialization buffers and filesystem overhead are outside that budget. Accumulated rounds require additional disk space and explicit retention. An entirely empty round fails with a reason; reaching a limit after publishing a batch succeeds and records the stop reason. `copy_seconds` measures CPU transfer **and serialization/file writes**, rather than device-to-host bandwidth alone.

The token map records the original sample, prompt group, rollout ID, source position and next-token target. Padding has no selected row; masked tokens are excluded without changing the training mask. Load manifests through `vime.utils.draft_feature_contract.manifest_from_dict`, which reconstructs nested contracts and validates offsets, shifts, identity, dtype and declared bytes.

Rollout provenance, target-forward version and head version have separate meanings. Missing rollout versions remain unknown (`[]`); mixed versions remain mixed. The actor/head label uses the weight updater's counter within this run. A fresh actor restores model/optimizer state before its initial publication; the publication counter can restart, so it is not a global checkpoint identifier. Publication errors propagate before the next driver round. Consumers requiring strict on-policy data must validate actual rollout provenance separately.

## Reproducible checks

Run CPU regressions without initializing Ray or CUDA:

```bash
python -m pytest -q tests/test_draft_feature_contract.py \
  tests/test_draft_feature_collector.py tests/test_draft_feature_metadata.py \
  tests/test_megatron_argument_validation.py
```

For Ouro, `tests/integration/run_draft_feature_ouro.py` wraps the existing `examples/ouro/train.py`. Pass the usual recipe arguments after that path; the wrapper replaces rollout-logprob reuse with the same target forward in both off/on cases, preserves real RLT generation, rewards, optimization and weight publication, and saves reference logits outside the consumer artifact directory. `--only-train-params-name-list lm_head.weight` is a smaller memory smoke configuration and must be identified as such in results.

For a complete validation, compare fixed replay with collection off/on, reconstruct selected logits with the exported same-version head, run six real generation/update/save/publication rounds, then resume in a new process and run namespace. Compare loss, entropy, gradients, optimizer state, RNG and checkpoints. Report collector overhead from matched runs; collection itself does not promise a rollout speedup. Keep checkpoints, feature tensors and raw logs outside the source tree.
