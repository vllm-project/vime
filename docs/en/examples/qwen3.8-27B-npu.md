# Qwen3.8-27B on Ascend

Use the [NPU image and dependency patches](../get_started/NPU.md). The checkpoint
uses the Qwen3.5 dense hybrid-attention architecture and its existing model bridge.
Download `Qwen/Qwen3.8-27B` into `/root/models/Qwen3.8-27B` and
`zhuzilin/dapo-math-17k` into `/root/datasets/dapo-math-17k`, then run:

```bash
bash scripts/run-qwen3.8-27B-npu.sh
```

This uses 16 logical NPUs: 8 for TP4/PP2 training and 8 for two TP4 rollout
engines. It runs 100 GRPO rounds with 32 prompts × 8 samples, response limit 4096,
V2 model runner, no MTP, CPU optimizer offload, and full/uniform/1 recomputation.
Override `MODEL_DIR`, `DATASET_DIR`, `CHECKPOINT_DIR`, or `NUM_ROLLOUT` as needed.
It saves model and optimizer checkpoints every 25 rounds. Reserve approximately
1.4 TiB for all four checkpoints, in addition to model and dataset storage.

`--qwen-gdn-tensor-parallel` partitions GDN K/V heads, projections, convolution,
and per-head state across TP ranks. Input and output normalization remain shared;
their gradients are summed across TP. Unequal Q/K/V segments are split and merged
separately for HF loading, weight updates, and distributed checkpoints. Both SP
and non-SP execution are supported; head TP currently requires CP=1. The flag is
opt-in and leaves existing replicated GDN execution unchanged when omitted.

The launcher starts an isolated Ray head and does not stop other jobs. Choose an
unused `RAY_PORT_BASE` (default 18479; it reserves ports through base+1099) and
HCCL port ranges when sharing a host. Run it in a dedicated NPU container; it
does not clean up an existing Ray cluster or silently resume a prior checkpoint.
Its own Ray head remains available for inspection after the job exits.

The smoke test downloads its inputs to `HF_HOME`, runs two short GRPO rounds on
16 NPUs, and exercises native HF loading, head TP/SP/PP, training, HCCL weight
updates, and the subsequent rollout without MTP:

```bash
python tests/test_qwen3.8_27B_npu.py
```

It is registered in the Buildkite NPU `smk` suite. Select `smk` at the NPU gate;
CPU/pre-commit success alone does not mean that the NPU test has run.
