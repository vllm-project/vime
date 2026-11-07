# Ouro GRPO with the shared rollout engine

This recipe uses VIME's existing Megatron optimizer, GRPO loss, gradient reduction
and distributed checkpoints. Ouro keeps one physical copy of every decoder layer,
all four sandwich norms, the inter-loop norm and the exit gate. The gate remains
frozen during this externally budgeted recipe and participates in conversion and
publication.

The companion is [engine draft #78](https://github.com/ThinkFlowLab/vllm-rlt/pull/78), the training-contract branch of
[0z5a/vllm-rlt](https://github.com/0z5a/vllm-rlt/tree/feat/training-contract), based on
ThinkFlowLab/vllm-rlt `ecb1f8b505b7e831815b40aec3b4598619cca23a`; tested companion
commit `f6cfefe5d7df8e31884b9396701245b934590bd6`.
The adapter calls public `LLM.generate`, `start_weight_update`, `update_weights`,
`finish_weight_update` and `get_weight_version`. It uses the same contract for
local and disaggregated prefill/decode (PD) engines. The previous private
`RLEngine` prototype is superseded.

## Launch

Make VIME, the companion and an existing Megatron source checkout importable.
The tested model is `ByteDance/Ouro-1.4B` at
`574fa66cb8bf5abdc979642d01cf2b79b16bfab1`. JSONL rows contain `prompt`, `label`
and `metadata.problem_id`.

```bash
export PYTHONPATH="$PWD:/path/to/vllm-rlt:$MCORE"
GPUS_PER_NODE=1 bash examples/ouro/run.sh /models/Ouro-1.4B train.jsonl /runs/ouro 5 \
  --ouro-depths 2 3 4 --ouro-cuda-graphs --ouro-kv-blocks 128 \
  --rollout-max-response-len 16 --seq-length 512 --max-position-embeddings 2048 \
  --ouro-export-hf /runs/ouro/export
```

The launcher runs one trainer directly when `GPUS_PER_NODE=1`; multiple local
trainer ranks use torchrun. `--debug-train-only` bypasses VIME's server/router
setup while this recipe still generates real online samples. TP=PP=CP=1, BF16,
sequence parallelism and gradient accumulation fusion disabled are required.
GRPO uses positive-temperature, full-vocabulary sampling without a reference,
critic or KL loss. The small launch defaults are execution checks rather than a
convergence protocol.

- `--ouro-depths 2 3 4` cycles a globally agreed decode budget by update ID.
- `--ouro-cuda-graphs` enables recurrent graph replay across full publications.
- `--ouro-exit-threshold 0` exercises native early exit within the budget.
- `--ouro-async-scheduling` uses the shared delayed-exit asynchronous path.
- `--ouro-speculative-tokens 3 --ouro-depths 4` uses the existing full-depth target
  policy with a K=2 draft. The upstream engine requires synchronous speculation.
- `--ouro-prefill-devices 0 --ouro-decode-devices 1 --ouro-max-num-seqs 1` selects real spawned PD
  workers. This pilot supports one trainer rank, which may share GPU 0 with P.
  PD and self-speculation cannot be combined in the upstream engine. Fixed
  batch shapes are required for the reported BF16 bitwise resume comparisons;
  dynamic BF16 batching can produce different scores and optimizer hashes.
- `--ouro-export-hf DIR` exports through VIME's ordinary HF saver.
- `--ouro-reward-function vime_plugins.ouro.reward.boxed_answer` enables exact
  integer boxed-answer scoring; the default is Deepscaler.
- `--ouro-eval-data FILE --ouro-eval-prompts N --ouro-eval-interval M` evaluates
  K=2/3/4 on separate prompts, or K=4 for speculation.

For a fresh-process resume, keep the same schedule, data, sampling and PD pool
flags, increase the update count and add:

```bash
--ouro-resume --load /runs/ouro/checkpoint --override-opt-param-scheduler
```

The override is needed when extending the requested constant-LR horizon. Model,
Adam, scheduler, RNG and the deterministic data/budget cursor are restored.
`ouro-plan.json` rejects incompatible execution settings. Completed saves remove
older iteration directories within that experiment's save directory.

On the tested A100 container, default UCX CUDA IPC did not deliver GPU writes in
an independent two-process probe. Per-command `UCX_TLS=tcp,cuda_copy` passed the
probe and PD checks with the installed NIXL; this is host-staged transport.
The recipe disables forced worker termination and closes successful PD workers
through the normal stop/acknowledgment protocol.

## Probability and conversion semantics

Every prompt uses full-depth prefill. Subsequent decode inputs use each returned
`exit_depths` value, including mixed early exits and speculative target depths.
The differentiable provider replays those physical depths and retains the last
exited KV at deeper planes. VIME's existing per-sample forward kwargs carry the
trace, including sequence padding and activation recomputation.

Outputs include selected-token processed logprobs, effective sampling parameters
and the committed policy version. VIME's existing loss applies the same rollout
temperature. Each GRPO group validates its prompt ID, K, version and configuration
hash. Failed or incomplete publication blocks generation until a newer full
update completes. Publication changes tensors in place and invalidates cached KV.

Ouro is registered in the general HF-to-Megatron and Megatron-to-HF converters.
Physical names are preserved after removing MCore/DDP wrapper prefixes; recurrent
loops do not duplicate parameter tensors. The same `HfWeightIteratorDirect` is
used for initial and per-update publication. Full HF exports include all 269
physical model tensors.

Metrics separate rollout, training, publication and whole-step time. Decode work
counts physical layer/token evaluations, including speculative verification rows;
trainer counts include padding and recomputation. These are work counters, not
measured backward FLOPs. PD GPU-hours count distinct devices once.
