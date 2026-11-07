# MTP Speculative Decoding for Qwen3.5-4B

Qwen3.5 checkpoints ship one multi-token-prediction (MTP) layer. This example
serves it as the speculative draft in vLLM (`"method": "mtp"`) during RL and,
with `SPEC_MODE=online`, also trains it alongside the policy and syncs it to
vLLM every step. The rollout, GRPO and optimizer settings are the same as in the
[DSpark example](../dspark/README.md), so the two can be compared.

| `SPEC_MODE` | Speculative decoding | MTP layer | `--ref-load` |
| --- | --- | --- | --- |
| `online` (default) | MTP, `MTP_K` tokens | trained (`--mtp-num-layers 1 --enable-mtp-training`) and synced every step | `${MODEL_DIR}_mtp_torch_dist` |
| `frozen` | MTP, `MTP_K` tokens | the checkpoint's layer, never updated | `${MODEL_DIR}_torch_dist` |
| `off` | none | — | `${MODEL_DIR}_torch_dist` |

`MTP_K` defaults to 2, the value the Qwen3.5-4B model card uses with vLLM.
Qwen3.5-4B reasons at length by default; with a 4096-token response limit most
dapo-math answers are cut off and score 0, so the policy hardly learns. For
short answers set `CHAT_TEMPLATE_KWARGS='{"enable_thinking":false}'` together
with `RM_TYPE=math`: the `deepscaler` reward only scores text after `</think>`,
which non-thinking answers do not contain.
Megatron clips the MTP layer's gradients by their own norm, so `train/grad_norm`
covers the policy only; the MTP loss is logged as `train/mtp_loss`.

## Prerequisites

Convert the HF checkpoint twice: once without and once with the MTP layer.

```bash
cd /root/vime
source scripts/models/qwen3.5-4B.sh

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} \
    --hf-checkpoint /root/Qwen3.5-4B \
    --save /root/Qwen3.5-4B_torch_dist

PYTHONPATH=/root/Megatron-LM python tools/convert_hf_to_torch_dist.py \
    ${MODEL_ARGS[@]} --mtp-num-layers 1 \
    --hf-checkpoint /root/Qwen3.5-4B \
    --save /root/Qwen3.5-4B_mtp_torch_dist
```

Download dapo-math-17k as in the DSpark example.

## Run

```bash
cd /root/vime
SPEC_MODE=online bash examples/mtp/run-qwen3.5-4B-mtp-colocate.sh
```

To compare the three modes with the same seed and steps, point the benchmark
tool at this example:

```bash
python tools/bench_spec_rollout.py plan --out /data/bench-mtp --layouts colocate \
    --example examples/mtp/run-qwen3.5-4B-mtp-{layout}.sh --env MODEL_DIR=/root/Qwen3.5-4B
python tools/bench_spec_rollout.py run --out /data/bench-mtp
python tools/bench_spec_rollout.py summarize --out /data/bench-mtp
```
