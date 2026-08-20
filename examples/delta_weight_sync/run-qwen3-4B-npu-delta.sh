#!/usr/bin/env bash
set -euo pipefail

# Run from an Ascend VIME environment. Adjust model/data paths and the shared/local
# directories for your deployment. Training uses NPUs 0-3; rollout uses NPUs 4-7.
python3 train.py \
  --train-backend megatron \
  --hf-checkpoint /path/to/Qwen3-4B \
  --load /path/to/Qwen3-4B \
  --ref-load /path/to/Qwen3-4B \
  --megatron-to-hf-mode bridge \
  --actor-num-nodes 1 --actor-num-gpus-per-node 4 \
  --rollout-num-gpus 4 --rollout-num-gpus-per-engine 4 \
  --tensor-model-parallel-size 4 \
  --update-weight-mode delta \
  --update-weight-transport disk \
  --update-weight-disk-dir /shared/vime-delta \
  --update-weight-local-checkpoint-dir /local-nvme/vime-rollout-checkpoint \
  --update-weight-delta-encoding xor \
  --update-weight-delta-checksum xxh3-128 \
  "$@"
