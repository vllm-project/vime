# Qwen3-30B-A3B with 8xH100


## Environment Preparation

Environment setup and checkpoint conversion are the same as for the Qwen3-4B model; see [Example: Qwen3-4B Model](qwen3-4B.md) and replace mentions of Qwen3-4B with Qwen3-30B-A3B.

Download the model and data:

```bash
# hf checkpoint
hf download Qwen/Qwen3-30B-A3B --local-dir /root/Qwen3-30B-A3B

# train data
hf download --repo-type dataset zhuzilin/dapo-math-17k \
  --local-dir /root/dapo-math-17k

# eval data
hf download --repo-type dataset zhuzilin/aime-2024 \
  --local-dir /root/aime-2024
```

To convert the huggingface checkpoint to torch_dist, please try:

```bash
cd vime/
pip install -e . --no-deps
source scripts/models/qwen3-30B-A3B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun --nproc-per-node 8 \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --hf-checkpoint /root/Qwen3-30B-A3B/ \
   --save /root/Qwen3-30B-A3B_torch_dist/
```

## Run Training

Execute the training script:

```bash
cd /root/vime
bash scripts/run-qwen3-30B-A3B.sh
```

### Parameter Introduction

Here, we will briefly introduce the MoE-related parts in the [run-qwen3-30B-A3B.sh](https://github.com/vllm-project/vime/blob/main/scripts/run-qwen3-30B-A3B.sh) script.

1.  To support running Qwen3-30B-A3B in an 8xH800 environment, we need to enable Megatron's CPU Adam to save GPU memory. The corresponding configuration is:

    ```bash
    OPTIMIZER_ARGS=(
       ...
       --optimizer-cpu-offload
       --overlap-cpu-optimizer-d2h-h2d
       --use-precision-aware-optimizer
    )
    ```

2.  Enable MoE optimization supported by Megatron. The current configuration is tp4, ep8:

    ```bash
    PERF_ARGS=(
       --tensor-model-parallel-size 4
       --sequence-parallel
       --pipeline-model-parallel-size 1
       --context-parallel-size 1
       --expert-model-parallel-size 8
       --expert-tensor-parallel-size 1
       ...
    )
    ```

3.  Enable MoE expert parallelism in vLLM. EP size is auto-derived as
    `tensor_parallel_size × data_parallel_size`, so for an 8-GPU engine
    `--vllm-enable-expert-parallel` alone gives you EP=8:

    ```bash
    VLLM_ARGS=(
       --rollout-num-gpus-per-engine 8
       --vllm-gpu-memory-utilization 0.7
       --vllm-enable-expert-parallel
       --vllm-cudagraph-capture-sizes 1 2 4 8 $(seq 16 8 256)
    )
    ```

    For DP on the attention block plus EP on the experts, combine
    `--vllm-data-parallel-size N` with `--vllm-enable-expert-parallel`.

### BF16 Training with FP8 Inference

vime also supports BF16 training with FP8 inference. For the Qwen3-30B-A3B model, just download the FP8 weights:

```bash
hf download Qwen/Qwen3-30B-A3B-FP8 --local-dir /root/Qwen3-30B-A3B-FP8
```

And replace `--hf-checkpoint` in the script with:

```bash
#--hf-checkpoint /root/Qwen3-30B-A3B
--hf-checkpoint /root/Qwen3-30B-A3B-FP8
```

This triggers FP8 inference. Currently we directly cast the BF16 weights to FP8; more precision-friendly quantization schemes will be added over time.

⚠️ The Megatron checkpoint used for training must still be the one originally converted from the BF16 huggingface weights (`--ref-load` / `--load` unchanged).

### Multi-Node Support

For a multi-node environment, the following modifications are necessary:

- Place the training model and data on storage accessible by every node (e.g. NFS).
- Set `MASTER_ADDR` to an address reachable by all nodes (not `127.0.0.1`), start Ray on each node manually, then submit training from the head (see [Quick Start — Multi-node training](../get_started/quick_start.md#multi-node-training-for-large-scale-moe-models)).
- Adjust `--actor-num-nodes` for `train.py` and the parallelism (TP/EP/CP) in `PERF_ARGS` to match the total GPU count.
- Remove the CPU Adam configuration, since multi-node uses a distributed optimizer, which significantly reduces the optimizer's memory footprint.

In addition, when the total number of GPUs is not a multiple or divisor of the total number of experts, you can enable vLLM's EPLB (Expert Parallelism Load Balancer) and configure redundant experts via `--vllm-eplb-config`. For example, in a 24-GPU scenario:

```bash
VLLM_ARGS=(
   --rollout-num-gpus-per-engine 24
   --vllm-gpu-memory-utilization 0.7
   --vllm-data-parallel-size 3
   --vllm-enable-expert-parallel
   --vllm-enable-eplb
   --vllm-eplb-config '{"num_redundant_experts": 16}'
)
```
