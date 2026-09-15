# Qwen3-30B-A3B with 8xH100


## Environment Preparation

The environment setup, model download, data, and checkpoint conversion are the same as for the Qwen3-4B model. You can refer to [Example: Qwen3-4B Model](qwen3-4B.md), replacing mentions of Qwen3-4B with Qwen3-30B-A3B.

To convert huggingface checkpoint to torch_dist, please try:

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

1.  To support running Qwen3-30B-A3B in an 8xH100 environment, we need to enable Megatron's CPU Adam to save GPU memory. The corresponding configuration is:

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

The following uses **2 nodes × 8 GPUs (16 GPUs total) in colocate mode** as the example. The only differences from single-node are "starting Ray across nodes" and "adjusting a few resource/parallelism arguments"; the training script itself is unchanged.

1. **Shared storage**: put the model, data, and checkpoints on a location that every node can access at the same path (e.g. NFS).

2. **Start Ray across nodes** (outside the training script, run manually on each node; see [Quick Start — Multi-node training](../get_started/quick_start.md#multi-node-training-for-large-scale-moe-models)):

   ```bash
   # Head node (node0); MASTER_ADDR must be a LAN IP, not 127.0.0.1
   export MASTER_ADDR=<head_lan_ip>
   ray start --head --node-ip-address ${MASTER_ADDR} --num-gpus 8 --disable-usage-stats

   # Every other node
   ray start --address=${MASTER_ADDR}:6379 --num-gpus 8
   ```

   Wait until `ray status` reports 16 GPUs before submitting. Because you started the cluster manually, make the script skip its process-management preamble — remove (or comment out) **both** the initial cleanup block (`ray stop --force`, `pkill -9 ray`, `pkill -9 python`, `pkill -9 redis`) **and** the `ray start --head ...` line. Otherwise running the script tears down the head you just started (and orphans the workers), so `ray job submit` to `http://127.0.0.1:8265` fails. Keep the rest of the script — it still sources the model args and runs `ray job submit`.

3. **Adjust script arguments** (`scripts/run-qwen3-30B-A3B.sh`):
   - Change `--actor-num-nodes` for `train.py` from `1` to `2` (keep `--actor-num-gpus-per-node` at 8). Under colocate, `--rollout-num-gpus` is auto-set to `actor_num_gpus_per_node × actor_num_nodes = 16`, so you don't set it manually.
   - Scale up the parallelism in `PERF_ARGS` for the doubled GPU count (e.g. raise TP or add DP); for concrete large-scale ratios see the bigger-cluster examples such as [GLM-4.7](glm4.7-355B-A32B.md) and [DeepSeek-R1](deepseek-r1.md).
   - `global-batch-size` must equal `rollout-batch-size × n-samples-per-prompt`.
   - (Optional) Multi-node uses a distributed optimizer, which lowers optimizer memory pressure, so you may drop the CPU Adam options (`--optimizer-cpu-offload`, etc.) from `OPTIMIZER_ARGS` for speed.

4. **Keep each vLLM engine within a single node**: prefer `--rollout-num-gpus-per-engine 8` (one engine per node) over `16` (a single engine spanning both nodes at TP=16). Cross-node TP is noticeably slower and more sensitive to per-token numerics; this value must divide the total rollout GPU count (16 here).

⚠️ Common issues:
- **Worker cannot join Ray / NCCL failures**: check `MASTER_ADDR`, container `/etc/hosts` (hostname must not map to `127.0.0.1`), and set `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME` on multi-NIC hosts.
- **`Not enough samples X for global_batch_size Y`**: keep `global-batch-size = rollout-batch-size × n-samples-per-prompt`.
- **Fewer than 8 GPUs per node (colocate)**: set `--num-gpus-per-node` explicitly.

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
