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

### Multi-Node Support

The following uses **two machines with 8 GPUs each (16 GPUs total)** as the starting example; scripts and parameters scale to **N nodes**. Key differences from single-node:

- Place weights, checkpoints, and data on storage visible to every node (e.g. NFS).
- Set `MASTER_ADDR` to the head **LAN IP** (not `127.0.0.1`).
- Omit CPU Adam (multi-node uses a distributed optimizer; do not use `--optimizer-cpu-offload`).
- `global-batch-size` must equal `rollout-batch-size × n-samples-per-prompt`.

#### Topology

| Component | Dual-node defaults |
|-----------|-------------------|
| Cluster | `ACTOR_NUM_NODES=2`, `ACTOR_NUM_GPUS_PER_NODE=8` |
| Megatron training | TP=8, EP=8, CP=2 (experts sharded across nodes) |
| vLLM rollout | Cross-node TP=16 (`rollout-num-gpus-per-engine = nodes × GPUs per node`) |
| Scheduling | Ray cluster + `--colocate` mode |

Convert checkpoints with Megatron parallelism matching training (dual-node: TP=8, EP=8). Checkpoint EP must match `--expert-model-parallel-size`, or `load_checkpoint` may hang or resharding may be extremely slow.

#### Start the Ray Cluster

Start Ray **outside** the training script on each node. Join all workers first; verify `ray status` reports the expected GPU count, then submit training from the head. Dual-node example:

```bash
# === Head node ===
export MASTER_ADDR=<head_lan_ip>
ray start --head --node-ip-address="${MASTER_ADDR}" --num-gpus 8 --disable-usage-stats \
   --dashboard-host=0.0.0.0 --dashboard-port=8265

# === Each worker node ===
export MASTER_ADDR=<head_lan_ip>
ray start --address="${MASTER_ADDR}:6379" --node-ip-address=<this_node_lan_ip> --num-gpus 8
```

See [Quick Start — Multi-node training](../get_started/quick_start.md#multi-node-training-for-large-scale-moe-models) for more details.

#### Run Training

After the Ray cluster is ready, on the **head node** set multi-node env vars and run the **same script as single-node** (`ACTOR_NUM_NODES>1` skips Ray startup and applies multi-node defaults):

```bash
export MASTER_ADDR=<head_lan_ip>
export ACTOR_NUM_NODES=2
export ACTOR_NUM_GPUS_PER_NODE=8
cd /root/vime
bash scripts/run-qwen3-30B-A3B.sh
```

2-step smoke test:

```bash
NUM_ROLLOUT=2 ENABLE_R3=0 bash scripts/run-qwen3-30B-A3B.sh
```

To scale to N nodes (e.g. 4×8), join all workers to Ray, set `ACTOR_NUM_NODES=4` on the head, and tune `MEGATRON_TP` / `MEGATRON_EP` / `MEGATRON_CP` / `ROLLOUT_NUM_GPUS_PER_ENGINE` for total GPU count.

#### Key Multi-Node Parameters

| Variable | Dual-node default | Description |
|----------|-------------------|-------------|
| `ACTOR_NUM_NODES` | 2 (default 1 for single-node) | Total nodes including head; script skips Ray startup when >1 |
| `ACTOR_NUM_GPUS_PER_NODE` | 8 | GPUs per node |
| `MEGATRON_TP` / `MEGATRON_EP` / `MEGATRON_CP` | 8 / 8 / 2 | Megatron parallelism |
| `ROLLOUT_NUM_GPUS_PER_ENGINE` | total GPUs | vLLM engine GPU count |
| `ENABLE_R3` | 1 | set to 0 to disable R3 |

Default batch: `rollout-batch-size=4`, `n-samples-per-prompt=2`, `global-batch-size=8`; vLLM uses `--vllm-moe-backend triton`.

#### Multi-Node Troubleshooting

- **Worker cannot join Ray / NCCL failures**: check `MASTER_ADDR`, container `/etc/hosts` (hostname must not map to `127.0.0.1`), `NCCL_SOCKET_IFNAME` / `GLOO_SOCKET_IFNAME`.
- **`Not enough samples X for global_batch_size Y`**: keep `global-batch-size` equal to `rollout-batch-size × n-samples-per-prompt`.
- **GPU memory full but no processes**: restart the container or run `ray stop --force` to clear stale vLLM contexts.

#### EPLB

When the total number of GPUs is not a multiple or divisor of the total number of experts, enable vLLM's EPLB (Expert Parallelism Load Balancer) and configure redundant experts via `--vllm-eplb-config`. For example, in a 24-GPU scenario:

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
