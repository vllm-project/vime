# Multi-node vLLM validation for PR #66

End-to-end harness that drives the `nnodes > 1` rollout topology introduced by PR #66 across a sweep of parallelism configurations on `Qwen3-30B-A3B` (30B MoE, 128 experts).

The same test file (`test_qwen3_30B_A3B_pr66_sweep.py`) is used for both single-host and cross-host configs — pick a config by setting `PR66_CONFIG=<id>`. Single-host configs run a single container and let slime build its own ray head; cross-host configs build the ray cluster externally (one container on each host with `--network host`) and pass `SLIME_SCRIPT_EXTERNAL_RAY=1`.

## Configs

| id | mode | hosts × GPU | `rollout_num_gpus_per_engine` | `num_gpus_per_node` | `nnodes_per_engine` | vLLM TP × PP | Megatron TP × PP × CP × EP |
|---|---|---|---|---|---|---|---|
| `ref` | colocate | 1 × 8  | 8  | 8 | 1 | 8 × 1  | 4 × 1 × 2 × 8 |
| `s1`  | colocate | 1 × 8  | 4  | 8 | 1 | 4 × 1  | 4 × 1 × 2 × 8 |
| `s2`  | colocate | 1 × 8  | 2  | 8 | 1 | 2 × 1  | 4 × 1 × 2 × 8 |
| `s3`  | colocate | 2 × 8  | 16 | 8 | **2** | 16 × 1 | 8 × 1 × 2 × 8 |
| `s4`  | colocate | 2 × 8  | 8  | 4 | **2** | 8 × 1  | 8 × 1 × 2 × 8 |
| `s5`  | colocate | 2 × 8  | 8  | 8 | 1 | 8 × 1  | 8 × 1 × 2 × 8 |
| `s6`  | disagg   | 1 × (4+4) | 4 | 4 | 1 | 4 × 1  | 4 × 1 × 1 × 4 |
| `s7`  | disagg   | 2 × (8+8) | 8 | 8 | 1 | 8 × 1  | 4 × 1 × 2 × 8 |
| `s8`  | colocate | 1 × 8  | 8  | 8 | 1 | 4 × **2** | 4 × 1 × 2 × 8 |
| `s9`  | colocate | 2 × 8  | 16 | 8 | **2** | 8 × **2** | 8 × 1 × 2 × 8 |
| `s10` | colocate | 1 × 8  | 8  | 8 | 1 | 8 × 1  | 4 × **2** × 1 × 4 |

The `nnodes_per_engine > 1` configs (`s3`, `s4`, `s9`) are the ones that exercise the new code in PR #66 — they cause `_compute_server_args` to emit `--nnodes 2 --node-rank …` and `_build_vllm_cmd_and_env` to add `--headless` + `--data-parallel-backend mp` + `--distributed-executor-backend mp`.

## Single-host run

```bash
# Inside one r3 container that has all 8 GPUs visible:
cd /root/slime
PR66_CONFIG=ref python examples/multinode_validation/test_qwen3_30B_A3B_pr66_sweep.py
```

(Replace `ref` with `s1`/`s2`/`s6`/`s8`/`s10` for the other single-host configs.)

## Cross-host run

The `launch_cross_host.sh` driver spawns one head container and one worker container, both with `--network host`, and waits on NFS-shared barrier files (`head-ready` → `worker-joined` → `test-done`).

```bash
HEAD_HOST=h200-0      WORKER_HOST=h200-1 \
HEAD_IP=172.27.48.114 WORKER_IP=172.27.55.233 \
IMAGE=inferactinc/public:vime-vllm-r3-latest \
LAN_IFACE=ens7 \
WORKSPACE=/path/to/nfs-shared/slime-workspace \
bash examples/multinode_validation/launch_cross_host.sh s3
```

`HEAD_HOST` / `WORKER_HOST` are SSH-reachable names; `HEAD_IP` / `WORKER_IP` are real LAN IPs on the interface NCCL/Gloo should use (`LAN_IFACE`).

The driver mounts `head.sh` / `worker.sh` from this directory into each container; they handle the `/etc/hosts` fix, build the ray cluster, run the test on the head, and tear down ray on `test-done`.

## Deployment requirements (cross-host)

These are environment requirements, not slime-code requirements; the PR's `nnodes > 1` path itself is correct.

1. **`--network host` on every container.** Docker bridge networks don't span hosts; vLLM's nnodes>1 setup needs cross-host TCP reachability for Ray + vLLM `master_addr` + Gloo full-mesh.

2. **Rewrite container `/etc/hosts`** to map the container's hostname to the real host IP instead of the default `127.0.0.1 <hostname>` / `127.0.1.1 <hostname>`. Gloo's full-mesh peer discovery advertises whatever IP the container resolves its own hostname to; without this rewrite cross-host peers try to connect to `127.0.1.1` and get `Connection refused`. Important: `sed -i` does not work on the bind-mounted `/etc/hosts` (writes a renamed file that no longer maps to the bind target). Use `cat > /etc/hosts` — see `head.sh` / `worker.sh`.

3. **Force NCCL / Gloo to use the LAN interface** when InfiniBand is unavailable (these env vars are set per-container by `launch_cross_host.sh`):
   - `NCCL_SOCKET_IFNAME=<lan_iface>`
   - `GLOO_SOCKET_IFNAME=<lan_iface>`
   - `NCCL_IB_DISABLE=1` if the IB interfaces are DOWN (otherwise NCCL retries IB and times out).

4. **`SLIME_SCRIPT_EXTERNAL_RAY=1` + `MASTER_ADDR=<head_ip>`** in the head's env before running `execute_train` — tells slime to skip its own `ray stop` / `ray start --head` and reuse the already-built cluster. Set by `head.sh`.

5. **`SLIME_HOST_IP=<real_host_ip>`** per container — `slime.utils.http_utils.get_host_info()` honors this env var; setting it bypasses any leftover `127.0.1.1` resolution in `ray._private.services.get_node_ip_address()`.

6. **`SLIME_VLLM_HEALTH_TIMEOUT_SEC=1200`** if running vLLM PP > 1 with a MoE model — the CUDA-graph capture phase under PP can run past the default 300 s health-check window. This env var is read by `vllm_engine._wait_server_healthy`.
