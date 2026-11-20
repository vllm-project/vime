# Observability

vime's default observability path is intentionally small: training metrics still go to W&B / TensorBoard; high-frequency vLLM Prometheus metrics are no longer uploaded to W&B; request timings from vLLM response `metrics` are stored in sample traces and aggregated once per rollout step as compact `perf/...` metrics.

## W&B / TensorBoard Metrics

W&B and TensorBoard still receive reward, loss, KL, entropy, eval, and other training metrics. vLLM request timing summaries are logged under `perf/`, for example:

```text
perf/request/e2e_latency/mean
perf/request/queue_time/median
perf/request/count
perf/request/profiled_count
perf/decode/throughput/mean
perf/prefill/queue_duration/mean
perf/prefill/ttft_duration/max
perf/prefill/transfer_speed_gb_s/mean
perf/decode/remote_kv_wait_duration/mean
perf/decode/allocation_wait_duration/mean
perf/decode/transfer_worker_duration/max
perf/decode/handshake_wait_worker_duration/mean
perf/decode/transfer_post_worker_duration/mean
```

These metrics are aggregated once per rollout step, not emitted once per request, so they should not slow W&B like uploading raw Prometheus metrics would.

Without PD, common `perf/request/...` metrics and available `perf/decode/throughput/...` metrics still exist. Detailed `perf/prefill/...` and `perf/decode/...duration` metrics only appear when vLLM returns the corresponding timing fields, which vime translates into `pd_*` trace attributes.

## Where Prometheus Data Is Stored

vime does not store per-second Prometheus data. vLLM exposes `/metrics` on each engine's HTTP port; the router exposes its own metrics on a separate Prometheus port. Prometheus scrapes those endpoints periodically and stores the time series in Prometheus's own TSDB.

That means:

- If Prometheus is not running, serving metrics are only available from the current vLLM process memory and endpoint output, with no historical storage.
- If Prometheus is running, history is stored under Prometheus's `--storage.tsdb.path`.
- vime does not upload these high-frequency metrics to W&B.

Useful vLLM metrics include:

```text
vllm:num_requests_waiting
vllm:num_requests_running
vllm:num_requests_waiting_by_reason
vllm:nixl_xfer_time_seconds_bucket
vllm:nixl_bytes_transferred_bucket
vllm:mooncake_xfer_time_seconds_bucket
vllm:mooncake_bytes_transferred_bucket
```

Connector metrics depend on the selected backend: `nixl_*` for NIXL, `mooncake_*` for Mooncake. These are useful in Prometheus / Grafana for live queue buildup, transfer volume, latency histograms, and other serving-side symptoms.

## Starting Prometheus

Prometheus must run while training is running because it can only scrape endpoints that are currently alive. It does not need to run inside the training Python process; run it as a side process in the same machine or job.

A minimal config is:

```yaml
global:
  scrape_interval: 10s

scrape_configs:
  - job_name: vime-vllm
    metrics_path: /metrics
    static_configs:
      - targets:
          - "ENGINE_IP:ENGINE_PORT"
  - job_name: vime-router
    metrics_path: /metrics
    static_configs:
      - targets:
          - "ROUTER_IP:ROUTER_PROMETHEUS_PORT"
```

List each engine's HTTP address under `vime-vllm`. For `vime-router`, use the router address and the separate `Prometheus port` printed by vime, not `--vllm-router-port`.

Start Prometheus with its TSDB path on persistent storage:

```bash
prometheus \
  --config.file=/path/to/prometheus.yml \
  --storage.tsdb.path=/path/to/prometheus-data \
  --storage.tsdb.retention.time=7d \
  --web.listen-address=0.0.0.0:9090
```

The vime image includes the `prometheus` binary, so this command can run directly inside the container. You can also start a side container from the same image as long as it can reach the configured engine and router scrape targets and mounts `/path/to/prometheus-data` on persistent storage.

If `--storage.tsdb.path` points to container-local disk, the data is lost when the container is removed. If it points to NFS, a persistent volume, or a job output directory, you can restart Prometheus with the same TSDB directory after training and query the historical time range in the Prometheus UI or Grafana. This is time-series replay, not full per-request trace replay; per-sample request timings still come from sample traces / debug rollout data.

## Trace Viewer

Debug rollout dumps saved with `--save-debug-rollout-data` include sample traces. The trace viewer reads vLLM timing attrs directly from those traces and uses `pd_*` fields to render synthetic `[P]` / `[D]` lanes.

```bash
python tools/trace_timeline_viewer.py /path/to/debug/rollout_0.pt
```

The default path does not require separate `ReqTimeStats(...)` logs, Loki, or a compaction tool.
