# 性能分析（Profiling）

在vime中，我们可以通过vLLM提供的profiling接口对**rollout（vLLM推理）**过程做详细的性能分析。Profiling针对vLLM engine侧，不是Megatron训练侧。

典型流程：

- 启动train（sleep_rollout + vllm-profiler-config）
- 等待vLLM engine与router就绪
- 从日志确认router/worker地址
- start_profile
- 发送少量推理请求
-（可选）stop_profile；或达到max_iterations后自动写入 trace
- 在torch_profiler_dir查看trace文件



## 1. 使Rollout进入等待状态（sleep_rollout）

为了更灵活地压测和profiling，通常让rollout在初始化完成后进入等待，而不是立即开始生成。

在 `train.py` 启动参数中替换 `rollout_function_path` 即可，无需改代码：

```bash
python train.py \
    --rollout-function-path vime.rollout.sleep_rollout.sleep \
    ... (其他参数)
```

该函数会让rollout进程进入无限循环等待，便于手动发HTTP请求或运行压测工具。

## 2. 启用vLLM Profiler（启动train时配置）

vLLM只有在启动时配置了`--profiler-config`，才会注册`/start_profile`与`/stop_profile`路由。在vime中通过**`--vllm-profiler-config`**转发给`vllm serve`子进程。

### 2.1 使用JSON整包传参

```bash
--vllm-profiler-config '{"profiler":"torch","torch_profiler_dir":"/root/logs/vllm_profile","max_iterations":3,"ignore_frontend":true}'
```

常用JSON字段：

| 字段 | 说明 |
|------|------|
| `profiler` | `"torch"` 或 `"cuda"` |
| `torch_profiler_dir` | trace输出目录（绝对路径） |
| `max_iterations` | worker记录超过N步后自动stop并写入 trace（条件为`> N`） |
| `ignore_frontend` | 建议`true`，仅profile worker，降低前端开销 |

**防止`stop_profile`时RPC超时：** vLLM APIServer与EngineCore/worker之间通过内部RPC通信。手动调用`stop_profile`把 trace 写出来可能耗时数分钟，而默认`VLLM_RPC_TIMEOUT`仅**10秒**（10000 ms），容易导致flush中断或trace不完整。Profiling时建议设为**30分钟**（1800000 ms）。

该变量须在**启动train、拉起vLLM之前**传入Ray worker环境（仅在本机shell `export`不一定会进入Ray job）。在`ray job submit`的`runtime-env-json`中写入，例如：

```bash
export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"

RUNTIME_ENV_JSON="{
  \"env_vars\": {
    \"PYTHONPATH\": \"/root/Megatron-LM\",
    \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
    \"VLLM_RPC_TIMEOUT\": \"${VLLM_RPC_TIMEOUT}\"
  }
}"

ray job submit --address=\"http://127.0.0.1:8265\" \
  --runtime-env-json=\"${RUNTIME_ENV_JSON}\" \
  -- python3 train.py \
  ... \
  --vllm-profiler-config '{\"profiler\":\"torch\",\"torch_profiler_dir\":\"/root/logs/vllm_profile\",...}'
```


### 2.2 验证是否生效

启动train后，在日志中确认以下三点（缺任一项说明profiler未正确启用）：

1. **参数已解析**：出现`vllm_profiler_config ... profiler='torch'`（及`torch_profiler_dir`路径）。
2. **已转发给vLLM子进程**：出现`Launching vLLM server: ... --profiler-config {"profiler":"torch",...}`。
3. **HTTP路由已注册**：vLLM启动时的路由列表中包含`/start_profile`与`/stop_profile`（否则`POST /start_profile`会返回404）。

## 3. 获取Router与Worker地址

vLLM engine（workers）注册在vllm-router上。启动日志示例：

```text
Router launched at 127.0.0.1:3521, Prometheus port: 4153
Ports for engine 0: {'host': '127.0.0.1', 'port': 15000, ...}
Starting vLLM server on http://127.0.0.1:15000
```

**注意：router端口每次job可能变化**（默认在3000–4000随机），不要沿用上次端口。可用curl验证：

```bash
curl http://127.0.0.1:3521/workers
```

返回每个worker的`url`与`is_healthy`。

## 4. 使用`tools/profile_rollout.py`

脚本通过router的`/workers`列表，对所有worker调用`/start_profile`或`/stop_profile`。

### 启动Profiling

```bash
cd /root/vime
python tools/profile_rollout.py \
    --router-url http://127.0.0.1:3521 \
    --action start
```

### 停止Profiling（可选）

若在`--vllm-profiler-config`中设置了`max_iterations`，worker在记录足够步数后会**自动stop并写入 trace**，实践中发完推理后常可直接在`torch_profiler_dir`看到trace，**不必**再手动`stop_profile`。需要提前结束采集时再执行：

```bash
python tools/profile_rollout.py \
    --router-url http://127.0.0.1:3521 \
    --action stop
```

## 5. 发送推理请求

在sleep_rollout等待期间，执行步骤如下：

1. `profile_rollout.py --action start`
2. 向router或**直连worker**发送少量completion请求（通常2～4条即可，trace会很大）
3. 如果依赖自动写入 trace，要注意 `max_iterations` 的停止条件是 `> N`。例如 `max_iterations=3` 时，需要发 4 条请求；否则请手动执行 `profile_rollout.py --action stop`
4. 在`torch_profiler_dir`查看trace

请求示例（`model`使用HF checkpoint路径）：

```bash
curl -X POST http://127.0.0.1:15000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"/root/models/Qwen3-4B","prompt":"Hello","max_tokens":32}'
```


## 6. 查看Trace

### Perfetto

1. 打开 [https://ui.perfetto.dev/](https://ui.perfetto.dev/)
2. **Open trace file**，选择`*.trace.json.gz`
3. 查看GPU kernel、CPU算子与时间线

### Chrome Tracing

浏览器访问`chrome://tracing`，Load加载trace文件。

### 分析工具

```bash
cd /root/vime
python tools/analyze_profile.py --profile-dir /root/logs/vllm_profile --all-ranks
```


## 7. 常见问题

| 现象 | 处理 |
|------|------|
| `POST /start_profile` 404 | 用JSON传`--vllm-profiler-config`；重启job |
| start成功但目录为空 | 确认curl打到worker且返回200；若 `max_iterations=3`，请发 4 条请求，或手动执行 `stop_profile` |
| router 503 | 确认当前job的router端口；改直连worker |
| stop很慢或超时 | 增大`VLLM_RPC_TIMEOUT`；减少请求条数 |

## 8. 完整可运行示例

以下脚本假设在**容器内**、vime仓库位于`/root/vime`，模型与数据在`/root/models`、`/root/data`。分两段：

1. **`launch_train_for_profiling`**：启动带profiler的train（sleep_rollout，单卡colocate最小示例，可按机器改GPU数）。
2. **`run_profiling_session`**：train就绪后，在**另一个终端**执行profiling。

将脚本保存为 `/root/vime/run_profiling_demo.sh` 后执行。

```bash
#!/usr/bin/env bash
#
# vime rollout profiling 完整示例
# 用法:
#   bash /root/vime/run_profiling_demo.sh launch    # 终端1：启动train
#   bash /root/vime/run_profiling_demo.sh profile   # 终端2：train就绪后抓trace
#
set -euo pipefail

VIME_ROOT="${VIME_ROOT:-/root/vime}"
HF_CKPT="${HF_CKPT:-/root/models/Qwen3-4B}"
REF_LOAD="${REF_LOAD:-/root/models/Qwen3-4B_torch_dist}"
PROMPT_DATA="${PROMPT_DATA:-/root/data/gsm8k/train.parquet}"
LOG_ROOT="${LOG_ROOT:-/root/logs/vime_profiling}"
PROFILE_DIR="${PROFILE_DIR:-/root/logs/vllm_profile}"
TRAIN_LOG="${LOG_ROOT}/train_profiling.log"
ROUTER_HOST="${ROUTER_HOST:-127.0.0.1}"

mkdir -p "${LOG_ROOT}" "${PROFILE_DIR}"

VLLM_PROFILER_CONFIG_JSON="$(printf \
  '{"profiler":"torch","torch_profiler_dir":"%s","max_iterations":3,"ignore_frontend":true}' \
  "${PROFILE_DIR}")"

launch_train_for_profiling() {
  cd "${VIME_ROOT}"

  # 清理旧 Ray / vLLM 进程（按需注释）
  ray stop --force || true
  pkill -9 vllm || true
  sleep 2

  ray start --head --node-ip-address 127.0.0.1 --num-gpus 2 --disable-usage-stats

  source "${VIME_ROOT}/scripts/models/qwen3-4B.sh"

  export VLLM_RPC_TIMEOUT="${VLLM_RPC_TIMEOUT:-1800000}"

  RUNTIME_ENV_JSON="{
    \"env_vars\": {
      \"PYTHONPATH\": \"/root/Megatron-LM\",
      \"CUDA_DEVICE_MAX_CONNECTIONS\": \"1\",
      \"VLLM_RPC_TIMEOUT\": \"${VLLM_RPC_TIMEOUT}\"
    }
  }"

  echo "=== Launching train; log: ${TRAIN_LOG} ==="
  echo "=== After engines are up, run: bash $0 profile ==="

  ray job submit --address="http://127.0.0.1:8265" \
    --runtime-env-json="${RUNTIME_ENV_JSON}" \
    -- python3 train.py \
      --train-backend megatron \
      --colocate \
      --actor-num-nodes 1 \
      --actor-num-gpus-per-node 1 \
      --rollout-num-gpus 1 \
      --rollout-num-gpus-per-engine 1 \
      --rollout-function-path vime.rollout.sleep_rollout.sleep \
      --hf-checkpoint "${HF_CKPT}" \
      --ref-load "${REF_LOAD}" \
      --prompt-data "${PROMPT_DATA}" \
      --input-key question \
      --label-key label \
      --apply-chat-template \
      --rm-type deepscaler \
      --num-rollout 1 \
      --rollout-batch-size 4 \
      --n-samples-per-prompt 1 \
      --rollout-max-response-len 512 \
      --global-batch-size 4 \
      --vllm-gpu-memory-utilization 0.7 \
      --vllm-profiler-config "${VLLM_PROFILER_CONFIG_JSON}" \
      ${MODEL_ARGS[@]} \
      2>&1 | tee "${TRAIN_LOG}"
}

discover_router_url() {
  local line port
  line="$(grep -E 'Router launched at' "${TRAIN_LOG}" | tail -1 || true)"
  if [[ -z "${line}" ]]; then
    echo "ERROR: Router not found in ${TRAIN_LOG}. Is train still starting?" >&2
    exit 1
  fi
  # Router launched at 127.0.0.1:3521, Prometheus port: ...
  port="$(echo "${line}" | sed -n 's/.*Router launched at [^:]*:\([0-9]*\).*/\1/p')"
  echo "http://${ROUTER_HOST}:${port}"
}

discover_worker_url() {
  local router_url="$1"
  python3 - <<'PY' "${router_url}"
import json, sys, urllib.request
router = sys.argv[1]
with urllib.request.urlopen(f"{router}/workers", timeout=10) as r:
    workers = json.load(r).get("workers", [])
if not workers:
    raise SystemExit("No workers registered")
print(workers[0]["url"])
PY
}

run_profiling_session() {
  cd "${VIME_ROOT}"

  local router_url worker_url model="${HF_CKPT}"
  router_url="$(discover_router_url)"
  worker_url="$(discover_worker_url "${router_url}")"

  echo "=== ROUTER=${router_url} WORKER=${worker_url} PROFILE_DIR=${PROFILE_DIR} ==="

  echo "=== 1/3 start_profile (all workers via router) ==="
  python tools/profile_rollout.py --router-url "${router_url}" --action start

  echo "=== 2/3 send completions (direct to worker; 4 requests so max_iterations=3 can auto-flush) ==="
  for i in 1 2 3 4; do
    response="$(curl -sS -X POST "${worker_url}/v1/completions" \
      -H "Content-Type: application/json" \
      -d "{\"model\":\"${model}\",\"prompt\":\"Hello ${i}\",\"max_tokens\":32}")"
    printf '%s\n' "${response:0:400}"
  done

  echo "=== 3/3 list trace files (max_iterations=3 auto-stop uses > N; add --action stop if needed) ==="
  sleep 2
  find "${PROFILE_DIR}" -type f \( -name '*.json*' -o -name 'profiler_out_*' \) | sort
  echo "Open *.trace.json.gz in https://ui.perfetto.dev/ or run:"
  echo "  python tools/analyze_profile.py --profile-dir ${PROFILE_DIR} --all-ranks"
}

case "${1:-}" in
  launch)  launch_train_for_profiling ;;
  profile) run_profiling_session ;;
  *)
    echo "Usage: $0 {launch|profile}" >&2
    exit 1
    ;;
esac
```

**操作步骤：**

```bash
# 终端1：启动train（等待vLLM与router就绪，日志出现Router launched at ...）
bash /root/vime/run_profiling_demo.sh launch

# 终端2：抓trace
bash /root/vime/run_profiling_demo.sh profile
```

按需修改脚本顶部的`/root/models/...`、`/root/data/...`与GPU布局（`actor-num-gpus-per-node`、`rollout-num-gpus`等）。
