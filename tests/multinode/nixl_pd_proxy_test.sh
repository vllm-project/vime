#!/bin/bash
# vLLM NIXL PD through slime's pd_proxy over /inference/v1/generate (the exact
# endpoint slime's rollout uses). Run INSIDE an r3 container with >=2 GPUs and
# /root/slime = the PR workspace. Launches prefill(GPU0)+decode(GPU1)+pd_proxy,
# registers the engines, sends one GenerateRequest through the proxy.
set -u
MODEL="${MODEL:-/root/models/Qwen3-0.6B}"
LOG=/root/runs/nixl_pd_proxy_test.log
exec > >(tee -a "$LOG") 2>&1
echo "=== NIXL PD proxy test $(date -u) model=$MODEL ==="
cd /root/slime || exit 3

CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=7557 \
  vllm serve "$MODEL" --port 8100 --enforce-eager --gpu-memory-utilization 0.4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","engine_id":"prefill0"}' \
  > /root/runs/pxy_prefill.log 2>&1 &
PF=$!
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=7558 \
  vllm serve "$MODEL" --port 8200 --enforce-eager --gpu-memory-utilization 0.4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","engine_id":"decode0"}' \
  > /root/runs/pxy_decode.log 2>&1 &
DC=$!
python3 -m slime.backends.vllm_utils.pd_proxy --host 127.0.0.1 --port 9000 > /root/runs/pxy_proxy.log 2>&1 &
PX=$!
echo "prefill=$PF decode=$DC proxy=$PX; waiting for health..."

for i in $(seq 1 120); do
  curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && \
  curl -sf http://127.0.0.1:8200/health >/dev/null 2>&1 && \
  curl -sf http://127.0.0.1:9000/health >/dev/null 2>&1 && break
  kill -0 $PF 2>/dev/null || { echo PREFILL_DIED; tail -20 /root/runs/pxy_prefill.log; exit 4; }
  kill -0 $DC 2>/dev/null || { echo DECODE_DIED; tail -20 /root/runs/pxy_decode.log; exit 4; }
  kill -0 $PX 2>/dev/null || { echo PROXY_DIED; tail -20 /root/runs/pxy_proxy.log; exit 4; }
  sleep 5
done
echo "healthy after $((i*5))s"

# Does the engine expose /inference/v1/generate?
echo "--- probe /inference/v1/generate route on prefill ---"
curl -s -o /dev/null -w "prefill /inference/v1/generate HTTP %{http_code}\n" \
  -X POST http://127.0.0.1:8100/inference/v1/generate -H 'Content-Type: application/json' -d '{}'

# Register engines with the proxy (slime engines do this automatically).
curl -s -X POST http://127.0.0.1:9000/workers -H 'Content-Type: application/json' \
  -d '{"url":"http://127.0.0.1:8100","worker_type":"prefill"}'; echo
curl -s -X POST http://127.0.0.1:9000/workers -H 'Content-Type: application/json' \
  -d '{"url":"http://127.0.0.1:8200","worker_type":"decode"}'; echo
echo "registered workers:"; curl -s http://127.0.0.1:9000/workers; echo

python3 - "$MODEL" <<'PYEOF'
import sys, json, requests
from transformers import AutoTokenizer
model = sys.argv[1]
tok = AutoTokenizer.from_pretrained(model)
ids = tok("The capital of France is", return_tensors=None)["input_ids"]
req = {"token_ids": ids, "sampling_params": {"max_tokens": 16, "temperature": 0.0}, "stream": False}
r = requests.post("http://127.0.0.1:9000/inference/v1/generate", json=req, timeout=180)
print("proxy HTTP", r.status_code)
j = r.json()
print("resp keys:", list(j.keys()))
ch = (j.get("choices") or [{}])[0]
out_ids = ch.get("token_ids") or ch.get("output_token_ids") or []
print("decoded:", repr(tok.decode(out_ids)) if out_ids else "(no token_ids; raw="+json.dumps(j)[:300]+")")
assert r.status_code == 200 and out_ids, "PD-proxy generate failed"
print("=== NIXL PD PROXY (/inference/v1/generate) PROOF: PASS ===")
PYEOF
RC=$?
echo "test rc=$RC"
kill $PF $DC $PX 2>/dev/null; sleep 2
exit $RC
