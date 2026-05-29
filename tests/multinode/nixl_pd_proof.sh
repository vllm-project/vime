#!/bin/bash
# Standalone vLLM NIXL 1P1D proof — run INSIDE an r3 container with >=2 GPUs.
# Launches a prefill engine (GPU0) + decode engine (GPU1), both NixlConnector
# kv_both, then relays one /v1/completions request P->D and checks output.
set -u
MODEL="${MODEL:-/root/models/Qwen3-0.6B}"
LOG=/root/runs/nixl_pd_proof.log
exec > >(tee -a "$LOG") 2>&1
echo "=== NIXL 1P1D proof $(date -u) model=$MODEL ==="

command -v vllm >/dev/null || { echo "no vllm"; exit 3; }
[ -d "$MODEL" ] || { echo "model dir missing: $MODEL"; exit 3; }

# Prefill on GPU0, side-channel port 7557; Decode on GPU1, side-channel port 7558.
CUDA_VISIBLE_DEVICES=0 VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=7557 \
  vllm serve "$MODEL" --port 8100 --enforce-eager --gpu-memory-utilization 0.4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","engine_id":"prefill0"}' \
  > /root/runs/nixl_prefill.log 2>&1 &
PF=$!
CUDA_VISIBLE_DEVICES=1 VLLM_NIXL_SIDE_CHANNEL_HOST=127.0.0.1 VLLM_NIXL_SIDE_CHANNEL_PORT=7558 \
  vllm serve "$MODEL" --port 8200 --enforce-eager --gpu-memory-utilization 0.4 \
  --kv-transfer-config '{"kv_connector":"NixlConnector","kv_role":"kv_both","engine_id":"decode0"}' \
  > /root/runs/nixl_decode.log 2>&1 &
DC=$!

echo "prefill PID=$PF decode PID=$DC; waiting for health..."
ok=0
for i in $(seq 1 120); do
  if curl -sf http://127.0.0.1:8100/health >/dev/null 2>&1 && curl -sf http://127.0.0.1:8200/health >/dev/null 2>&1; then ok=1; break; fi
  if ! kill -0 $PF 2>/dev/null; then echo "PREFILL died"; tail -30 /root/runs/nixl_prefill.log; exit 4; fi
  if ! kill -0 $DC 2>/dev/null; then echo "DECODE died"; tail -30 /root/runs/nixl_decode.log; exit 4; fi
  sleep 5
done
[ $ok -eq 1 ] || { echo "health timeout"; tail -20 /root/runs/nixl_prefill.log /root/runs/nixl_decode.log; kill $PF $DC; exit 4; }
echo "both healthy after $((i*5))s; running P->D relay"

python3 - "$MODEL" <<'PYEOF'
import sys, json, requests
model = sys.argv[1]
prompt = "The capital of France is"
# 1) prefill: 1 token, mark for remote decode
pf = requests.post("http://127.0.0.1:8100/v1/completions", json={
    "model": model, "prompt": prompt, "max_tokens": 1, "temperature": 0.0,
    "kv_transfer_params": {"do_remote_decode": True, "do_remote_prefill": False,
                           "remote_engine_id": None, "remote_block_ids": None,
                           "remote_host": None, "remote_port": None},
}, timeout=120).json()
print("PREFILL resp keys:", list(pf.keys()))
ktp = pf.get("kv_transfer_params")
print("PREFILL kv_transfer_params:", json.dumps(ktp))
assert ktp, "prefill did not return kv_transfer_params — NIXL handshake not engaged"
# 2) decode: full gen, consume remote prefill KV
dc = requests.post("http://127.0.0.1:8200/v1/completions", json={
    "model": model, "prompt": prompt, "max_tokens": 16, "temperature": 0.0,
    "kv_transfer_params": ktp,
}, timeout=120).json()
txt = dc["choices"][0]["text"]
print("DECODE text:", repr(txt))
assert txt.strip(), "decode produced empty output"
print("=== NIXL 1P1D PROOF: PASS ===")
PYEOF
RC=$?
echo "relay rc=$RC"
kill $PF $DC 2>/dev/null
sleep 2
exit $RC
