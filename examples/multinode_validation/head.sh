#!/bin/bash
# Head container entrypoint for cross-host PR #66 multi-node vLLM validation.
#
# Expected env (set by the docker run on the launcher side):
#   PR66_CONFIG=s3|s4|s5|s7|s9                config id from the test file
#   HEAD_IP=<real LAN IP of this host>        e.g. 172.27.48.114
#
# Mounts expected:
#   /root/slime                               the slime workspace (the PR branch)
#   /root/runs                                NFS-shared per-run dir (for barrier files + log)
#   /root/runs/barrier/                       NFS-shared barrier dir (auto-created)
#
# Notes:
#   - /etc/hosts in docker maps the container hostname to 127.0.1.1 by default;
#     Gloo's full-mesh peer discovery then advertises 127.0.1.1 to remote peers,
#     who get "Connection refused" since their loopback has no such listener.
#     We rewrite /etc/hosts in-place via `cat >` (sed -i breaks the bind mount).
#   - SLIME_SCRIPT_EXTERNAL_RAY=1 + MASTER_ADDR=<head_ip> tell slime to reuse
#     this externally-built Ray cluster instead of running `ray start --head` itself.

# Mirror all stdout/stderr to NFS log (survives container --rm).
exec > >(tee -a /root/runs/run.log) 2>&1
set -e

: "${PR66_CONFIG:?PR66_CONFIG env var is required (e.g. s3)}"
: "${HEAD_IP:?HEAD_IP env var is required (real LAN IP of head host)}"

BARRIER_DIR=/root/runs/barrier
mkdir -p $BARRIER_DIR

# 1. Fix hostname → real host IP in /etc/hosts (Gloo full-mesh reachability).
TMP_HOSTS=$(mktemp)
grep -v "^127.0.1.1" /etc/hosts > "$TMP_HOSTS"
echo "${HEAD_IP} $(hostname)" >> "$TMP_HOSTS"
cat "$TMP_HOSTS" > /etc/hosts
echo "[head] /etc/hosts after fix:"; cat /etc/hosts

# 2. Start Ray head; signal the barrier file so the worker can join.
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 2

ray start --head --node-ip-address=${HEAD_IP} --port=6379 \
    --num-gpus=8 --disable-usage-stats \
    --dashboard-host=0.0.0.0 --dashboard-port=8265 2>&1 | tail -10

touch $BARRIER_DIR/head-ready
echo "[head] signaled head-ready, waiting for worker (timeout 600s)..."

# 3. Wait for the worker to join the cluster.
for i in $(seq 1 600); do
    if [ -f $BARRIER_DIR/worker-joined ]; then
        echo "[head] worker joined at iter $i"
        break
    fi
    sleep 1
done
if [ ! -f $BARRIER_DIR/worker-joined ]; then
    echo "[head] ERROR: worker never joined" >&2
    exit 2
fi

ray status

# 4. Run the test. SLIME_SCRIPT_EXTERNAL_RAY tells slime not to ray-stop/start.
cd /root/slime
export SLIME_SCRIPT_EXTERNAL_RAY=1
export MASTER_ADDR=${HEAD_IP}
python tests/test_qwen3_30B_A3B_pr66_sweep.py
TEST_EXIT=$?

# 5. Signal worker to exit and stop Ray.
touch $BARRIER_DIR/test-done
ray stop --force 2>/dev/null || true
exit $TEST_EXIT
