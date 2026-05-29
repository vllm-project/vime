#!/bin/bash
# Worker container entrypoint for cross-host PR #66 multi-node vLLM validation.
# See head.sh for the design overview.
#
# Expected env:
#   HEAD_IP=<real LAN IP of the head host>      e.g. 172.27.48.114
#   WORKER_IP=<real LAN IP of this host>        e.g. 172.27.55.233
#
# This script does not run the test itself; it joins the Ray cluster, waits
# until the head writes the ``test-done`` barrier, then exits.

exec > >(tee -a /root/runs/worker.log) 2>&1
set -e

: "${HEAD_IP:?HEAD_IP env var is required}"
: "${WORKER_IP:?WORKER_IP env var is required (real LAN IP of worker host)}"

BARRIER_DIR=/root/runs/barrier

# 1. /etc/hosts fix (same reason as head.sh).
TMP_HOSTS=$(mktemp)
grep -v "^127.0.1.1" /etc/hosts > "$TMP_HOSTS"
echo "${WORKER_IP} $(hostname)" >> "$TMP_HOSTS"
cat "$TMP_HOSTS" > /etc/hosts
echo "[worker] /etc/hosts after fix:"; cat /etc/hosts

# 2. Cleanup any leftover ray state.
ray stop --force 2>/dev/null || true
pkill -9 ray 2>/dev/null || true
sleep 2

# 3. Wait for head to be up.
for i in $(seq 1 600); do
    if [ -f $BARRIER_DIR/head-ready ]; then
        echo "[worker] head ready at iter $i"
        break
    fi
    sleep 1
done
if [ ! -f $BARRIER_DIR/head-ready ]; then
    echo "[worker] ERROR: head never ready" >&2
    exit 2
fi

# 4. Join the Ray cluster.
ray start --address=${HEAD_IP}:6379 --node-ip-address=${WORKER_IP} \
    --num-gpus=8 --disable-usage-stats 2>&1 | tail -10

touch $BARRIER_DIR/worker-joined
ray status

# 5. Sleep until head signals test-done (or 4h ceiling).
for i in $(seq 1 14400); do
    if [ -f $BARRIER_DIR/test-done ]; then
        echo "[worker] test done at iter $i"
        break
    fi
    sleep 1
done

ray stop --force 2>/dev/null || true
echo "[worker] exit"
