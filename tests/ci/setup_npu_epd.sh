#!/usr/bin/env bash
# ECCPUConnector transfers CPU image features; this is independent of Ray NIXL.
set -euo pipefail
# Pin both wheels and preserve the image's torch/torch-npu/numpy stack.
python -m pip install --only-binary=:all: --no-deps 'nixl==1.4.1' 'nixl-cu12==1.4.1'
export UCX_TLS=tcp,sm,self
python tests/ci/npu_pd_environment.py --encoder
