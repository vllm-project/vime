# Ascend disk weight synchronization

Run these smoke cases in an Ascend environment built with this branch's
`docker/npu_patch/vllm.patch` and `vllm-ascend.patch`. The existing Dockerfile
and patch series already apply both files. Python dependencies are declared in
`requirements.txt` (`zstandard`, `xxhash`, and `blake3`).

Both cases use Qwen3-4B, four training NPUs and four rollout NPUs, with three
rollout iterations by default. Set `DATA_ROOT` to a directory containing
`models/Qwen3-4B` and `datasets/dapo-math-17k/dapo-math-17k.jsonl`, as in the
existing Qwen3-4B NPU example. Run one case at a time on allocated devices.

```bash
ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DATA_ROOT=/root \
  bash scripts/run-qwen3-4B-full-disk.sh

ASCEND_RT_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 DATA_ROOT=/root \
  bash scripts/run-qwen3-4B-delta-disk.sh
```

Set `UPDATE_WEIGHT_DISK_DIR` to a dedicated directory shared at the same path
between trainer and rollout hosts. The `/tmp` default is for a single host.
For delta mode, set `UPDATE_WEIGHT_LOCAL_CHECKPOINT_DIR` to a dedicated writable
directory on each rollout host. Existing delta-stream files are cleared when
capturing the initial baseline; do not share these directories between jobs.
`NUM_ROLLOUT`, `RAY_GCS_PORT`, `RAY_DASHBOARD_PORT`, and `RAY_TEMP_DIR` can be
overridden. The launchers do not terminate existing processes. They require a
clean, dedicated Ray runtime; stop the runtime you started after the run.

Full flow: the Megatron actor selects `UpdateWeightFromDisk`; all ranks take
part in HF conversion, rank zero writes safetensors shards and their index,
then rank zero pauses rollout, flushes its cache and calls
`VLLMEngine.update_weights_from_disk`. `/collective_rpc` dispatches
`reload_weights(weights_path=...)` to the NPU workers before generation resumes.

Delta flow: the first update captures the original HF checkpoint as version
zero. Later updates publish Zstandard-compressed XOR or overwrite deltas and
checksums. `VLLMEngine.pull_weights` dispatches `/collective_rpc` to every NPU
worker. The mainline local-checkpoint helper locks each host's checkpoint,
applies each version once and verifies the resulting bytes; the engine then
reloads that local checkpoint through the same full reload interface.

Disk synchronization currently requires non-colocated training and rollout.
The existing default collective transport remains unchanged. These launchers
exercise end-to-end training; protocol unit tests additionally check exported
full shards, both delta encodings, unchanged versions and repeated pulls:

```bash
python3 -m pytest -q tests/unit/backends/megatron_utils/update_weight/test_disk_weight_sync.py
```
