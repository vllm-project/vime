# Disk delta weight update smoke test

`test_delta_disk_weight_update.py` runs Qwen3-0.6B with XOR followed by
overwrite. Each encoding runs two rollout/training rounds in a separate
temporary directory. The second rollout exercises inference after the first
disk delta reload. The test checks versions 000001 and 000002, base versions,
encoding, zstd compression, Adler-32 metadata, nonempty delta shards, and the
final recovered tensors against the trainer's published checksums.

## Configuration

- Model: `Qwen/Qwen3-0.6B`; local HF weights: `/home/vllm/weights/Qwen3-0.6B`.
- Dataset: `zhuzilin/gsm8k`, `/root/datasets/gsm8k/train.parquet`;
  prompt field `messages`, answer field `label`, math reward.
- Four devices: one training device and three single-device rollout engines;
  tensor/pipeline/context/expert parallel sizes are one.
- Two rounds per encoding, four selected prompts per round, four samples per
  prompt, global batch size 16. Dynamic sampling uses batches of eight prompts
  and retains groups with nonzero reward standard deviation; generated sample
  count may exceed the 16 samples retained for training.
- Maximum response length 1024, temperature 0.8, GRPO, entropy coefficient 0.01,
  Adam learning rate 1e-6. Graph capture maximum size 32; eager is not enabled.
- NPU loads reference weights directly from HF; platforms with checkpoint
  conversion use `/dev/shm/Qwen3-0.6B_torch_dist`.

## Run

From the repository root in an environment configured for the project:

```bash
python3 tests/test_delta_disk_weight_update.py
```

The script downloads model/data, prepares the reference checkpoint where
required, and runs both encodings. It needs four free devices and write access
to `/home/vllm/weights`, `/root/datasets`, and the temporary directory. The test
removes temporary checkpoints after each encoding; checksum summaries remain
in stdout, so retain the log.

### Remote Ascend reproduction

The verified run used `root@192.168.13.190`, container `cjy-vime-0623`,
physical devices 8–11, CANN 9.0, PyTorch/torch_npu 2.10, and the patched vLLM
and vllm-ascend checkouts below. The inline Python only propagates these
container-specific dependency paths into the launcher runtime; the test
itself is unchanged. No separate shell script is needed.

```bash
ssh root@192.168.13.190
docker exec -it cjy-vime-0623 bash
source /home/w00899129/vllm-code/vllm_vime/vllm-ascend/vllm_ascend/_cann_ops_custom/vendors/custom_transformer/bin/set_env.bash
export PYTHONPATH=/home/w00899129/pr431-e2e-20261008/vime:/home/w00899129/pr431-e2e-20261008/vllm:/home/w00899129/pr431-e2e-20261008/vllm-ascend:/home/vllm/c00944022/vime-proj/MegatronAdaptor:/home/vllm/c00944022/vime-proj/TransformerEngineNPU:/home/vllm/c00944022/0623/Megatron-LM:${PYTHONPATH:-}
export VIME_PLATFORM=npu VLLM_VERSION=0.28.0 VLLM_BATCH_INVARIANT=0
export ASCEND_RT_VISIBLE_DEVICES=8,9,10,11
export RAY_EXPERIMENTAL_NOSET_ASCEND_RT_VISIBLE_DEVICES=1
export HF_ENDPOINT=https://hf-mirror.com HF_HUB_DISABLE_XET=1 WANDB_MODE=disabled
cd /home/w00899129/pr431-e2e-20261008/vime
python3 -u - <<'PY' 2>&1 | tee delta_disk_weight_update.log
import os
import runpy
from vime.utils.external_utils.launch import PLATFORMS

for name in ("PYTHONPATH", "LD_LIBRARY_PATH", "ASCEND_CUSTOM_OPP_PATH",
             "ASCEND_RT_VISIBLE_DEVICES", "VLLM_VERSION"):
    if name in os.environ:
        PLATFORMS["npu"].env[name] = os.environ[name]
runpy.run_path("tests/test_delta_disk_weight_update.py", run_name="__main__")
PY
```

## Verified results, 2026-10-09

| Encoding | Rollout rounds | Final local version | Final delta tensors checked | Result | Job elapsed |
| --- | ---: | --- | ---: | --- | ---: |
| XOR | 2 | 000002 | 240 | Passed | 7m 57s |
| overwrite | 2 | 000002 | 238 | Passed | 7m 40s |

Table note: tensor counts refer to changed tensors in the final delta, not all
model tensors (the baseline snapshot contains 310 tensors). Job elapsed is
Ray submission-success timestamp to Ray job-success timestamp, including
worker startup, graph compilation, rollout, training and weight updates;
it excludes downloads and the final Python checksum verification.
XOR ran 10:31:11.854–10:39:08.966 and overwrite ran
10:39:43.193–10:47:23.372, Asia/Shanghai. Both jobs succeeded.

| Encoding | Post-training update v1 | Post-training update v2 | Mean response tokens, rounds 1 / 2 | Raw reward, rounds 1 / 2 | Truncation, rounds 1 / 2 |
| --- | ---: | ---: | --- | --- | --- |
| XOR | 11.7 s | 4.4 s | 587.25 / 915.06 | 0.6875 / 0.5625 | 12.5% / 43.75% |
| overwrite | 11.2 s | 3.9 s | 581.75 / 741.31 | 0.5625 / 0.7500 | 12.5% / 31.25% |

Table note: update times are the logged `Timer update_weights` intervals after
each training round, including disk publication and engine synchronization;
they are not isolated engine reload times. Initial baseline synchronization
was 4.0 s for XOR and 2.3 s for overwrite and is excluded from these columns.
Response length, raw reward and truncation describe the dynamically selected
16-sample training batches, not an unbiased GSM8K evaluation. These small runs
verify functionality, not accuracy equivalence or a performance ranking.

Final stdout:

```text
Disk delta validation passed: {'encoding': 'xor', 'rollout_rounds': 2, 'local_version': '000002', 'checked_tensors': 240}
Disk delta validation passed: {'encoding': 'overwrite', 'rollout_rounds': 2, 'local_version': '000002', 'checked_tensors': 238}
```

Remote log: `/home/w00899129/pr431-qwen3-06b-delta-20261009/run.log`.
Ray jobs: `raysubmit_1ENHsN8gGFUPhXU8` (XOR),
`raysubmit_z5tPvPJ3B8sY2GvX` (overwrite).
Nonfatal `AscendRotaryEmbedding: Failed to load weights` and Ray disk-space
warnings were present; the jobs and checksum assertions still passed.
