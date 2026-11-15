# DeepSeek-V4-Flash-Base integration status

`scripts/models/deepseek-v4-flash.sh` selects Megatron's native `dsv4_hybrid`
spec for a 43-layer model with MTP disabled. The conversion code loads the
local Flash-Base checkpoint's `layers.*` FP8 blocks into BF16 Megatron weights
and exports native `.weight` / `.scale` pairs for vLLM. Hash-MoE `tid2eid` and
learned-router `expert_bias` buffers are included in direct loads and weight
sync. No V4 attention, compressor, indexer, or mHC kernel is vendored here.

The reference checkpoint is `/home/ljj/models/DeepSeek-V4-Flash-Base`. The
checkpoint uses FP8 E4M3 weights, including routed experts, with UE8M0 block
scales. Packed FP4 expert checkpoints are not supported by this converter.
Validation dependency pins are Megatron `ae76c60e6`, vLLM `83c078d95`, and
fast-hadamard-transform `f134af63d`. The latter is installed by the Dockerfile;
without it, Megatron's compressed-layer forward fails.

Validated on NVIDIA H20-3e:

- Checkpoint mapping, parser options, FP8 decoding/quantization, and conversion
  regressions: 82 passing tests in the pinned vLLM base image. The native
  four-layer name check passed earlier in a Transformer Engine environment;
  it skips when Transformer Engine is unavailable.
- Megatron's Flash attention parity tests: six forward/backward cases covering
  window-only, ratio-4, ratio-128, and packed THD. Two hash-MoE integration
  tests passed.
- Full-width layer 2 (ratio 4, hash router): 544 checkpoint tensors loaded,
  finite loss and 543 finite gradient tensors. Layer 3 (ratio 128, learned
  router): 538 tensors loaded, finite loss and 537 finite gradient tensors.
- The full 46-shard, 274 GiB checkpoint loaded into four H20 GPUs with vLLM's
  Triton MoE backend. A short rollout generated four tokens with logprobs.

Reproduce the targeted checks inside the pinned training and vLLM environments:

```bash
export PYTHONPATH=/home/ljj/vime-dsv4/.test-deps:/home/ljj/vime-dsv4/.deps/Megatron-LM:/home/ljj/vime-dsv4
export DSV4_CHECKPOINT=/home/ljj/models/DeepSeek-V4-Flash-Base
python -m pytest tests/test_deepseek_v4_weights.py tests/test_hf_to_megatron.py
python -m pytest tests/test_deepseek_v4_native_mapping.py
python tools/dsv4_checkpoint_layer_smoke.py "$DSV4_CHECKPOINT" --layer-index 2
python tools/dsv4_checkpoint_layer_smoke.py "$DSV4_CHECKPOINT" --layer-index 3
python tools/dsv4_rollout_smoke.py "$DSV4_CHECKPOINT" --tensor-parallel-size 4
```

The converter regression also runs on one H20 with the pinned base image:

```bash
srun -p h20 --gres=gpu:1 --time=00:20:00 \
  --container-image=vllm/vllm-openai@sha256:cb12c3016938a4ee98ed176ac833984089460689a0982cff4b5a60bb45b3cded \
  --container-mounts=/home/ljj:/home/ljj --container-workdir=/home/ljj/vime-dsv4 \
  bash -lc 'export PYTHONPATH=/home/ljj/vime-dsv4/.test-deps:/home/ljj/vime-dsv4/.deps/Megatron-LM:/home/ljj/vime-dsv4; export DSV4_CHECKPOINT=/home/ljj/models/DeepSeek-V4-Flash-Base; python3 -m pytest -q tests/test_deepseek_v4_weights.py tests/test_hf_to_megatron.py tests/test_deepseek_v4_native_mapping.py'
```

These checks do not constitute full RL support. Roadmap gates still open are
vLLM indexer top-k capture and training replay, fixed-data router/indexer
replay parity, full HF-to-Megatron and reverse model parity, FP8 online refit
versus fresh load with cache/version checks, ten-step CP=1 GRPO and resume,
and the 100-step release recipe. Megatron's native V4 attention currently
requires TP=1, so full training topology must use PP/EP instead of TP.
