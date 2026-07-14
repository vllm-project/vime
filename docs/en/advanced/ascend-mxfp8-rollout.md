# Ascend W8A8-MXFP8 rollout

Vime supports online W8A8-MXFP8 rollout weight updates on Ascend 950 through
vLLM and vLLM-Ascend. Training weights remain in BF16 during IPC or NCCL
transfer. Each rollout worker quantizes eligible Linear and fused MoE weights
before vLLM loads them, then vLLM-Ascend restores its inference layout.

## Support matrix

| Capability | Status |
| --- | --- |
| Ascend 950 W8A8-MXFP8 rollout | Supported |
| Colocated IPC online updates | Supported |
| Decoupled NCCL online updates | Supported |
| Linear and fused MoE weights described by ModelSlim | Supported |
| MXFP4 | Not supported in this feature |
| External vLLM server | Not supported in this feature |
| Disk/delta reload | Not supported in this feature |

## Version baseline

This integration targets the local `vllm-ascend/main` API at commit
`8bedc666`. It relies on:

- `AscendModelSlimConfig`;
- the `W8A8_MXFP8` Linear and fused MoE schemes;
- `torch_npu.npu_dynamic_mx_quant`;
- vLLM's checkpoint-format weight-update transaction and worker-extension API.

Use compatible vLLM, vLLM-Ascend, torch-npu, CANN, and ModelSlim versions for
your Ascend 950 environment. This Vime repository does not install the Ascend
software stack.

## Model preparation

The rollout checkpoint directory must contain the ModelSlim file
`quant_model_description.json`. Generate the complete file with the ModelSlim
version matched to your model and vLLM-Ascend environment. It must identify
the intended layers as `W8A8_MXFP8` and use group size 32. For example, a
complete model-specific file contains metadata and entries resembling:

```json
{
  "quant_method": "ascend",
  "group_size": 32,
  "model.layers.0.self_attn.q_proj.weight": "W8A8_MXFP8",
  "model.layers.0.self_attn.k_proj.weight": "W8A8_MXFP8",
  "model.layers.0.input_layernorm.weight": "FLOAT"
}
```

The snippet is illustrative, not a complete description for a model. Every
required model layer must be represented using the names expected by the
current vLLM-Ascend ModelSlim parser. vLLM-Ascend rejects
`--quantization ascend` when the description file is missing or incompatible.

The initial rollout checkpoint may contain MXFP8 inference weights, but the
online actor updates sent by Vime are BF16. Do not quantize these tensors on
the trainer side.

## Configuration

Enable the existing vLLM quantization argument:

```bash
VLLM_ARGS=(
  --vllm-quantization ascend
  --vllm-gpu-memory-utilization 0.7
)
```

### Colocated IPC

Use the normal colocated topology and include `--colocate` in the Vime
invocation:

```bash
python train.py \
  --colocate \
  --hf-checkpoint /path/to/rollout-checkpoint \
  --vllm-quantization ascend \
  --vllm-gpu-memory-utilization 0.7 \
  ...
```

Vime installs the MXFP8 worker extension and the native vLLM IPC transfer
backend. BF16 named tensors are shared through IPC and quantized in each
rollout worker.

### Decoupled NCCL

Use the normal non-colocated topology (omit `--colocate`):

```bash
python train.py \
  --hf-checkpoint /path/to/rollout-checkpoint \
  --vllm-quantization ascend \
  --vllm-gpu-memory-utilization 0.7 \
  ...
```

Vime installs the same MXFP8 worker extension and selects the native vLLM NCCL
transfer backend. Only the transport changes; quantization and layout handling
remain worker-local and identical to the IPC path.

## Weight-update lifecycle

For both transports, Vime performs the following transaction:

1. Start vLLM's checkpoint-format weight update, which restores model-format
   metadata while preserving runtime tensor storage.
2. Prepare the Vime MXFP8 worker extension and wrap the common
   `model.load_weights()` boundary.
3. Transfer BF16 tensors through IPC or NCCL.
4. Quantize eligible weights with `npu_dynamic_mx_quant` and load both the FP8
   weight and its `*_scale` tensor.
5. Finish native vLLM layerwise processing and reapply the vLLM-Ascend MXFP8
   inference layout.
6. Publish the new Vime weight version only after finish and finalization both
   succeed.

Other quantization configurations retain vLLM's existing behavior.

## Ascend 950 validation

The following acceptance procedure must be run in the target Ascend 950
environment for both deployment modes:

1. Start a small Vime recipe with `--update-weights-interval 1` and the MXFP8
   arguments above.
2. Confirm the vLLM-Ascend server loads the ModelSlim description and reaches
   healthy state.
3. Allow at least two actor update/rollout iterations to complete.
4. Confirm each iteration completes the vLLM start/update/finish transaction
   without missing-weight, missing-scale, shape, or dtype errors.
5. Confirm rollout generation succeeds after each update and that Vime reports
   increasing weight versions.
6. Repeat once with `--colocate` (IPC) and once without it (NCCL).

The CPU tests in this repository validate configuration, conversion, lifecycle,
and transport contracts with stubs. They do not replace this hardware test.

## Troubleshooting

`ModelSlim Quantization Config Not Found`
: Ensure `quant_model_description.json` is inside the directory passed through
  `--hf-checkpoint` and was generated for the current model.

Missing `weight_scale` or shape mismatch
: Verify all quantized layer names and fused-module mappings in the ModelSlim
  file. Confirm `group_size` is 32 and the installed vLLM-Ascend matches the
  baseline API.

Worker extension conflict
: Remove a custom `--vllm-worker-extension-cls`. This feature requires Vime's
  MXFP8 extension and fails closed instead of silently replacing another
  extension.

External server rejected
: Let Vime launch the rollout servers. Independently launched external vLLM
  servers do not receive Vime's MXFP8 worker extension in this feature.

