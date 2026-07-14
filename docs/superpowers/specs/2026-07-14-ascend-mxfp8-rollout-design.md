# Ascend W8A8-MXFP8 Rollout Support Design

## Goal

Add online W8A8-MXFP8 rollout weight updates for Ascend 950 to Vime, using
the APIs and weight layouts provided by the local `vllm-ascend/main`
baseline. Both Vime-managed colocated CUDA/NPU IPC and decoupled NCCL weight
transfer must share the same quantization implementation.

## Scope

The feature supports:

- Vime-managed vLLM rollout servers using vLLM-Ascend.
- `AscendModelSlimConfig` configurations containing `W8A8_MXFP8` layers.
- Online conversion of BF16 training weights with
  `torch_npu.npu_dynamic_mx_quant`.
- Linear and fused MoE weights supported by vLLM-Ascend's W8A8-MXFP8
  schemes.
- Colocated IPC and decoupled NCCL online weight synchronization.

The feature does not support MXFP4, disk/delta checkpoint reload, or an
independently launched external vLLM server in this iteration.

## User Interface and Prerequisites

Users enable the backend through Vime's existing vLLM argument forwarding:

```bash
--vllm-quantization ascend
```

The rollout checkpoint directory must satisfy the local vLLM-Ascend main
contract. In particular, it must contain `quant_model_description.json` with
the intended layers marked `W8A8_MXFP8` and `group_size` set to 32. The model
weights supplied by the trainer remain BF16; Vime quantizes each online update
inside the rollout worker.

## Architecture

Vime will add an Ascend-specific MXFP8 worker adapter alongside its vLLM
backend. Imports of `vllm_ascend` and `torch_npu` will remain lazy so that the
existing CUDA and ROCm paths can import and run without those packages.

Activation is fail-closed. The adapter is active only when the worker's
quantization configuration is an `AscendModelSlimConfig` and its quantization
description contains `W8A8_MXFP8`. Other Ascend quantization schemes and all
non-Ascend configurations retain the native vLLM behavior.

The adapter hooks the worker-side weight loading lifecycle rather than either
transport implementation. Consequently, IPC and NCCL continue to carry BF16
named tensors and both reach one MXFP8 implementation at the worker's
`model.load_weights()` boundary.

## Weight-Update Lifecycle

For each online update:

1. Vime starts the native vLLM weight-update transaction.
2. The MXFP8 adapter walks supported vLLM-Ascend linear and fused MoE modules
   and invokes `restore_weights_for_rl_loading()` where the inference layout
   is currently transformed.
3. IPC or NCCL transfers BF16 named tensors without protocol changes.
4. At the common model-loading boundary, the adapter identifies parameters
   whose target modules use the vLLM-Ascend W8A8-MXFP8 scheme.
5. Each eligible tensor is converted with
   `torch_npu.npu_dynamic_mx_quant(..., axis=-1,
   dst_type=torch_npu.float8_e4m3fn)`. The generated weight is loaded under
   the original parameter name and its flattened uint8 scale is loaded using
   the vLLM-Ascend `*_scale` parameter name.
6. BF16-only parameters pass through unchanged.
7. After every transfer chunk succeeds, the worker calls the applicable
   vLLM-Ascend `process_weights_after_loading()` methods to recreate the
   Ascend 950 inference layouts.
8. Only a successful native finish operation allows Vime to advance its
   weight version.

## Components

The implementation will keep responsibilities separated:

- An Ascend MXFP8 utility module detects the quantization configuration,
  maps named parameters to target modules, produces quantized weight/scale
  pairs, and performs layout restore/reapply operations.
- A small worker lifecycle integration installs the utility at vLLM server
  startup and delegates transport to native vLLM.
- Existing IPC and NCCL trainer-side senders remain format-agnostic. Their
  tests gain contracts showing both routes use the same worker lifecycle.
- A feature guide documents setup, supported modes, configuration, version
  baseline, limitations, and hardware validation.

## Error Handling

- Optional Ascend dependencies are imported only after MXFP8 activation.
- Enabling `quantization=ascend` without a valid ModelSlim description is
  rejected with a message naming the missing or incompatible configuration.
- Unsupported target parameter shapes or missing scale parameters raise
  before the update is marked complete.
- Quantization or model loading errors preserve the original exception. The
  post-load layout transformation and Vime weight-version advancement do not
  run after a failed update.
- Repeated restore and process operations rely on vLLM-Ascend's idempotence
  markers (`_mxfp8_transformed` and `_mxfp8_original_shapes`).

## Testing and Validation

CPU-runnable unit tests will use stubs for optional NPU dependencies and cover:

- Positive and negative MXFP8 configuration detection.
- Linear and fused MoE parameter selection.
- BF16 passthrough for unquantized parameters.
- `npu_dynamic_mx_quant` arguments and generated `*_scale` names/shapes.
- Restore-before-load and process-after-load ordering.
- Failure behavior and idempotent lifecycle calls.
- IPC and NCCL contracts reaching the same MXFP8 integration without changing
  their wire formats.

Relevant existing Vime weight-transfer and vLLM backend tests will be rerun.
Static checks will cover all changed Python and documentation files.

The local development machine has no Ascend 950 device, so completion will not
claim hardware execution. The feature guide will provide an Ascend end-to-end
smoke procedure that verifies initial server load, at least two online weight
updates, rollout generation after each update, and worker logs showing the
restore/quantize/reapply lifecycle for both IPC and NCCL deployments.

## Compatibility

The implementation baseline is the locally checked-out
`vllm-ascend/main` at design time (`8bedc666`). The integration will prefer
capability checks over broad version branching. CUDA FP8, ROCm, unquantized
rollout, existing compressed-tensor handling, and non-MXFP8 Ascend schemes
must retain their current behavior.
