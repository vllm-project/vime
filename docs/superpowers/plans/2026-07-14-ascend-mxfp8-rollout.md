# Ascend W8A8-MXFP8 Rollout Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add documented W8A8-MXFP8 online rollout weight updates for Ascend 950 to both Vime-managed IPC and NCCL vLLM-Ascend deployments.

**Architecture:** A focused Ascend utility module lazily detects `AscendModelSlimConfig`, quantizes eligible BF16 Linear/MoE weights with `torch_npu.npu_dynamic_mx_quant`, and exposes unique worker-extension RPC methods. Vime's engine orchestration injects that extension when `--vllm-quantization ascend` is requested and composes it with vLLM's native checkpoint-format start/update/finish transaction, leaving IPC and NCCL wire formats unchanged.

**Tech Stack:** Python 3.11+, PyTorch/torch-npu, vLLM weight transfer, vLLM-Ascend ModelSlim quantization, Ray RPC, pytest, Ruff, Sphinx/MyST documentation.

## Global Constraints

- Baseline vLLM-Ascend is local `main` commit `8bedc666`.
- Support only `W8A8_MXFP8`; MXFP4 is outside this plan.
- Support Vime-managed colocated IPC and decoupled NCCL; external servers and disk/delta reload are outside this plan.
- Keep `vllm_ascend` and `torch_npu` imports lazy so CPU, CUDA, and ROCm imports remain unaffected.
- Require the vLLM-Ascend ModelSlim `quant_model_description.json` contract with `group_size: 32`.
- Do not claim Ascend 950 hardware validation from this development machine.

---

### Task 1: Ascend MXFP8 quantization utilities

**Files:**
- Create: `vime/backends/vllm_utils/ascend_mxfp8.py`
- Create: `tests/utils/test_ascend_mxfp8.py`

**Interfaces:**
- Produces: `is_ascend_mxfp8_config(quant_config: object) -> bool`.
- Produces: `quantize_mxfp8_weights(weights, model, dtype) -> Iterator[tuple[str, torch.Tensor]]`.
- Produces: `prepare_mxfp8_modules_for_reload(model) -> int` and `finalize_mxfp8_modules_after_reload(model) -> int`.
- Consumes: vLLM-Ascend schemes exposing `restore_weights_for_rl_loading(layer)` and `process_weights_after_loading(layer)`.

- [ ] **Step 1: Write failing detection, mapping, quantization, and lifecycle tests**

Create CPU-only tests using scoped `sys.modules` stubs. The core assertions must include:

```python
def test_quantize_mxfp8_linear_weight_emits_weight_and_scale(fake_runtime):
    model, source = fake_runtime.linear_model_and_weight()
    output = list(mx.quantize_mxfp8_weights([("layers.0.proj.weight", source)], model, torch.bfloat16))
    assert [name for name, _ in output] == ["layers.0.proj.weight", "layers.0.proj.weight_scale"]
    fake_runtime.dynamic_mx_quant.assert_called_once_with(
        source.to(torch.bfloat16), axis=-1, dst_type=fake_runtime.float8_dtype
    )


def test_non_mxfp8_weight_passes_through_unchanged(fake_runtime):
    model, source = fake_runtime.float_model_and_weight()
    assert list(mx.quantize_mxfp8_weights([("layers.0.proj.weight", source)], model, torch.bfloat16)) == [
        ("layers.0.proj.weight", source)
    ]


def test_prepare_uses_scheme_restore_when_native_metadata_has_not_restored_shapes(fake_runtime):
    model, scheme = fake_runtime.transformed_model(native_shapes_restored=False)
    assert mx.prepare_mxfp8_modules_for_reload(model) == 1
    scheme.restore_weights_for_rl_loading.assert_called_once()


def test_prepare_only_resets_marker_when_vllm_native_reload_already_restored_shapes(fake_runtime):
    model, scheme = fake_runtime.transformed_model(native_shapes_restored=True)
    assert mx.prepare_mxfp8_modules_for_reload(model) == 1
    scheme.restore_weights_for_rl_loading.assert_not_called()
    assert model.layers[0].proj._mxfp8_transformed is False
```

Also cover `AscendModelSlimConfig` positive/negative detection, packed-module name mapping, fused MoE weights, scale flattening/squeezing, and idempotent finalize behavior.

- [ ] **Step 2: Run the new test module and verify RED**

Run: `pytest -q tests/utils/test_ascend_mxfp8.py`

Expected: collection fails because `vime.backends.vllm_utils.ascend_mxfp8` does not exist.

- [ ] **Step 3: Implement the minimal lazy utility module**

Implement the following public structure, keeping optional imports inside functions:

```python
MXFP8_QUANT_TYPE = "W8A8_MXFP8"


def is_ascend_mxfp8_config(quant_config: object) -> bool:
    try:
        from vllm_ascend.quantization.modelslim_config import AscendModelSlimConfig
    except ImportError:
        return False
    return isinstance(quant_config, AscendModelSlimConfig) and MXFP8_QUANT_TYPE in getattr(
        quant_config, "quant_description", {}
    ).values()


def quantize_mxfp8_weights(weights, model, dtype=torch.bfloat16):
    import torch_npu

    for name, value in weights:
        if not _is_mxfp8_weight(name, model):
            yield name, value
            continue
        quantized, scale = torch_npu.npu_dynamic_mx_quant(
            value.to(dtype), axis=-1, dst_type=torch_npu.float8_e4m3fn
        )
        scale = scale.flatten(-2, -1).squeeze(-1)
        yield name, quantized
        yield name + "_scale", scale
```

Implement `_module_from_param_name` using the model's `packed_modules_mapping`, recognize a scheme by the `restore_weights_for_rl_loading` capability, and make prepare/finalize handle both native-vLLM-restored shapes and still-transformed vLLM-Ascend tensors.

- [ ] **Step 4: Run utility tests and verify GREEN**

Run: `pytest -q tests/utils/test_ascend_mxfp8.py`

Expected: all tests pass with no real `vllm_ascend` or `torch_npu` installation.

- [ ] **Step 5: Run lint and commit Task 1**

Run: `ruff check vime/backends/vllm_utils/ascend_mxfp8.py tests/utils/test_ascend_mxfp8.py`

Expected: exit 0.

Commit:

```bash
git add vime/backends/vllm_utils/ascend_mxfp8.py tests/utils/test_ascend_mxfp8.py
git commit -m "feat: add Ascend MXFP8 rollout utilities"
```

### Task 2: vLLM worker extension lifecycle

**Files:**
- Modify: `vime/backends/vllm_utils/ascend_mxfp8.py`
- Modify: `tests/utils/test_ascend_mxfp8.py`

**Interfaces:**
- Consumes: the Task 1 utility functions.
- Produces: `AscendMXFP8WorkerExtension.prepare_ascend_mxfp8_weight_update() -> dict`.
- Produces: `AscendMXFP8WorkerExtension.finalize_ascend_mxfp8_weight_update(success: bool = True) -> dict`.

- [ ] **Step 1: Write failing worker-extension tests**

Add tests that build a fake worker with `model_runner.model.load_weights` and a fake MXFP8 config:

```python
def test_worker_prepare_wraps_common_model_load_boundary(fake_worker):
    result = fake_worker.prepare_ascend_mxfp8_weight_update()
    fake_worker.model_runner.model.load_weights([("layers.0.proj.weight", torch.ones(2, 2))])
    assert result == {"active": True, "modules": 1}
    assert fake_worker.original_loader_names == ["layers.0.proj.weight", "layers.0.proj.weight_scale"]


def test_worker_finalize_restores_original_loader_and_reapplies_layout(fake_worker):
    original = fake_worker.model_runner.model.load_weights
    fake_worker.prepare_ascend_mxfp8_weight_update()
    result = fake_worker.finalize_ascend_mxfp8_weight_update(success=True)
    assert fake_worker.model_runner.model.load_weights == original
    assert result == {"active": True, "modules": 1}


def test_worker_finalize_failure_restores_loader_without_processing(fake_worker):
    fake_worker.prepare_ascend_mxfp8_weight_update()
    result = fake_worker.finalize_ascend_mxfp8_weight_update(success=False)
    assert result == {"active": True, "modules": 0}
    fake_worker.scheme.process_weights_after_loading.assert_not_called()
```

Also assert that non-MXFP8 ModelSlim configs return `{"active": False, "modules": 0}` and that nested prepare calls raise a clear `RuntimeError`.

- [ ] **Step 2: Run worker-extension tests and verify RED**

Run: `pytest -q tests/utils/test_ascend_mxfp8.py -k worker`

Expected: failures because `AscendMXFP8WorkerExtension` is absent.

- [ ] **Step 3: Implement the worker extension**

Add a class with unique method names so vLLM's extension conflict guard accepts it:

```python
class AscendMXFP8WorkerExtension:
    def prepare_ascend_mxfp8_weight_update(self) -> dict:
        model_runner = self.model_runner
        if not is_ascend_mxfp8_config(model_runner.vllm_config.quant_config):
            return {"active": False, "modules": 0}
        if hasattr(self, "_vime_mxfp8_original_load_weights"):
            raise RuntimeError("Ascend MXFP8 weight update is already active")

        model = model_runner.model
        modules = prepare_mxfp8_modules_for_reload(model)
        original_load_weights = model.load_weights

        def load_quantized(weights):
            return original_load_weights(
                quantize_mxfp8_weights(weights, model, model_runner.vllm_config.model_config.dtype)
            )

        self._vime_mxfp8_original_load_weights = original_load_weights
        model.load_weights = load_quantized
        return {"active": True, "modules": modules}

    def finalize_ascend_mxfp8_weight_update(self, success: bool = True) -> dict:
        original = getattr(self, "_vime_mxfp8_original_load_weights", None)
        if original is None:
            return {"active": False, "modules": 0}
        self.model_runner.model.load_weights = original
        del self._vime_mxfp8_original_load_weights
        modules = finalize_mxfp8_modules_after_reload(self.model_runner.model) if success else 0
        return {"active": True, "modules": modules}
```

- [ ] **Step 4: Run the whole utility/extension suite and verify GREEN**

Run: `pytest -q tests/utils/test_ascend_mxfp8.py`

Expected: all tests pass.

- [ ] **Step 5: Commit Task 2**

```bash
git add vime/backends/vllm_utils/ascend_mxfp8.py tests/utils/test_ascend_mxfp8.py
git commit -m "feat: add MXFP8 vLLM worker lifecycle"
```

### Task 3: Vime engine orchestration for IPC and NCCL

**Files:**
- Modify: `vime/backends/vllm_utils/vllm_engine.py`
- Modify: `tests/utils/test_vllm_engine.py`

**Interfaces:**
- Consumes: `vime.backends.vllm_utils.ascend_mxfp8.AscendMXFP8WorkerExtension` by qualified name.
- Produces: server args containing `worker_extension_cls` for both `weight_transfer_config.backend == "ipc"` and `"nccl"` when quantization is `ascend`.
- Produces: prepare/finalize collective RPC calls around native vLLM update transactions.

- [ ] **Step 1: Write failing server-argument and transaction tests**

Add parameterized tests:

```python
@pytest.mark.parametrize(("colocate", "backend"), [(True, "ipc"), (False, "nccl")])
def test_ascend_quantization_installs_same_worker_extension(vllm_args, monkeypatch, colocate, backend):
    monkeypatch.setattr(mod, "_VLLM_SERVER_FIELDS", frozenset({"quantization", "worker_extension_cls"}))
    vllm_args.colocate = colocate
    vllm_args.vllm_quantization = "ascend"
    server_args, _ = mod._compute_server_args(
        vllm_args, rank=0, dist_init_addr=None, host="127.0.0.1", port=8000
    )
    assert server_args["weight_transfer_config"]["backend"] == backend
    assert server_args["worker_extension_cls"] == mod._ASCEND_MXFP8_WORKER_EXTENSION


def test_mxfp8_start_calls_native_then_prepare(vllm_engine, monkeypatch):
    vllm_engine._uses_ascend_mxfp8 = True
    calls = record_requests(vllm_engine, monkeypatch)
    vllm_engine.start_weight_update(is_checkpoint_format=True)
    assert calls == [
        ("start_weight_update", {"is_checkpoint_format": True}),
        ("collective_rpc", {"method": "prepare_ascend_mxfp8_weight_update", "kwargs": {}}),
    ]


def test_mxfp8_finish_promotes_pending_version_only_after_finalize(vllm_engine, monkeypatch):
    vllm_engine._uses_ascend_mxfp8 = True
    vllm_engine._weight_version = "old"
    vllm_engine._pending_weight_version = "new"
    calls = record_requests(vllm_engine, monkeypatch)
    vllm_engine.finish_weight_update()
    assert calls[-1] == (
        "collective_rpc",
        {"method": "finalize_ascend_mxfp8_weight_update", "kwargs": {"success": True}},
    )
    assert vllm_engine._weight_version == "new"
```

Also test rejection of an independently launched external engine, conflict with a user-supplied worker extension, prepare/finalize failure cleanup, pending version behavior for both tensor and distributed update methods, and unchanged non-Ascend behavior.

- [ ] **Step 2: Run focused engine tests and verify RED**

Run: `pytest -q tests/utils/test_vllm_engine.py -k 'ascend or mxfp8'`

Expected: failures for missing extension injection and lifecycle calls.

- [ ] **Step 3: Implement extension selection and RPC orchestration**

Add:

```python
_ASCEND_MXFP8_WORKER_EXTENSION = (
    "vime.backends.vllm_utils.ascend_mxfp8.AscendMXFP8WorkerExtension"
)


def _requested_quantization(args, vllm_overrides: dict | None) -> str | None:
    if vllm_overrides and "quantization" in vllm_overrides:
        return vllm_overrides["quantization"]
    return getattr(args, "vllm_quantization", None)
```

In `_compute_server_args`, after applying overrides, reject external rollout for `ascend`, reject a conflicting `worker_extension_cls`, and install `_ASCEND_MXFP8_WORKER_EXTENSION`. Ensure `_build_subprocess_env` includes the Vime package root whenever that extension is selected.

In `VLLMEngine`, set `_uses_ascend_mxfp8`, track `_pending_weight_version`, and add a private collective-RPC helper. Native start must run before prepare. Native finish must run before successful finalize. On failures, call finalize with `success=False` without masking the original exception. Tensor/NCCL updates store a pending version under MXFP8 and retain immediate version updates for every existing backend.

- [ ] **Step 4: Run engine tests and verify GREEN**

Run: `pytest -q tests/utils/test_vllm_engine.py`

Expected: all existing and new tests pass.

- [ ] **Step 5: Run both transport contract suites**

Run:

```bash
pytest -q tests/utils/test_update_weight_from_tensor.py tests/utils/test_update_weight_from_distributed.py
```

Expected: all tests pass, proving both transports still use native start/update/finish wire formats.

- [ ] **Step 6: Commit Task 3**

```bash
git add vime/backends/vllm_utils/vllm_engine.py tests/utils/test_vllm_engine.py
git commit -m "feat: enable MXFP8 for IPC and NCCL rollout"
```

### Task 4: Feature support documentation

**Files:**
- Create: `docs/en/advanced/ascend-mxfp8-rollout.md`
- Create: `docs/zh/advanced/ascend-mxfp8-rollout.md`
- Modify: `docs/en/index.rst`
- Modify: `docs/zh/index.rst`

**Interfaces:**
- Documents: `--vllm-quantization ascend`, ModelSlim metadata, IPC/NCCL deployment modes, version baseline, limitations, troubleshooting, and hardware smoke validation.

- [ ] **Step 1: Write the English and Chinese feature guides**

Each guide must contain this support matrix and matching configuration example:

```markdown
| Capability | Status |
| --- | --- |
| Ascend 950 W8A8-MXFP8 rollout | Supported |
| Colocated IPC online updates | Supported |
| Decoupled NCCL online updates | Supported |
| MXFP4 | Not supported in this feature |
| External vLLM server | Not supported in this feature |
| Disk/delta reload | Not supported in this feature |
```

```bash
VLLM_ARGS=(
  --vllm-quantization ascend
  --vllm-gpu-memory-utilization 0.7
)
```

Document the requirement for `quant_model_description.json`, `W8A8_MXFP8`, and `group_size: 32`; list vLLM-Ascend baseline `8bedc666`; explain that BF16 crosses IPC/NCCL and online quantization occurs in each rollout worker. Include two-update smoke procedures for colocated and decoupled deployments and label them as commands to run on Ascend hardware, not locally verified results.

- [ ] **Step 2: Add both guides to the Advanced Features toctrees**

Add `advanced/ascend-mxfp8-rollout.md` to both `docs/en/index.rst` and `docs/zh/index.rst` immediately after `advanced/low-precision.md`; add `advanced/low-precision.md` to either index first if it is currently omitted.

- [ ] **Step 3: Verify documentation structure**

Run:

```bash
python -m compileall -q vime
git diff --check
```

Expected: both commands exit 0.

If the documentation dependencies are installed, also run `make -C docs html`; otherwise record the missing dependency and do not install unrelated packages.

- [ ] **Step 4: Commit Task 4**

```bash
git add docs/en/advanced/ascend-mxfp8-rollout.md docs/zh/advanced/ascend-mxfp8-rollout.md docs/en/index.rst docs/zh/index.rst
git commit -m "docs: add Ascend MXFP8 rollout guide"
```

### Task 5: Regression verification and delivery audit

**Files:**
- Modify only if a verification failure reveals an in-scope defect.

**Interfaces:**
- Verifies all preceding tasks and the approved design specification.

- [ ] **Step 1: Run focused feature tests**

Run:

```bash
pytest -q \
  tests/utils/test_ascend_mxfp8.py \
  tests/utils/test_vllm_engine.py \
  tests/utils/test_update_weight_from_tensor.py \
  tests/utils/test_update_weight_from_distributed.py
```

Expected: zero failures.

- [ ] **Step 2: Run static verification**

Run:

```bash
ruff check \
  vime/backends/vllm_utils/ascend_mxfp8.py \
  vime/backends/vllm_utils/vllm_engine.py \
  tests/utils/test_ascend_mxfp8.py \
  tests/utils/test_vllm_engine.py
python -m compileall -q vime
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 3: Audit scope and repository state**

Run:

```bash
git status --short --branch
git log --oneline --decorate main..HEAD
git diff --stat main...HEAD
```

Expected: branch is `rollout_support_mx`; committed changes are limited to the design/plan, MXFP8 utility and engine integration, tests, and bilingual documentation. Persistent `.planning/` working-memory files may remain untracked during execution but are not product deliverables.

- [ ] **Step 4: Record the hardware-validation boundary**

State in the final handoff that CPU mock and contract tests passed locally, while the documented Ascend 950 smoke procedure still needs execution in the target hardware environment.
