# main vs slime-44d29ee-as-vime：10 块差距

排除 `projects/`、`reference/`、已废弃 `docker/patch/v0.5.*`。geo3k 已在 main（只有图片更新，归 sync）。

---

## 1. Scripts（23 个缺失）→ PR #260

**PR**: [#260](https://github.com/vllm-project/vime/pull/260)（Open）— scripts: complete slime-exact translation of all 29 run scripts

- 基础模型 10 个：glm4-9B, moonlight-16B, qwen3-32B, qwen3.5-27B, qwen3.5-35B-sft, qwen3-4B-base-sft, mimo-7B-rl-eagle, deepseek-r1, qwen2.5-0.5B-gb10-smoke, qwen2.5-0.5B-reproducibility
- 大模型 5 个：qwen3-235B, qwen3-235B-sft, qwen3-next-80B, glm4.7-355B, glm5-744B
- 新模型 2 个：kimi-k2-Instruct, kimi-k2-Thinking
- Low precision 6 个：fp8-4b, fp8-30b, int4-30b, int4-moonlight, int4-235B, int4-kimi
- 平台/杂项 2 个：qwen3-4B-amd, models/mimo-7B-rl.sh

---

## 2. Examples（8 个缺失目录）

| 目录 | 文件数 | PR |
|------|--------|-----|
| `delta_weight_sync/` | 2 | [#150](https://github.com/vllm-project/vime/pull/150)（重开） |
| `retool/` | 8 | 需新建 |
| `search-r1/` | 8 | 需新建 |
| `on_policy_distillation/` | 3 | 需新建 |
| `multi_agent/` | 2 | 需新建 |
| `eval_multi_task/` | 3 | 需新建 |
| `strands_vllm/` | 4 | 需新建 |
| `train_infer_mismatch_helper/` | 1 | 需新建 |

已在 main 不算差距：tau-bench（#142 已合入）、coding_agent_rl（已有，分支多 1 个 SWE 脚本）、fully_async（缺 1 个变体脚本，随 #260 补）、geo3k_vlm/geo3k_vlm_multi_turn（已有，图片更新归 sync）。

---

## 3. Delta Weight Sync（核心源码，最大单块）→ PR #150（重开）

| 文件 | 类型 | 行数 |
|------|------|------|
| `update_weight_from_distributed_delta.py` | 新文件 | 870 行 |
| `vllm.py`（megatron_utils/DeltaSpec helper） | 新文件 | 43 行 |
| `update_weight_from_distributed.py` | 修改，接入 delta 路径 | +52 -148 |
| `arguments.py`（utils） | +60 行 delta 相关参数 | |
| `tests/test_delta_weight_update.py` | 新文件 | 143 行 |
| `examples/delta_weight_sync/` | 2 文件 | |

对应 slime #1806/#1946/#1991。

---

## 4. vllm_engine / vllm_rollout / arguments 对账

**结论：main is correct。** #264（in-process launch）和 #246（native /update_weights）用不同方式实现了 slime 的功能，且功能上超越了 slime 快照。分支的旧代码（ServerArgs, /generate endpoint）已过时。

- [x] `vllm_engine.py` — main #264 in-process launch 替代了 subprocess `vllm serve`，功能超集
- [x] `vllm_rollout.py` — main #246/#247 使用 vLLM native API，功能无遗漏
- [x] `vllm_utils/arguments.py` — main 用 `AsyncEngineArgs` prefix wrapping，覆盖所有 vllm flags
- [x] `on_policy_distillation.py` — main 用 `http_utils.post`（已在 #232 sync），功能一致
- [x] `vllm_streaming_rollout.py` — main 有更详细的 docstring 和 abort 处理，超集

---

## 5. Agent Adapter 差异

**结论：main is correct。** 分支使用 `/generate` + `X-SMG-Routing-Key` + `output_token_logprobs`（sglang API shape 的机械翻译残留）。Main 使用 `/inference/v1/generate` + `x-session-id` + `choices[].token_ids/logprobs.content`（实际 vLLM API shape）。`parsing.py` 分支 import `vllm.srt.parser`（不存在的路径，是 sglang 路径的机械重命名），main 正确 import `sglang.srt.parser`。

- [x] `agent/adapters/common.py` — main 用 vLLM 实际 API shape，正确
- [x] `agent/adapters/anthropic.py` — main 正确
- [x] `agent/adapters/openai.py` — main 正确
- [x] `agent/parsing.py` — main 正确（import sglang.srt.parser，而非不存在的 vllm.srt.parser）
- [x] `test_agent_adapters.py` — main 正确

---

## 6. ROCm 支持 → PR #273

| 文件 | PR |
|------|-----|
| `rocm_checkpoint_writer.py`（新文件，27 行） | [#273](https://github.com/vllm-project/vime/pull/273) |
| `ray/train_actor.py`（+11 -7，ROCm pynvml skip） | #273 |
| `Dockerfile.rocm` + `amd_patch/latest/*` | #273 |
| `docs/en/platform_support/amd_tutorial.md` | #273 或单独 doc PR |

---

## 7. slime sync 残留（4 个 slime PR 未完整同步）→ 补入 #107

快照分支对应 slime commit `44d29ee`（即 slime #2013）。#1987 是 slime 公开仓库的 bulk import（初始提交），44d29ee 之间共 14 个 slime PR。逐行 diff 溯源后，残留 diff 来自以下 4 个 slime PR：

> 注：之前错误地将 #2027/#2050/#2016/#2021/#2081/#2088 列为残留，实际这 6 个 PR 都在 44d29ee **之后**合入 slime，不在快照范围内。

### slime [#1987](https://github.com/THUDM/slime/pull/1987) — bulk import（初始提交，已在 #107 TRACKED）

所有基础文件的初始版本。以下文件 main 上的版本与快照有 diff，但已被 #107 追踪：

| 文件 | diff | 说明 | 状态 |
|------|------|------|------|
| `vime/utils/trace_utils.py` | +9 | VLLM_TRACE_META_KEYS — 分支用 sglang response shape，main 用 vLLM shape | ✅ main is correct |
| `vime/utils/wandb_utils.py` | +6 | 分支用 `sgl_engine` metric key，main 用 `vllm_engine` | ✅ main is correct |
| `vime/utils/logging_utils.py` | +1 | `# ref: vLLM` 注释 | ✅ trivial, skip |
| `vime/utils/train_metric_utils.py` | +7 | `extra_metrics` 参数 | ✅ **已同步** |
| `vime/rollout/fully_async_rollout.py` | +2 | `vLLM` → `vllm` docstring casing | ✅ **已同步** |
| `vime/ray/actor_group.py` | +2 | NCCL_CUMEM_ENABLE 注释 rewording | ✅ trivial, skip |

### slime [#2013](https://github.com/THUDM/slime/pull/2013) — Revert rename rollout_ids to group_ids（**未被 #107 追踪**）

44d29ee 本身就是这个 PR。20 文件，17 缺失：

| 文件 | 状态 |
|------|------|
| `vime/ray/rollout.py` | 缺失 |
| `vime/backends/megatron_utils/actor.py` | ✅ extra_metrics plumbing 已同步（delta 分支属 #150） |
| `vime/backends/megatron_utils/data.py` | ✅ extra_metrics param 已同步 |
| `vime/backends/megatron_utils/model.py` | ✅ 已在 main（comment + save 签名已同步，ROCm patch 属 #273） |
| `vime/utils/types.py` | 缺失（`vllm_speculative_algorithm` rename — main 用 `vllm_speculative_config` 是 vLLM 正确名） |
| `vime/rollout/_fanout_test_helpers.py` | ✅ **已同步** |
| `vime/rollout/forge_load.py` | ✅ **已同步** |
| `docs/en/get_started/agent.md` (+zh) | 缺失 |
| `docs/en/get_started/customization.md` (+zh) | 缺失 |
| `examples/coding_agent_rl/README.md`, `generate.py` | 缺失 |
| `examples/multi_agent/agent_system.py` | 缺失 |
| `tests/test_dp_schedule.py`, `test_qwen2.5_0.5B_fanout_short.py`, `test_sample.py` | 缺失 |
| `vime/agent/trajectory.py`, `vime/backends/megatron_utils/loss.py`, `vime/utils/dp_schedule.py` | 已同步 |

### slime [#1929](https://github.com/THUDM/slime/pull/1929) — MiniMax M2.5 support（已在 #107 TRACKED，mega-B）

| 文件 | 状态 |
|------|------|
| `vime/backends/megatron_utils/megatron_to_hf/__init__.py` | ✅ **已同步**（+31 行 q_lora_rank 兼容 + tensor cache） |
| `vime/backends/megatron_utils/megatron_to_hf/minimax_m2.py` | 已同步 |
| scripts/run-minimax-m2.sh 等 5 文件 | 在 #260 覆盖 |

### slime [#1967](https://github.com/THUDM/slime/pull/1967) — Fix PYTHONBUFFERED typo to PYTHONUNBUFFERED=1（**已同步 via #232**，但未完整）

#232 同步了 #1967 的核心修复，但 #1967 同时改了所有 scripts/examples 脚本（46 文件）的 `PYTHONBUFFERED` → `PYTHONUNBUFFERED`。这些脚本/example 文件的修复随 #260（scripts）和各 example PR 一起带入。

| 文件 | 状态 |
|------|------|
| `vime/utils/external_utils/command_utils.py` | ✅ main is correct（main 的 pkill pattern 更精确，分支的是机械翻译残留） |
| 46 个 scripts/examples .sh 文件 | 在 #260 + example PRs 覆盖 |

### main-only 代码（分支没有，需确认是否保留）

| 文件 | 行数 | 说明 |
|------|------|------|
| `fp8_helpers.py` | 72 行 | main 独有，分支删了。确认是否 #246 已吸收其功能 |
| `arguments.py`（megatron_utils）| -25 行 | main 多了 dist-ckpt-optim 单节点 OOM warning |
| `qwen3_vl.py` | -47 行 | main 多了 vision model 转换逻辑 |

### 行动项

1. slime #2013 补入 [#107](https://github.com/vllm-project/vime/issues/107) tracking 表，作为下一批 sync
2. slime #1929 的 `megatron_to_hf/__init__.py` 改动确认是否已在 mega-B 覆盖，如未覆盖则补入
3. slime #1967 的 `command_utils.py` 残留随 #260 或单独小 PR 补入
4. #1987 bulk import 残留（6 个文件小改）归入 #107 下一批 sync

---

## 8. Tests 差异（~20 个文件修改 + 1 个新文件）

| 子类 | 文件 | 内容 | PR |
|------|------|------|-----|
| delta test | `test_delta_weight_update.py`（新，143 行） | delta weight sync 测试 | #150 |
| CI runner | `ci/gpu_lock_exec.py`（+55 -5） | subprocess 替代 execvp + signal handling | 需新建 PR |
| GPU test 参数 | ~15 个 test_*.py（每个 ±10 行） | ckpt 参数化、slime sync 参数更新 | #107 sync batch |
| plugin contracts | 3 个 test_plugin_*.py（±几行） | sync 更新 | #107 sync batch |

---

## 9. Docs（~25 页缺失）→ PR #220（更新）

| 子类 | 页数 | PR |
|------|------|-----|
| example 文档 en+zh | 12 | [#220](https://github.com/vllm-project/vime/pull/220) |
| advanced 文档 en+zh（OPD, low-precision, delta-sync） | 6 | #220 + 随对应 example PR |
| blog en+zh（introducing_vime, release_v0.1.0） | 4 | 需新建 PR |
| get_started/agent.md en+zh | 2 | #220 |
| platform/amd_tutorial.md | 1 | #273 |

---

## 10. Docker / CI / 杂项

| 内容 | PR | 状态 |
|------|-----|------|
| Dockerfile.rocm + amd_patch/latest/* | [#273](https://github.com/vllm-project/vime/pull/273) | 待 #273 |
| npu_patch/* | [#230](https://github.com/vllm-project/vime/pull/230) / [#256](https://github.com/vllm-project/vime/pull/256) | 待 NPU PRs |
| Dockerfile.gb10 + patch/gb10/* | 需新建 PR | 待做 |
| conda-ci.yml + build_conda.sh | 需新建 PR | 待做 |
| docker/Dockerfile | — | ✅ main is correct（#253 升级到 vLLM 0.23.0 base，分支用旧 sglang base） |
| docker/version.txt | — | ✅ main is correct（main 版本更新） |
| docker/patch/latest/vllm.patch | 需确认 | 待确认是否 #253 已 obsolete |
| requirements.txt | 本 PR | ✅ **已同步**（rm cloudpickle, vllm-router≥0.2.3） |
| pyproject.toml | — | ✅ 已在 main（`[tool.ruff.lint]` section 已存在） |
| tools/convert_torch_dist_to_hf_parallel.py | 本 PR | ✅ **已同步**（`vllm_enable_ep_moe = False`） |
| tools/convert_hf_to_torch_dist.py | #273 | 待 #273（ROCm patch） |
| CONTRIBUTING.md | — | ✅ main is correct（main 有完整 contributor guide，分支是 slime Z.ai 版） |
| train.py / train_async.py | 本 PR | ✅ **已同步**（casing fix） |
