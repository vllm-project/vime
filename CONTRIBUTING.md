# Contributing to vime

[中文版](#贡献指南)

Thank you for your interest in contributing to vime! This document describes our project principles, collaboration guidelines, and the development workflow.

## Table of Contents

- [About vime](#about-vime)
- [Project Principles](#project-principles)
- [How Can I Contribute?](#how-can-i-contribute)
- [Development Setup](#development-setup)
- [Development Workflow](#development-workflow)
- [Code Style](#code-style)
- [Pull Requests & Code Reviews](#pull-requests--code-reviews)
- [Reporting Bugs](#reporting-bugs)
- [Requesting Features](#requesting-features)
- [License](#license)

## About vime

**vime** is a reinforcement learning (RL) post-training framework built on [**vLLM**](https://github.com/vllm-project/vllm) and [**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM) as its core backends. It is derived from [slime](https://github.com/THUDM/slime).

## Project Principles

We aim to keep vime **lightweight**, **stable**, and **easy to maintain**:

- **Lightweight** — Focus on the RL training loop and rollout integration; avoid turning the repo into a monolithic application framework.
- **Stable** — Prefer changes that are testable, reviewable, and safe for production-scale training runs.
- **Concise code** — Match existing patterns; avoid unnecessary abstraction or large refactors without clear benefit.
- **Hardware support** — vime targets **GPU** (CUDA) and **NPU** (e.g., Ascend). To keep the codebase concise, GPU- and NPU-related changes are currently maintained on **separate branches** instead of folding both paths into heavy abstractions on one branch.

## How Can I Contribute?

We welcome all kinds of contributions. Please note:

- **New features or capabilities** — **Open an Issue first** to discuss the problem, design, scope, and verification plan before starting a large PR.
- **Bug reports and bug fixes** — Welcome. For both Issues and PRs, include **complete information to reproduce** the problem (see [Reporting Bugs](#reporting-bugs)).
- **Performance improvements** — Welcome when you can show benchmarks or tests that CI or standard training runs can validate.
- **Documentation and examples** — Guides, training scripts, and typo fixes are welcome; update English and Chinese user-facing docs when applicable.
- **Tests and CI** — Unit tests, integration tests, and workflow improvements are appreciated.

## Development Setup

```bash
# Clone the repository
git clone https://github.com/vllm-project/vime.git
cd vime

# Install dependencies (adjust for your CUDA / NPU environment)
pip install -r requirements.txt

# Install pre-commit hooks
pip install pre-commit
pre-commit install
```

For environment details (Docker, vLLM versions), see [docs/en/get_started/quick_start.md](docs/en/get_started/quick_start.md) and [docker/README.md](docker/README.md).

## Development Workflow

### 1. Create a Branch

```bash
# Feature (after Issue discussion)
git checkout -b feature/your-feature-name

# Bug fix
git checkout -b fix/your-bug-fix

# Documentation
git checkout -b docs/your-doc-change
```

### 2. Make Changes

- Follow existing code patterns in the repository
- Keep changes scoped; prefer extending vLLM/Megatron integration points over duplicating logic
- Add or update tests when behavior changes
- Update documentation if user-facing behavior changes (EN + ZH when applicable)

### 3. Validate

```bash
# Run pre-commit checks (lint + format)
pre-commit run --all-files --show-diff-on-failure --color=always

# Run relevant tests (examples)
pytest tests/
```

### 4. Submit a Pull Request

Push your branch and open a PR against `main`. Link related issues (e.g., `Fixes #123`). Use `git commit -s` so each commit includes `Signed-off-by:` (see [Pull Requests & Code Reviews](#pull-requests--code-reviews)).

## Code Style

- **Formatter / linter**: [Ruff](https://docs.astral.sh/ruff/) and [isort](https://pycqa.github.io/isort/) via [pre-commit](https://pre-commit.com/) (see [.pre-commit-config.yaml](.pre-commit-config.yaml))
- **Imports**: No wildcard imports (`from x import *`)
- **Logging**: Use project logging utilities; avoid ad-hoc `print()` in library code
- **Megatron / vLLM boundaries**: Keep Megatron training logic and vLLM rollout logic in their respective backend modules; avoid cross-cutting refactors that blur module responsibilities

## Pull Requests & Code Reviews

vime follows the [vLLM contributing guidelines](https://docs.vllm.ai/en/latest/contributing/) for PR titles, commit sign-off, and AI-assisted contributions.

### DCO and Signed-off-by

When contributing changes, you must agree to the [Developer Certificate of Origin (DCO)](https://github.com/vllm-project/vllm/blob/main/DCO). Each commit must include a `Signed-off-by:` line.

Use `-s` with `git commit` to add it automatically:

```bash
git commit -s -m "[Rollout] your message"
```

You can also enable automatic sign-off in your IDE (e.g., VS Code: `Git: Always Sign Off` / `git.alwaysSignOff`).

### AI Assisted Contributions

vime adopts the same requirements as [vLLM: AI Assisted Contributions](https://docs.vllm.ai/en/latest/contributing/#ai-assisted-contributions).

Before making an AI assisted contribution, you must:

1. **Be involved**: Do not submit "pure agent" PRs. The human submitter is responsible for reviewing all changed lines, validating behavior end-to-end, and running relevant tests.
2. **Ensure significance**: Avoid one-off "busywork" PRs (single typo, isolated style cleanup, one mutable default fix, etc.). Bundle mechanical cleanups into a clear, systematic scope.

When AI tools provide non-trivial assistance in generating or modifying code, you must:

1. **Review thoroughly**: You remain responsible for all code you submit. Review and understand AI-generated code with the same care as code you write manually.
2. **Disclose in PR**: Always mention when a pull request includes AI-generated code. Add a note in the PR description.
3. **Mark commits**: Add attribution using commit trailers such as `Co-authored-by:` (other projects use `Assisted-by:` or `Generated-by:`). For example:

```
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

AI-assisted code must meet all quality standards: proper testing, documentation, adherence to style guides, and thorough review. Attribution helps reviewers evaluate contributions in context and maintains legal clarity for the project.

### PR Title and Classification

Prefix the **PR title** with one of the following (aligned with [vLLM PR classification](https://docs.vllm.ai/en/latest/contributing/#pr-title-and-classification)):

- `[Bugfix]` — Bug fixes
- `[CI/Build]` — CI or build improvements
- `[Doc]` — Documentation fixes and improvements
- `[Training]` — Megatron training loop, weight sync, or trainer-side logic
- `[Rollout]` — vLLM rollout, router, or inference integration
- `[Core]` — Framework orchestration (data buffer, Ray actors, shared arguments)
- `[Example]` — Training scripts and examples under `examples/` or `scripts/`
- `[Hardware][Vendor]` — Hardware-specific changes (e.g., `[Hardware][Ascend]` for NPU)
- `[Misc]` — Changes that do not fit the above; use sparingly

If a PR spans multiple areas, include all relevant prefixes (e.g., `[Bugfix][Rollout]`).

### Before Submitting

- [ ] `pre-commit run --all-files` passes
- [ ] Relevant tests pass (`pytest tests/` or the CI job your change affects)
- [ ] Documentation updated if behavior or flags changed
- [ ] PR title uses the correct prefix; commits include `Signed-off-by:`
- [ ] For **new features**: linked Issue with maintainer agreement on scope
- [ ] If AI-assisted: disclosed in the PR description and attributed in commits when applicable

### Code Quality

- Pass all linter checks (`pre-commit`)
- Add or update tests when behavior changes
- Update `docs/` (EN + ZH when user-facing) for behavior or CLI changes
- Keep PRs focused; for large architectural changes (>500 LOC excluding tests/config), **open an Issue first** to discuss design

### Review Process

1. **Automated CI** — Pre-commit and PR tests run on GitHub Actions
2. **Code review** — Maintainers review for correctness, scope, and maintainability
3. **Feedback** — Address review comments and re-request review when ready
4. **Merge** — Maintainer merges after approval

### Tips for a Good PR

- Describe **what**, **why**, and **how**
- For training or rollout changes, note GPU/NPU target, vLLM version, and how you verified
- For bug fixes, include logs and minimal repro commands

## Reporting Bugs

Use the [Bug Report template](.github/ISSUE_TEMPLATE/bug_report.yml) when available, or open an issue with **complete** reproduction details. Issues without enough information may be closed. Please include:

- **Environment** — OS, Python, PyTorch, CUDA or NPU stack (e.g., CANN / `torch_npu`), GPU/NPU model and count, vLLM and Megatron versions
- **Steps to reproduce** — Minimal commands or script; note whether the issue is in training, rollout, or both
- **Expected vs actual behavior**
- **Logs** — Full traceback and relevant Ray / vLLM / Megatron output
- **Configuration** — Launch script, CLI args, or config snippets (redact secrets)
- **Hardware mode** — GPU or NPU; colocate vs disaggregated if relevant

Before submitting:

- [ ] Read [docs/en/get_started/qa.md](docs/en/get_started/qa.md) and [README.md](README.md)
- [ ] Search [existing issues](https://github.com/vllm-project/vime/issues) for duplicates
- [ ] Reproduce on the latest `main` when possible

## Requesting Features

**Please open an Issue before implementing non-trivial features.** This helps us align on:

- Problem statement and use case
- Fit with vime's lightweight, vLLM-centric roadmap
- Verification plan (CI, unit test, or documented training run)
- Impact on GPU and NPU code paths

In the Issue, briefly cover the problem, proposed approach (which module: training / rollout / buffer), alternatives you considered, and how you plan to verify correctness and performance.

Large features that cannot be verified in CI or routine tests are harder to maintain long-term and may be deferred.

## License

By contributing to vime, you agree that your contributions will be licensed under the [Apache License 2.0](LICENSE).

---

## 贡献指南

感谢你对vime的关注与支持！本文档说明项目理念、协作方式与开发流程。

## 目录

- [关于vime](#关于vime)
- [项目理念](#项目理念)
- [如何参与贡献](#如何参与贡献)
- [开发环境](#开发环境)
- [开发流程](#开发流程)
- [代码风格](#代码风格)
- [Pull Request与代码审查](#pull-request与代码审查)
- [报告Bug](#报告bug)
- [功能建议](#功能建议)
- [许可证](#许可证)

## 关于vime

**vime**是以[**vLLM**](https://github.com/vllm-project/vllm)与[**Megatron-LM**](https://github.com/NVIDIA/Megatron-LM)为核心后端的RL后训练框架，源自[slime](https://github.com/THUDM/slime)。

## 项目理念

我们希望vime保持**轻量**、**稳定**、**代码简洁**：

- **轻量** — 聚焦RL训练主链路与rollout集成，避免演化成臃肿的一体化应用框架。
- **稳定** — 优先合入可测试、可评审、对大规模训练安全的代码变更。
- **代码简洁** — 遵循现有模式；无明确收益时避免过度抽象或大范围重构。
- **硬件支持** — vime支持**GPU**（CUDA）与**NPU**（如Ascend）。为保证代码简洁，当前通过**独立的分支**分别承载GPU与NPU相关改动，而非在单分支中用大量抽象同时兼容。

## 如何参与贡献

欢迎各类贡献，请注意：

- **新功能或特性** — **请先开Issue讨论**问题背景、方案、范围与验证方式，再开始较大的PR。
- **Bug报告与Bug修复** — 欢迎。无论提Issue还是PR，都请提供**完整、可复现**的信息（见[报告Bug](#报告bug)）。
- **性能优化** — 欢迎，需能通过benchmark或CI/常规训练流程验证。
- **文档与示例** — 指南、训练脚本、勘误等均可；用户可见文档建议在适用时中英同步更新。
- **测试与CI** — 单测、集成测试与工作流改进均欢迎。

## 开发环境

```bash
git clone https://github.com/vllm-project/vime.git
cd vime

pip install -r requirements.txt

pip install pre-commit
pre-commit install
```

环境与Docker、vLLM版本说明见[docs/zh/get_started/quick_start.md](docs/zh/get_started/quick_start.md)与[docker/README.md](docker/README.md)。

## 开发流程

### 1. 创建分支

```bash
git checkout -b feature/your-feature-name   # 功能（Issue讨论后）
git checkout -b fix/your-bug-fix            # Bug修复
git checkout -b docs/your-doc-change        # 文档
```

### 2. 修改代码

- 遵循仓库内现有代码风格与组织方式
- 优先扩展vLLM/Megatron集成点，避免重复实现
- 行为变更请补充测试
- 用户可见变更请更新文档（适用时中英同步）

### 3. 本地验证

```bash
pre-commit run --all-files --show-diff-on-failure --color=always
pytest tests/
```

### 4. 提交Pull Request

推送到你的分支，向`main`发起PR，并关联Issue（如`Fixes #123`）。提交时使用`git commit -s`添加`Signed-off-by:`（见[Pull Request与代码审查](#pull-request与代码审查)）。

## 代码风格

- **格式化/静态检查**：通过[.pre-commit-config.yaml](.pre-commit-config.yaml)运行Ruff、isort等
- **禁止**通配符导入（`from x import *`）
- **日志**：使用项目内日志工具，库代码中避免随意`print()`
- **模块边界**：Megatron训练与vLLM rollout逻辑保持在各自backend模块，避免模糊职责的大范围重构

## Pull Request与代码审查

vime在PR标题、提交签署与AI辅助贡献方面对齐[vLLM贡献指南](https://docs.vllm.ai/en/latest/contributing/)。

### DCO与Signed-off-by

贡献代码即表示你同意[Developer Certificate of Origin (DCO)](https://github.com/vllm-project/vllm/blob/main/DCO)。每个commit须包含`Signed-off-by:`行。

使用`git commit -s`可自动添加：

```bash
git commit -s -m "[Rollout] your message"
```

也可在IDE中开启自动sign-off（如VS Code：`Git: Always Sign Off` / `git.alwaysSignOff`）。

### AI辅助贡献

与[vLLM：AI Assisted Contributions](https://docs.vllm.ai/en/latest/contributing/#ai-assisted-contributions)要求一致。

在借助AI提交贡献前，你必须：

1. **Be involved（亲自参与）**：不要提交「纯Agent」PR。人类提交者须审查全部变更行、端到端验证行为，并运行相关测试。
2. **Ensure significance（确保有意义）**：避免琐碎的、无实质内容的PR（单独改typo、孤立格式清理、单个mutable default修复等）。机械性清理应合并为有明确、成体系的范围。

当AI工具在生成或修改代码时提供了实质性协助，你必须：

1. **Review thoroughly（充分审查）**：你对所提交的全部代码负责。须以与手写代码同等的认真程度阅读并理解AI生成代码。
2. **Disclose in PR（在PR中披露）**：若PR包含AI生成代码，务必在PR描述中说明。
3. **Mark commits（标注commit）**：使用`Co-authored-by:`等commit trailer标注来源（其他项目亦使用`Assisted-by:`、`Generated-by:`）。示例：

```
Your commit message here

Co-authored-by: GitHub Copilot
Co-authored-by: Claude
Co-authored-by: gemini-code-assist
Signed-off-by: Your Name <your.email@example.com>
```

AI辅助代码须满足全部质量标准：充分测试、完善文档、遵守风格规范，并经充分人工审查。标注有助于审查者结合上下文评估贡献，并保证项目在法律层面的清晰性。

### PR标题与分类

**PR标题**须使用下列前缀之一（与[vLLM PR分类](https://docs.vllm.ai/en/latest/contributing/#pr-title-and-classification)一致，并按vime模块做了细化）：

- `[Bugfix]` — Bug修复
- `[CI/Build]` — CI或构建相关
- `[Doc]` — 文档修复与改进
- `[Training]` — Megatron训练、权重同步等训练侧逻辑
- `[Rollout]` — vLLM rollout、router或推理集成
- `[Core]` — 框架编排（data buffer、Ray actor、公共参数等）
- `[Example]` — `examples/`、`scripts/`下的训练脚本与示例
- `[Hardware][Vendor]` — 硬件相关（如NPU：`[Hardware][Ascend]`）
- `[Misc]` — 其他；请尽量少用

若PR跨多个模块，可组合前缀（如`[Bugfix][Rollout]`）。

### 提交前检查

- [ ] `pre-commit run --all-files`通过
- [ ] 相关测试通过
- [ ] 行为或参数变更已更新文档
- [ ] PR标题前缀正确；commit含`Signed-off-by:`
- [ ] **新功能**已关联并经维护者认可的Issue
- [ ] 若使用AI：已在PR描述中说明，并在commit中按需标注

### 代码质量

- 通过`pre-commit`等静态检查
- 行为变更须补充或更新测试
- 用户可见变更须更新`docs/`（适用时中英同步）
- PR保持聚焦；较大架构改动（除测试/配置外超过约500行）须**先开Issue**讨论方案

### 审查流程

1. CI自动运行pre-commit与PR测试
2. 维护者审查正确性、范围与可维护性
3. 根据反馈修改并在就绪后请求再次审查
4. 通过后由维护者合并

### 优质PR建议

- 写清**做了什么**、**为什么**、**怎么验证**
- 训练或rollout相关改动请注明GPU/NPU、vLLM版本与验证方式
- Bug修复请附日志与最小复现命令

## 报告Bug

请使用[Bug Report模板](.github/ISSUE_TEMPLATE/bug_report.yml)，或在Issue中提供**完整**复现信息。信息不足、无法复现的Issue可能会被关闭，请包含：

- **环境** — 操作系统、Python、PyTorch、CUDA或NPU栈（如CANN / `torch_npu`）、GPU/NPU型号与数量、vLLM与Megatron版本
- **复现步骤** — 最小命令或脚本；说明属于训练、rollout或两者
- **期望与实际行为**
- **日志** — 完整traceback及Ray / vLLM / Megatron相关输出
- **配置** — 启动脚本、CLI参数或配置片段（请脱敏）
- **硬件模式** — GPU或NPU；是否colocate等

提交前请确认：

- [ ] 已阅读[docs/zh/get_started/qa.md](docs/zh/get_started/qa.md)与[README_zh.md](README_zh.md)
- [ ] 已搜索[已有Issue](https://github.com/vllm-project/vime/issues)避免重复
- [ ] 尽可能在最新`main`上复现

## 功能建议

**重大或复杂的新功能请先开Issue讨论**，便于对齐：

- 问题背景与使用场景
- 是否符合vime轻量、以vLLM为中心的方向
- 验证方式（CI、单测或文档化训练流程）
- 对GPU/NPU路径的影响

Issue中建议简要说明：问题背景、方案概要（影响training / rollout / buffer哪一块）、备选方案，以及如何验证正确性与性能。

难以在CI或常规流程中持续验证的大功能，长期维护成本较高，可能会延后合入。

## 许可证

向vime贡献即表示你同意将贡献置于[Apache License 2.0](LICENSE)下授权。
