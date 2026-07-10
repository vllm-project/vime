# Vime

[English](./README.md) · [代码仓库](https://github.com/vllm-project/vime)

[![文档](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.vllm.ai/projects/vime/zh-cn/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vllm-project/vime)

**Vime** 是基于 [slime](https://github.com/THUDM/slime) 的 RL scaling 用 LLM post-training 框架。在保留 slime 训练栈与数据生成设计的同时，默认以 [**vLLM**](https://github.com/vllm-project/vllm)（配合 [vllm-router](https://github.com/vllm-project/router)）作为 rollout 后端。Vime 提供两大核心能力：

1. **高性能训练**：通过连接 Megatron 与 vLLM，支持各种模式的高效训练；
2. **灵活的数据生成**：通过自定义数据生成接口以及 server based engine，实现任意的训练数据生成流程。

Vime 继承了 slime 广泛的模型支持，包括：

- Qwen 系列（Qwen3.6、Qwen3.5、Qwen3Next、Qwen3MoE、Qwen3、Qwen2.5）；
- DeepSeek V3 系列（DeepSeek V3、V3.1、DeepSeek R1）；
- Llama 3。

讨论渠道：

- [Slack](https://vllm-dev.slack.com/archives/C0B8W5QFL22/p1780899164831779)
- [微信群](./imgs/wechat_group.png)

## 定位

vLLM 社区横向支持许多 LLM post-training 框架，包括（按字母顺序）[NeMo RL](https://github.com/NVIDIA-NeMo/RL)、[OpenRLHF](https://github.com/openrlhf/openrlhf)、[prime-rl](https://github.com/PrimeIntellect-ai/prime-rl)、[SkyRL](https://github.com/NovaSky-AI/SkyRL)、[verl](https://github.com/verl-project/verl) 等。我们创建了 Vime 项目，旨在将 slime 经过验证的训练范式无缝引入 vLLM 生态系统，提供一个可用于生产的桥梁，对齐两个项目的快速迭代节奏。我们希望有不同需求的用户都能在 vLLM 生态中找到适合自己工作流的选择。vLLM 社区会一如既往地支持这些 post-training 框架中的 vLLM 集成。

## 目录

- [Vime](#vime)
  - [定位](#定位)
  - [目录](#目录)
  - [架构总览](#架构总览)
  - [快速开始](#快速开始)
  - [参数说明](#参数说明)
  - [开发指南](#开发指南)
  - [slime doc](#slime-doc)
  - [FAQ](#faq)
  - [致谢](#致谢)
  - [引用](#引用)

## 架构总览

![arch](./imgs/arch.png)

**模块说明**：

- **training (Megatron)**：负责主训练流程，从 Data Buffer 读取数据，训练完后将参数同步至 rollout 模块；
- **rollout (vLLM + router)**：启动 vLLM 推理引擎并路由生成请求，产出新数据（含 reward/verifier），存储至 Data Buffer；
- **data buffer**：桥梁模块，管理 prompt 初始化、自定义数据与 rollout 生成方法。

## 快速开始

有关环境配置、数据准备、训练启动和关键代码分析的完整快速开始指南，请参考：

- [快速开始指南](./docs/zh/get_started/quick_start.md)

我们还提供了一些未在快速开始中覆盖的使用示例，请查看 [examples](examples/)。

## 参数说明

Vime 的参数分为三类：

1. **Megatron 参数**：Vime 会读取 Megatron 中的全部参数，可通过传入如 `--tensor-model-parallel-size 2` 的方式配置 Megatron；
2. **vLLM 参数**：vLLM server 与 engine 相关选项以 `--vllm-` 为前缀（例如 `--vllm-gpu-memory-utilization`）。路由相关选项分两类前缀：vllm-router 自身的选项以 `--router-` 传入（例如 `--router-policy round_robin`、`--router-request-timeout-secs`），Vime 侧用于告诉 Vime *router 在哪里* 的编排参数则以 `--vllm-router-` 为前缀（`--vllm-router-ip`、`--vllm-router-port`）。完整参数见 [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py)。
3. **框架参数**：与 Vime 编排相关的开关（rollout GPU、数据路径、RL 算法等），见 [vime/utils/arguments.py](vime/utils/arguments.py)。

`--rollout-num-gpus-per-engine` 对应每个 vLLM engine 的 tensor parallel size。默认 rollout 入口为 `vime.rollout.vllm_rollout.generate_rollout`。

完整使用说明请查阅 [使用文档](docs/zh/get_started/usage.md)。

## 开发指南

- **欢迎贡献！** 若有功能建议、性能调优或使用体验反馈，欢迎提交 Issue / PR。

- 使用 [pre-commit](https://pre-commit.com/) 保证提交代码风格：

  ```bash
  apt install pre-commit -y
  pre-commit install

  # 运行 pre-commit 保证代码风格
  pre-commit run --all-files --show-diff-on-failure --color=always
  ```

- 调试技巧请参考 [debug 指南](docs/zh/developer_guide/debug.md)

## slime doc

Vime 由 slime 衍生而来。以下上游资源与本仓库文档仍沿用 slime 命名，可作为共享概念（Megatron 集成、定制化、高级主题）的参考：

[![Documentation](https://img.shields.io/badge/slime_文档-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

- 上游仓库：[THUDM/slime](https://github.com/THUDM/slime)
- 本仓库英文文档：[docs/en/](docs/en/)
- 本仓库中文文档：[docs/zh/](docs/zh/)

## FAQ

常见问题请见 [Q&A](docs/zh/get_started/qa.md)

## 致谢

Vime 构建于开源 RL 生态的想法与基础设施之上。特别感谢 [slime](https://github.com/THUDM/slime) 社区——Vime 直接构建于其出色工作之上；也感谢 [SkyRL](https://github.com/NovaSky-AI/SkyRL) 与 [verl](https://github.com/verl-project/verl)，我们参考了它们的优秀工作。Vime 由 vLLM 社区维护。

## 引用

```bibtex
@misc{vime,
  author       = {Vime Contributors},
  title        = {Vime: An LLM post-training framework with vLLM for RL Scaling},
  year         = {2026},
  howpublished = {\url{https://github.com/vllm-project/vime}},
  urldate      = {2026-06}
}
```
