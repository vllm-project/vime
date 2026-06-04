# Vime

[中文版](./README_zh.md) · [Repository](https://github.com/vllm-project/vime)

**Vime** is an LLM post-training framework for RL scaling, built on [slime](https://github.com/THUDM/slime). It keeps slime's training stack and data-generation design while using [**vLLM**](https://github.com/vllm-project/vllm) (with [vllm-router](https://github.com/vllm-project/router)) as the default rollout backend. Vime provides two core capabilities:

1. **High-performance training**: Efficient training in various modes by connecting Megatron with vLLM;
2. **Flexible data generation**: Arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

Vime inherits broad model support from slime, including:

- Qwen series (Qwen3.6, Qwen3.5, Qwen3Next, Qwen3MoE, Qwen3, Qwen2.5);
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

## Table of Contents

- [Architecture Overview](#architecture-overview)
- [Quick Start](#quick-start)
- [Arguments Walkthrough](#arguments-walkthrough)
- [Developer Guide](#developer-guide)
- [slime doc](#slime-doc)
- [FAQ & Acknowledgements](#faq--acknowledgements)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions**:

- **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
- **rollout (vLLM + router)**: Launches vLLM inference engines and routes generation requests; produces new data (including rewards/verifier outputs) and stores it in the Data Buffer.
- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods.

## Quick Start

For a comprehensive quick start guide covering environment setup, data preparation, training startup, and key code analysis, please refer to:

- [Quick Start Guide](./docs/en/get_started/quick_start.md)

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

## Arguments Walkthrough

Arguments in Vime are divided into three categories:

1. **Megatron arguments**: Vime reads all arguments in Megatron. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2. **vLLM arguments**: vLLM server and engine options are exposed with a `--vllm-` prefix (for example, `--vllm-gpu-memory-utilization`). Router options live under two prefixes: vllm-router's native options are passed with `--router-` (for example, `--router-policy round_robin`, `--router-request-timeout-secs`), while Vime-side orchestration knobs that tell Vime *where* the router lives use `--vllm-router-` (`--vllm-router-ip`, `--vllm-router-port`). See [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py) for the full surface.
3. **Framework-specific arguments**: Shared Vime orchestration flags (rollout GPUs, data paths, RL algorithms, etc.). Please refer to [vime/utils/arguments.py](vime/utils/arguments.py).

`--rollout-num-gpus-per-engine` sets the tensor parallel size of each vLLM engine. The default rollout entry is `vime.rollout.vllm_rollout.generate_rollout`.

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Developer Guide

- **Contributions are welcome!** If you have suggestions for new features, performance tuning, or feedback on user experience, feel free to submit an Issue or PR.

- Use [pre-commit](https://pre-commit.com/) to ensure code style consistency for your commits:

```bash
apt install pre-commit -y
pre-commit install

# run pre-commit to ensure code style consistency
pre-commit run --all-files --show-diff-on-failure --color=always
```

- For debugging tips, please refer to the [Debugging Guide](docs/en/developer_guide/debug.md)

## slime doc

Vime is derived from slime. The following upstream resources and in-repo guides still use the slime naming and remain the reference for shared concepts (Megatron integration, customization, advanced topics):

[![Documentation](https://img.shields.io/badge/slime_docs-latest-brightgreen.svg?style=flat)](https://thudm.github.io/slime/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/THUDM/slime)

- Upstream repository: [THUDM/slime](https://github.com/THUDM/slime)
- English docs in this repo: [docs/en/](docs/en/)
- Chinese docs in this repo: [docs/zh/](docs/zh/)

## FAQ & Acknowledgements

- For frequently asked questions, please see the [Q&A](docs/en/get_started/qa.md)
- Special thanks to the **slime** community and the open-source projects that slime depends on, including **SGLang**, **Megatron-LM**, and others that Vime builds upon.

Citation:

```bibtex
@misc{vime,
  author       = {Vime Contributors},
  title        = {Vime: An LLM post-training framework with vLLM for RL Scaling},
  year         = {2026},
  howpublished = {\url{https://github.com/vllm-project/vime}},
  note         = {Based on slime. GitHub repository.},
  urldate      = {2026-05-25}
}
```
