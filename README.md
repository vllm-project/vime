# Vime

[中文版](./README_zh.md) · [Repository](https://github.com/vllm-project/vime)

[![Documentation](https://img.shields.io/badge/docs-latest-brightgreen.svg?style=flat)](https://docs.vllm.ai/projects/vime/en/latest/)
[![Ask DeepWiki](https://deepwiki.com/badge.svg)](https://deepwiki.com/vllm-project/vime)

**Vime** is an LLM post-training framework for RL scaling, built on [slime](https://github.com/THUDM/slime). It keeps slime's training stack and data-generation design while using [**vLLM**](https://github.com/vllm-project/vllm) (with [vllm-router](https://github.com/vllm-project/router)) as the default rollout backend. Vime provides two core capabilities:

1. **High-performance training**: Efficient training in various modes by connecting Megatron with vLLM;
2. **Flexible data generation**: Arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

Vime inherits broad model support from slime, including:

- Qwen series (Qwen3.6, Qwen3.5, Qwen3Next, Qwen3MoE, Qwen3, Qwen2.5);
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

Discussion channels:

- [slack](https://vllm-dev.slack.com/archives/C0B8W5QFL22/p1780899164831779)
- [wechat group](./imgs/wechat_group.png)

## Positioning

The vLLM community horizontally supports many LLM post-training frameworks, including (in alphabetical order) [NeMo RL](https://github.com/NVIDIA-NeMo/RL), [OpenRLHF](https://github.com/openrlhf/openrlhf), [prime-rl](https://github.com/PrimeIntellect-ai/prime-rl), [SkyRL](https://github.com/NovaSky-AI/SkyRL), [verl](https://github.com/verl-project/verl), and so on. We built the Vime project to seamlessly bring slime's proven training paradigm into the vLLM ecosystem, offering a production-ready bridge that aligns both projects' rapid release cycles. We hope that users with different needs can find the right vLLM-ecosystem choice for their workflows. The vLLM community will continue to support the vLLM integration in these post-training frameworks.

## Table of Contents

- [Vime](#vime)
  - [Positioning](#positioning)
  - [Table of Contents](#table-of-contents)
  - [Architecture Overview](#architecture-overview)
  - [Quick Start](#quick-start)
    - [Agentic RL examples](#agentic-rl-examples)
  - [Arguments Walkthrough](#arguments-walkthrough)
  - [Code Reading Path](#code-reading-path)
  - [Developer Guide](#developer-guide)
  - [slime doc](#slime-doc)
  - [FAQ](#faq)
  - [Acknowledgements](#acknowledgements)
  - [Citation](#citation)

## Architecture Overview

![arch](./imgs/arch.png)

**Module Descriptions**:

- **training (Megatron)**: Responsible for the main training process, reads data from the Data Buffer, and synchronizes parameters to the rollout module after training.
- **rollout (vLLM + router)**: Launches vLLM inference engines and routes generation requests; custom generate functions can wrap generation with multi-turn loops, tool calls, environment/sandbox interaction, and verifier-based rewards.
- **data buffer**: A bridge module that manages prompt initialization, custom data, and rollout generation methods, including agentic workflows that produce samples through the same interface.

## Quick Start

For a comprehensive quick start guide covering environment setup, data preparation, training startup, and key code analysis, please refer to:

- [Quick Start Guide](./docs/en/get_started/quick_start.md)

We also provide examples for some use cases not covered in the quick start guide; please check [examples](examples/).

### Agentic RL examples

Agentic workloads use the standard rollout / Data Buffer loop through Vime's customization interfaces; they are not a separate framework:

- [`examples/multi_agent`](examples/multi_agent/README.md): Multi-agent generation through `--custom-generate-function-path`.
- [`examples/fully_async`](examples/fully_async/README.md): Fully asynchronous rollout for long-tail agent generation.
- [`examples/coding_agent_rl`](examples/coding_agent_rl/README.md): End-to-end coding-agent RL with Claude Code or Codex, sandboxed tool use, test-based rewards, and token-correct trajectory segments.

See the [Agentic RL Training Roadmap](docs/en/get_started/agent.md) and [Customization Guide](docs/en/get_started/customization.md). The coding-agent example ships an E2B-compatible backend, while the shared `vime.agent.sandbox.Sandbox` contract can be implemented for Docker, Modal, or local VMs.

## Arguments Walkthrough

Arguments in Vime are divided into three categories:

1. **Megatron arguments**: Vime reads all arguments in Megatron. You can configure Megatron by passing arguments like `--tensor-model-parallel-size 2`.
2. **vLLM arguments**: vLLM server and engine options are exposed with a `--vllm-` prefix (for example, `--vllm-gpu-memory-utilization`). Router options live under two prefixes: vllm-router's native options are passed with `--router-` (for example, `--router-policy round_robin`, `--router-request-timeout-secs`), while Vime-side orchestration knobs that tell Vime *where* the router lives use `--vllm-router-` (`--vllm-router-ip`, `--vllm-router-port`). See [vime/backends/vllm_utils/arguments.py](vime/backends/vllm_utils/arguments.py) for the full surface.
3. **Framework-specific arguments**: Shared Vime orchestration flags (rollout GPUs, data paths, RL algorithms, etc.). Please refer to [vime/utils/arguments.py](vime/utils/arguments.py).

`--rollout-num-gpus-per-engine` sets the tensor parallel size of each vLLM engine. The default rollout entry is `vime.rollout.vllm_rollout.generate_rollout`.

For complete usage instructions, please refer to the [Usage Documentation](docs/en/get_started/usage.md).

## Code Reading Path

Start from the training loop and follow the calls only as deep as needed:

```text
train.py: train
├─ vime/ray/placement_group.py       Ray resource and worker initialization
├─ vime/ray/rollout.py              RolloutManager.generate: rollout orchestration
│  └─ vime/rollout/vllm_rollout.py  Sample generation and reward computation
└─ vime/ray/actor_group.py          RayTrainGroup.async_train: training dispatch
   └─ vime/backends/megatron_utils/actor.py
      ├─ model.py                    Megatron model execution
      └─ loss.py                     RL losses and advantages
```

On a first pass, treat `vime/utils/arguments.py` as the configuration entry point. The deployment details in `vime/backends/vllm_utils/` and the weight-sync implementations under `vime/backends/megatron_utils/update_weight/` can wait until you need to change those areas.

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

## FAQ

For frequently asked questions, please see the [Q&A](docs/en/get_started/qa.md)

## Acknowledgements

Vime builds on ideas and infrastructure from the open-source RL ecosystem. We especially thank the [slime](https://github.com/THUDM/slime) community, whose great work Vime is directly built on. We also thank [SkyRL](https://github.com/NovaSky-AI/SkyRL) and [verl](https://github.com/verl-project/verl), whose excellent work we referenced. Vime is maintained by the vLLM community.

## Citation

```bibtex
@misc{vime,
  author       = {Vime Contributors},
  title        = {Vime: An LLM post-training framework with vLLM for RL Scaling},
  year         = {2026},
  howpublished = {\url{https://github.com/vllm-project/vime}},
  urldate      = {2026-06}
}
```
