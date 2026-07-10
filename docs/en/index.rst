vime Documentation
====================

vime is an LLM post-training framework for RL scaling, providing two core capabilities:

- High-Performance Training: Supports efficient training in various modes by connecting Megatron with vLLM;
- Flexible Data Generation: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

vime is built on `slime <https://github.com/THUDM/slime>`_, the RL framework behind GLM-4.7, GLM-4.6 and GLM-4.5. vime keeps slime's training stack and data-generation design while using vLLM as the default rollout backend, and inherits broad model support from slime, including:

- Qwen3 series (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 series;
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

.. toctree::
   :maxdepth: 1
   :caption: Get Started

   get_started/quick_start.md
   get_started/usage.md
   get_started/customization.md
   get_started/qa.md

.. toctree::
   :maxdepth: 1
   :caption: Dense

   examples/qwen3-4B.md

.. toctree::
   :maxdepth: 1
   :caption: MoE

   examples/qwen3-30B-A3B.md
   examples/glm5.2-744B-A40B.md
   examples/glm4.7-355B-A32B.md
   examples/deepseek-r1.md

.. toctree::
   :maxdepth: 1
   :caption: Advanced Features

   advanced/speculative-decoding.md
   advanced/reproducibility.md
   advanced/fault-tolerance.md
   advanced/observability.md
   advanced/pd-disaggregation.md
   advanced/external-rollout-engines.md
   advanced/delta-weight-sync.md
   advanced/vllm-config.md
   advanced/megatron-config.md
   advanced/arch-support-beyond-megatron.md

.. toctree::
   :maxdepth: 1
   :caption: Other Usage

   _examples_synced/fully_async/README.md
   _examples_synced/multi_agent/README.md

.. toctree::
   :maxdepth: 1
   :caption: Developer Guide

   developer_guide/ci.md
   developer_guide/debug.md
   developer_guide/trace.md
   developer_guide/profiling.md
