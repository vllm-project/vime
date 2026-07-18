vime Documentation
====================

vime is an LLM post-training framework for RL scaling, providing two core capabilities:

- High-Performance Training: Supports efficient training in various modes by connecting Megatron with vLLM;
- Flexible Data Generation: Enables arbitrary training data generation workflows through custom data generation interfaces and server-based engines.

vime is built on `slime <https://github.com/THUDM/slime>`_, the RL framework behind GLM-4.7, GLM-4.6 and GLM-4.5. vime keeps slime's training stack and data-generation design while using vLLM as the default rollout backend, and inherits broad model support from slime, including:

- Qwen3 series (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 series;
- DeepSeek V3 series (DeepSeek V3, V3.1, DeepSeek R1);
- Llama 3.

Start by Use Case
-----------------

- New to vime: :doc:`get_started/quick_start`
- Configure training and rollout arguments: :doc:`get_started/usage`
- Add custom generation, reward, or rollout functions: :doc:`get_started/customization`
- Build agentic RL workflows: :doc:`get_started/agent`
- Configure production vLLM rollout topology: :doc:`advanced/vllm-config`
- Connect external rollout engines: :doc:`advanced/external-rollout-engines`
- Sync weights as byte-level deltas: :doc:`advanced/delta-weight-sync`
- Use PD disaggregation: :doc:`advanced/pd-disaggregation`
- Use BF16 training with FP8 rollout or FP8 KV cache: :doc:`advanced/low-precision`
- Understand CI and reliability coverage: :doc:`developer_guide/ci`
- Debug, trace, and profile long-running jobs: :doc:`developer_guide/debug`, :doc:`developer_guide/trace`, :doc:`developer_guide/profiling`

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
   examples/gemma4.md
   examples/glm4-9B.md

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

.. toctree::
   :maxdepth: 1
   :caption: Hardware Platforms

   platform_support/amd_tutorial.md
   platform_support/ascend_tutorial.md
