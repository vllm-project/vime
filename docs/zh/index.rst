vime 文档
====================

vime 是一个面向 RL Scaling 的 LLM 后训练框架，提供两大核心能力：

- 高性能训练：通过连接 Megatron 与 vLLM，支持多种模式下的高效训练；
- 灵活的数据生成：通过自定义数据生成接口与基于服务器的引擎，实现任意训练数据生成流程。

vime 构建于 `slime <https://github.com/THUDM/slime>`_ 之上，slime 正是 GLM-4.7、GLM-4.6、GLM-4.5 背后的 RL 训练框架。vime 沿用了 slime 的训练栈与数据生成设计，并默认采用 vLLM 作为 rollout 后端，同时继承了 slime 广泛的模型支持，包括：

- Qwen3 系列 (Qwen3Next, Qwen3MoE, Qwen3), Qwen2.5 系列；
- DeepSeek V3 系列 (DeepSeek V3, V3.1, DeepSeek R1)；
- Llama 3。

.. toctree::
   :maxdepth: 1
   :caption: 开始使用

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
   :caption: 高级特性

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
   :caption: 其他用法

   _examples_synced/fully_async/README.md
   _examples_synced/multi_agent/README.md

.. toctree::
   :maxdepth: 1
   :caption: 开发指南

   developer_guide/ci.md
   developer_guide/debug.md
   developer_guide/trace.md
   developer_guide/profiling.md
