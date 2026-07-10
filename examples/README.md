# Examples

These examples provide concrete examples to leverage vime in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

- **[eval_multi_task](./eval_multi_task)**: Example for supporting evaluation multiple tasks with different configs.
- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs on a single-turn reasoning task using GRPO on the GEO3K dataset.
- **[geo3k_vlm_multi_turn](./geo3k_vlm_multi_turn)**: VLM multi-turn training on Geo3k dataset.
- **[low_precision](./low_precision)**: Examples of FP8 training and inference for improved throughput and stability.
- **[mem_agent](./mem_agent)**: MemAgent long-context RL — chunk-wise memory update, HotpotQA GRPO training, and RULER-HQA evaluation.
- **[multi_agent](./multi_agent)**: Example of running multi-agent RL with `vime`.
- **[tau-bench](./tau-bench)**: Multi-turn tool-use agent training in tau-bench environments.
- **[train_infer_mismatch_helper](./train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
