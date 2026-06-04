# Examples

These examples provide concrete examples to leverage vime in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

- **[eval_multi_task](./eval_multi_task)**: Example for supporting evaluation multiple tasks with different configs.
- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs on a single-turn reasoning task using GRPO on the GEO3K dataset.
- **[geo3k_vlm_multi_turn](./geo3k_vlm_multi_turn)**: VLM multi-turn training on Geo3k dataset.
- **[multi_agent](./multi_agent)**: Example of running multi-agent RL with `vime`.
- **[reproducibility](./reproducibility)**: Guides on achieving bitwise experiment reproduction using deterministic modes.
- **[train_infer_mismatch_helper](./train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
