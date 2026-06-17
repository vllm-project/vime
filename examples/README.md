# Examples

These examples provide concrete examples to leverage vime in your own RL workflow. Some examples are just demonstrative, but most of them are verifiable with a concrete performance score.

## Directory Structure

- **[coding_agent_rl](./coding_agent_rl)**: End-to-end SWE coding-agent RL: a real coding agent (claude-code) edits code in a per-sample sandbox, and the resulting `git diff` is graded against the dataset's test harness.
- **[fully_async](./fully_async)**: Demonstrates fully asynchronous rollout generation for higher efficiency.
- **[geo3k_vlm](./geo3k_vlm)**: Training VLMs on a single-turn reasoning task using GRPO on the GEO3K dataset.
- **[geo3k_vlm_multi_turn](./geo3k_vlm_multi_turn)**: VLM multi-turn training on Geo3k dataset.
- **[multi_agent](./multi_agent)**: Example of running multi-agent RL with `vime`.
- **[tau-bench](./tau-bench)**: Multi-turn tool-use agent training in tau-bench environments.
- **[train_infer_mismatch_helper](./train_infer_mismatch_helper)**: Algorithmic methods for rollout correction (e.g., TIS, MIS).
