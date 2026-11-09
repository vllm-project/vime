# Fixed-depth recurrent GRPO

Use the [Nanbeige recipe](../nanbeige) with `--algorithm grpo`, or invoke
`examples/looped_grpo/run.py` with the same model/data/revision/output arguments.
The launcher shares the [recurrent training entry](../looped_ppo).

GRPO trains an actor with VIME's existing clipped policy loss and per-prompt reward
normalization. Defaults use four prompts with four completions each. Group size
must be at least two. Complete, ordered groups are checked before math reward
calculation; each completion keeps its seed, fixed-depth trace and policy identity.
No additional batch-wide advantage whitening is applied. Equal rewards within
a group yield zero advantages and may leave policy weights unchanged.

Fresh resume restores only `checkpoints/actor`, the dataset cursor and sample
indices, then publishes the full actor before generating. Compare tokens,
rewards, traces and actor/optimizer/scheduler/RNG state with uninterrupted training.
