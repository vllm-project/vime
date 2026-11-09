# Categorical Flow-DPPO extension

This extends the asymmetric divergence gate from [Flow-DPPO](https://arxiv.org/abs/2606.11025)
to categorical token policies. The original paper uses Gaussian denoising
transitions. This implementation instead computes the complete categorical
`KL(old || new)` over the vocabulary at every response position, conditional on
the saved recurrent trace. It does not use selected-token approximate KL as its
divergence budget.

An outward update has positive advantage with ratio above one, or negative
advantage with ratio below one. Its gradient is blocked only when full KL exceeds
the budget. Corrective updates remain active. PPO ratio clipping is replaced by
this gate; PPO's existing value loss and GAE remain.

The frozen actor forward stores full response distributions on CPU before any
optimizer update. Packed microbatches carry these distributions to the loss.
The initial implementation requires TP=CP=1 and full-vocabulary sampling.

Use the [recurrent PPO recipe](../../examples/looped_ppo/README.md) and append:

```bash
--flow-dppo-divergence-budget 0.01 --num-steps-per-rollout 2 --global-batch-size 2
```

Keep these settings identical for uninterrupted and save/resume runs. Record
`categorical_kl`, blocked fraction, policy/value gradients and actual mathematical
rewards. Report full E2E time including the frozen-policy distribution forward;
this additional work is part of the categorical extension's measured cost.
