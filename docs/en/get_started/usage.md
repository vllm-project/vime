# Usage Guide


## Introduction to vime Parameters

When using vime, parameters are primarily passed for the following purposes:

1.  To allocate a portion of the GPUs in the cluster for training and another portion for inference.
2.  To load Megatron for the training portion.
3.  To load vLLM for the inference portion.
4.  To configure the hyperparameters required for RL training.

Following this order, we need to configure these parameters:

### Cluster Resource Allocation

There are four main parameters for cluster resource allocation:

  - `--actor-num-nodes`: The number of nodes required for RL actor training.
  - `--actor-num-gpus-per-node`: The number of GPUs per node for RL actor training.
  - `--rollout-num-gpus`: The total number of GPUs required for rollout (inference). Set it to `0` to still parse vLLM arguments and launch the router without launching local vLLM servers.
  - `--rollout-num-gpus-per-engine`: The total worker GPU count for one inference engine. It equals vLLM's `tensor_parallel_size` only when data and pipeline parallelism are both 1. For example, if one model is served across 2 nodes and 16 GPUs, this value should be 16.

With the default configuration, we use these parameters to allocate `actor_num_nodes * actor_num_gpus_per_node` GPUs for training and `rollout_num_gpus` GPUs for inference via Ray, thus achieving a separation of training and inference resources.

For co-located training and inference, you also need to configure:

  - `--colocate`: Enables co-located training and inference. By default, this makes the number of GPUs for training and inference equal. You can explicitly set a different positive `--rollout-num-gpus`, for example to use more rollout GPUs than actor GPUs; the extra GPUs are used as rollout-only resources. If `--rollout-num-gpus 0` is set explicitly, vime launches only the router and no local vLLM servers.

Additionally, vime supports Prefill and Decode disaggregation (PD Disaggregation). You can set the number of servers used for Prefill by setting the `--prefill-num-servers` argument.

### Choosing Training Backend

vime currently uses Megatron-LM as its training backend. The compatibility option
`--train-backend megatron` may still be supplied explicitly.

### Loading Megatron

Unlike tools such as vLLM or Hugging Face Trainer, Megatron cannot directly read Hugging Face checkpoints. Instead, the user must configure the parameters for the model to be trained and load Megatron's own checkpoint format.

Generally, we need to perform three preparatory steps:

  - Configure model parameters.
  - Configure parallelism and other optimizations.
  - Configure the checkpoint to be loaded.

For details on some of Megatron's customizations and the principles behind how vime incorporates Megatron, please see the "How to Use Megatron" section.

#### Configuring Model Parameters

Taking qwen3 4B as an example, we need these parameters:

```bash
MODEL_ARGS=(
   --num-layers 36
   --hidden-size 2560
   --ffn-hidden-size 9728
   --swiglu
   --vocab-size 151936
   --disable-bias-linear
   # attn head
   --num-attention-heads 32
   --group-query-attention
   --num-query-groups 8
   --kv-channels 128
   --qk-layernorm
   # norm
   --normalization "RMSNorm"
   --norm-epsilon 1e-6
   # rope
   --use-rotary-position-embeddings
   --rotary-base 1000000
)
```

We provide configurations for common models in [scripts/models](../../../scripts/models), which you can reuse directly. If you are also using Megatron for pre-training/SFT, you can directly reuse the model configurations from your pre-training/SFT setup.

Note:

  - vime will load all parameters of Megatron found in the `PYTHONPATH`, so you can find parameters and their descriptions within the Megatron in your environment.
  - vime uses data packing (also known as varlen or thd) for training. There is no need to configure `--seq-length` or `--max-positional-embedding`, as these parameters do not affect the maximum context length of the trained model.

#### Setting Up Parallelism and Recomputation

Megatron is currently the most comprehensively optimized training framework. A major reason for using Megatron is to pursue its excellent performance. Here is a brief introduction to configuring Megatron's parallelism and recomputation.

  - Here we list Megatron's parallelism strategies. For a more detailed discussion on the trade-offs between these strategies, please refer to more specialized discussions:
      - `--tensor-model-parallel-size`: TP
      - `--sequence-parallel`: Megatron's SP is an optimization for TP. It is recommended to always enable SP when using TP.
      - `--pipeline-model-parallel-size`: PP
      - `--context-parallel-size`: Megatron's CP, also known as sequence parallelism, generally corresponds to ring attention.
      - `--expert-model-parallel-size`: EP for MoE, where each GPU has `num_experts / ep_size` experts.
      - `--expert-tensor-parallel-size`: Megatron supports using a different `tp_size` for the MoE experts than for other parts of the model, which we generally call ETP.
  - For recomputation, the following flags are commonly configured in Megatron:
      - `--recompute-granularity`: This can be set to `full` or `selective`. `full` means complete recomputation, while `selective` recomputes less. If not configured, no recomputation is done.
      - `--recompute-method`: `uniform` is generally sufficient.
      - `--recompute-num-layers`: The number of layers per group for recomputation. A value of 1 is usually fine.

#### Loading Megatron Checkpoints

Megatron supports several of its custom checkpoint formats. Here are two of the more common ones:

  - The once mainstream `torch` format (corresponding to `--ckpt-format torch`).
  - The currently recommended `torch_dist` format (corresponding to `--ckpt-format torch_dist`).

The `torch` format is Megatron's older storage format. Its structure consists of directories like `mp_rank_xxx`, where each directory corresponds to the checkpoint stored by each rank under a specific parallel partitioning. Because of this, when loading a `torch` format checkpoint, you must ensure that the checkpoint's parallelism strategy matches that of the training task.

We recommend using the `torch_dist` format because it supports automatic parallel sharding, meaning that training tasks with different parallelism settings can share the same checkpoint, which is much more convenient. `torch_dist` is also the default format in the open-source Megatron. A `torch_dist` format checkpoint typically contains a set of `.distcp` files. When using `torch_dist`, you can convert from Hugging Face to `torch_dist` and vice versa using the checkpoint conversion method described in the [README](../../../README.md).

In terms of storage structure, a Megatron checkpoint typically looks like this, assuming the storage path is `/ckpt/`:

```bash
--/ckpt/
    |-- latest_checkpointed_iteration.txt
    |-- iter_0000100/
         |-- _0_0.distcp
         |-- _0_1.distcp
         |-- ...
    |-- iter_0000200/
    |-- iter_0000300/
    |-- ...
```

The `latest_checkpointed_iteration.txt` file records the latest training step. When loading a model, you should not directly pass `/ckpt/iter_xxxxxxx`, but rather pass `/ckpt/` and use `--ckpt-step` to select the corresponding training step (if `--ckpt-step` is not used, the step will be read from `latest_checkpointed_iteration.txt`).

When using vime, there are three parameters for loading and saving checkpoints:

  - `--ref-load`: The Megatron checkpoint for the reference model.
  - `--load`: The Megatron checkpoint for the actor. If `--load` is not set, or if the specified directory does not exist or does not contain `latest_checkpointed_iteration.txt`, the actor will be initialized from the `--ref-load` checkpoint.
  - `--save`: The path where the actor's checkpoints are saved.

Note:

  - Regardless of the checkpoint storage method (i.e., however `--ckpt-format` is set), Megatron can load both `torch` and `torch_dist` formats.

### Loading vLLM

Loading vLLM is very simple. You only need:

  - `--hf-checkpoint`: The Hugging Face checkpoint used to initialize vLLM.

Note:

  - Before the first training step, vime will synchronize the parameters from Megatron to vLLM. Therefore, the `--hf-checkpoint` does not need to contain the latest training parameters, and you do not need to change the HF checkpoint when resuming training.
  - By default, vLLM reads the maximum context length from the `config.json` in the Hugging Face checkpoint. You can use the `--vllm-max-model-len` parameter to override this value to support longer inference.
  - During co-located training and inference, although Megatron and vLLM will offload sequentially, they still need to leave some memory for each other. You need to adjust vLLM's total VRAM usage by reducing `--vllm-gpu-memory-utilization`.
  - vime supports passing through vllm-router parameters by adding a `router` prefix to the original parameter name. For example, vllm-router's `--balance-abs-threshold` parameter should be set as `--router-balance-abs-threshold`. Since vllm-router uses cache-aware routing by default, it may cause uneven request distribution. You can set `--router-balance-abs-threshold 0` to force balanced distribution, but this may affect prefix cache hit rate in multi-turn conversation scenarios. For multi-turn sessions that require session affinity, set `--router-policy consistent_hash` and send a stable `x-session-id` for each session.
  - If vLLM engines are pre-launched by an external system, connect to them with `--rollout-external-engine-addrs host1:port host2:port`. When the trainer and engines cannot form an NCCL weight-update group, use `--update-weight-mode full --update-weight-transport disk --update-weight-disk-dir /shared/fs/updates`; vime writes a complete HF checkpoint and asks vLLM to hot-load it through `update_weights_from_disk`. For large models or cross-cluster deployments, use `--update-weight-mode delta --update-weight-transport disk` instead. See [External Rollout Engines Roadmap](../advanced/external-rollout-engines.md) and [Delta Weight Sync](../advanced/delta-weight-sync.md).

For details on some of vLLM's customizations and the principles behind how vime incorporates vLLM, please see the "How to Use vLLM" section.

### Data Format

Raw data is managed by the DataSource. The built-in DataSource loads `--prompt-data` when provided; use `--data-source-path` for custom data management.

vime supports `.jsonl` and `.parquet` files; reading Parquet requires `pyarrow`. Each record in either format should contain the fields selected by `--input-key` and `--label-key`. An expanded JSONL record looks like this:

```json
{
  "prompt": [
    {
      "content": "Solve the following math problem step by step. The last line of your response should be of the form Answer: \\boxed{$Answer} where $Answer is the answer to the problem.\n\nIn triangle $ABC$, $\\sin \\angle A = \\frac{4}{5}$ and $\\angle A < 90^\\circ$. Let $D$ be a point outside triangle $ABC$ such that $\\angle BAD = \\angle DAC$ and $\\angle BDC = 90^\\circ$. Suppose that $AD = 1$ and that $\\frac{BD}{CD} = \\frac{3}{2}$. If $AB + AC$ can be expressed in the form $\\frac{a\\sqrt{b}}{c}$ where $a, b, c$ are pairwise relatively prime integers, find $a + b + c$.\n\nRemember to put your answer on its own line after \"Answer:\".",
      "role": "user",
      "step_loss_mask": 1,
    }
  ],
  "label": "34"
}
```

This corresponds to the following configuration:

```bash
  --input-key prompt
  --label-key label
  --apply-chat-template
```

Please note that the `step_loss_mask` (default=1) here is for SFT phase. If it is set to 0, the turn will not contibute to the final loss; if it is set to 1, vime will use the normal `loss_mask`.
Additionally, we provide a `metadata_key`, which defaults to `"metadata"`. When read, vime will load the metadata from the data, which can be helpful for custom data generation or creating custom reward models.

If one run mixes multiple data sources, put `source_name` in the sample metadata:

```json
{
  "prompt": "...",
  "label": "...",
  "metadata": {
    "source_name": "math"
  }
}
```

The recommended contract is to put the source identifier in `metadata["source_name"]`; vime also recognizes a dynamically set `sample.source` from custom data sources. When rollout samples are converted to training data, vime carries one `source_names` entry per sample to the training side. The source lookup order is dynamic `sample.source`, then `metadata["source_name"]`; if neither is set, the source is `"unknown"`. This is useful for custom rewards, filters, logging, and future per-source routing such as OPD teacher selection.

### Hyperparameters for RL Training

- `--advantage-estimator`: Specifies the RL algorithm for the training process. Currently supported algorithms include:
    - `grpo` ([https://arxiv.org/abs/2402.03300](https://arxiv.org/abs/2402.03300))
    - `gspo` ([https://arxiv.org/abs/2507.18071](https://arxiv.org/abs/2507.18071))
    - `cispo` ([https://arxiv.org/abs/2506.13585](https://arxiv.org/abs/2506.13585))
    - `reinforce_plus_plus` and `reinforce_plus_plus_baseline` ([https://arxiv.org/abs/2501.03262](https://arxiv.org/abs/2501.03262))
    - `ppo` ([https://arxiv.org/abs/1707.06347](https://arxiv.org/abs/1707.06347))

  Note: On-policy distillation (OPD) is now orthogonal to the advantage estimator. Use `--use-opd` and `--opd-kl-coef` to enable OPD on top of any estimator.
- `--calculate-per-token-loss`: By default, vime calculates loss on a per-sample basis, i.e., `mean(sum(sample_i) / len(sample_i))`. Enable this flag to calculate loss on a per-token basis, i.e., `sum(sum(sample_i)) / sum(len(sample_i))`.
- `--use-tis`: Enable this setting to use TIS (Truncated Importance Sampling) (https://fengyao.notion.site/off-policy-rl).
- `--use-score-centering`: Enable [Score Centering](https://arxiv.org/abs/2609.20807), optionally combined with TIS. See [Score Centering](#score-centering) below.

#### GRPO Algorithm

GRPO (Group Relative Policy Optimization) is an RL algorithm proposed in DeepSeek-Math. Its core idea is to compute advantage through intra-group relative comparisons, eliminating the need for a separate critic model.

To use GRPO, set:

```bash
--advantage-estimator grpo
```

Key features of GRPO:

- **No Critic Model Required**: GRPO samples multiple responses for the same prompt and estimates advantage by computing relative rewards within the group, avoiding the overhead of training and maintaining a critic model.
- **Resource Efficient**: Since no critic model is needed, GPU resources can be fully utilized for actor training and inference.
- **Simple to Use**: Easy configuration - just set `--advantage-estimator grpo`.

Related parameters:

- `--n-samples-per-prompt`: Number of responses sampled per prompt for intra-group comparison.
- `--normalize-advantages`: Whether to normalize advantages.
- `--eps-clip`: PPO-style clip range.

#### PPO Algorithm

PPO (Proximal Policy Optimization) is a classic RL algorithm that uses a critic model to estimate the value function for computing advantages.

To use PPO, set:

```bash
--advantage-estimator ppo
```

**Note: In PPO, the critic and actor share the same training GPU group.** You do not need to reserve a separate set of GPUs for the critic. Specifically:

- PPO creates separate actor and critic training process groups, but places them on the same train placement group.
- The critic training scale follows the actor configuration, and the actor / critic Megatron parallel topology must currently stay identical.
- PPO forces train-side offload so that actor and critic can wake up and release memory on the same GPUs in turn.
- There are currently no separate CLI arguments for configuring critic training resources; the critic node count and GPUs per node are derived from the actor configuration.


PPO-related parameters:

- `--megatron-config-path`: YAML config for role-specific Megatron overrides, such as setting critic-specific `load`, `save`, `lr`, or warmup parameters.
- `--num-critic-only-steps`: Number of steps to train only the critic at the beginning of training.
- `--eps-clip`: PPO clip range.
- `--value-clip`: Clip range for value loss.
- `--kl-coef`: KL penalty coefficient for reward shaping.

#### Score Centering

[Score Centering Stabilizes Off-policy Reinforcement Learning](https://arxiv.org/abs/2609.20807) introduces an additive correction to reduce drift caused by training-inference mismatch. It can also be combined with importance sampling. In vime, score centering (SC) is supported by the Megatron backend with non-streaming vLLM rollouts.

Add the following options to an existing RL launch:

```bash
--use-score-centering \
--score-centering-top-k 128 \
--pg-loss-type reinforce \
--advantage-estimator grpo \
--disable-grpo-std-normalization \
--calculate-per-token-loss \
--rollout-temperature 1.0 \
--rollout-top-p 1.0 \
--rollout-top-k -1 \
--entropy-coef 0 \
--kl-coef 0
```

- `--use-score-centering`: Enable the REINFORCE score-centering objective. If `--pg-loss-type` is omitted, SC selects `reinforce` automatically. Without SC, the existing PPO/CISPO default is preserved. SC cannot be combined with `--pg-loss-type ppo` or the GSPO/CISPO advantage estimators; PPO clipping parameters do not affect the REINFORCE objective.
- `--score-centering-top-k`: Applies only when `--rollout-top-p 1`. Number of sampler top-k token IDs and logprobs retained for each response token; defaults to 128 and must fit the model vocabulary. The head retains its full-vocabulary probability mass. The remaining sampler mass is modeled as proportional to the current trainer's tail mass.
- `--use-tis`: Optional and independent of SC. Combine the two to center the weighted scores using `--tis-clip-low` and `--tis-clip`. The built-in `vime.backends.megatron_utils.loss.icepop_function` is also supported through `--custom-tis-function-path`; arbitrary custom TIS callbacks are not supported with SC. REINFORCE uses detached current-trainer/sampler weights, while PPO preserves its old-trainer/sampler weights.

**Sampling requirements:** Use a positive temperature, `0 < top_p <= 1`, `top_k=-1`, `min_p=0`, and no repetition/frequency/presence penalties or constrained decoding. Keep `VLLM_RETURN_ORIGINAL_LOGPROB` unset or false on all sampler workers when temperature differs from one or top_p is below one. Per-request temperature or top_p changes and streaming SC are unsupported. Evaluation does not request SC data and may use its own sampling settings.

**Combining with top-p replay:** Change `--rollout-top-p 1.0` above to, for example, `--rollout-top-p 0.9`. SC automatically sums over the complete replay support and ignores `--score-centering-top-k`. Rollout returns all support IDs and their post-truncation, normalized sampler logprobs. The trainer normalizes on the same support and computes the correction `sum(stop_gradient(q * weight) * log p)`. This uses no tail approximation and excludes tokens outside the support. Full trainer logits cannot reconstruct the sampler probabilities, so the original probabilities must still be stored. Payload size varies with the support and can greatly exceed fixed top-k heads when top-p approaches one. Exactness is relative to the recorded replay support, including replay's existing rule for retaining sampled boundary tokens.

**vLLM support:** For `top_p=1`, Vime requests `k+1` native vLLM top logprobs and retains the highest-probability `k`. For `top_p<1`, it requests all logprobs, intersects them with vLLM's sampling mask, and normalizes the recorded support. Custom generators should call `score_centering_request` from `vime.utils.score_centering` and pass equivalent metadata to `Sample.append_response_tokens`.

Top-p probabilities stay on CPU with replay IDs/offsets and support partial rollouts, masked tool tokens, DP, TP, and both CP layouts. Exact top-p SC has a larger response payload because vLLM currently returns the full logprob vector before Vime selects the replay support.

Sampler heads survive partial-rollout continuation, masked tool tokens, DP partitioning, microbatch selection, TP and both CP layouts. With the R3 spill hook, they share its file lifetime. Logged metrics include `sc_correction`, `sc_sampler_head_mass`, `sc_train_head_mass` and `sc_importance_weight`.

### Advanced Megatron Configuration (--megatron-config-path)

For PPO workflows, you can use `--megatron-config-path` with a YAML file to override Megatron arguments separately for actor and critic. Common use cases include setting a different critic `lr`, or giving actor and critic different `load` / `save` paths.

```yaml
megatron:
  - name: default
    role: actor
    overrides:
      lr: 1e-6
  - name: default
    role: critic
    overrides:
      lr: 1e-5
```

> **Note:** This configuration currently only supports PPO, and in current PPO the actor and critic must use the same Megatron parallel topology. The recommended pattern is to keep parallelism-related settings in the shared CLI arguments and put only role-specific differences in YAML. See [Megatron Config: Role-Based Training Overrides](../advanced/megatron-config.md) for details.

## Custom Rollout Function

vime supports customizing data generation (rollout) to various degrees.

  - By default, it uses the `generate_rollout` function from [vime/rollout/vllm_rollout.py](https://github.com/vllm-project/vime/blob/main/vime/rollout/vllm_rollout.py) for data generation. This file implements an asynchronous (asyncio) data generation flow based on vLLM and supports features like dynamic sampling and partial rollout.

  - You can completely replace the default `generate_rollout` by using the `--rollout-function-path` parameter. You just need to ensure that the function signature passed via `--rollout-function-path` is as follows:

    ```python
    def generate_rollout(args, rollout_id, data_source, evaluation=False) -> RolloutFnTrainOutput | RolloutFnEvalOutput:
        """
        Args:
            args: the whole args
            rollout_id: int, the id of the rollout, used for deterministic data generation
            data_source: the data source to get and store samples
            evaluation: bool, whether the rollout is for evaluation or not
        
        Returns:
            RolloutFnTrainOutput | RolloutFnEvalOutput: the output of the rollout
        """
            ...
            return output
    ```

    Where:

      - `args`: The complete arguments used for the vime run.

      - `rollout_id`: The ID of the current data generation round, used to ensure data order when resuming training.

      - `data_source`: A globally unique data source in vime, which can be used to get initial prompts, data IDs, and store partially generated samples for later use.

      - `evaluation`: A boolean indicating if the rollout is for evaluation. You can configure a separate evaluation function using `--eval-function-path`.

      - The returned `Sample` type is defined in [vime/utils/types.py](https://github.com/vllm-project/vime/blob/main/vime/utils/types.py). When implementing, you need to ensure the following fields are correctly set:

          - `tokens`: The tokens for the prompt + response.
          - `response_length`: The total length of the response. For multi-turn tasks, this is the length of the tokens remaining after the first-turn prompt.
          - `reward`: The reward for this data sample.
        - `status`: The status of this data sample (e.g., `Sample.Status.COMPLETED`, `Sample.Status.TRUNCATED`, `Sample.Status.ABORTED`, `Sample.Status.FAILED`).
          - `loss_mask` should be the same length as `response_length`, with `1` for tokens that should be included in the loss calculation and `0` for those that should be masked out.

  - In some cases, you may only need to replace the data generation logic. You can do this using `--custom-generate-function-path`. A simplified implementation of this function is as follows:

    ```python
    async def generate(args, sample: Sample, sampling_params) -> Sample:
        global TOKENIZER
        if TOKENIZER is None:
            TOKENIZER = AutoTokenizer.from_pretrained(args.hf_checkpoint, trust_remote_code=True)

        # send request to router
        prompt_token_ids = TOKENIZER(sample.prompt, add_special_tokens=False)["input_ids"]
        output = await post(
            f"http://{args.vllm_router_ip}:{args.vllm_router_port}/inference/v1/generate",
            {
                "token_ids": prompt_token_ids,
                "sampling_params": {"max_tokens": sampling_params["max_new_tokens"]},
            }
        )

        choice = output["choices"][0]
        response_token_ids = list(choice.get("token_ids") or [])

        # set sample
        sample.tokens = prompt_token_ids + response_token_ids
        sample.response_length = len(response_token_ids)
        finish_reason = choice.get("finish_reason") or "stop"
        if finish_reason == "length":
            sample.status = Sample.Status.TRUNCATED
        elif finish_reason in ("abort", "cancelled"):
            sample.status = Sample.Status.ABORTED
        else:
            sample.status = Sample.Status.COMPLETED
        sample.response = TOKENIZER.decode(response_token_ids) if response_token_ids else ""

        return sample
    ```

    For a more complete version, please refer to [vime/rollout/vllm_rollout.py](https://github.com/vllm-project/vime/blob/main/vime/rollout/vllm_rollout.py).

  - Sometimes, you may also need to support a custom reward model. This can be configured by setting `--custom-rm-path`.

### Persistent rollout queue and distributed fully async

The default rollout transport is Ray `object-store`, which does not require a
shared directory. `--rollout-data-transport nixl` selects Ray's NIXL tensor transport.
For persistent queues and packed tensor storage across machines, use
[straw](../advanced/straw.md) with a shared JuiceFS directory.

To enable distributed fully async rollout with straw, add:

```bash
--rollout-function-path vime.rollout.fully_async_rollout.generate_rollout_fully_async \
--rollout-data-transport straw \
--rollout-data-dir /shared/run/rollout_data
```

All nodes must mount the directory at the same absolute path and install `straw-queue>=0.1.2`.

With straw transport, `--use-rollout-routing-replay` stores completed samples' R3 routes in straw with their owning sample group. SC tensors are also stored in the same publication when `--use-score-centering` is enabled, including SC without R3. The default object-store transport keeps these tensors in memory unless the existing disk-spill hook is configured; that hook remains available for object-store rollouts. User sample hooks run first; R3, SC and sample metadata are published together into shared pack files. Later queue publications reuse their references. Large group bundles are split by the native record-count limit. Aborted prefixes are persisted by the queue continuation path. `--rollout-queue-online-gc` is a separate opt-in: after training acknowledges completion, straw can reclaim sealed packs whose owners have all released them. It is disabled by default.

One job-owned, non-restarting Ray actor coordinates task leases and a serialized dataset producer. Dataset cursor advancement and task submission are committed together. Workers receive small task references and read prompt groups directly from shared storage. This replaces the former in-memory distributed index allocator; no per-reader index ranges are abandoned on failure. Seeded shuffle and sample/group identities remain those of the existing data source.

`get_samples(n)` and `add_samples(groups)` retain their interfaces. Custom producers can pass `source.reader_config("unique_reader_id")` to a remote process and call `config.open()` there. IDs must be unique; `owner` is reserved. Returned partial groups are persisted and become available to every reader; there is no reader-local sample buffer. Readers renew leases; a stopped reader returns unfinished tasks, while lost workers' tasks become available after lease expiry. Newly generated retries have new attempts. Recovery uses the normal `--load`/`--save` checkpoint flow, with optional `--ckpt-step`; the selected checkpoint restores the queue branch and dataset cursor, so no separate queue-resume flag is needed.

Fully async starts one generation process per Ray node with CPU resources. Each reserves one CPU and receives a share of the configured concurrency in complete prompt groups. Faster workers may fill a global batch without waiting for slower ones. Local generation queues and collector prefetch remain bounded. Complete groups pass through the existing dynamic filter before publication; rejected groups contribute filter metrics. Distributed fully async still rejects `--rollout-all-samples-process-path`. The synchronous all-samples hook retains its existing Samples and calling order.

The default logical result is a complete filtered group; physical segment boundaries do not define training groups. Workers encode Samples explicitly and store large tensors as typed dependencies. The format uses immutable committed extents in append-only `.pack` files, relative references and checksums, with no pickle payload protocol. Unsupported custom field types fail explicitly. See the [straw architecture, deployment and recovery guide](../advanced/straw.md).

Built-in producers return a collection manifest. Legacy custom rollouts can still return Sample lists; Manager publishes and accepts one compatibility collection. New producers may return an accepted `RawRolloutRef`; Manager validates its receipt and passes it to the same BatchBuilder without rewriting its payload. BatchBuilder preserves reward/conversion hooks and DP scheduling, records the selection plan, and publishes all rank shards before exposing `TrainBatchRef`s. All rank references identify the same batch and plan.

The vime adapter currently binds the shared root at the same absolute path on all nodes and checks visibility in both directions. The underlying store's references are relocatable. r3/sc dependencies are adopted into queue storage before references escape the producer; old rollout spill cleanup cannot remove them. Queue files are retained by default. Offline inspection/cleanup requires stopped writers, coordinator and readers; ordinary reads or batch-ready events never delete data. Debug archives can reuse immutable Straw records; `.pt` exports remain available. Evaluation follows the existing path.

Straw checkpoints commit model, optimizer/RNG, queue, builder and dataset-cursor state together. `--load` restores the latest valid joint checkpoint by default; `--ckpt-step` selects a step in the active branch history. Restore creates an isolated branch from that snapshot, leaving the source run unchanged and excluding later-produced samples. Repeated rollback follows the active branch and cannot cross a fork point. If a model checkpoint predates a straw queue snapshot, recovery starts with an empty queue and restores the saved dataset cursor when available; a damaged or incomplete joint snapshot fails rather than silently falling back. This is durable queue/data state; it does not restore GPU KV cache or generation RNG.

R3 training reads only the CP/TP rows assigned to the current rank. Batched continuation publication, loading and state persistence reuse authenticated indices within a bounded read session; receipt lookups use batch RPCs. These optimizations preserve checksums, WAL durability and GC ownership. Read sessions do not replace ownership pins.

Distributed fully async producers stop admitting new groups while weights synchronize, then resume if another training rollout remains. In-flight results still finish and persist.

Joint checkpoint restore defers background GC until the saved training consumer state and its restored storage ownership are durable. Unavailable checkpoint data fails restoration before GC starts.

Restored reader buffers keep an explicit storage reference independent of filter decisions. A later data-source checkpoint moves that reference only after all active and not-yet-started consumers are captured in a complete, retained, durable snapshot. Retiring an older checkpoint can then release its obsolete tensors without losing resumable prefixes. A failed save keeps the previous reference; checkpoint retirement remains the caller's responsibility.

The manager still materializes the selected batch for conversion, so manager memory and shared-storage bandwidth remain limits. `--rollout-io-concurrency` bounds publication I/O submissions. straw's storage, journal and GC implementation is Rust; vime retains Python sample conversion and training integration. Measure throughput for the intended model, concurrency and shared filesystem before changing deployment defaults.

## How to Use vLLM

vime runs vLLM in server mode and talks to it over HTTP.

### Parameter Configuration

vime incorporates almost all vLLM parameters by forwarding vLLM's `EngineArgs` CLI flags. When setting a vLLM parameter, you need to add the `--vllm-` prefix. For example:

  - In co-located training and inference, you often need to limit GPU memory utilization. Pass it as `--vllm-gpu-memory-utilization`.
  - During training, if you want vLLM to infer beyond the maximum context length specified in the Hugging Face checkpoint's `config.json`, you need to use `--max-model-len`, which becomes `--vllm-max-model-len` in vime.
  - For multi-node large EP inference, you might need `--enable-expert-parallel`, `--data-parallel-size`, etc. These can be passed as `--vllm-enable-expert-parallel` and `--vllm-data-parallel-size` respectively.

Some parameters related to vime's resource scheduling are configured by vime itself, for example:

  - `--tensor-parallel-size` in vime is set using `--rollout-num-gpus-per-engine`.
  - `--model` in vime is set using `--hf-checkpoint`.

The way vLLM parameters are integrated into vime can be found in [vime/backends/vllm_utils/arguments.py](https://github.com/vllm-project/vime/blob/main/vime/backends/vllm_utils/arguments.py).

### How to Use the Router

vime uses [vllm-router](https://github.com/vllm-project/router) to manage the vLLM engines during the training process. You can configure the address of the router using `--vllm-router-ip` and `--vllm-router-port`. If not configured, a router will be started by default within the cluster.

After starting, all vLLM engines will register with the router. When actually generating data, you only need to send HTTP requests to the router, which will perform load balancing and forward the requests to the engines.

When you configure an external router using `--vllm-router-ip` and `--vllm-router-port`, vime will not start an internal router. Instead, it will register all its engines with this external router. You can then use this external router's address to implement more complex data generation workflows. Note that the router supports OpenAI-compatible APIs.

### Advanced Engine Configuration (--vllm-config)

For advanced deployments, you can use `--vllm-config` with a YAML file to configure server groups, multi-model serving, and selective weight updates.

**Multi-model deployment** allows serving multiple models simultaneously (e.g., an actor model that receives weight updates and a frozen reference/reward model):

```yaml
vllm:
  - name: actor
    update_weights: true          # receives weight updates from training (default)
    server_groups:
      - worker_type: regular
        num_gpus: 8
        num_gpus_per_engine: 4
  - name: ref
    model_path: /path/to/ref_model
    update_weights: false          # frozen, no weight updates
    server_groups:
      - worker_type: regular
        num_gpus: 4
        num_gpus_per_engine: 2
```

Each model gets its own router. The per-model router info is accessible via `args.vllm_model_routers` (a dict mapping model name to `(ip, port)` tuples). Custom rollout functions can use `get_model_url(args, "ref")` from `vime.rollout.vllm_rollout` to route requests to a specific model.

**Server group features:**
- `worker_type`: `regular`, `prefill`, `decode`, or `placeholder` (reserves GPU slots without creating engines)
- `overrides`: Dict of vLLM `EngineArgs` field overrides applied on top of `--vllm-*` CLI args
- `num_gpus_per_engine`: Per-group total worker GPU count override

## How to Use Megatron

vime supports different and lightly modified versions of Megatron by reusing common functions from the `megatron.training` directory, such as `parse_args`, `save_checkpoint`, and `load_checkpoint`. Therefore, when using it, you must ensure that Megatron is accessible in the `PYTHONPATH`, for example, by adding `export PYTHONPATH=/root/Megatron-LM` at runtime.

### Parameter Configuration

vime directly imports all parameters of the Megatron in the current environment by using `from megatron.training.arguments import parse_args`. If the version of Megatron you are using has parameters defined outside of `parse_args`, you can configure them by passing them in, similar to how it's done in [train.py](https://github.com/vllm-project/vime/blob/main/train.py), for example:

```python
if __name__ == "__main__":
    try:
        from pretrain_gpt import extra_args_provider
    except:
        extra_args_provider = None
    args = parse_args(extra_args_provider)
    train(args)
```

### Custom Parameters

In some customized Megatron implementations, special operations need to be performed during initialization or before/after a training step. We have added the following plugins for this purpose:

  - `--custom-megatron-init-path`: Adds some initialization calls.
  - `--custom-megatron-before-log-prob-hook-path`: Is called before calculating the log probability.
  - `--custom-megatron-before-train-step-hook-path`: Is called before each training step. You could use this to mix in special training losses, for example.
