# PD 分离

PD Disaggregation 将 vLLM rollout 中的 Prefill worker 和 Decode worker 拆开部署。它特别适合 multi-turn、long-context 和 agentic RL：这些 workload 中，prompt processing 和 token generation 的计算/显存特征往往完全不同。

## 什么时候使用

建议在以下场景使用 PD 分离：

- rollout context 很长，或会随着多轮交互持续增长；
- decode 阶段占据主要 rollout 时间；
- multi-turn session 需要更好的 prefix-cache locality；
- prefill 和 decode 需要不同 TP、显存或 runtime 设置；
- 希望 rollout topology 更接近生产 serving，而不是单一 uniform inference group。

对于短单轮任务，默认 regular vLLM engine layout 通常更简单。

## 配置路径

vime 支持两种 PD 配置方式。

### 简单路径：`--prefill-num-servers`

如果只有单个 actor model，并且只需要简单 PD layout，可以设置：

```bash
--prefill-num-servers 1
```

这是轻量路径，适合只想拆开 prefill/decode、但不需要分别调每个 group 的场景。

### 高级路径：`--vllm-config`

生产级 rollout topology 推荐使用 [vLLM Config](vllm-config.md)。它可以独立配置 prefill 和 decode group，也能表达 EPD layout、heterogeneous server group、multi-model serving 和 per-group vLLM override。

示例：

```yaml
vllm:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: prefill
        num_gpus: 4
        num_gpus_per_engine: 2
        overrides:
          chunked_prefill_size: 8192
      - worker_type: decode
        num_gpus: 12
        num_gpus_per_engine: 4
        overrides:
          mem_fraction_static: 0.88
```

启动：

```bash
python train.py \
  --vllm-config vllm_pd.yaml \
  --rollout-num-gpus 16 \
  ...
```

## EPD：拆分视觉编码器

对视觉语言模型来说，vision tower 是第三种独立的 workload：突发式、由图像数量驱动、遇到纯文本样本时完全空闲。加入 `worker_type: encoder` 组即可把它放到专门的引擎上，由这些引擎把图像 embedding 写入 encoder cache，language 引擎直接消费，不再自己跑 vision tower。

EPD 与 PD 正交，两者可以组合：

```yaml
vllm:
  - name: actor
    update_weights: true
    server_groups:
      - worker_type: encoder
        num_gpus: 2
      - worker_type: prefill
        num_gpus: 6
      - worker_type: decode
        num_gpus: 8
```

vime 会先启动 encoder 组，再启动 prefill/decode 组，并为后者注入 `language_only: true` 和 encoder URL。encoder 引擎不会注册到 router 的 worker 列表中——只有 prefill 和 decode 承接路由流量。

去掉 `decode` 组就得到 `encoder` + `regular`，即只拆 vision tower、不做 PD。

完整的角色矩阵、自定义 rollout function 需要调用的 `prime_encoder`、如何替换 encoder cache connector 以及当前限制，见 [EPD 分离](vllm-config.md#3-epd-分离视觉编码器拆分)。

## 为什么这对 RL 重要

RL rollout 往往不是一批短 completion。Agentic 和 verifier-based workload 常见特征包括：

- 来自 tool/environment history 的长 prompt；
- 每个 sample 多轮交互；
- decode latency long tail；
- session-local prefix cache 机会；
- actor、reference、reward、judge model 资源需求不同。

PD 让 vime 在不改变 training loop 的情况下，使用更贴合真实 serving workload 的 rollout topology。

## 运维注意事项

- 新的复杂部署优先使用 `--vllm-config`，而不是 `--prefill-num-servers`。
- multi-turn agent 建议开启 router session affinity，使同一 sample 的多轮请求可以复用 prefix cache。见 [Session-Affinity Routing](vllm-config.md#session-affinity-routing-for-multi-turn-agents)。
- `--rollout-num-gpus` 应等于 vLLM config 中描述的 GPU 总数。
- 不要在同一个 model entry 中混用 `regular` worker 和 `prefill`/`decode` worker。`encoder` 是例外——它可以与任一布局组合。
- 当 prompt processing 和 token generation 的瓶颈不同时，分别调 prefill 和 decode 的 TP。

## 相关文档

- [vLLM Config](vllm-config.md)
- [Agentic RL Training Roadmap](../get_started/agent.md)
- [Trace Viewer](../developer_guide/trace.md)
