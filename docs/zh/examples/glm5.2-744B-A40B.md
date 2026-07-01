# 18xGB300 训练 GLM-5.2 744B-A40B

这里是使用 18 个 tray、72 张 GB300 训练 [GLM-5.2](https://z.ai/blog/glm-5.2) 的推荐配置示例。

这个配置使用 GLM-5.2 的 BF16 checkpoint 做 Megatron 训练，使用 FP8 checkpoint 做 vLLM rollout。下面假设 Hugging Face 上会提供两个地址：

- BF16: `zai-org/GLM-5.2`
- FP8: `zai-org/GLM-5.2-FP8`

## 环境准备

搭建环境与下载数据的方法可以参考 [示例：Qwen3-4B](qwen3-4B.md)。多机启动前，请确保所有节点都能访问同一个模型根目录和数据根目录。

### 下载模型

```bash
hf download zai-org/GLM-5.2 --local-dir $BASE_DIR/GLM-5.2
hf download zai-org/GLM-5.2-FP8 --local-dir $BASE_DIR/GLM-5.2-FP8
```

开源 GLM-5.2 的 config 使用 `model_type: glm_moe_dsa`，vime 将其映射到 DeepSeek-V3.2 的 bridge（`vime_plugins.mbridge.deepseek_v32`），因为两者共享相同的 DSA 权重布局。

### 转换 Checkpoint

训练侧需要把 BF16 Hugging Face checkpoint 转换为 Megatron 可加载的 torch_dist 格式。torch_dist 格式支持重新切分，所以转换时的并行布局**不需要**与训练一致；我们使用一个能满足 Megatron expert group 约束（在转换的节点数下成立）的布局即可。

可以在 4 台机器 / 16 卡上分别执行：

```bash
cd /mnt/weka/aoshen/vime/projects/vime-pr286
pip install -e . --no-deps
source scripts/models/glm5.2-744B-A40B.sh
PYTHONPATH=/root/Megatron-LM/ torchrun \
   --nproc-per-node 4 \
   --master-addr ${MASTER_ADDR} --master-port 12345 \
   --nnodes=4 --node-rank ${NODE_RANK} \
   tools/convert_hf_to_torch_dist.py \
   ${MODEL_ARGS[@]} \
   --tensor-model-parallel-size 4 \
   --pipeline-model-parallel-size 2 \
   --decoder-last-pipeline-num-layers 40 \
   --expert-model-parallel-size 8 \
   --expert-tensor-parallel-size 1 \
   --hf-checkpoint $BASE_DIR/GLM-5.2/ \
   --save $BASE_DIR/GLM-5.2_torch_dist/
```

其中 `MASTER_ADDR` 是 node0 的 IP，`NODE_RANK` 表示当前机器编号。

`MODEL_ARGS` 里包含 `--allgather-cp` 这个 vime 自定义参数，所以 `tools/convert_hf_to_torch_dist.py` 也注册了它（转换时是 no-op）。在 32 卡上，Megatron 要求 `expert_tp(1) * expert_model_parallel * pp` 能整除 world size，因此转换时使用 `EP=16`（`1*16*2=32`）。由于 torch_dist 支持重新切分，转换出的 checkpoint 仍然可以在训练时以 `EP=32` 加载。

## 执行训练

从 node0 执行：

```bash
cd /mnt/weka/aoshen/vime/projects/vime-pr286
export MODEL_ROOT=/mnt/weka/models
export DATA_ROOT=/mnt/weka/aoshen/data/dapo-math-17k-hf
export MASTER_ADDR=<node0-ip>
export HOSTFILE=$MODEL_ROOT/hostfile  # 每行一个 worker IP，共 18 个节点
bash scripts/run-glm5.2-744B-A40B.sh
```

如果不设置 `HOSTFILE`，需要手动在其他节点加入 Ray 集群。

### 参数简介

#### 模型配置

`scripts/models/glm5.2-744B-A40B.sh` 使用 GLM-5.2 的 DSA + cross-layer index sharing 配置：256 个 routed experts、top-8 激活、1 个 shared expert，模型共 78 层（3 层 dense + 75 层 MoE）。

DSA index sharing 的 schedule（例如 `index_topk_freq=4`、`index_skip_topk_offset=3`）从 Hugging Face config 中读取。Megatron 侧使用共享的 `vime_plugins.models.glm5.glm5:get_glm5_spec` provider，并开启：

```bash
--allgather-cp
```

这会让 DSA + context parallel 使用 allgather-CP layout，并在 index-share provider 中对 index K/V 做 CP group gather。

#### 训练并行

默认脚本按 18 个 tray、72 卡配置：

```bash
PERF_ARGS=(
   --tensor-model-parallel-size 4
   --pipeline-model-parallel-size 9
   --decoder-first-pipeline-num-layers 10
   --decoder-last-pipeline-num-layers 12
   --context-parallel-size 2
   --expert-model-parallel-size 8
   --expert-tensor-parallel-size 1
   ...
)
```

`TP=4 * PP=9 * CP=2 = 72` 卡构成一个训练组（`DP=1`）。expert group 约束 `expert_tp(1) * EP(8) * PP(9) = 72` 正好整除 world size（`expert_dp=1`）。

DSA cross-layer index sharing 要求每个 pipeline stage 都必须**从 computing layer 开始**。在 `index_topk_freq=4` / `index_skip_topk_offset=3` 下，computing layer 是第 1、2、3、7、11、...、75 层。如果直接 `78/9` 均分，stage 会从 skip layer 开始，触发 `get_glm5_spec` 里的 index-share 断言。因此我们使用 `--decoder-first-pipeline-num-layers 10` 和 `--decoder-last-pipeline-num-layers 12`，中间 7 个 stage 各 `(78-10-12)/7 = 8` 层。各 stage 的起始全局层为 1、11、19、27、35、43、51、59、67，全部是 computing layer。

#### BF16 训练 + FP8 Rollout

训练脚本直接在 `CKPT_ARGS` 和 `ROLLOUT_ARGS` 中写入默认路径，风格与其他示例脚本保持一致：

```bash
CKPT_ARGS=(
   --hf-checkpoint $BASE_DIR/GLM-5.2-FP8
   --ref-load $BASE_DIR/GLM-5.2_torch_dist
   --load $BASE_DIR/GLM-5.2_vime
   --save $BASE_DIR/GLM-5.2_vime
   --save-interval 20
)

ROLLOUT_ARGS=(
   --prompt-data $BASE_DIR/dapo-math-17k/dapo-math-17k.jsonl
   ...
)
```

`--hf-checkpoint` 提供 vLLM rollout 所需的 FP8 权重和 tokenizer；`--ref-load` 是从 BF16 checkpoint 转换出的 Megatron torch_dist 权重。需要调试 BF16 rollout 时，可以直接把脚本里的 `--hf-checkpoint` 改成 `$BASE_DIR/GLM-5.2`。

#### vLLM 配置

rollout 侧每个 engine 跨 2 个 tray（8 张 GPU，TP=8），prefill 和 decode 共卡。72 张 GPU 共 9 个 engine，全部开启 expert parallel 和 MTP speculative decoding。切分通过 `--vllm-config` YAML 配置:

```yaml
vllm:
  - name: default
    server_groups:
      - worker_type: regular
        num_gpus: 72
        num_gpus_per_engine: 8
```

rollout 配置使用 FP8 权重、MTP4 speculative decoding:

```bash
VLLM_ARGS=(
   --rollout-num-gpus-per-engine 8
   --vllm-gpu-memory-utilization 0.85
   --vllm-max-model-len 131072
   --vllm-enable-expert-parallel
   --vllm-enable-ep-weight-filter
   --vllm-speculative-config '{"method":"mtp","num_speculative_tokens":4}'
   ...
)
```

MTP speculative decoding 直接使用模型自带的 next-token-prediction 层（GLM-5.2 checkpoint 自带 MTP 层），因此不需要单独的 draft model。独立 benchmark 测试中 MTP4 相比无 MTP 基线，输出吞吐提升 76.7%，TPOT 从 28.59 ms 降至 13.10 ms，acceptance rate 70.16%。

#### 网络

rollout 的跨节点通信使用 Ray runtime env 里的网络接口配置。脚本默认使用 `SOCKET_IFNAME=bond0.225`，如环境不同可在启动前设置 `SOCKET_IFNAME`，它会同时写入 `GLOO_SOCKET_IFNAME`、`TP_SOCKET_IFNAME` 和 `NCCL_SOCKET_IFNAME`。
