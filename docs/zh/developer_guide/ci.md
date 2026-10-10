# CI（持续集成）

Vime 使用 Buildkite 进行持续集成。提交到仓库中的 pipeline 是
`.buildkite/pipeline.yml`。

## 始终运行的检查

每个 pull request 都会运行以下 CPU step：

| Step | 覆盖范围 |
|---|---|
| `pre-commit` | 格式化、lint 与仓库规则 |
| `plugin-contracts` | customization contract 与 CPU 测试 |
| `agent-adapter` | agent adapter 行为 |
| `upstream-sync-cpu` | 从上游同步的 CPU 测试 |
| `utils` | `tests/utils` |

权威命令与队列配置位于 `.buildkite/pipeline.yml`。

## GPU 套件

CPU step 通过后，Buildkite build 会显示名为 `Run GPU test suites?` 的
block step。可以选择一个或多个套件：

- `short`
- `vllm-config`
- `megatron`
- `vime-customized`
- `precision`
- `ckpt`

`.buildkite/gpu_suites.py` 会把所选套件展开为每个测试一个 Buildkite
job。验证 Dockerfile 或 vLLM patch 修改时，通过 `VIME_CI_IMAGE` 指定不可变的
候选镜像 digest；未设置时使用 `vllm/vime:latest`，且 PR 合入前不得更新该标签。

`megatron` 套件包含 `test_straw_checkpoint_fork.py`，覆盖 checkpoint
步骤选择、回退分支和带索引的 debug 归档。

`test_qwen2.5_0.5B_pipeline_rl.py` 使用 4 张 GPU，运行 fully async rollout
和三个真实 GRPO step。探针检查同一请求跨权重更新持续生成，并确认训练改变了
策略权重。固定矩阵覆盖 `--flush-cache-interval 0` 下的 NCCL 和 full disk
权重同步，以及 NCCL 下 interval `2` 的周期性刷新。`test_pipeline_rl.py`
在 CPU 上检查刷新周期。这些测试不衡量学习效果或吞吐增益。

### Agent 训练

`tests/test_agent_sunabako_codex_e2e.py` 使用 **Codex 0.162.1**、**Claude Code 2.1.296**、
Qwen3.8-27B 和两个独立镜像中的小米任务：

- `format-code-task-000003`：给 smol-evm 补上缺失的 `SHL` 左移指令。
- `format-code-task-000045`：让 BootBot 保留对象形式的 quick replies。

每一步都使用这两道题，**每题 4 条，共 8 个独立 agent 并行**，分别在新沙箱中评分，
然后执行一次 GRPO 优化器更新。**第一步使用 Codex，第二步使用 Claude Code**，
第二步使用更新后的模型，对相同两道题重新采样。两个 CLI 共用 trace 拼接和训练路径。
总共运行 16 条轨迹、训练 2 步，不按 reward 过滤或额外重采。
两道题的原始代码都必须无法通过官方测试；失败轨迹保留 reward 0，与 reward 1 的轨迹
一起参与训练。每道题的 4 条结果独立归一化，使用组内均值与样本标准差加 epsilon；
整组同分时，advantage 正常为 0。

测试会从实际训练张量中核对每个 token 的 advantage，并记录 8 个训练 rank 上的两次
优化器调用。梯度与参数必须有限；非零梯度必须带来参数变化。采样使用 temperature=1、
`top_p=0.95`，禁用 top-k，启用 SC，并校验原始 token、mask、sampler 概率及 nucleus
replay 数据。Qwen3.8-27B 是 dense 模型，因此关闭 R3。
推理通过 vLLM 的 `mtp` 方法启用 checkpoint 中的 MTP head（3 个 speculative token），
并检查日志中确实出现已接受的 draft token。
上下文额度为 64K；vLLM 在内部管理 lookahead 状态。Claude Code 使用与 example 相同的
六个代码工具：Bash、Read、Edit、Write、Glob 和 Grep。

每个 agent 上限 600 秒，Buildkite job 连同准备阶段上限 35 分钟。
连续的模型调用会保留原始 token 和采样分布，合并成训练轨迹；历史实际改写时才分支。
CI 与正式 example 一样，训练每条真实轨迹中的全部片段，保持 token、logprob 和
reward 原样；`agents/<sample-index>/agent-full.pt` 保存完整轨迹。
agent 只收到原始题目和仓库，隐藏测试保留在独立评分沙箱中。

模型、选中的两个任务镜像、两个 CLI 安装包和 TileLang/Triton 编译结果均使用缓存。
镜像下载支持断点续传与完整性校验，`--prepare-only` 在申请 GPU 锁之前准备资源。
首次下载可能超过计时 CI 的上限。代理变量会传入容器，本机 Ray 和沙箱流量绕过代理。
Buildkite 的 `megatron` 套件包含这项 GPU e2e，自动 `agent-adapter`
步骤覆盖 CPU 合约检查。

`tests/ci/setup_agent_e2e.sh` 按 `examples/coding_agent_rl/requirements-sunabako.txt`
安装 PyPI 发布的 `sunabako==0.1.1` wheel，并在启动 Ray head 前准备认证 token；
已有 token 会复用，其内容不会打印到 CI 日志。该版本通过 `uid_range_size` 支持 native 用户，
本地节点为 8 个沙箱分别保留 65,536 个 UID/GID，
state 挂载到 `/workspace`，保证映射后的用户能遍历父目录。测试在特权 Buildkite test pod
中运行，无需内层 Docker daemon 或 PRoot。RSS 模式只用于有界功能验证，
**不证明总内存硬限制**；生产环境仍需可写、已委派的 cgroup，缺失时拒绝启动。

已有沙箱集群时，安装相同 requirements 后运行：

```bash
(umask 077; ray get-auth-token --generate >/dev/null)
HF_CHECKPOINT=/path/to/Qwen3.8-27B \
RAY_AUTH_MODE=token \
SUNABAKO_CLUSTER=/path/to/cluster.json \
SUNABAKO_IMAGES=/path/to/images.json \
ADAPTER_PUBLIC_HOST=<training-node-ip> \
python tests/test_agent_sunabako_codex_e2e.py
```

镜像映射必须在每台沙箱节点包含两个任务，并提供至少 8 个并发沙箱的容量。只在无硬 cgroup 的功能测试中，显式设置
`SUNABAKO_ALLOW_TEST_MEMORY=1`。可用 `VIME_AGENT_TEST_DATA` 复用已下载的小米 parquet
和镜像映射，用 `VIME_AGENT_CODEX_NATIVE_TARBALL` 和 `VIME_AGENT_CC_NATIVE_TARBALL`
复用官方平台安装包，无需在沙箱中下载工具链。
`VIME_AGENT_TEST_RUN_DIR` 必须是新目录，会保留 CLI 日志、评分输出、rollout/train
张量、参数更新证据与 `result.json`。测试与其他 E2E 一样调用 `U.execute_train()`，
通过 `extra_env_vars` 传入 agent 环境。Ray CLI 直接向 Buildkite 控制台输出训练日志；
训练结束后将该任务日志保存为 `train.log`，用于检查 MTP 和训练指标。
Buildkite 在成功或失败后都会上传这些产物。

### Megatron 手动重启

`test_qwen2.5_0.5B_training_recovery.py` 使用 4 张 GPU，在同一个 Ray 集群中先后运行两次训练任务。第一次使用 TP=1，故意触发真实的 CUDA OOM；确认训练任务退出后推理服务仍能响应，再改成 TP=2 重新提交，此时 DP 大小也会改变。

测试检查是否复用了健康的 vLLM 进程、路由器和 GPU 资源，重放的批次内容是否一致，以及训练调度器进度、非零且有限的梯度和最终 checkpoint。固定测试列表包括：

| 数据保存方式 | RolloutManager 状态 | 检查内容 |
|---|---|---|
| straw，开启在线 GC | 保持存活 | 重新连接训练进程，重放已经训练但尚未保存到 checkpoint 的批次。 |
| straw，开启在线 GC | 失败后被杀掉 | 新 manager 接回原推理集群，并重放同样的批次。 |
| straw，已有模型和优化器 checkpoint，使用 Megatron YAML 配置 | 训练过程中被杀掉 | 从 checkpoint 恢复，核对配置与恢复状态。 |
| Rollout 调试文件 | 失败后被杀掉 | 从调试文件恢复数据，新 manager 接回原推理集群。 |
| straw，使用 disk-delta 同步权重 | 失败后被杀掉 | 以恢复后的权重发布新的完整基准，再继续 delta 更新。 |
| straw，使用 PD/NIXL 推理 | 失败后被杀掉 | 卡住 prefill actor，在连接重置超时后替换它，并保留健康的 decode actor。 |

`test_qwen3_30B_A3B_training_recovery.py` 使用 8 张 GPU，在 MoE 模型、R3 和 stateless Adam 配置下覆盖 OOM、checkpoint 恢复及 manager 丢失。它不保存优化器张量，但会检查 scheduler 进度，比对 TP/DP 改变前后的持久化路由字节，并在恢复后完成训练。Dense 测试覆盖普通 Adam 的优化器 checkpoint 恢复。

无论是否传入兼容参数 `--use-fault-tolerance`，内部推理健康检查都会启用。CPU 测试还包括：`test_training_recovery.py` 的配置与 checkpoint 边界检查，`test_disk_delta_recovery.py` 的权重更新应答丢失，以及 `test_rollout_manager_recovery.py` 中真实 Ray manager 的 SIGKILL 和转换应答丢失。

### Rollout 收尾时清理失效引擎

`test_qwen2.5_0.5B_rollout_health.py` 在 rollout 收尾前停掉真实的 vLLM HTTP 服务，但保留它在路由器中的注册信息。两个四卡场景分别保留或杀掉对应的 Ray actor，检查收尾是否有超时限制、返回训练前是否注销失效服务、更新权重时能否恢复引擎，以及训练能否保存最终 checkpoint。

两种情况都不传 `--use-fault-tolerance`，并将后台检查间隔和首次等待时间设为 600 秒，以验证收尾检查会立即执行，不必等待后台定时检查。

## 注册测试

- 始终运行的 CPU 测试加入 `.buildkite/pipeline.yml` 中对应的命令。
- GPU 测试加入 `.buildkite/gpu_suites.py` 中对应的套件，并同步更新
  `.buildkite/pipeline.yml` 显示的测试数量。
- `.buildkite/README.md` 必须与 pipeline 行为保持一致。

触发远程 Buildkite job 前，应先在本地运行完全相同的命令。GPU 测试失败
时，先使用相同镜像和环境在 H200 节点复现并修复；本地通过后再重跑远程
套件。
