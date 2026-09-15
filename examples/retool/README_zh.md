# Retool：从 SFT 到 RL
本示例 **(Retool)** 演示了如何使用 retool 功能进行带工具调用的语言模型生成。

## 概述
Retool 示例提供了：

- 沙盒环境中的安全 Python 代码执行
- 工具注册表，用于管理可用工具
- 与语言模型生成的集成
- 工具使用的奖励计算

## 文件说明
- `generate_with_retool.py`：主生成函数，支持工具调用
- `tool_sandbox.py`：工具执行与安全管理
- `sft_data_processing.py`：处理 SFT 数据集

## 奖励设计
RL 奖励函数（`generate_with_retool.reward_func`）在数学正确性奖励的基础上采用了**工具感知的奖励塑形**策略：

| 答案 | 是否使用工具 | 奖励值 | 说明 |
|------|-------------|--------|------|
| ✅ 正确 | 否 | 1.0 | 纯推理——理想情况 |
| ✅ 正确 | 是 | 1.0 + min(0.2, 轮次 × 0.05) | 有效工具使用的额外奖励（上限 0.2） |
| ❌ 错误 | 否 | 0.0 | 中性——模型未尝试使用工具 |
| ❌ 错误 | 是 | min(0.1, 轮次 × 0.02) | 小额正奖励以鼓励探索 |

该设计鼓励模型在 RL 训练早期**探索工具调用**，同时避免模型通过偏好工具调用而非正确答案来进行奖励投机。随着训练推进和准确率提升，正确性奖励将占主导，工具奖励变为辅助项。

## 使用方法
### 1. 环境搭建
```bash
git clone -b ascend https://github.com/vllm-project/vime.git
cd vime
docker build -f docker/Dockerfile.npu -t vime-ascend:latest .
```

```bash
# 更新 vime 镜像
export IMAGE=vime-ascend:latest

docker run -d --name vime-npu -it --net=host --shm-size=1024g \
    --privileged=true \
    --cap-add=SYS_PTRACE \
    --device=/dev/davinci_manager \
    --device=/dev/hisi_hdc \
    --device=/dev/devmm_svm \
    -v /usr/local/Ascend/driver:/usr/local/Ascend/driver \
    -v /usr/local/dcmi:/usr/local/dcmi \
    -v /usr/local/bin/npu-smi:/usr/local/bin/npu-smi \
    -v /usr/local/sbin:/usr/local/sbin \
    -v /home:/home \
    -v /mnt:/mnt \
    -v /tmp:/tmp \
    -v /data:/data \
    -v /usr/share/zoneinfo/Asia/Shanghai:/etc/localtime \
    $IMAGE

docker exec -it vime-npu bash
```

### 2. 下载
```bash
# SFT 部分，你也可以直接使用后续模型进行 RL 而跳过 SFT。
hf download --repo-type dataset JoeYing/ReTool-SFT  --local-dir /path/to/ReTool-SFT
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /path/to/Qwen3-4B-Instruct-2507

# RL 部分
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /path/to/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /path/to/aime-2024
# 如果你想跳过 SFT，可下载我们的 SFT 模型
hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir /path/to/qwen3-4b-sft
```

### 3. 数据预处理
```bash
cd /root/vime
# 将保存路径替换为 /path/to/ReTool-SFT.parquet
python examples/retool/sft_data_processing.py
```

### 4. SFT
```bash
cd /root/vime
# 替换模型和数据加载/保存路径
python examples/retool/retool_qwen3_4b_sft.sh
```

### 5. RL
```bash
cd /root/vime
# 替换模型和数据加载/保存路径
python examples/retool/retool_qwen3_4b_rl.sh
```
