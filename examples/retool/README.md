# Retool: from SFT to RL
This example _**(Retool)**_ demonstrates how to use the retool functionality for tool-enabled language model generation.

## Overview
The retool example provides:

- Safe Python code execution in a sandbox environment
- Tool registry for managing available tools
- Integration with language model generation
- Reward calculation for tool usage

## Files
- `generate_with_retool.py`: Main generation function with tool support
- `tool_sandbox.py`: Tool execution and safety management
- `sft_data_processing.py`: Process SFT dataset

## Reward Design
The RL reward function (`generate_with_retool.reward_func`) uses a **tool-aware reward shaping** strategy on top of the math accuracy reward:

| Answer | Tool Used | Reward | Rationale |
|--------|-----------|--------|-----------|
| ✅ Correct | No | 1.0 | Pure reasoning — the ideal case |
| ✅ Correct | Yes | 1.0 + min(0.2, turns × 0.05) | Bonus for effective tool use (capped at 0.2) |
| ❌ Wrong | No | 0.0 | Neutral — model didn't attempt tools |
| ❌ Wrong | Yes | min(0.1, turns × 0.02) | Small positive to encourage exploration |

This design encourages the model to **explore tool calling** during early RL training without letting it reward-hack by preferring tool calls over correct answers. As training progresses and accuracy improves, the accuracy reward dominates and the tool bonus becomes a tiebreaker.

## Usage
### 1. Setup
```
git clone -b ascend https://github.com/vllm-project/vime.git
cd vime
docker build -f docker/Dockerfile.npu -t vime-ascend:latest .
```
```
# Update the vime image
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
### 2. Download
```
# For SFT part, you can use later model to RL directly and skip SFT.
hf download --repo-type dataset JoeYing/ReTool-SFT  --local-dir /path/to/ReTool-SFT
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /path/to/Qwen3-4B-Instruct-2507

# For RL part
hf download --repo-type dataset zhuzilin/dapo-math-17k --local-dir /path/to/dapo-math-17k
hf download --repo-type dataset zhuzilin/aime-2024  --local-dir /path/to/aime-2024
# download our SFT model if you want to skip SFT
hf download font-info/qwen3-4b-sft-SGLang-RL --local-dir /path/to/qwen3-4b-sft
```
### 3. Preprocessing
```
cd /root/vime
# Replace the save path with /path/to/ReTool-SFT.parquet
python examples/retool/sft_data_processing.py
```
### 4. SFT
```
cd /root/vime
# Replace the model and data loading/saving paths
python examples/retool/retool_qwen3_4b_sft.sh
```
### 5. RL
```
cd /root/vime
# Replace the model and data loading/saving paths
python examples/retool/retool_qwen3_4b_rl.sh
```
