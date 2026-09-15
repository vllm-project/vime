# Search-R1 lite

本示例 **(Search-R1 lite)** 演示了如何使用 vime 进行带工具调用的语言模型生成，具备搜索/检索能力，是基于 [Search-R1](https://github.com/PeterGriffinJin/Search-R1) 的最小复现。

## 概述

Search-R1 示例提供了：

- **多轮对话**：支持工具调用（搜索/回答）
- **双搜索后端支持**：本地稠密检索器（FAISS + E5）或 Google 搜索（serper.dev）
- **基于 GRPO 的强化学习训练**：使用精确匹配（EM）计算问答任务奖励
- **TIS（轨迹重要性采样）**：用于处理训推不一致问题
- **格式感知奖励**：同时评估答案正确性和输出结构

## 文件说明

| 文件                                          | 描述 |
|---------------------------------------------|------|
| `generate_with_search.py`                   | 主生成函数，支持多轮搜索 + 回答工具调用，以及奖励函数 |
| `google_search_server.py`                   | 通过 serper.dev API 实现 Google 搜索后端 |
| `local_search_server.py`                    | 本地搜索后端，封装检索服务 |
| `qa_em_format.py`                           | 问答精确匹配评分，包含格式验证和检索正确性检查 |
| `run_qwen3_4b_npu.sh`                       | 训练启动脚本（NPU 8卡，Qwen3-4B-Instruct-2507，GRPO） |
| `local_dense_retriever/retrieval_server.py` | 稠密检索服务器（FAISS + E5 模型） |
| `local_dense_retriever/download.py`         | 从 HuggingFace 下载 wiki-18 索引和语料库 |

---

## 使用方法

### 1. 环境搭建

```bash
git clone -b ascend https://github.com/vllm-project/vime.git
cd vime
docker build -f docker/Dockerfile.npu -t vime-ascend:latest .
```

```bash
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

```bash
# 检索依赖
pip install faiss-cpu==1.13.2
```

### 2. 下载

#### 2.1 数据

**方式 A：在线自动下载**

```bash
cd /root
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1
pip install -e . --no-deps
pip install tensordict
pip install chardet

# 设置工作目录
WORK_DIR=/root/Search-R1
LOCAL_DIR=/path/to/nq_hotpotqa_train

# 处理多数据集搜索格式训练文件
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# （可选）处理多数据集搜索格式测试文件
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

**方式 B：离线手动下载**

- 从 Hugging Face 仓库下载完整数据集资源：[PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)
- 将所有下载的文件上传到目标目录 `LOCAL_DIR=/path/to/nq_hotpotqa_train`

#### 2.2 模型

**方式 A：在线自动下载**

```bash
hf download Qwen/Qwen3-4B-Instruct-2507 --local-dir /path/to/Qwen3-4B-Instruct-2507
```

**方式 B：离线手动下载**

- 从 Hugging Face 仓库下载完整模型权重：[Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- 将所有下载的模型文件上传到目标目录 `MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507`

#### 2.3 本地检索服务器（可选）

> 仅在使用本地搜索后端而非 Google 搜索时需要。

**(1) 数据**

**方式 A：在线自动下载**

```bash
# 下载索引和语料库（约 60-70 GB 下载量）
SAVE_PATH=/path/to/Index
python /root/vime/examples/search-r1/local_dense_retriever/download.py --save_path $SAVE_PATH
```

**方式 B：离线手动下载**

- 从 Hugging Face 仓库下载完整索引资源：[PeterJinGo/wiki-18-e5-index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index) 和 [PeterJinGo/wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
- 将所有下载的文件上传到目标目录 `SAVE_PATH=/path/to/Index`

> **注意：**无论采用上述哪种下载方式（方案 A 或方案 B），在数据下载完成后，均须执行以下命令以完成索引分片的合并以及语料库的解压。

```bash
SAVE_PATH=/path/to/Index
cat $SAVE_PATH/part_* > $SAVE_PATH/e5_Flat.index
gzip -d $SAVE_PATH/wiki-18.jsonl.gz
```

**(2) 模型**

**方式 A：在线自动下载**

```bash
hf download intfloat/e5-base-v2 --local-dir /path/to/e5-base-v2
```

**方式 B：离线手动下载**

- 从 Hugging Face 仓库下载完整模型权重：[intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- 将所有下载的模型文件上传到目标目录 `MODEL_DIR=/path/to/e5-base-v2`

### 3. 配置搜索后端

`generate_with_search.py` 文件同时支持**本地**搜索和 **Google** 搜索后端。通过 `SEARCH_R1_CONFIGS` 字典进行配置：

```python
SEARCH_R1_CONFIGS = {
    # ============== 通用配置 ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 8,

    # ============== 搜索后端选择 ==============
    "search_backend": "local",  # 选项："local" 或 "google"

    # ============== 本地搜索配置 ==============
    # （仅在 search_backend="local" 时使用）
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # 本地检索服务器 URL
        "proxy": None,
    },

    # ============== Google 搜索配置 ==============
    # （仅在 search_backend="google" 时使用）
    "google": {
        "api_key": "your_api_key_here",  # 替换为你的 serper.dev API 密钥
        "snippet_only": True,
        "proxy": None,
    },

    # ============== 对数概率收集 ==============
    "return_logprob": True,  # 设为 True 以收集对数概率（TIS 所需）

    # ============== 奖励模型配置 ==============
    "format_score": 0.2,
}
```

#### 使用本地搜索

- 设置 `"search_backend": "local"`
- 在 `"local"` 部分配置本地检索服务器 URL
- 在运行训练脚本之前启动本地搜索服务器

```bash
# 设置路径
SAVE_PATH=/path/to/Index
INDEX_FILE=$SAVE_PATH/e5_Flat.index
CORPUS_FILE=$SAVE_PATH/wiki-18.jsonl
RETRIEVER_NAME=e5
RETRIEVER_PATH=/path/to/e5-base-v2

# 启动检索服务器
python /root/vime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $INDEX_FILE\
    --corpus_path $CORPUS_FILE\
    --topk 3 \
    --retriever_name $RETRIEVER_NAME \
    --retriever_model $RETRIEVER_PATH
```

> **注意：**
> - 首次启动将下载模型并加载索引，可能需要几分钟
> - 正常启动时间（不含下载）：1-2 分钟
> - 本地搜索引擎的 Python 进程不会在 Shell 关闭时终止
> - 重启服务器：使用 `lsof -i :8000` 查找进程 PID，然后终止并重启

#### 使用 Google 搜索

- 设置 `"search_backend": "google"`
- 在 `"google"` 部分配置你的 serper.dev API 密钥
- 从 [serper.dev](https://serper.dev/) 获取你的 API 密钥

### 4. 运行训练

```bash
cd /root/vime
# 替换模型和数据加载/保存路径
bash examples/search-r1/run_qwen3_4b_npu.sh
```
