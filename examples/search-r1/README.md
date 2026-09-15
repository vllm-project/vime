# Search-R1 lite

This example **(Search-R1 lite)** demonstrates how to use vime for tool-enabled language model generation with search/retrieval capabilities, based on a minimal reproduction of [Search-R1](https://github.com/PeterGriffinJin/Search-R1).

## Overview

The Search-R1 example provides:

- **Multi-turn conversation** with tool-calling (search/answer actions)
- **Dual search backend support**: local dense retriever (FAISS + E5) or Google Search (serper.dev)
- **GRPO-based RL training** with exact-match (EM) reward for QA tasks
- **TIS (Trajectory Importance Sampling)** support for handling train/inference mismatch
- **Format-aware reward** that evaluates both answer correctness and output structure

## Files

| File                                        | Description                                                                        |
|---------------------------------------------|------------------------------------------------------------------------------------|
| `generate_with_search.py`                   | Main generation function with multi-turn search + answer tool-calling, and reward function |
| `google_search_server.py`                   | Google Search backend via serper.dev API                                           |
| `local_search_server.py`                    | Local search backend that wraps the retrieval server                               |
| `qa_em_format.py`                           | QA exact-match scoring with format validation and retrieval correctness check      |
| `run_qwen3_4b_npu.sh`                       | Training launch script (NPU 8-card, Qwen3-4B-Instruct-2507, GRPO)                                        |
| `local_dense_retriever/retrieval_server.py` | Dense retriever server (FAISS + E5 model)                                          |
| `local_dense_retriever/download.py`         | Download wiki-18 index and corpus from HuggingFace                                 |

---

## Usage

### 1. Setup

```bash
git clone -b ascend https://github.com/vllm-project/vime.git
cd vime
docker build -f docker/Dockerfile.npu -t vime-ascend:latest .
```

```bash
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

```bash
# For retriever
pip install faiss-cpu==1.13.2
```

### 2. Download

#### 2.1 Data

**Option A: Online Auto Download**

```bash
# Training data (NQ + HotpotQA)
cd /root
git clone https://github.com/PeterGriffinJin/Search-R1.git
cd Search-R1
pip install -e . --no-deps
pip install tensordict
pip install chardet

# Set your working directory
WORK_DIR=/root/Search-R1
LOCAL_DIR=/path/to/nq_hotpotqa_train

# Process multiple dataset search format train file
DATA=nq,hotpotqa
python $WORK_DIR/scripts/data_process/qa_search_train_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA

# (Optional) Process multiple dataset search format test file
DATA=nq,triviaqa,popqa,hotpotqa,2wikimultihopqa,musique,bamboogle
python $WORK_DIR/scripts/data_process/qa_search_test_merge.py \
    --local_dir $LOCAL_DIR \
    --data_sources $DATA
```

**Option B: Offline Manual Download**

- Download full dataset assets from Hugging Face repo: [PeterJinGo/nq_hotpotqa_train](https://huggingface.co/datasets/PeterJinGo/nq_hotpotqa_train)
- Upload all downloaded files to your target directory `LOCAL_DIR=/path/to/nq_hotpotqa_train`

#### 2.2 Model

**Option A: Online Auto Download**

```bash
hf download Qwen/Qwen3-4B-Instruct- --local-dir /path/to/Qwen3-4B-Instruct-2507
```

**Option B: Offline Manual Download**

- Download full model weights from Hugging Face repo: [Qwen/Qwen3-4B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507)
- Upload all downloaded model files to your target directory `MODEL_DIR=/path/to/Qwen3-4B-Instruct-2507`

#### 2.3 Local Retrieval Server (Optional)

> Only needed if using the local search backend instead of Google Search.

**(1) Data**

**Option A: Online Auto Download**

```bash
# Download index and corpus (~60-70 GB download)
SAVE_PATH=/path/to/Index
python /root/vime/examples/search-r1/local_dense_retriever/download.py --save_path $SAVE_PATH
```

**Option B: Offline Manual Download**

- Download full index assets from Hugging Face repo: [PeterJinGo/wiki-18-e5-index](https://huggingface.co/datasets/PeterJinGo/wiki-18-e5-index) and [PeterJinGo/wiki-18-corpus](https://huggingface.co/datasets/PeterJinGo/wiki-18-corpus)
- Upload all downloaded files to your target directory `SAVE_PATH=/path/to/Index`

> No matter which download option you use (Option A or B), you must run the following commands after the download completes to merge the index shards and unpack the corpus dataset.

```bash
SAVE_PATH=/path/to/Index
cat $SAVE_PATH/part_* > $SAVE_PATH/e5_Flat.index
gzip -d $SAVE_PATH/wiki-18.jsonl.gz
```

**(2) Model**

**Option A: Online Auto Download**

```bash
hf download intfloat/e5-base-v2 --local-dir /path/to/e5-base-v2
```

**Option B: Offline Manual Download**

- Download full model weights from Hugging Face repo: [intfloat/e5-base-v2](https://huggingface.co/intfloat/e5-base-v2)
- Upload all downloaded model files to your target directory `MODEL_DIR=/path/to/e5-base-v2`

### 3. Configure Search Backend

The `generate_with_search.py` file supports both **local** search and **Google** search backends. Configure via the `SEARCH_R1_CONFIGS` dictionary:

```python
SEARCH_R1_CONFIGS = {
    # ============== General Configuration ==============
    "max_turns": 2,
    "topk": 3,
    "search_concurrency": 8,

    # ============== Search Backend Selection ==============
    "search_backend": "local",  # Options: "local" or "google"

    # ============== Local Search Configuration ==============
    # (Only used when search_backend="local")
    "local": {
        "search_url": "http://127.0.0.1:8000/retrieve",  # URL of your local retrieval server
        "proxy": None,
    },

    # ============== Google Search Configuration ==============
    # (Only used when search_backend="google")
    "google": {
        "api_key": "your_api_key_here",  # Replace with your actual serper.dev API key
        "snippet_only": True,
        "proxy": None,
    },

    # ============== Log Probability Collection ==============
    "return_logprob": True,  # Set to True to collect log probabilities (required for TIS)

    # ============== Reward Model Configuration ==============
    "format_score": 0.2,
}
```

#### Using Local Search

- Set `"search_backend": "local"`
- Configure `"local"` section with your local retrieval server URL
- Start your local search server before running the training script

```bash
# Set paths
SAVE_PATH=/path/to/Index
INDEX_FILE=$SAVE_PATH/e5_Flat.index
CORPUS_FILE=$SAVE_PATH/wiki-18.jsonl
RETRIEVER_NAME=e5
RETRIEVER_PATH=/path/to/e5-base-v2

# Start the retrieval server
python /root/vime/examples/search-r1/local_dense_retriever/retrieval_server.py \
    --index_path $INDEX_FILE\
    --corpus_path $CORPUS_FILE\
    --topk 3 \
    --retriever_name $RETRIEVER_NAME \
    --retriever_model $RETRIEVER_PATH
```

> **Note:**
> - First startup will download the model and load the index, which may take a few minutes
> - Normal startup time (excluding downloads): 1-2 minutes
> - The local search engine's Python process will not terminate when the shell closes
> - To restart the server: `lsof -i :8000` to find the PID, then kill it and restart

#### Using Google Search

- Set `"search_backend": "google"`
- Configure `"google"` section with your serper.dev API key
- Get your API key from [serper.dev](https://serper.dev/)

### 4. Run Training

```bash
cd /root/vime
# Replace the model and data loading/saving paths
bash examples/search-r1/run_qwen3_4b_npu.sh
```
