# Environment setup

Start in the {ref}`interactive Quick Start <lab>`. This page prepares the environment used by every generated recipe; the [tutorial](experiment-guide.md) explains the full RL workflow.

## Basic Environment Setup

Use the provided Docker images to get a compatible vLLM, Megatron, and patch stack.

### Hardware and Image Selection

| Hardware | Image | CUDA |
| --- | --- | --- |
| H100/H200 | `vllm/vime:latest` | 13.0 |
| B200/B300 (Blackwell, x86) | `vllm/vime:latest` | 13.0 |

Vime uses a CUDA 13 image for both hardware groups. See the [Docker guide](https://github.com/vllm-project/vime/blob/main/docker/README.md) for source pins and build options.

GPU CI primarily covers H100/H200. Check each recipe's hardware and parallelism requirements before running it on other platforms.

For AMD, see the [AMD tutorial](../platform_support/amd_tutorial.md).

### Pull and Start Docker Container

Please execute the following commands to pull the latest image and start an interactive container:

```shell
# Pull the latest image
docker pull vllm/vime:latest

# Start the container
docker run --rm --gpus all --ipc=host --shm-size=16g \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -it vllm/vime:latest /bin/bash
```

### Install vime

vime is already installed in the docker image. To update to the latest version, please execute the following command:

```bash
# Path can be adjusted according to actual situation
cd /root/vime
git pull
pip install -e . --no-deps
```

## Mount the experiment paths

Mount a data directory when starting the container, for example `-v /shared/data:/data`, and use those container-visible paths in the lab. Keep the repository and generated configuration directory visible at the same path on all Ray hosts. Install the same dependencies on every node.

Return to the [tutorial preparation and launch steps](experiment-guide.md#5-prepare-once-convert-then-run) to download, convert, and train the selected model. Use [parameter reference](usage.md) for existing launchers.
