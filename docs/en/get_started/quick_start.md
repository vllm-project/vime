# Environment setup

Start in the {ref}`interactive Quick Start <lab>`. This page prepares the environment used by every generated recipe; the [tutorial](experiment-guide.md) explains the full RL workflow.

## Basic Environment Setup

Use the provided Docker images to get a compatible vLLM, Megatron, and patch stack.

### Hardware and Image Selection

| Hardware | Image | CUDA |
| --- | --- | --- |
| H100/H200 | `vllm/vime:latest` | 13.0 |
| B200/B300 (Blackwell, x86) | `vllm/vime:latest` | 13.0 |

New builds use the same CUDA 13 image for Hopper and Blackwell. On supported data-center GPUs, the image automatically enables CUDA 13 forward-compatibility libraries for older drivers, including 535 and 570, without replacing the host driver. See the [Docker guide](https://github.com/vllm-project/vime/blob/main/docker/README.md) for build options and driver compatibility.

GPU CI primarily covers H100/H200. Check each recipe's hardware and parallelism requirements before running it on other platforms.

For AMD, see the [AMD tutorial](../platform_support/amd_tutorial.md).

### Pull and Start Docker Container

On a Linux x86_64 GPU host, prepare Git, Docker, a compatible NVIDIA driver and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html). Run these commands on the **host** to pull the image, clone vime and start a container:

```shell
VIME_IMAGE=vllm/vime:latest
docker pull "$VIME_IMAGE"
git clone https://github.com/vllm-project/vime.git
cd vime
mkdir -p ../vime-data
docker run --rm -it --gpus all --ipc=host \
  --shm-size=16g --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$PWD:/root/vime" -v "$PWD/../vime-data:/data" \
  -w /root/vime "$VIME_IMAGE" /bin/bash
```

### Install vime

The image already contains vime and its dependencies. The bind mount replaces the image's checkout with your host checkout. Run the following **inside the container** to install that checkout without replacing the image's pinned dependencies, then check imports and GPU visibility:

```bash
pip install -e . --no-deps
python -c "import vime, vllm, megatron.core; import torch; print('GPUs:', torch.cuda.device_count())"
```

The GPU count should match the devices exposed to the container. Save the builder's `experiment.sh` in the host `vime` directory; it is immediately visible at `/root/vime/experiment.sh` inside the container. Models, data and checkpoints written under `/data` persist in the host's sibling `vime-data` directory after the container exits. To reopen the environment, repeat the `docker run` command from the same checkout and the editable install inside the new container.

## Mount the experiment paths

The Docker commands above mount a persistent data directory at `/data`, matching the builder's defaults. To use an existing dataset or shared storage, replace that mount with, for example, `-v /shared/data:/data`, and use the container-visible paths in the lab. Keep the repository and generated configuration directory visible at the same path on all Ray hosts. Install the same dependencies on every node and configure networking so the Ray nodes can reach each other; the single-host Docker command above does not set up a multi-node network.

Return to the [tutorial preparation and launch steps](experiment-guide.md#5-prepare-once-convert-then-run) to download, convert, and train the selected model. Use [parameter reference](usage.md) for existing launchers.
