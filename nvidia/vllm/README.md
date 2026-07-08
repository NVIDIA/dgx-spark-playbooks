# vLLM for Inference

> Install and use vLLM on DGX Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Run on two Sparks](#run-on-two-sparks)
- [Run on multiple Sparks through a switch](#run-on-multiple-sparks-through-a-switch)
- [Run Agent Ready Qwen3.6 35B Model with vLLM](#run-agent-ready-qwen36-35b-model-with-vllm)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

vLLM is an inference engine designed to run large language models efficiently. The key idea is **maximizing throughput and minimizing memory waste** when serving LLMs.

- It uses a memory-efficient attention algoritm called **PagedAttention** to handle long sequences without running out of GPU memory.
- New requests can be added to a batch already in process through **continuous batching** to keep GPUs fully utilized.
- It has an **OpenAI-compatible API** so applications built for the OpenAI API can switch to a vLLM backend with little or no modification.

## What you'll accomplish

You'll set up vLLM high-throughput LLM serving on DGX Spark with Blackwell architecture,
either using a pre-built Docker container or building from source with custom LLVM/Triton
support for ARM64.

## What to know before starting

- Experience building and configuring containers with Docker
- Familiarity with CUDA toolkit installation and version management
- Understanding of Python virtual environments and package management
- Knowledge of building software from source using CMake and Ninja
- Experience with Git version control and patch management

## Prerequisites

- DGX Spark device with ARM64 processor and Blackwell GPU architecture
- CUDA 13.0 toolkit installed: `nvcc --version` shows CUDA toolkit version.
- Docker installed and configured: `docker --version` succeeds
- NVIDIA Container Toolkit installed
- Python 3.12 available: `python3.12 --version` succeeds
- Git installed: `git --version` succeeds
- Network access to download packages and container images


## Model Support Matrix

The following models are supported with vLLM on Spark. All listed models are available and ready to use:

| Model | Quantization | Support Status | HF Handle |
|-------|-------------|----------------|-----------|
| **DiffusionGemma 26B A4B IT** | BF16 | ✅ | [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |
| **DiffusionGemma 26B A4B IT** | NVFP4 | ✅ | [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | BF16 | ✅ | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | FP8 | ✅ | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8) |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | NVFP4 | ✅ | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4) |
| **Gemma 4 31B IT** | Base | ✅ | [`google/gemma-4-31B-it`](https://huggingface.co/google/gemma-4-31B-it) |
| **Gemma 4 31B IT** | NVFP4 | ✅ | [`nvidia/Gemma-4-31B-IT-NVFP4`](https://huggingface.co/nvidia/Gemma-4-31B-IT-NVFP4) |
| **Gemma 4 26B A4B IT** | Base | ✅ | [`google/gemma-4-26B-A4B-it`](https://huggingface.co/google/gemma-4-26B-A4B-it) |
| **Gemma 4 E4B IT** | Base | ✅ | [`google/gemma-4-E4B-it`](https://huggingface.co/google/gemma-4-E4B-it) |
| **Gemma 4 E2B IT** | Base | ✅ | [`google/gemma-4-E2B-it`](https://huggingface.co/google/gemma-4-E2B-it) |
| **Nemotron-3-Super-120B** | NVFP4 | ✅ | [`nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4) |
| **GPT-OSS-20B** | MXFP4 | ✅ | [`openai/gpt-oss-20b`](https://huggingface.co/openai/gpt-oss-20b) |
| **GPT-OSS-120B** | MXFP4 | ✅ | [`openai/gpt-oss-120b`](https://huggingface.co/openai/gpt-oss-120b) |
| **Llama-3.1-8B-Instruct** | FP8 | ✅ | [`nvidia/Llama-3.1-8B-Instruct-FP8`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-FP8) |
| **Llama-3.1-8B-Instruct** | NVFP4 | ✅ | [`nvidia/Llama-3.1-8B-Instruct-NVFP4`](https://huggingface.co/nvidia/Llama-3.1-8B-Instruct-NVFP4) |
| **Llama-3.3-70B-Instruct** | NVFP4 | ✅ | [`nvidia/Llama-3.3-70B-Instruct-NVFP4`](https://huggingface.co/nvidia/Llama-3.3-70B-Instruct-NVFP4) |
| **Qwen3-8B** | FP8 | ✅ | [`nvidia/Qwen3-8B-FP8`](https://huggingface.co/nvidia/Qwen3-8B-FP8) |
| **Qwen3-8B** | NVFP4 | ✅ | [`nvidia/Qwen3-8B-NVFP4`](https://huggingface.co/nvidia/Qwen3-8B-NVFP4) |
| **Qwen3-14B** | FP8 | ✅ | [`nvidia/Qwen3-14B-FP8`](https://huggingface.co/nvidia/Qwen3-14B-FP8) |
| **Qwen3-14B** | NVFP4 | ✅ | [`nvidia/Qwen3-14B-NVFP4`](https://huggingface.co/nvidia/Qwen3-14B-NVFP4) |
| **Qwen3-32B** | NVFP4 | ✅ | [`nvidia/Qwen3-32B-NVFP4`](https://huggingface.co/nvidia/Qwen3-32B-NVFP4) |
| **Qwen2.5-VL-7B-Instruct** | NVFP4 | ✅ | [`nvidia/Qwen2.5-VL-7B-Instruct-NVFP4`](https://huggingface.co/nvidia/Qwen2.5-VL-7B-Instruct-NVFP4) |
| **Qwen3-VL-Reranker-2B** | Base | ✅ | [`Qwen/Qwen3-VL-Reranker-2B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-2B) |
| **Qwen3-VL-Reranker-8B** | Base | ✅ | [`Qwen/Qwen3-VL-Reranker-8B`](https://huggingface.co/Qwen/Qwen3-VL-Reranker-8B) |
| **Qwen3-VL-Embedding-2B** | Base | ✅ | [`Qwen/Qwen3-VL-Embedding-2B`](https://huggingface.co/Qwen/Qwen3-VL-Embedding-2B) |
| **Phi-4-multimodal-instruct** | FP8 | ✅ | [`nvidia/Phi-4-multimodal-instruct-FP8`](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-FP8) |
| **Phi-4-multimodal-instruct** | NVFP4 | ✅ | [`nvidia/Phi-4-multimodal-instruct-NVFP4`](https://huggingface.co/nvidia/Phi-4-multimodal-instruct-NVFP4) |
| **Phi-4-reasoning-plus** | FP8 | ✅ | [`nvidia/Phi-4-reasoning-plus-FP8`](https://huggingface.co/nvidia/Phi-4-reasoning-plus-FP8) |
| **Phi-4-reasoning-plus** | NVFP4 | ✅ | [`nvidia/Phi-4-reasoning-plus-NVFP4`](https://huggingface.co/nvidia/Phi-4-reasoning-plus-NVFP4) |
| **Nemotron3-Nano** | BF16 | ✅ | [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16) |
| **Nemotron3-Nano** | FP8 | ✅ | [`nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8`](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) |

> [!NOTE]
> The Phi-4-multimodal-instruct models require `--trust-remote-code` when launching vLLM.

> [!NOTE]
> You can use the NVFP4 Quantization documentation to generate your own NVFP4-quantized checkpoints for your favorite models. This enables you to take advantage of the performance and memory benefits of NVFP4 quantization even for models not already published by NVIDIA.

Reminder: not all model architectures are supported for NVFP4 quantization.

## Time & risk

* **Duration:** 30 minutes for Docker approach
* **Risks:** Container registry access requires internal credentials
* **Rollback:** Container approach is non-destructive.
* **Last Updated:** 06/12/2026
  * Add Agent ready model recipe for Qwen3.6 35B

## Instructions

## Step 1. Use model specific deployment guide

Certain models require special deployment configurations. Please refer to their respective model cards to run on DGX Spark:

| Model | Quantization | HF Model Card Link |
|-------|-------------|----------------|
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | BF16 | https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16 |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | FP8 | https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8 |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | NVFP4 | https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4 |

## Step 2. Configure Docker permissions

To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.

Open a new terminal and test Docker access. In the terminal, run:
```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with sudo .

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 3. Pull vLLM container image

Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

```bash
## HuggingFace token (required)
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"

export LATEST_VLLM_VERSION=<latest_container_version>
## example
## export LATEST_VLLM_VERSION=26.05.post1-py3

export HF_MODEL_HANDLE=<HF_HANDLE>
## example
## export HF_MODEL_HANDLE=openai/gpt-oss-20b

docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
```

For DiffusionGemma models, use vLLM custom container:
```bash
docker pull vllm/vllm-openai:gemma
```

For Gemma 4 model family, use vLLM custom container:
```bash
docker pull vllm/vllm-openai:gemma4-cu130
```

## Step 4. Test vLLM in container

Launch the container and start vLLM server with a test model to verify basic functionality.

```bash
docker run -it --gpus all -p 8000:8000 \
nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
vllm serve ${HF_MODEL_HANDLE}
```

To run the server in detached (non-interactive) mode:

```bash
docker run -d --gpus all -p 8000:8000 --name vllm-server \
  nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
  vllm serve ${HF_MODEL_HANDLE}
```

Wait for the server to become ready (model loading may take several minutes):

```bash
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done' || { echo "Server failed to start within 900s"; docker logs vllm-server | tail -50; exit 1; }
```

To run DiffusionGemma models (e.g. `google/diffusiongemma-26B-A4B-it`):
```bash
docker run -it \
  -p 8000:8000 \
  --gpus all \
  --shm-size=16g \
  -e HF_TOKEN="$HF_TOKEN" \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  vllm/vllm-openai:gemma ${HF_MODEL_HANDLE} \
  --gpu-memory-utilization 0.8 \
  --max-model-len 262144 \
  --attention-backend TRITON_ATTN \
  --max-num-seqs 10 \
  --diffusion-config '{"canvas_length":256}' \
  --override-generation-config '{"max_new_tokens": null}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4 \
  --reasoning-parser gemma4 \
  --enable-prefix-caching \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --load-format fastsafetensors

## For BF16 checkpoint add "--moe-backend triton" for better performance
```

To run models from Gemma 4 model family, (e.g. `google/gemma-4-31B-it`):
```bash
docker run -it --gpus all -p 8000:8000 \
vllm/vllm-openai:gemma4-cu130 ${HF_MODEL_HANDLE}
```

Expected output should include:
- Model loading confirmation
- Server startup on port 8000
- GPU memory allocation details

You can verify the server is ready by checking the health endpoint:

```bash
curl http://localhost:8000/health
```

In another terminal, test the server:

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "'"${HF_MODEL_HANDLE}"'",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
}'
```

Expected response should contain `"content": "204"` or similar mathematical calculation.

## Step 5. Cleanup and rollback

For container approach (non-destructive):

NGC Container:
```bash
docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm
```

Upstream Container:
```bash
docker stop "<container name>"
docker rm "<container name>"
docker rmi "<container image name>"
```

## Step 6. Next steps

- **Production deployment:** Configure vLLM with your specific model requirements
- **Performance tuning:** Adjust batch sizes and memory settings for your workload
- **Monitoring:** Set up logging and metrics collection for production use
- **Model management:** Explore additional model formats and quantization options

## Run on two Sparks

## Step 1. Configure network connectivity

Follow the network setup instructions from the [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

> **Heads up:** the `discover-sparks` script in the linked playbook writes its SSH key to `~/.ssh/` and fails if the directory does not exist yet. Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both nodes first if you have never used SSH on them.

## Step 2. Download cluster deployment script

Obtain the vLLM cluster deployment script on both nodes. This script orchestrates the Ray cluster setup required for distributed inference.

```bash
## Download on both nodes — pinned to a known-good commit so upstream changes
## can't silently break this playbook against the 26.05-py3 image.
wget https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh

## Patch the script to pip-install ray inside the container before ray starts.
## The 26.05-py3 NGC image ships without ray (upstream made it an optional CUDA dep);
## the install takes ~10s on first container launch.
sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' run_cluster.sh

chmod +x run_cluster.sh
```

## Step 3. Pull the NVIDIA vLLM image from NGC

First, configure docker. If this is your first time using docker, run:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

After this, you should be able to run docker commands without `sudo`.

Pull the image **on both nodes**:

```bash
docker pull nvcr.io/nvidia/vllm:26.05-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
```

## Step 4. Start Ray head node

Launch the Ray cluster head node on Node 1. This node coordinates the distributed inference and serves the API endpoint.

```bash
## On Node 1, start head node. Run inside tmux/screen so an SSH drop doesn't
## tear down the cluster (run_cluster.sh has an EXIT trap that stops the container).

## Get the IP address of the high-speed interface
## Use the interface that shows "(Up)" from ibdev2netdev (enp1s0f0np0 or enp1s0f1np1)
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

Leave this terminal open — closing it stops the head node and tears down the cluster.

## Step 5. Start Ray worker node

Open a second terminal, SSH to Node 2 (`ssh user@<NODE_2_IP>`), and join the Ray cluster as a worker. Replace `<NODE_1_IP_ADDRESS>` below with the QSFP-side IP from Node 1 (run `echo $VLLM_HOST_IP` on Node 1 to print it). Run inside tmux/screen on Node 2 as well.

```bash
## On Node 2, join as worker

## Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1

## Get Node 2's own IP address
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

## Set this to Node 1's QSFP IP (see step header)
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>

## Set the image tag (same as Step 3)
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```

## Step 6. Verify cluster status

Confirm both nodes are recognized and available in the Ray cluster.

```bash
## On Node 1 (head node)
## Find the vLLM container name (it will be node-<random_number>)
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"

docker exec $VLLM_CONTAINER ray status
```

Expected output shows 2 nodes with available GPU resources.

## Step 7. Download Llama 3.3 70B model

Llama 3.3 70B is a gated model — first accept its license at <https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct> and create an HF access token with read permission. Then authenticate inside the container so the cache lands at `/root/.cache/huggingface` (mounted from `~/.cache/huggingface`).

```bash
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  hf auth login
  hf download meta-llama/Llama-3.3-70B-Instruct'
```

## Step 8. Launch inference server for Llama 3.3 70B

Start the vLLM inference server with tensor parallelism across both nodes.

```bash
## On Node 1, enter container and start server
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max-model-len 2048 \
    --gpu-memory-utilization 0.8 --distributed-executor-backend ray'
```

## Step 9. Test 70B model inference

Verify the deployment with a sample inference request. Run this on Node 1 itself; from an external client, replace `localhost` with Node 1's reachable IP.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

Expected output includes a generated haiku response.

## Step 10. (Optional) Deploy Llama 3.1 405B model

> [!WARNING]
> 405B model has insufficient memory headroom for production use.

Download the quantized 405B model for testing purposes only. Runs inside the head container so the cache lands in the mounted HF directory.

```bash
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  hf download hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4'
```

## Step 11. (Optional) Launch 405B inference server

Start the server with memory-constrained parameters for the large model.

```bash
## On Node 1, launch with restricted parameters
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
    --tensor-parallel-size 2 --max-model-len 64 --gpu-memory-utilization 0.9 \
    --max-num-seqs 1 --max-num-batched-tokens 64 \
    --distributed-executor-backend ray'
```

Startup is slow for 405B — expect several minutes of model-loading logs across both nodes. The server is ready to take traffic once you see:

```
INFO:     Application startup complete.
INFO:     Uvicorn running on http://0.0.0.0:8000 (Press CTRL+C to quit)
```

## Step 12. (Optional) Test 405B model inference

Verify the 405B deployment with constrained parameters. As in Step 9, run on Node 1 or replace `localhost` with Node 1's reachable IP from an external client.

```bash
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

## Step 13. Validate deployment

Perform comprehensive validation of the distributed inference system.

```bash
## Check Ray cluster health
docker exec $VLLM_CONTAINER ray status

## Verify server health endpoint
curl http://localhost:8000/health

## Monitor GPU utilization on both nodes (DGX Spark has unified memory,
## so the --query-gpu memory fields report N/A; use raw nvidia-smi instead).
nvidia-smi
```

## Step 14. Next steps

The Ray dashboard runs on port 8265 of the head node. It binds to the container's network (host networking), so it is only directly reachable from Node 1 itself. From an external workstation, tunnel it over SSH:

```bash
## From your workstation:
ssh -L 8265:localhost:8265 nvidia@<NODE_1_IP>
## then open http://localhost:8265 in a local browser
```

Consider for production:
- Health checks and automatic restarts
- Log rotation for long-running services
- Persistent model caching across restarts
- Alternative quantization methods (FP8, INT4)

## Run on multiple Sparks through a switch

## Step 1. Configure network connectivity

Follow the network setup instructions from the [Multi Sparks through switch](https://build.nvidia.com/spark/multi-sparks-through-switch) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connections between Sparks and Switch
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification
- NCCL Bandwidth test

## Step 2. Download cluster deployment script

Download the vLLM cluster deployment script on all nodes. This script orchestrates the Ray cluster setup required for distributed inference.

```bash
## Download on all nodes
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/ray_serving/run_cluster.sh
chmod +x run_cluster.sh
```

## Step 3. Pull the NVIDIA vLLM Image from NGC

Do this step on all nodes.

First, you will need to configure docker to pull from NGC
If this is your first time using docker run:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

After this, you should be able to run docker commands without using `sudo`.

Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

```bash
docker pull nvcr.io/nvidia/vllm:26.02-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
```

## Step 4. Start Ray head node

Launch the Ray cluster head node on Node 1. This node coordinates the distributed inference and serves the API endpoint.

```bash
## On Node 1, start head node

## Get the IP address of the high-speed interface
## Use the interface that shows "(Up)" from ibdev2netdev (enp1s0f0np0 or enp1s0f1np1)
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

echo "Using interface $MN_IF_NAME with IP $VLLM_HOST_IP"

bash run_cluster.sh $VLLM_IMAGE $VLLM_HOST_IP --head ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$VLLM_HOST_IP
```

## Step 5. Start Ray worker nodes

Connect rest of the nodes to the Ray cluster as a worker nodes. This provides additional GPU resources for tensor parallelism.

```bash
## On other Nodes, join as workers

## Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1

## Get Node's own IP address
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')

## IMPORTANT: Set HEAD_NODE_IP to Node 1's IP address
## You must get this value from Node 1 (run: echo $VLLM_HOST_IP on Node 1)
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>

echo "Worker IP: $VLLM_HOST_IP, connecting to head node at: $HEAD_NODE_IP"

bash run_cluster.sh $VLLM_IMAGE $HEAD_NODE_IP --worker ~/.cache/huggingface \
  -e VLLM_HOST_IP=$VLLM_HOST_IP \
  -e UCX_NET_DEVICES=$MN_IF_NAME \
  -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
  -e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
  -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
  -e TP_SOCKET_IFNAME=$MN_IF_NAME \
  -e RAY_memory_monitor_refresh_ms=0 \
  -e MASTER_ADDR=$HEAD_NODE_IP
```
> **Note:** Replace `<NODE_1_IP_ADDRESS>` with the actual IP address from Node 1, specifically the QSFP interface enp1s0f1np1 configured in the [Multi Sparks through switch](https://build.nvidia.com/spark/multi-sparks-through-switch) playbook.

## Step 6. Verify cluster status

Confirm all nodes are recognized and available in the Ray cluster.

```bash
## On Node 1 (head node)
## Find the vLLM container name (it will be node-<random_number>)
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"

docker exec $VLLM_CONTAINER ray status
```

Expected output shows all nodes with available GPU resources.

## Step 7. Download MiniMax M2.5 model

If you are running with four or more sparks, you can comfortably run this model with tensor parallelism. Authenticate with Hugging Face and download the model.

```bash
## On all nodes, from within the docker containers created in previous steps, run the following
hf auth login
hf download MiniMaxAI/MiniMax-M2.5
```

## Step 8. Launch inference server for MiniMax M2.5

Start the vLLM inference server with tensor parallelism across all nodes.

```bash
## On Node 1, enter container and start server
## Assuming that you run on a 4 node cluster, set --tensor-parallel-size as 4
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 --max-model-len 129000 --max-num-seqs 4 --trust-remote-code'
```

## Step 9. Test MiniMax M2.5 model inference

Verify the deployment with a sample inference request.

```bash
## Test from Node 1 or external client.
## If testing with external client change localhost to the Node 1 Mgmt IP address.
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.5",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

## Step 10. Validate deployment

Perform comprehensive validation of the distributed inference system.

```bash
## Check Ray cluster health
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

## Verify server health endpoint on Node 1
curl http://localhost:8000/health

## Monitor GPU utilization on all nodes
nvidia-smi
```

## Step 11. Next steps

Access the Ray dashboard for cluster monitoring and explore additional features:

```bash
## Ray dashboard available at:
http://<head-node-ip>:8265

## Consider implementing for production:
## - Health checks and automatic restarts
## - Log rotation for long-running services
## - Persistent model caching across restarts
## - Other models which can fit on the cluster with different quantization methods (FP8, NVFP4)
```

## Run Agent Ready Qwen3.6 35B Model with vLLM

## Step 1. Configure Docker permissions

To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.

Open a new terminal and test Docker access. In the terminal, run:
```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with sudo .

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Pull vLLM container image

```bash
docker pull vllm/vllm-openai:latest
```

## Step 3. Launch the Agent Ready Qwen3.6 35B server

Launch the container and start the vLLM server with the agent-ready
`nvidia/Qwen3.6-35B-A3B-NVFP4` recipe. The `vllm/vllm-openai` image entrypoint is
`vllm serve`, so the model handle and flags are passed directly as container arguments.

```bash
## HuggingFace token (required to download the model)
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"

docker run -it --gpus all -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  vllm/vllm-openai:latest \
  nvidia/Qwen3.6-35B-A3B-NVFP4 \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 1 \
  --trust-remote-code \
  --kv-cache-dtype fp8 \
  --attention-backend flashinfer \
  --moe-backend marlin \
  --gpu-memory-utilization 0.4 \
  --max-model-len 262144 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 8192 \
  --enable-chunked-prefill \
  --async-scheduling \
  --enable-prefix-caching \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
  --load-format fastsafetensors \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice
```

Expected output should include:
- Model loading confirmation
- Server startup on port 8000
- GPU memory allocation details

In another terminal, test the server:

```bash
curl http://localhost:8000/v1/chat/completions \
-H "Content-Type: application/json" \
-d '{
    "model": "nvidia/Qwen3.6-35B-A3B-NVFP4",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
}'
```

Expected response should contain `"content": "204"` or similar mathematical calculation.


## Step 4. Cleanup and rollback

For container approach (non-destructive):

```bash
docker rm $(docker ps -aq --filter ancestor=vllm/vllm-openai:latest)
docker rmi vllm/vllm-openai:latest
```

## Troubleshooting

## Common issues for running on a single Spark

| Symptom | Cause | Fix |
|---------|--------|-----|
| CUDA version mismatch errors | Wrong CUDA toolkit version | Reinstall CUDA 12.9 using exact installer |
| Container registry authentication fails | Invalid or expired GitLab token | Generate new auth token |
| SM_121a architecture not recognized | Missing LLVM patches | Verify SM_121a patches applied to LLVM source |
| CUDA out of memory | Insufficient GPU memory | Reduce --max-model-len and --max-num-seqs parameters |

## Common Issues for running on two Sparks
| Symptom | Cause | Fix |
|---------|--------|-----|
| Node 2 not visible in Ray cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| Model download fails | Authentication or network issue | Re-run `huggingface-cli login`, check internet access |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your HuggingFace token; and request access to the gated model on your web browser |
| CUDA out of memory | Insufficient GPU memory | Reduce --max-model-len and --max-num-seqs parameters |
| Container startup fails | Missing ARM64 image | Rebuild vLLM image following ARM64 instructions |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
