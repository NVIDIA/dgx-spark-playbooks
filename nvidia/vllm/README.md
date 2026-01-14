# vLLM for Inference

> Install and use vLLM on DGX Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Run on two Sparks](#run-on-two-sparks)
  - [Step 11. (Optional) Launch 405B inference server](#step-11-optional-launch-405b-inference-server)
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
| **GPT-OSS-20B** | MXFP4 | ✅ | `openai/gpt-oss-20b` |
| **GPT-OSS-120B** | MXFP4 | ✅ | `openai/gpt-oss-120b` |
| **Llama-3.1-8B-Instruct** | FP8 | ✅ | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| **Llama-3.1-8B-Instruct** | NVFP4 | ✅ | `nvidia/Llama-3.1-8B-Instruct-FP4` |
| **Llama-3.3-70B-Instruct** | NVFP4 | ✅ | `nvidia/Llama-3.3-70B-Instruct-FP4` |
| **Qwen3-8B** | FP8 | ✅ | `nvidia/Qwen3-8B-FP8` |
| **Qwen3-8B** | NVFP4 | ✅ | `nvidia/Qwen3-8B-FP4` |
| **Qwen3-14B** | FP8 | ✅ | `nvidia/Qwen3-14B-FP8` |
| **Qwen3-14B** | NVFP4 | ✅ | `nvidia/Qwen3-14B-FP4` |
| **Qwen3-32B** | NVFP4 | ✅ | `nvidia/Qwen3-32B-FP4` |
| **Qwen2.5-VL-7B-Instruct** | NVFP4 | ✅ | `nvidia/Qwen2.5-VL-7B-Instruct-FP4` |
| **Phi-4-multimodal-instruct** | FP8 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP8` |
| **Phi-4-multimodal-instruct** | NVFP4 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP4` |
| **Phi-4-reasoning-plus** | FP8 | ✅ | `nvidia/Phi-4-reasoning-plus-FP8` |
| **Phi-4-reasoning-plus** | NVFP4 | ✅ | `nvidia/Phi-4-reasoning-plus-FP4` |
| **Nemotron3-Nano** | BF16 | ✅ | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16` |
| **Nemotron3-Nano** | FP8 | ✅ | `nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8` |

> [!NOTE]
> The Phi-4-multimodal-instruct models require `--trust-remote-code` when launching vLLM.

> [!NOTE]
> You can use the NVFP4 Quantization documentation to generate your own NVFP4-quantized checkpoints for your favorite models. This enables you to take advantage of the performance and memory benefits of NVFP4 quantization even for models not already published by NVIDIA.

Reminder: not all model architectures are supported for NVFP4 quantization.

## Time & risk

* **Duration:** 30 minutes for Docker approach
* **Risks:** Container registry access requires internal credentials
* **Rollback:** Container approach is non-destructive.
* **Last Updated:** 01/02/2026
  * Add supported Model Matrix (25.11-py3)
  * Improve cluster setup instructions

## Instructions

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

Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm

```bash
export LATEST_VLLM_VERSION=<latest_container_version>

## example
## export LATEST_VLLM_VERSION=25.11-py3

docker pull nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
```

For Nemotron3-Nano model support, please use release version 25.12.post1-py3

```bash
docker pull nvcr.io/nvidia/vllm:25.12.post1-py3
```

## Step 3. Test vLLM in container

Launch the container and start vLLM server with a test model to verify basic functionality.

```bash
docker run -it --gpus all -p 8000:8000 \
nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
vllm serve "Qwen/Qwen2.5-Math-1.5B-Instruct"
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
    "model": "Qwen/Qwen2.5-Math-1.5B-Instruct",
    "messages": [{"role": "user", "content": "12*17"}],
    "max_tokens": 500
}'
```

Expected response should contain `"content": "204"` or similar mathematical calculation.

## Step 4. Cleanup and rollback

For container approach (non-destructive):

```bash
docker rm $(docker ps -aq --filter ancestor=nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION})
docker rmi nvcr.io/nvidia/vllm
```


To remove CUDA 12.9:

```bash
sudo /usr/local/cuda-12.9/bin/cuda-uninstaller
```

## Step 5. Next steps

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

## Step 2. Download cluster deployment script

Obtain the vLLM cluster deployment script on both nodes. This script orchestrates the Ray cluster setup required for distributed inference.

```bash
## Download on both nodes
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/online_serving/run_cluster.sh
chmod +x run_cluster.sh
```

## Step 3. Pull the NVIDIA vLLM Image from NGC

First, you will need to configure docker to pull from NGC
If this is your first time using docker run:
```bash
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```

After this, you should be able to run docker commands without using `sudo`.


```bash
docker pull nvcr.io/nvidia/vllm:25.11-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.11-py3
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


## Step 5. Start Ray worker node

Connect Node 2 to the Ray cluster as a worker node. This provides additional GPU resources for tensor parallelism.

```bash
## On Node 2, join as worker

## Set the interface name (same as Node 1)
export MN_IF_NAME=enp1s0f1np1

## Get Node 2's own IP address
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
> **Note:** Replace `<NODE_1_IP_ADDRESS>` with the actual IP address from Node 1, specifically the QSFP interface nep1s0f1np1 configured in the [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks) playbook.

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

Authenticate with Hugging Face and download the recommended production-ready model.

```bash
## From within the same container where `ray status` ran, run the following
hf auth login
hf download meta-llama/Llama-3.3-70B-Instruct
```

## Step 8. Launch inference server for Llama 3.3 70B

Start the vLLM inference server with tensor parallelism across both nodes.

```bash
## On Node 1, enter container and start server
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max_model_len 2048'
```

## Step 9. Test 70B model inference

Verify the deployment with a sample inference request.

```bash
## Test from Node 1 or external client
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

Download the quantized 405B model for testing purposes only.

```bash
## On Node 1, download quantized model
huggingface-cli download hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4
```

### Step 11. (Optional) Launch 405B inference server

Start the server with memory-constrained parameters for the large model.

```bash
## On Node 1, launch with restricted parameters
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
    --tensor-parallel-size 2 --max-model-len 256 --gpu-memory-utilization 1.0 \
    --max-num-seqs 1 --max_num_batched_tokens 256'
```

## Step 12. (Optional) Test 405B model inference

Verify the 405B deployment with constrained parameters.

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
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

## Verify server health endpoint
curl http://192.168.100.10:8000/health

## Monitor GPU utilization on both nodes
nvidia-smi
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Step 14. Next steps

Access the Ray dashboard for cluster monitoring and explore additional features:

```bash
## Ray dashboard available at:
http://<head-node-ip>:8265

## Consider implementing for production:
## - Health checks and automatic restarts
## - Log rotation for long-running services
## - Persistent model caching across restarts
## - Alternative quantization methods (FP8, INT4)
```

## Troubleshooting

## Common issues for running on a single Spark

| Symptom | Cause | Fix |
|---------|--------|-----|
| CUDA version mismatch errors | Wrong CUDA toolkit version | Reinstall CUDA 12.9 using exact installer |
| Container registry authentication fails | Invalid or expired GitLab token | Generate new auth token |
| SM_121a architecture not recognized | Missing LLVM patches | Verify SM_121a patches applied to LLVM source |

## Common Issues for running on two Sparks
| Symptom | Cause | Fix |
|---------|--------|-----|
| Node 2 not visible in Ray cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| Model download fails | Authentication or network issue | Re-run `huggingface-cli login`, check internet access |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your HuggingFace token; and request access to the gated model on your web browser |
| CUDA out of memory with 405B | Insufficient GPU memory | Use 70B model or reduce max_model_len parameter |
| Container startup fails | Missing ARM64 image | Rebuild vLLM image following ARM64 instructions |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
