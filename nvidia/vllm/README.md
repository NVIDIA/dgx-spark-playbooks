# Install and use vLLM

> Use a container or build vLLM from source for Spark

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


## Time & risk

* **Duration:** 30 minutes for Docker approach
* **Risks:** Container registry access requires internal credentials
* **Rollback:** Container approach is non-destructive.

## Instructions

## Step 1. Pull vLLM container image

Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm?version=25.09-py3
```
docker pull nvcr.io/nvidia/vllm:25.09-py3
```

## Step 2. Test vLLM in container

Launch the container and start vLLM server with a test model to verify basic functionality.

```bash
docker run -it --gpus all -p 8000:8000 \
nvcr.io/nvidia/vllm:25.09-py3 \
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

## Step 3. Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| CUDA version mismatch errors | Wrong CUDA toolkit version | Reinstall CUDA 12.9 using exact installer |
| Container registry authentication fails | Invalid or expired GitLab token | Generate new auth token |
| SM_121a architecture not recognized | Missing LLVM patches | Verify SM_121a patches applied to LLVM source |


## Step 4. Cleanup and rollback

For container approach (non-destructive):

```bash
docker rm $(docker ps -aq --filter ancestor=******:5005/dl/dgx/vllm*)
docker rmi ******:5005/dl/dgx/vllm:main-py3.31165712-devel
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

Follow the network setup instructions from the Connect two Sparks playbook to establish connectivity between your DGX Spark nodes.

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
docker pull nvcr.io/nvidia/vllm:25.09-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:25.09-py3
```


## Step 4. Start Ray head node

Launch the Ray cluster head node on Node 1. This node coordinates the distributed inference and serves the API endpoint.

```bash
## On Node 1, start head node
export MN_IF_NAME=enP2p1s0f1np1
bash run_cluster.sh $VLLM_IMAGE 192.168.100.10 --head ~/.cache/huggingface \
-e VLLM_HOST_IP=192.168.100.10 \
-e UCX_NET_DEVICES=$MN_IF_NAME \
-e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
-e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
-e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
-e TP_SOCKET_IFNAME=$MN_IF_NAME \
-e RAY_memory_monitor_refresh_ms=0 \
-e MASTER_ADDR=192.168.100.10
```


## Step 5. Start Ray worker node

Connect Node 2 to the Ray cluster as a worker node. This provides additional GPU resources for tensor parallelism.

```bash
## On Node 2, join as worker
export MN_IF_NAME=enP2p1s0f1np1
bash run_cluster.sh $VLLM_IMAGE 192.168.100.10 --worker ~/.cache/huggingface \
-e VLLM_HOST_IP=192.168.100.11 \
-e UCX_NET_DEVICES=$MN_IF_NAME \
-e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
-e OMPI_MCA_btl_tcp_if_include=$MN_IF_NAME \
-e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
-e TP_SOCKET_IFNAME=$MN_IF_NAME \
-e RAY_memory_monitor_refresh_ms=0 \
-e MASTER_ADDR=192.168.100.10
```

## Step 6. Verify cluster status

Confirm both nodes are recognized and available in the Ray cluster.

```bash
## On Node 1 (head node)
docker exec node ray status
```

Expected output shows 2 nodes with available GPU resources.

## Step 7. Download Llama 3.3 70B model

Authenticate with Hugging Face and download the recommended production-ready model.

```bash
## On Node 1, authenticate and download
huggingface-cli login
huggingface-cli download meta-llama/Llama-3.3-70B-Instruct
```

## Step 8. Launch inference server for Llama 3.3 70B

Start the vLLM inference server with tensor parallelism across both nodes.

```bash
## On Node 1, enter container and start server
docker exec -it node /bin/bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
--tensor-parallel-size 2 --max_model_len 2048
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

> **Warning:** 405B model has insufficient memory headroom for production use.

Download the quantized 405B model for testing purposes only.

```bash
## On Node 1, download quantized model
huggingface-cli download hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4
```

### Step 11. (Optional) Launch 405B inference server

Start the server with memory-constrained parameters for the large model.

```bash
## On Node 1, launch with restricted parameters
docker exec -it node /bin/bash
vllm serve hugging-quants/Meta-Llama-3.1-405B-Instruct-AWQ-INT4 \
--tensor-parallel-size 2 --max-model-len 256 --gpu-memory-utilization 1.0 \
--max-num-seqs 1 --max_num_batched_tokens 256
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
docker exec node ray status

## Verify server health endpoint
curl http://192.168.100.10:8000/health

## Monitor GPU utilization on both nodes
nvidia-smi
docker exec node nvidia-smi --query-gpu=memory.used,memory.total --format=csv
```

## Step 14. Cleanup and rollback

Remove temporary configurations and containers when testing is complete.

> **Warning:** This will stop all inference services and remove cluster configuration.

```bash
## Stop containers on both nodes
docker stop node
docker rm node

## Remove network configuration on both nodes
sudo ip addr del 192.168.100.10/24 dev enP2p1s0f1np1  # Node 1
sudo ip addr del 192.168.100.11/24 dev enP2p1s0f1np1  # Node 2
sudo ip link set enP2p1s0f1np1 down
```

## Step 15. Next steps

Access the Ray dashboard for cluster monitoring and explore additional features:

```bash
## Ray dashboard available at:
http://192.168.100.10:8265

## Consider implementing for production:
## - Health checks and automatic restarts
## - Log rotation for long-running services
## - Persistent model caching across restarts
## - Alternative quantization methods (FP8, INT4)
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Node 2 not visible in Ray cluster | Network connectivity issue | Verify QSFP cable connection, check IP configuration |
| Model download fails | Authentication or network issue | Re-run `huggingface-cli login`, check internet access |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your HuggingFace token; and request access to the gated model on your web browser |
| CUDA out of memory with 405B | Insufficient GPU memory | Use 70B model or reduce max_model_len parameter |
| Container startup fails | Missing ARM64 image | Rebuild vLLM image following ARM64 instructions |

> **Note:** DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
