# Serve LLMs with vLLM

> High-throughput serving for 30+ models, with continuous batching and an OpenAI-compatible API


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Hardware platform launch notes](#hardware-platform-launch-notes)
  - [Base configuration (most models)](#base-configuration-most-models)
  - [Agent-ready models](#agent-ready-models)
  - [Watch startup](#watch-startup)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendations by hardware platform](#recommendations-by-hardware-platform)
  - [Verify your server](#verify-your-server)
  - [Next steps](#next-steps)
- [Multi-node DGX Spark](#multi-node-dgx-spark)
  - [Docker permissions](#docker-permissions)
  - [Step 1. Confirm network connectivity](#step-1-confirm-network-connectivity)
  - [Step 2. Download the cluster deployment script](#step-2-download-the-cluster-deployment-script)
  - [Step 3. Pull the NGC vLLM image](#step-3-pull-the-ngc-vllm-image)
  - [Step 4. Start the Ray head node (Node 1)](#step-4-start-the-ray-head-node-node-1)
  - [Step 5. Start the Ray worker node (Node 2)](#step-5-start-the-ray-worker-node-node-2)
  - [Step 6. Verify cluster status](#step-6-verify-cluster-status)
  - [Step 7. Download Llama 3.3 70B](#step-7-download-llama-33-70b)
  - [Step 8. Launch inference server (tensor parallel across both nodes)](#step-8-launch-inference-server-tensor-parallel-across-both-nodes)
  - [Step 9. Test inference](#step-9-test-inference)
  - [Step 1. Confirm network connectivity](#step-1-confirm-network-connectivity)
  - [Step 2. Download the cluster deployment script (all nodes)](#step-2-download-the-cluster-deployment-script-all-nodes)
  - [Step 3. Pull the NGC vLLM image (all nodes)](#step-3-pull-the-ngc-vllm-image-all-nodes)
  - [Step 4. Start the Ray head node (Node 1)](#step-4-start-the-ray-head-node-node-1)
  - [Step 5. Start the Ray worker nodes (all other nodes)](#step-5-start-the-ray-worker-nodes-all-other-nodes)
  - [Step 6. Verify cluster status](#step-6-verify-cluster-status)
  - [Step 7. Download MiniMax M2.5](#step-7-download-minimax-m25)
  - [Step 8. Launch inference server (tensor parallel = node count)](#step-8-launch-inference-server-tensor-parallel-node-count)
  - [Step 9. Test inference](#step-9-test-inference)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

vLLM is an inference engine designed to run large language models efficiently. The key idea is **maximizing throughput and minimizing memory waste** when serving LLMs.

- **PagedAttention** handles long sequences without running out of GPU memory.
- **Continuous batching** keeps GPUs fully utilized by adding new requests to batches already in progress.
- An **OpenAI-compatible API** lets applications built for the OpenAI API switch to a vLLM backend with little or no modification.

## What you'll accomplish

Serve a **model** with vLLM on your **supported hardware platform** using a pre-built container and an OpenAI-compatible endpoint.

## What to know before starting

**Required:**

- Basic Docker container usage
- Familiarity with REST APIs

**Optional:**

- Basic networking and SSH between nodes (multi-node capable hardware only)

> [!TIP]
> For DGX Spark multi-node serving, use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure the interconnect and inter-device SSH. A successful Cluster Assistant run satisfies that prerequisite.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. The same base single-node workflow applies across supported hardware platforms. **Multi-node serving in this playbook is DGX Spark only** (see the Multi-node serving tab).

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | ✅ (QSFP + Ray) |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |
| **RTX PRO** | Ubuntu 22.04 / 24.04 (Linux) | Dedicated VRAM | — |


## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for your chosen model (see [vLLM Recipes](https://recipes.vllm.ai/browse) for your hardware platform)
- Multi-node capable hardware: QSFP connectivity and passwordless SSH between nodes

**Software requirements**

- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured
- HuggingFace account with an access token (for gated / private model downloads)
- Network access to NGC and HuggingFace
- NGC vLLM container image for your hardware platform — see Instructions

## Find model recipes

Browse tested vLLM launch settings for your hardware platform on [vLLM Recipes](https://recipes.vllm.ai/browse). Each recipe includes copyable `vllm serve` commands, container images, and tuning notes for that model on your hardware.

For **more recipes**, open the filtered catalogs below:

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [recipes.vllm.ai — DGX Spark](https://recipes.vllm.ai/browse?panel=open&hw=dgx_spark_gb10) |
| **DGX Station** | [recipes.vllm.ai — DGX Station](https://recipes.vllm.ai/browse?panel=open&hw=dgx_station_gb300) |
| **RTX PRO** | [recipes.vllm.ai — RTX PRO](https://recipes.vllm.ai/browse?panel=open&hw=rtx_pro_6000) |

Use the **Instructions** tab for container setup and a base `docker run` workflow. For agentic workloads, see the **Agent-ready Models** tab.

> [!NOTE]
> **Memory determines what you can run.** Large models need substantially more memory and may require CPU offload. If a model is not listed for your hardware platform, check whether it fits in available memory and try the base configuration in **Instructions**.

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to model download)
- **Risk level:** Low
  - Model download requires HuggingFace authentication
  - Some containers require NGC credentials
- **Rollback:** Stop and remove the container to restore state (non-destructive)
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant guidance for multi-node serving
  - 08/03/2026: Step 5 API test uses `max_tokens: 2048` so reasoning-enabled recipes return a visible answer
  - 07/27/2026: Multi-node serving scoped to DGX Spark only; Find model recipes links out to Spark / Station / RTX PRO filtered catalogs

## Instructions

> [!NOTE] These instructions target **Linux** (containerized vLLM). WSL and Windows Native are not applicable to the containerized vLLM workflow at this time.

## Step 1. Set up Docker permissions

To manage containers without `sudo`, add your user to the `docker` group. Open a terminal and test Docker access:

```shell
docker ps
```

If you see a permission-denied error, add your user to the docker group (skip if it already works):

```shell
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set up environment variables

Find your model's HuggingFace handle and launch settings on [vLLM Recipes](https://recipes.vllm.ai/browse) for your hardware platform. Set these so the vLLM container can download and serve your model:

```shell
## HuggingFace token (required for gated / private models)
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"

## Model to serve (HuggingFace handle from vLLM Recipes for your hardware platform)
export MODEL_HANDLE="<HF_HANDLE>"

## Tag for the vLLM image (recommended in the vLLM Recipes), then pull
export VLLM_IMAGE=vllm/vllm-openai:latest 
docker pull "$VLLM_IMAGE"

## Maximum context length (prompt + output). Size to your workload and VRAM.
export MAX_MODEL_LEN=131072
```

## Step 3. Start the vLLM server

### Hardware platform launch notes

Container flags differ slightly by hardware platform. `--gpus all` is correct on all supported hardware platforms unless noted below. Apply the note for your hardware platform to any recipe below:

| Hardware platform | Launch notes |
| :---- | :---- |
| **DGX Spark** | Unified memory (UMA). If you hit memory pressure even within capacity, flush the buffer cache (see Troubleshooting). For multi-node serving, use the **Multi-node serving** tab (multi-node capable hardware only). |
| **DGX Station** | Add `--ipc host`. `--gpus all` uses the GB300; to pin the GB300 when both GPUs are present, use `--gpus '"device=N"'` where `N` is the GB300 device id from `nvidia-smi`. |

### Base configuration (most models)

Recommended starting point for any model that fits in memory on a single node. 

```shell
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  "$VLLM_IMAGE" \
  vllm serve "$MODEL_HANDLE" \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.8
```

Settings used:

- `--max-model-len` — maximum context length (prompt + output) per request. Larger values reserve more GPU memory for the KV cache; size it to your workload.  
- `--gpu-memory-utilization 0.8` — fraction of GPU memory vLLM may use for weights and KV cache. `0.8` leaves headroom; raise toward `0.95` on a dedicated GPU to fit more KV cache.

### Agent-ready models

For agentic workloads (tool calling, reasoning, long multi-turn sessions), see the **Agent-ready Models** tab for hardware-platform recommendations and launch guidance.

### Watch startup

Check the server logs for startup progress:

```shell
docker logs -f vllm-server
```

Expected output includes:

- Model download progress (first run only)  
- Model loading into GPU memory  
- `Application startup complete.`

Or wait for the health endpoint to come up (model loading can take several minutes):

```shell
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "Server failed to start within 900s"; docker logs vllm-server | tail -50; exit 1; }
```

## Step 4. Test the API

Send a test request to verify the server:

```shell
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 4096
  }'
```

The response should contain a `choices` array with the model's answer in `message.content`.

> Recipes that enable a reasoning parser (or models that think by default) spend part of the completion budget on a thinking pass before the answer. Use a large enough `max_tokens` (this example uses `4096`) so generation can finish with `finish_reason: stop` and a non-null `content`. If you lower the budget too far, you may see `finish_reason: length` with thinking text only (often under `reasoning` or `reasoning_content`) and `content: null`.

## Step 5. Stop the container

Stop and remove the container when you are done testing (non-destructive — your model cache is preserved):

```shell
docker rm -f vllm-server 2>/dev/null || true
```

Optionally remove the image and cached model. The container downloads weights as root into the mounted hub cache, so the cached model files are root-owned and need `sudo` to delete:

```shell
docker rmi "<docker image name>" 2>/dev/null || true
sudo rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"
```

## Next steps

- **Production deployment:** configure vLLM for your specific model and workload  
- **Performance tuning:** adjust batch sizes, `--max-model-len`, and memory settings  
- **Monitoring:** set up logging and metrics collection  
- **Agent-ready models:** tool-calling and reasoning workloads — see the **Agent-ready Models** tab  
- **Scale out:** serve larger models across multiple nodes on multi-node capable hardware — see the **Multi-node serving** tab

## Agent-ready Models

## Agent-ready models

Agent-ready models are tuned for **agentic workloads** — tool calling, reasoning traces, and long multi-turn sessions. Use this tab to pick a recommended model for your hardware platform and follow the launch guidance below.

Follow the launch recipe below for your platform to set up the recommended model.

### Recommendations by hardware platform

| Hardware platform | Recommended agent-ready model | MODEL_HANDLE | Recipe |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (NVFP4) | `nvidia/Qwen3.6-35B-A3B-NVFP4` | [Launch recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-35B-A3B?hardware=dgx_spark_gb10&features=tool_calling%2Creasoning) |
| **DGX Station** | DeepSeek-V4-Flash | `deepseek-ai/DeepSeek-V4-Flash` | [Launch recipe](https://recipes.vllm.ai/deepseek-ai/DeepSeek-V4-Flash?hardware=dgx_station_gb300) |
| **RTX PRO** | Qwen3.6 27B | `nvidia/Qwen3.6-27B-NVFP4` | [Launch recipe](https://recipes.vllm.ai/Qwen/Qwen3.6-27B?hardware=rtx_pro_6000) |

### Verify your server

After you launch a recipe above, confirm startup using **Watch startup** in the **Instructions** tab, then test with **Step 4. Test the API**.

### Next steps

- **General serving workflow:** Docker setup, health checks, and API testing — see the **Instructions** tab  
- **Scale out:** multi-node serving on multi-node capable hardware — see the **Multi-node serving** tab

## Multi-node DGX Spark

## Multi-node serving

Serve models larger than a single node can hold by pooling GPUs across multiple **multi-node capable hardware** systems with a Ray cluster and tensor parallelism. Two topologies are covered:

- **Two nodes (direct QSFP cable)** — connect two nodes back-to-back.  
- **Four or more nodes through a QSFP switch** — scale out over a switch fabric.

>   
> This tab applies to **multi-node capable hardware** only. Other supported hardware platforms serve models on a single node (see the Instructions tab).

## Prerequisites

### Docker permissions

If `docker ps` fails with a permission error, complete [Step 1 in the Instructions tab](http://instructions.md) on every node in the cluster before continuing.

---

## A. Two nodes (direct QSFP cable)

### Step 1. Confirm network connectivity

> [!TIP]
> If [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) successfully configured your two-Spark cluster, this step is complete. Do not repeat the manual connection playbook; continue to Step 2.

If you have not used Cluster Assistant, follow the [Connect multiple nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-multiple-sparks) setting up two node cluster - physical cabling, network configuration, passwordless SSH, and connectivity verification.

> **Manual setup only:** the connectivity script writes its SSH key to `~/.ssh/` and fails if the directory does not exist. Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both nodes first if you have never used SSH on them.

### Step 2. Download the cluster deployment script

On **both nodes**, download and patch the Ray cluster script:

```shell
wget https://raw.githubusercontent.com/vllm-project/vllm/51c1ee9b7c8acbba4899a8ebffd390685d171946/examples/ray_serving/run_cluster.sh

sed -i 's|^RAY_START_CMD="ray start|RAY_START_CMD="pip install -q --root-user-action=ignore '\''ray[default]>=2.9'\'' \&\& ray start|' run_cluster.sh

chmod +x run_cluster.sh
```

### Step 3. Pull the NGC vLLM image

Pull the image **on both nodes**:

```shell
docker pull nvcr.io/nvidia/vllm:26.05-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.05-py3
```

### Step 4. Start the Ray head node (Node 1)

Run inside tmux/screen so an SSH drop doesn't tear down the cluster (`run_cluster.sh` has an EXIT trap that stops the container).

Set `MN_IF_NAME` to the QSFP interface name from your completed cluster setup (validated example on multi-node capable hardware: `enp1s0f1np1`). Substitute if your interface differs.

```shell
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

### Step 5. Start the Ray worker node (Node 2)

Open a second terminal, SSH to Node 2, and join the cluster. Replace `<NODE_1_IP_ADDRESS>` with Node 1's QSFP IP (run `echo $VLLM_HOST_IP` on Node 1). Run inside tmux/screen on Node 2 as well. Use the same `MN_IF_NAME` guidance as Step 4.

```shell
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
export HEAD_NODE_IP=<NODE_1_IP_ADDRESS>
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

### Step 6. Verify cluster status

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
echo "Found container: $VLLM_CONTAINER"
docker exec $VLLM_CONTAINER ray status
```

Expected output shows 2 nodes with available GPU resources.

### Step 7. Download Llama 3.3 70B

Llama 3.3 70B is gated — accept its license at [https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct](https://huggingface.co/meta-llama/Llama-3.3-70B-Instruct) and create an HF token with read permission. Authenticate inside the container so the cache lands at `/root/.cache/huggingface`:

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  hf auth login
  hf download meta-llama/Llama-3.3-70B-Instruct'
```

### Step 8. Launch inference server (tensor parallel across both nodes)

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve meta-llama/Llama-3.3-70B-Instruct \
    --tensor-parallel-size 2 --max-model-len 2048 \
    --distributed-executor-backend ray'
```

The server is ready when you see `Application startup complete.`

### Step 9. Test inference

Run on Node 1; from an external client, replace `localhost` with Node 1's reachable IP.

```shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "meta-llama/Llama-3.3-70B-Instruct",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

---

## B. Four or more nodes through a QSFP switch

Same Ray + tensor-parallel workflow as Section A, scaled to more nodes over a QSFP switch. Set `--tensor-parallel-size` equal to your node count.

> **Topology note:** the four-or-more-node path uses a different validated container image and `run_cluster.sh` source than the two-node path above. Follow the steps in this section exactly — do not mix image tags or script versions between topologies.

### Step 1. Confirm network connectivity

> [!TIP]
> For exactly four DGX Spark systems, if [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) reports success, this step is complete. Do not repeat the manual connection playbook; continue to Step 2. For more than four systems, use the manual setup path below.

If you have not used Cluster Assistant, follow the [Connect multiple nodes through a switch](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook for QSFP cabling, interface configuration, passwordless SSH, connectivity verification, and the NCCL bandwidth test.

### Step 2. Download the cluster deployment script (all nodes)

On **every node**, download the Ray cluster script:

```shell
wget https://raw.githubusercontent.com/vllm-project/vllm/refs/heads/main/examples/ray_serving/run_cluster.sh
chmod +x run_cluster.sh
```

### Step 3. Pull the NGC vLLM image (all nodes)

```shell
docker pull nvcr.io/nvidia/vllm:26.02-py3
export VLLM_IMAGE=nvcr.io/nvidia/vllm:26.02-py3
```

### Step 4. Start the Ray head node (Node 1)

Run inside tmux/screen so an SSH drop doesn't tear down the cluster.

Set `MN_IF_NAME` to the QSFP interface name from your completed cluster setup (validated example on multi-node capable hardware: `enp1s0f1np1`). Substitute if your interface differs.

```shell
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

Leave this terminal open — closing it stops the head node and tears down the cluster.

### Step 5. Start the Ray worker nodes (all other nodes)

Repeat the block below on **each worker node** (Nodes 2 through N). SSH to each node in turn, run inside tmux/screen, and replace `<NODE_1_IP_ADDRESS>` with Node 1's QSFP interface IP from the switch playbook. Use the same `MN_IF_NAME` guidance as Step 4.

```shell
export MN_IF_NAME=enp1s0f1np1
export VLLM_HOST_IP=$(ip -4 addr show $MN_IF_NAME | grep -oP '(?<=inet\s)\d+(\.\d+){3}')
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

### Step 6. Verify cluster status

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status
```

Expected output shows all nodes with available GPU resources.

### Step 7. Download MiniMax M2.5

With four or more nodes you can run this model with tensor parallelism. Authenticate and download inside the head-node container (the cache is shared across the cluster):

```shell
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  hf auth login
  hf download MiniMaxAI/MiniMax-M2.5'
```

### Step 8. Launch inference server (tensor parallel = node count)

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec -it $VLLM_CONTAINER /bin/bash -c '
  vllm serve MiniMaxAI/MiniMax-M2.5 \
    --tensor-parallel-size 4 --max-model-len 129000 --max-num-seqs 4 --trust-remote-code \
    --distributed-executor-backend ray'
```

Set `--tensor-parallel-size` to match your node count (example above uses 4).

### Step 9. Test inference

Run on Node 1; from an external client, replace `localhost` with Node 1's reachable IP.

```shell
curl http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "MiniMaxAI/MiniMax-M2.5",
    "prompt": "Write a haiku about a GPU",
    "max_tokens": 32,
    "temperature": 0.7
  }'
```

---

## Validate and monitor (both topologies)

```shell
export VLLM_CONTAINER=$(docker ps --format '{{.Names}}' | grep -E '^node-[0-9]+$')
docker exec $VLLM_CONTAINER ray status

curl http://localhost:8000/health

nvidia-smi
```

On hardware platforms with unified memory, `nvidia-smi --query-gpu` memory fields report `N/A` — use plain `nvidia-smi` instead.

The **Ray dashboard** runs on port 8265 of the head node under host networking, so it is only directly reachable from Node 1. Tunnel it from a workstation:

```shell
ssh -L 8265:localhost:8265 nvidia@<NODE_1_IP>
## then open http://localhost:8265
```

## Next steps

Consider for production:

- Health checks and automatic restarts  
- Log rotation for long-running services  
- Persistent model caching across restarts  
- Alternative quantization (FP8, NVFP4, INT4) to fit more models on the cluster

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| HuggingFace authentication failure, gated model access denied, or model download hangs/fails | All hardware platforms | Missing/invalid token, restricted model access, or network issue | Export `HF_TOKEN` before running docker; regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens) and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated) if needed; check internet connection and verify the token is valid |
| CUDA out of memory | All hardware platforms | Context length too large / model too big | Reduce `--max-model-len` and `--max-num-seqs`, or lower `--gpu-memory-utilization` |
| Server not responding on port 8000 | All hardware platforms | Port already in use | Check with `lsof -i :8000`; use `-p 8001:8000` for a different port |
| NGC authentication fails | All hardware platforms | Invalid or missing credentials | Run `docker login nvcr.io` with your NGC API key |
| `rm: cannot remove '.../.cache/huggingface/hub/models--...': Permission denied` | All hardware platforms | The container downloads weights as root into the mounted hub cache, so cached model files are root-owned | Remove with `sudo rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"` |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |
| Container startup fails / missing ARM64 image | DGX Spark | Image not built for ARM64 | Use the default NGC image for your hardware platform from the Instructions tab |
| Model runs on wrong GPU | DGX Station | Default GPU selection with two GPUs | Use `--gpus '"device=N"'` to pin the GB300 (`N` from `nvidia-smi`) |
| EngineCore failed / FlashInfer "Buffer overflow when allocating memory for batch_prefill_tmp_v" | DGX Station | CUDA graph capture failure during batch prefill | Use the recommended container image: `nvcr.io/nvidia/vllm:26.01-py3` |
| Node not visible in Ray cluster | multi-node capable hardware | Network connectivity issue | Verify QSFP cable connection and IP configuration; see Multi-node serving tab |
| Chat completion returns `content: null` with `finish_reason: length` | `max_tokens` exhausted on the thinking pass before an answer | Raise `max_tokens` in the request (Step 4 uses `4096`) so reasoning-enabled recipes can finish with a visible answer |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> **Monitoring GPU memory with UMA.** Because of unified memory, `nvidia-smi --query-gpu` memory fields report `N/A`. Use plain `nvidia-smi` instead.
