# Serve LLMs with NVIDIA TensorRT-LLM

> Lower-latency responses and higher throughput for the largest models

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16](#nemotron-3-nano-omni-30b-a3b-reasoning-bf16)
  - [Llama 3.1 8B Instruct (default path)](#llama-31-8b-instruct-default-path)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendation by hardware platform](#recommendation-by-hardware-platform)
  - [Before you launch](#before-you-launch)
- [Multi-node serving](#multi-node-serving)
  - [Prerequisites](#prerequisites)
  - [Step 1. Identify the interconnect](#step-1-identify-the-interconnect)
  - [Step 2. Create the OpenMPI hostfile](#step-2-create-the-openmpi-hostfile)
  - [Step 3. Download the container entrypoint](#step-3-download-the-container-entrypoint)
  - [Step 4. Start a container on every node](#step-4-start-a-container-on-every-node)
  - [Step 5. Copy the hostfile to the primary container](#step-5-copy-the-hostfile-to-the-primary-container)
  - [Step 6. Configure the distributed server](#step-6-configure-the-distributed-server)
  - [Step 7. Download the model](#step-7-download-the-model)
  - [Step 8. Start distributed serving](#step-8-start-distributed-serving)
  - [Step 9. Test the API](#step-9-test-the-api)
  - [Step 10. Stop the distributed containers](#step-10-stop-the-distributed-containers)
- [Open WebUI](#open-webui)
  - [Step 1. Verify the API server](#step-1-verify-the-api-server)
  - [Step 2. Start Open WebUI](#step-2-start-open-webui)
  - [Step 3. Open the interface](#step-3-open-the-interface)
  - [Step 4. Stop Open WebUI](#step-4-stop-open-webui)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA TensorRT-LLM is an open-source library for optimizing and accelerating large language model inference on NVIDIA GPUs. It combines optimized kernels, memory management, quantization, and parallelism strategies to reduce latency and increase throughput.

An OpenAI-compatible server lets applications use optimized models through familiar chat-completions endpoints.

## What you'll accomplish

You'll validate TensorRT-LLM, run a model, and expose it through an OpenAI-compatible API on your **hardware platform**.

You can also connect Open WebUI or scale serving across multi-node capable hardware.

## What to know before starting

**Required:**

- Command-line experience
- Basic Docker container usage
- Familiarity with model identifiers and REST APIs

**Optional:**

- Experience with PyTorch, quantization, and inference tuning
- Basic networking and passwordless SSH between nodes (multi-node capable hardware only)

> [!TIP]
> For DGX Spark multi-node serving, use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure the interconnect and inter-device SSH. A successful Cluster Assistant run satisfies that prerequisite.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, default container image, and whether multi-node serving applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB unified memory | Single-node: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13`<br>Multi-node: `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5` | ✅ (high-speed interconnect and MPI) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see the matrix above
- Sufficient memory for the selected model and context length
- Sufficient available storage for container images and model weights
- For distributed serving, multi-node capable hardware with a configured interconnect and passwordless SSH

**Software requirements**

- NVIDIA driver and GPU visibility: `nvidia-smi`
- Docker: `docker --version`
- NVIDIA Container Toolkit configured for Docker
- Hugging Face account and access token for gated or private models
- Network access to NGC and Hugging Face
- TCP port `8355` available for the OpenAI-compatible server

## Find model recipes

Start with the validated model matrix below and the launch examples in **Instructions**. For other models, use the TensorRT-LLM documentation under **Resources** and the model card to confirm architecture support, precision, parser options, and memory settings.

Use the **Agent-ready Models** tab for the recommended agentic model and the **Multi-node serving** tab for distributed recipes.

## Supported models

These model checkpoints are validated starting points for the supported hardware platform. Confirm that the selected model fits available memory before downloading it.

| Model | Quantization | Hugging Face handle |
|-------|--------------|---------------------|
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | BF16 | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16` |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | FP8 | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-FP8` |
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | NVFP4 | `nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-NVFP4` |
| **Nemotron-3-Super-120B** | NVFP4 | `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| **GPT-OSS-20B** | MXFP4 | `openai/gpt-oss-20b` |
| **GPT-OSS-120B** | MXFP4 | `openai/gpt-oss-120b` |
| **Llama-3.1-8B-Instruct** | FP8 | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| **Llama-3.1-8B-Instruct** | NVFP4 | `nvidia/Llama-3.1-8B-Instruct-FP4` |
| **Llama-3.3-70B-Instruct** | NVFP4 | `nvidia/Llama-3.3-70B-Instruct-FP4` |
| **Qwen3-8B** | FP8 / NVFP4 | `nvidia/Qwen3-8B-FP8` / `nvidia/Qwen3-8B-FP4` |
| **Qwen3-14B** | FP8 / NVFP4 | `nvidia/Qwen3-14B-FP8` / `nvidia/Qwen3-14B-FP4` |
| **Qwen3-32B** | NVFP4 | `nvidia/Qwen3-32B-FP4` |
| **Qwen3-30B-A3B** | NVFP4 | `nvidia/Qwen3-30B-A3B-FP4` |
| **Qwen3-235B-A22B (multi-node)** | NVFP4 | `nvidia/Qwen3-235B-A22B-FP4` |
| **Phi-4-multimodal-instruct** | FP8 / NVFP4 | `nvidia/Phi-4-multimodal-instruct-FP8` / `nvidia/Phi-4-multimodal-instruct-FP4` |
| **Phi-4-reasoning-plus** | FP8 / NVFP4 | `nvidia/Phi-4-reasoning-plus-FP8` / `nvidia/Phi-4-reasoning-plus-FP4` |
| **Llama-4-Scout-17B-16E-Instruct** | NVFP4 | `nvidia/Llama-4-Scout-17B-16E-Instruct-FP4` |

Not every architecture supports every quantization format. Check the TensorRT-LLM documentation and model card before converting a checkpoint.

## Ancillary files

The multi-node workflow uses an entrypoint script shipped with this playbook:

- [`trtllm-mn-entrypoint.sh`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-trt-llm/assets/trtllm-mn-entrypoint.sh) — configures SSH inside each TensorRT-LLM container

## Time & risk

- **Estimated time:** 60 MIN (longer on first run due to container and model downloads)
- **Risk level:** Medium
  - Model downloads may require authentication and substantial storage
  - Memory requirements vary by model, precision, context length, and batch size
- **Rollback:** Stop and remove the serving container; keep the model cache for a non-destructive rollback
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant guidance for multi-node serving
  - 07/31/2026: Restored Nemotron Omni serve recipe; dual single-node / multi-node container tags; agent-ready recommendation with launch settings pending validation

## Instructions

> [!NOTE]
> These instructions use Linux and a containerized TensorRT-LLM workflow.

## Step 1. Set up Docker permissions

Test whether your account can manage Docker:

```bash
docker ps
```

If the command returns a permission-denied error, add your user to the `docker` group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

## Step 2. Verify GPU access

Confirm that the host and a TensorRT-LLM container can access the GPU:

```bash
nvidia-smi

export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"
docker run --rm --gpus all "$TRTLLM_IMAGE" nvidia-smi
```

Both commands should list an NVIDIA GPU without container-runtime errors.

## Step 3. Set environment variables

Create a Hugging Face token at <https://huggingface.co/settings/tokens>. A token is required for gated or private models.

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc13"

mkdir -p "$HOME/.cache/huggingface"
```

Choose another validated checkpoint from the **Supported models** table in **Overview** when needed.

## Step 4. Validate TensorRT-LLM

Verify the Python package inside the container:

```bash
docker run --rm --gpus all "$TRTLLM_IMAGE" \
  python -c "import tensorrt_llm; print(tensorrt_llm.__version__)"
```

Expected output includes a TensorRT-LLM version string.

Run a short generation to validate model download, engine initialization, and GPU execution:

```bash
docker run --rm -it \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download "$MODEL_HANDLE" &&
    python examples/llm-api/quickstart_advanced.py \
      --model_dir "$MODEL_HANDLE" \
      --prompt "Paris is great because" \
      --max_tokens 64
  '
```

Expected output includes generated text following the prompt.

## Step 5. Start an OpenAI-compatible server

Launch `trtllm-serve` on port `8355`. Leave the serving terminal open; model loading can take several minutes on the first run.

### Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16

This example writes **`nano_v3.yaml`** for KV cache, MoE, and CUDA graph settings, then starts the server with Nemotron Omni reasoning and tool parsers:

```bash
export MODEL_HANDLE="nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16"

docker run --name trtllm-server --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download "$MODEL_HANDLE" &&
    cat > nano_v3.yaml <<EOF
kv_cache_config:
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.80
  mamba_ssm_cache_dtype: float32
moe_config:
  backend: CUTLASS
cuda_graph_config:
  enable_padding: true
  max_batch_size: 1
max_batch_size: 1
EOF
    PYTORCH_ALLOC_CONF=expandable_segments:True \
    trtllm-serve serve "$MODEL_HANDLE" \
      --host 0.0.0.0 \
      --port 8355 \
      --trust_remote_code \
      --reasoning_parser nano-v3 \
      --tool_parser qwen3_coder \
      --extra_llm_api_options nano_v3.yaml
  '
```

### Llama 3.1 8B Instruct (default path)

Use this path for a smaller first-run validation with the default `MODEL_HANDLE` from Step 3:

```bash
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"

docker run --name trtllm-server --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download "$MODEL_HANDLE" &&
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --host 0.0.0.0 \
      --port 8355 \
      --max_batch_size 64 \
      --trust_remote_code \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

For `openai/gpt-oss-20b` or `openai/gpt-oss-120b`, add this setup at the start of the container's `bash -c` script (quickstart or serve), before `hf download`:

```bash
export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs"
mkdir -p "$TIKTOKEN_ENCODINGS_BASE"
wget -P "$TIKTOKEN_ENCODINGS_BASE" \
  https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken
wget -P "$TIKTOKEN_ENCODINGS_BASE" \
  https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
```

## Step 6. Test the API

From a second terminal, set `MODEL_HANDLE` to the model you served in Step 5 and send a chat request:

```bash
export MODEL_HANDLE="<MODEL_HANDLE_FROM_STEP_5>"

curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
    "max_tokens": 64
  }'
```

The JSON response should contain a `choices` array with generated text.

From another machine, replace `localhost` with the reachable IP address of your hardware platform.

## Step 7. Run multimodal inference

To validate image understanding, use a supported vision-language checkpoint:

```bash
export MODEL_HANDLE="nvidia/Phi-4-multimodal-instruct-FP4"

docker run --rm -it \
  --gpus all \
  --ipc host \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  "$TRTLLM_IMAGE" \
  bash -c '
    python examples/llm-api/quickstart_multimodal.py \
      --model_type phi4mm \
      --model_dir "$MODEL_HANDLE" \
      --modality image \
      --media "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png" \
      --prompt "What is happening in this image?" \
      --load_lora \
      --auto_model_name Phi4MMForCausalLM
  '
```

Expected output describes the image.

## Step 8. Stop the server

Press `Ctrl+C` in the serving terminal. Because the container uses `--rm`, Docker removes it after it stops.

To remove downloaded model data and the container image:

> [!WARNING]
> These commands delete cached model weights and require downloading them again for a future run.

```bash
sudo chown -R "$USER:$USER" "$HOME/.cache/huggingface"
rm -rf "$HOME/.cache/huggingface"
docker rmi "$TRTLLM_IMAGE"
```

## Next steps

- Use **Agent-ready Models** to select the recommended model for agentic workloads.
- Use **Open WebUI** to add a browser chat interface.
- Use **Multi-node serving** to serve a larger model across multi-node capable hardware.
- Tune batch size, context length, and KV cache allocation for your workload.

## Agent-ready Models

## Agent-ready models

Agent-ready models are suited to tool calling, reasoning, and long multi-turn sessions. Use this tab to choose the recommended starting point for your hardware platform.

### Recommendation by hardware platform

| Hardware platform | Recommended agent-ready model |
| ----------------- | ----------------------------- |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (NVFP4) |

### Before you launch

1. Complete Steps 1–4 in **Instructions** to validate Docker, GPU access, and TensorRT-LLM.
2. After the server is running, use the API test in Step 6 of **Instructions**.

Launch settings for **Agent-ready Qwen3.6-35B-A3B (NVFP4)** — including the Hugging Face handle, container image, and `trtllm-serve` flags — are pending validation. Do not substitute parser or runtime settings from another model family.

When validated launch settings are available, they will appear in this tab as a copy-paste serve recipe.

## Multi-node serving

## Multi-node serving

Use this tab to serve a model across **multi-node capable hardware** with OpenMPI and TensorRT-LLM. Complete Steps 1–4 in **Instructions** on every node first.

### Prerequisites

- Multi-node capable hardware with a configured high-speed interconnect
- The same user account and home-directory layout on every node
- Passwordless SSH between nodes
- Docker and NVIDIA Container Toolkit on every node
- A Hugging Face token with access to the selected model

> [!TIP]
> For a supported DGX Spark cluster, use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure the interconnect and inter-device SSH. If it reports success, do not repeat a manual connection playbook; continue to Step 1.

### Step 1. Identify the interconnect

On each node, identify the interface used for multi-node traffic and record its IPv4 address:

```bash
ip -br -4 address

export MN_IF_NAME="<INTERCONNECT_INTERFACE>"
export MN_IP_ADDRESS="$(ip -4 addr show "$MN_IF_NAME" | awk '/inet / {print $2}' | cut -d/ -f1)"
echo "$MN_IF_NAME $MN_IP_ADDRESS"
```

Verify that every node can reach every other node over these addresses.

### Step 2. Create the OpenMPI hostfile

On the primary node, create `~/openmpi-hostfile` with one interconnect IP address per line:

```bash
cat > "$HOME/openmpi-hostfile" <<'EOF'
<PRIMARY_NODE_IP>
<WORKER_NODE_IP>
EOF
```

Add another line for each additional worker. Test passwordless SSH from the primary node to each address before continuing.

### Step 3. Download the container entrypoint

On every node, download the entrypoint script shipped with this playbook:

```bash
curl -fL \
  "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-trt-llm/assets/trtllm-mn-entrypoint.sh" \
  -o "$HOME/trtllm-mn-entrypoint.sh"
chmod +x "$HOME/trtllm-mn-entrypoint.sh"
```

### Step 4. Start a container on every node

On every node, set the local interconnect interface and start the container. Multi-node serving uses container tag `1.3.0rc5` (single-node Instructions use `1.3.0rc13`):

```bash
export MN_IF_NAME="<INTERCONNECT_INTERFACE>"
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc5"

docker run -d --rm \
  --name trtllm-multinode \
  --gpus all \
  --network host \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="$MN_IF_NAME" \
  -e NCCL_SOCKET_IFNAME="$MN_IF_NAME" \
  -e OMPI_MCA_btl_tcp_if_include="$MN_IF_NAME" \
  -e OMPI_MCA_orte_default_hostfile=/etc/openmpi-hostfile \
  -e OMPI_MCA_rmaps_ppr_n_pernode=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT=1 \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM=1 \
  -e CPATH=/usr/local/cuda/include \
  -e TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$HOME/.ssh:/tmp/.ssh:ro" \
  -v "$HOME/trtllm-mn-entrypoint.sh:/usr/local/bin/trtllm-mn-entrypoint.sh:ro" \
  "$TRTLLM_IMAGE" \
  /usr/local/bin/trtllm-mn-entrypoint.sh
```

Confirm that `trtllm-multinode` is running on every node:

```bash
docker ps --filter name=trtllm-multinode
```

### Step 5. Copy the hostfile to the primary container

On the primary node:

```bash
docker cp "$HOME/openmpi-hostfile" \
  trtllm-multinode:/etc/openmpi-hostfile
```

Verify that MPI reaches every container:

```bash
docker exec trtllm-multinode \
  mpirun --hostfile /etc/openmpi-hostfile hostname
```

Expected output contains one hostname per node.

### Step 6. Configure the distributed server

On the primary node, create the runtime configuration:

```bash
docker exec trtllm-multinode bash -c 'cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
EOF'
```

### Step 7. Download the model

Set a model validated for distributed serving, then download it across the nodes:

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
export MODEL_HANDLE="nvidia/Qwen3-235B-A22B-FP4"

docker exec \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -it trtllm-multinode \
  bash -c 'mpirun -x HF_TOKEN -x MODEL_HANDLE \
    bash -c "hf download \$MODEL_HANDLE"'
```

### Step 8. Start distributed serving

Set `TP_SIZE` to the number of participating GPUs. For one GPU per node, this is the node count:

```bash
export TP_SIZE=2

docker exec \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TP_SIZE="$TP_SIZE" \
  -it trtllm-multinode \
  bash -c '
    mpirun -x HF_TOKEN -x MODEL_HANDLE -x TP_SIZE -x TRITON_PTXAS_PATH \
      trtllm-llmapi-launch trtllm-serve "$MODEL_HANDLE" \
        --tp_size "$TP_SIZE" \
        --backend pytorch \
        --max_num_tokens 32768 \
        --max_batch_size 4 \
        --extra_llm_api_options /tmp/extra-llm-api-config.yml \
        --port 8355
  '
```

Expected output includes the server-ready message.

> [!NOTE]
> You might see a warning such as `UCX WARN network device '…' is not available`. You can ignore it when inference succeeds and only one interconnect port is in use.

### Step 9. Test the API

Run on the primary node:

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Explain tensor parallelism in two sentences."}],
    "max_tokens": 64
  }'
```

The response should contain a `choices` array with generated text.

### Step 10. Stop the distributed containers

Run on every node:

```bash
docker stop trtllm-multinode
```

The mounted Hugging Face cache remains available for future runs.

## Open WebUI

## Add Open WebUI

Open WebUI provides a browser chat interface for the OpenAI-compatible TensorRT-LLM server.

### Step 1. Verify the API server

Complete the serving workflow in **Instructions** or **Multi-node serving**. Confirm that the API responds on port `8355`:

```bash
curl -f http://localhost:8355/v1/models
```

Also confirm that port `8080` is available for Open WebUI.

### Step 2. Start Open WebUI

Run this command on the node that exposes the TensorRT-LLM API:

```bash
docker run -d \
  --name open-webui \
  --restart always \
  --network host \
  --add-host=host.docker.internal:host-gateway \
  -e OPENAI_API_BASE_URL="http://localhost:8355/v1" \
  -v open-webui:/app/backend/data \
  ghcr.io/open-webui/open-webui:main
```

If the TensorRT-LLM server uses a different host or port, update `OPENAI_API_BASE_URL`.

### Step 3. Open the interface

On the hardware platform, open:

```text
http://localhost:8080
```

From another machine, replace `localhost` with the reachable IP address of the hardware platform. Select the served model from the model menu and start a chat.

### Step 4. Stop Open WebUI

Stop and remove the container while preserving chat data:

```bash
docker stop open-webui
docker rm open-webui
```

To also delete saved Open WebUI data:

> [!WARNING]
> Removing the volume permanently deletes chat history and Open WebUI settings.

```bash
docker volume rm open-webui
```

Optionally remove the downloaded image:

```bash
docker rmi ghcr.io/open-webui/open-webui:main
```

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Docker returns `permission denied` | User is not in the Docker group | Run `sudo usermod -aG docker "$USER" && newgrp docker` |
| Container fails to access the GPU | NVIDIA Container Toolkit is not configured | Run `nvidia-ctk runtime configure --runtime=docker`, restart Docker, and repeat the GPU check in Instructions |
| Hugging Face reports that a model is gated or inaccessible | Token is missing, invalid, or lacks model access | Export a valid `HF_TOKEN`, request access on the model page, and retry |
| Model download stalls or fails | Network interruption or insufficient storage | Check network access and free disk space, then rerun `hf download "$MODEL_HANDLE"` |
| `CUDA out of memory` during loading or serving | Model, batch size, KV cache, or context length exceeds available memory | Use a smaller or more strongly quantized model; reduce `free_gpu_memory_fraction`, batch size, or token limits |
| Out of memory occurs during parallel weight loading | Host memory pressure from concurrent loading | Set `TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1` before starting the container |
| `import tensorrt_llm` fails | Wrong image or incomplete container pull | Pull the image from the Supported hardware platforms table and repeat Step 4 in Instructions |
| Server does not respond on port `8355` | Model is still loading, server exited, or port is occupied | Check the serving terminal or `docker logs`; run `lsof -i :8355`; free the port or choose another one |
| Open WebUI cannot list models | API base URL is incorrect or TensorRT-LLM is not reachable | Verify `curl http://localhost:8355/v1/models` and set `OPENAI_API_BASE_URL` to the reachable `/v1` endpoint |
| MPI hostname test returns only one hostname | Inter-node networking or SSH is not working | Verify the OpenMPI hostfile, interconnect addresses, passwordless SSH, and container SSH service on every node |
| Multi-node container exits immediately | Entrypoint script is missing, not executable, or SSH port is occupied | Verify the script mount and permissions; inspect `docker logs trtllm-multinode`; free the configured SSH port (default: `2233`) |
| `invalid mount config for type 'bind'` | A mounted host path does not exist | Create the Hugging Face cache and `.ssh` directories and verify `trtllm-mn-entrypoint.sh` exists before starting the container |
| Distributed launch returns `task: non-zero exit (255)` | A worker container is unreachable over SSH | Check container logs on every node and repeat the MPI hostname test |
| Distributed serving reports `ptxas fatal` | Runtime Triton kernel compilation cannot locate `ptxas` | Confirm `TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas` is exported through `mpirun` |
| Docker reports a Root CA certificate validation error | System clock or certificates are out of sync | Enable time synchronization with `sudo timedatectl set-ntp true`, then retry |

> [!NOTE]
> On hardware platforms with unified memory, the operating-system page cache can contribute to memory pressure. If a workload fits rated capacity but still fails, stop other memory-intensive applications and flush the host buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For current known issues, see the TensorRT-LLM documentation under **Resources**.
