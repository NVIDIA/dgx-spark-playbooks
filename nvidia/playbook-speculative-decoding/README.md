# Speed Up Inference with Speculative Decoding

> Full output quality with drafted tokens confirmed by the target model

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Option 1: EAGLE-3](#option-1-eagle-3)
  - [Option 2: Draft-Target](#option-2-draft-target)
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
  - [Step 6. Launch Eagle3 speculative decoding](#step-6-launch-eagle3-speculative-decoding)
  - [Step 7. Validate the API](#step-7-validate-the-api)
  - [Step 8. Cleanup](#step-8-cleanup)
  - [Step 9. Next steps](#step-9-next-steps)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Speculative decoding speeds up text generation by using a **small, fast draft path** to propose several tokens ahead, then having the **larger target model** verify or correct them in parallel. The target model does not need to emit every token one step at a time, which reduces latency while preserving output quality.

This playbook uses NVIDIA TensorRT-LLM with two approaches:

- **EAGLE-3** — a drafting head generates speculative tokens internally
- **Draft-Target** — a smaller draft model accelerates a larger target model

## What you'll accomplish

You'll run speculative decoding with TensorRT-LLM on your **hardware platform**, serve an OpenAI-compatible endpoint, and optionally scale larger models across **multi-node capable hardware**.

## What to know before starting

**Required:**

- Experience with Docker and containerized applications
- Familiarity with TensorRT-LLM serving and API endpoints
- Basic understanding of GPU memory management for large language models

**Optional:**

- Understanding of speculative decoding concepts
- Basic networking and passwordless SSH between nodes (multi-node capable hardware only)

> [!TIP]
> For DGX Spark multi-node serving, use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure the interconnect and inter-device SSH. A successful Cluster Assistant run satisfies that prerequisite.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, default container image, and whether multi-node serving applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB unified memory | `nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12` | ✅ (high-speed interconnect and MPI) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see the matrix above
- Sufficient memory for the selected target model, draft path, and KV cache
- Sufficient available storage for container images and model weights
- For distributed serving, multi-node capable hardware with a configured interconnect and passwordless SSH

**Software requirements**

- NVIDIA driver and GPU visibility: `nvidia-smi`
- Docker: `docker --version`
- NVIDIA Container Toolkit configured for Docker
- Hugging Face account and access token for gated or private models
- Network access to NGC and Hugging Face

## Find model recipes

Start with the EAGLE-3 and Draft-Target examples in **Instructions**. For additional speculative decoding methods, model pairs, and tuning guidance, use the TensorRT-LLM documentation under **Resources**.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [TensorRT-LLM Speculative Decoding](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) |

Use the **Instructions** tab for the base single-node workflows. For agentic workloads, see the **Agent-ready Models** tab. For models that exceed single-node memory, see the **Multi-node serving** tab.

## When to use multi-node

A single supported hardware platform can run models such as GPT-OSS-120B with EAGLE-3 or Llama-3.3-70B with Draft-Target, as shown in **Instructions**.

Larger models such as **Qwen3-235B-A22B** can exceed single-node memory once weights, KV cache, and the Eagle3 draft head are loaded. On **multi-node capable hardware**, tensor parallelism (for example `TP=2`) splits layer weights across nodes so the model runs as one logical instance. Speculative decoding (Eagle3) on top further accelerates generation by drafting and verifying multiple tokens per step.

The **Multi-node serving** tab covers only the distributed setup and Eagle3 launch for that larger model.

## Ancillary files

The multi-node workflow uses an entrypoint script shipped with this playbook:

- [`trtllm-mn-entrypoint.sh`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-speculative-decoding/assets/trtllm-mn-entrypoint.sh) — configures SSH inside each TensorRT-LLM container

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to model downloads)
- **Risk level:** Medium
  - GPU memory exhaustion with large models or long context
  - Container registry access and network timeouts during downloads
- **Rollback:** Stop and remove the serving container; keep the model cache for a non-destructive rollback
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant guidance for multi-node serving
  - 07/31/2026: Supported hardware matrix, Find model recipes, Agent-ready Models, and multi-node Eagle3 serving with a selectable interconnect interface

## Instructions

> [!NOTE]
> These instructions use Linux and a containerized TensorRT-LLM workflow.

## Step 1. Configure Docker permissions

To manage containers without `sudo`, your user must be in the `docker` group. If you skip this step, prefix Docker commands with `sudo`.

Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error connecting to the Docker daemon socket, add your user to the `docker` group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

## Step 2. Verify GPU access

Confirm that the host and a TensorRT-LLM container can access the GPU:

```bash
nvidia-smi

export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"
docker run --rm --gpus all "$TRTLLM_IMAGE" nvidia-smi
```

Both commands should list an NVIDIA GPU without container-runtime errors.

## Step 3. Set environment variables

Create a Hugging Face token at <https://huggingface.co/settings/tokens>. A token is required for gated or private models.

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"

mkdir -p "$HOME/.cache/huggingface"
```

## Step 4. Run speculative decoding methods

Choose **Option 1 (EAGLE-3)** or **Option 2 (Draft-Target)**. Run one option at a time on a free port.

### Option 1: EAGLE-3

Run EAGLE-3 speculative decoding:

```bash
docker run \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface/" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  "$TRTLLM_IMAGE" \
  bash -c '
    hf download openai/gpt-oss-120b && \
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml'
```

Once the server is running, test it from another terminal:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "Solve the following problem step by step. If a train travels 180 km in 3 hours, and then slows down by 20% for the next 2 hours, what is the total distance traveled? Show all intermediate calculations and provide a final numeric answer.",
    "max_tokens": 300,
    "temperature": 0.7
  }'
```

**Key features of EAGLE-3 speculative decoding**

- **Simpler deployment** — a built-in drafting head generates speculative tokens instead of managing a separate draft model
- **Better draft acceptance** — features fused from multiple layers improve draft-token acceptance
- **Faster generation** — multiple tokens are verified in parallel per forward pass

### Option 2: Draft-Target

Run Draft-Target speculative decoding:

```bash
docker run \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/:/root/.cache/huggingface/" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  "$TRTLLM_IMAGE" \
  bash -c "
#    # Download models
    hf download nvidia/Llama-3.3-70B-Instruct-FP4 && \
    hf download nvidia/Llama-3.1-8B-Instruct-FP4 \
    --local-dir /opt/Llama-3.1-8B-Instruct-FP4/ && \

#    # Create configuration file
    cat <<EOF > extra-llm-api-config.yml
print_iter_log: false
disable_overlap_scheduler: true
speculative_config:
  decoding_type: DraftTarget
  max_draft_len: 4
  speculative_model_dir: /opt/Llama-3.1-8B-Instruct-FP4/
kv_cache_config:
  enable_block_reuse: false
EOF

#    # Start TensorRT-LLM server
    trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --kv_cache_free_gpu_memory_fraction 0.9 \
      --extra_llm_api_options ./extra-llm-api-config.yml
  "
```

Once the server is running, test it from another terminal:

```bash
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-FP4",
    "prompt": "Explain the benefits of speculative decoding:",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Key features of Draft-Target**

- **Efficient resource usage** — an 8B draft model accelerates a 70B target model
- **Flexible configuration** — adjustable draft token length for optimization
- **Memory efficient** — FP4 quantized models reduce memory footprint
- **Compatible models** — Llama family models with consistent tokenization

## Step 5. Cleanup

Stop the Docker container when finished:

```bash
docker ps
docker stop <container_id>
```

Optional: remove downloaded model cache entries you no longer need from `$HOME/.cache/huggingface/`.

## Step 6. Next steps

- Experiment with different `max_draft_len` values (1, 2, 3, 4, 8)
- Monitor token acceptance rates and throughput improvements
- Test with different prompt lengths and generation parameters
- For models that need more memory than one node provides, use the **Multi-node serving** tab
- Read more in the [TensorRT-LLM Speculative Decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)

## Agent-ready Models

## Agent-ready models

Agent-ready models are suited to tool calling, reasoning, and long multi-turn sessions. Use this tab to choose the recommended starting point for your hardware platform.

### Recommendation by hardware platform

| Hardware platform | Recommended agent-ready model |
| ----------------- | ----------------------------- |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (NVFP4) |

### Before you launch

1. Complete Steps 1–3 in **Instructions** to validate Docker, GPU access, and environment variables.
2. After a speculative decoding server is running, use the API tests in Step 4 of **Instructions**.

Launch settings for **Agent-ready Qwen3.6-35B-A3B (NVFP4)** with TensorRT-LLM speculative decoding — including the Hugging Face handle, container image, and `trtllm-serve` flags — are pending validation. Do not substitute parser or runtime settings from another model family.

When validated launch settings are available, they will appear in this tab as a copy-paste serve recipe.

## Multi-node serving

## Multi-node serving

Use this tab to serve a large model with Eagle3 speculative decoding across **multi-node capable hardware** using OpenMPI and TensorRT-LLM. Complete Steps 1–3 in **Instructions** on every node first.

### Prerequisites

- Multi-node capable hardware with a configured high-speed interconnect
- The same user account and home-directory layout on every node
- Passwordless SSH between nodes
- Docker and NVIDIA Container Toolkit on every node
- A Hugging Face token with access to the selected models

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

The hostfile tells MPI which nodes participate in distributed execution. Because `mpirun` launches from the primary node, create the hostfile on the primary node only.

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
  "https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-speculative-decoding/assets/trtllm-mn-entrypoint.sh" \
  -o "$HOME/trtllm-mn-entrypoint.sh"
chmod +x "$HOME/trtllm-mn-entrypoint.sh"
```

### Step 4. Start a container on every node

On every node, set the local interconnect interface and start the container:

```bash
export MN_IF_NAME="<INTERCONNECT_INTERFACE>"
export TRTLLM_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc12"
export TRTLLM_MN_CONTAINER=trtllm-multinode

docker run -d --rm \
  --name "$TRTLLM_MN_CONTAINER" \
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

Confirm that the container is running on every node:

```bash
docker ps --filter name=trtllm-multinode
```

Wait until the entrypoint finishes preparing SSH on every node before continuing. The container log ends with a message that SSH is starting:

```bash
docker logs "$TRTLLM_MN_CONTAINER"
```

> [!NOTE]
> If more than one interconnect port is up and you want to use both, set `UCX_NET_DEVICES` and `NCCL_SOCKET_IFNAME` to a comma-separated list of those interface names.

### Step 5. Copy the hostfile to the primary container

On the primary node:

```bash
docker cp "$HOME/openmpi-hostfile" \
  "$TRTLLM_MN_CONTAINER":/etc/openmpi-hostfile
```

Verify that MPI reaches every container:

```bash
docker exec "$TRTLLM_MN_CONTAINER" \
  mpirun --hostfile /etc/openmpi-hostfile hostname
```

Expected output contains one hostname per node.

### Step 6. Launch Eagle3 speculative decoding

Eagle3 speculative decoding accelerates inference by predicting multiple tokens ahead, then validating them in parallel.

#### Set your Hugging Face token

```bash
export HF_TOKEN="<YOUR_HUGGINGFACE_TOKEN>"
```

#### Download the Eagle3 speculative model to every node

Run this on the primary node. `mpirun` uses the hostfile to download the draft model on every node:

```bash
docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -it "$TRTLLM_MN_CONTAINER" bash -c "
    mpirun --hostfile /etc/openmpi-hostfile -x HF_TOKEN bash -c 'hf download nvidia/Qwen3-235B-A22B-Eagle3 --local-dir /opt/Qwen3-235B-A22B-Eagle3/'
"
```

#### Create the Eagle3 speculative decoding configuration

On the primary node, create the runtime configuration. It enables Eagle speculative decoding with 3 draft tokens and conservative memory settings.

```bash
docker exec -it "$TRTLLM_MN_CONTAINER" bash -c "cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
enable_chunked_prefill: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: /opt/Qwen3-235B-A22B-Eagle3/
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
"
```

#### Launch the server with Eagle3 speculative decoding

**Run on the primary node only.** `mpirun` coordinates execution across nodes. Adjust `--max_num_tokens` as needed.

```bash
export MODEL_HANDLE="nvidia/Qwen3-235B-A22B-FP4"
export TP_SIZE=2

docker exec \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -e HF_TOKEN="$HF_TOKEN" \
  -e TP_SIZE="$TP_SIZE" \
  -it "$TRTLLM_MN_CONTAINER" bash -c '
    mpirun --hostfile /etc/openmpi-hostfile \
           -x CPATH=/usr/local/cuda/include \
           -x TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
           -x HF_TOKEN \
           -x MODEL_HANDLE \
           -x TP_SIZE \
           trtllm-llmapi-launch \
           trtllm-serve \
           "$MODEL_HANDLE" \
           --backend pytorch \
           --tp_size "$TP_SIZE" \
           --max_num_tokens 1024 \
           --extra_llm_api_options /tmp/extra-llm-api-config.yml \
           --port 8355 --host 0.0.0.0
'
```

Expected output when the endpoint is ready includes the application startup complete message.

### Step 7. Validate the API

**Run on the primary node.** The server listens there:

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-235B-A22B-FP4",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

Expected: a JSON response with generated text. This confirms the multi-node TensorRT-LLM server with Eagle3 speculative decoding is working.

### Step 8. Cleanup

#### Stop the containers

**Run on every node:**

```bash
docker stop "$TRTLLM_MN_CONTAINER"
```

The containers are removed automatically because of the `--rm` flag.

#### (Optional) Remove downloaded models

If you need to free disk space, remove unused Hugging Face cache entries for the models you downloaded. Skip this if you plan to rerun the setup.

### Step 9. Next steps

- **Adjust draft length:** Modify `max_draft_len` in the configuration (try values between 2–5)
- **Optimize batch size:** Adjust `max_batch_size` in `cuda_graph_config` for throughput-latency tradeoffs
- **Learn more:** Review the [TensorRT-LLM Speculative Decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html)
- **Benchmark performance:** Compare inference with and without speculative decoding

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `CUDA out of memory` error | Insufficient GPU memory for model, draft path, or KV cache | Reduce `free_gpu_memory_fraction` / `kv_cache_free_gpu_memory_fraction`, lower `max_draft_len` or batch size, or use a smaller / more strongly quantized model |
| Container fails to start | Docker GPU support issues | Verify the NVIDIA Container Toolkit is installed and `--gpus=all` is supported; rerun the GPU check in Instructions |
| Model download fails | Network or authentication issues | Check Hugging Face authentication (`HF_TOKEN`) and network connectivity |
| Cannot access gated repo for URL | Hugging Face model access is restricted | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) |
| Server does not respond | Port conflicts, firewall, or model still loading | Check whether port `8000` (single-node) or `8355` (multi-node) is free; inspect container logs |
| `mpirun` fails with SSH connection refused | SSH is not configured between containers or nodes | Verify passwordless SSH between nodes and that the multi-node entrypoint finished starting SSH inside each container |
| `mpirun` hangs or times out connecting to a remote node | Hostfile IPs do not match interconnect addresses | Verify IPs in `/etc/openmpi-hostfile` match the interconnect interfaces with `ip addr show` |
| NCCL error: "Socket operation on non-socket" | Wrong network interface specified | Ensure `NCCL_SOCKET_IFNAME` and `UCX_NET_DEVICES` match the active interconnect interface used in Multi-node serving |
| `Permission denied (publickey)` during mpirun | SSH keys not exchanged between containers | Re-run the multi-node entrypoint setup or verify `/root/.ssh/authorized_keys` in each container |
| Model download fails silently in multi-node setup | `HF_TOKEN` not propagated to `mpirun` | Pass `-e HF_TOKEN="$HF_TOKEN"` to `docker exec` and `-x HF_TOKEN` to `mpirun` |

> [!NOTE]
> On hardware platforms with unified memory, the operating-system page cache can contribute to memory pressure. If a workload fits rated capacity but still fails, stop other memory-intensive applications and flush the host buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For current known issues, see the TensorRT-LLM documentation under **Resources**.
