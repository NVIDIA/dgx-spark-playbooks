# Deploy NVIDIA NIM for LLM Inference

> Prebuilt, GPU-optimized model containers with a ready-to-use HTTP endpoint

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA NIM is containerized software for fast, reliable AI model serving and inference on NVIDIA GPUs. This playbook shows how to run NIM microservices for LLMs on your hardware platform through a simple Docker workflow: authenticate with NVIDIA's registry, launch the NIM inference microservice, and validate the OpenAI-compatible HTTP endpoint.

## What you'll accomplish

You'll launch a NIM container on your **hardware platform** to expose a GPU-accelerated HTTP endpoint for chat completions. These instructions use the Nemotron 3.5 Lightning 30B-A3B NIM as the default example; additional NIM containers are available in the NGC catalog (see **Find model recipes**).

## What to know before starting

**Required:**

- Working in a terminal environment
- Using Docker commands and GPU-enabled containers
- Basic familiarity with REST APIs and curl commands

**Optional:**

- Understanding of NVIDIA GPU environments and CUDA

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | `nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:latest` | — |

   For full hardware x NIM support details, see [NIM for LLMs support matrix](https://docs.nvidia.com/nim/large-language-models/latest/reference/support-matrix.html).

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for your chosen NIM (varies by model and precision). On DGX Spark, Nemotron 3.5 Lightning needs roughly 30 GB (NVFP4), 32 GB (W4A16), or 66 GB (BF16) of unified memory at TP=1, before KV cache headroom
- At least 19–63 GB available storage for model caching, depending on the precision you pull (NVFP4 is smallest, BF16 largest)

**Software requirements**

- NVIDIA drivers installed: `nvidia-smi`
- Docker with NVIDIA Container Toolkit configured:
```bash
  docker run -it --gpus=all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```
- NGC account with an API key from [NGC API Key setup](https://ngc.nvidia.com/setup/api-key):
```bash
  echo $NGC_API_KEY | grep -E '^[a-zA-Z0-9]{86}=='
```
- Network access to NGC (`nvcr.io`) to pull containers and download model assets
- Port 8000 available for the NIM HTTP endpoint

## Find model recipes

Browse NIM containers for your hardware platform in the [NVIDIA NGC catalog](https://catalog.ngc.nvidia.com) and the [NIM for LLMs supported models](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html) list. Each container page includes pull and run guidance for that model.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [Nemotron 3.5 Lightning 30B-A3B NIM](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/nemotron-3.5-lightning-30b-a3b) · [Llama 3.1 8B Instruct NIM for DGX Spark](https://catalog.ngc.nvidia.com/orgs/nim/teams/meta/containers/llama-3.1-8b-instruct-dgx-spark) · [Qwen3-32B NIM for DGX Spark](https://catalog.ngc.nvidia.com/orgs/nim/teams/qwen/containers/qwen3-32b-dgx-spark) · [NIM for LLMs supported models](https://docs.nvidia.com/nim/large-language-models/latest/supported-models.html) |

Use the **Instructions** tab for the base Docker workflow with the default Nemotron 3.5 Lightning 30B-A3B NIM.

> [!NOTE]
> **Nemotron 3.5 Lightning does not use a `-dgx-spark` container variant.** Unlike the Llama and Qwen3 recipes above, it ships as a single multi-platform container that lists `NVIDIA-GB10` among its verified GPUs and supports all published profiles there. See [Get Started with Nemotron 3.5 Lightning](https://docs.nvidia.com/nim/large-language-models/latest/get-started/advanced/get-started-nemotron-3.5-lightning.html) for profile selection and pinned variant tags.

> [!NOTE]
> **Memory and disk determine what you can run.** Larger NIMs need more unified memory and cache space. If a model is not listed for your hardware platform, check the container page and supported-models list before downloading.

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to model download)
- **Risk level:** Low
  - Large model downloads may take significant time depending on network speed
  - GPU memory requirements vary by model size
  - Container startup time depends on model loading
- **Rollback:** Stop and remove the container with `docker stop <CONTAINER_NAME> && docker rm <CONTAINER_NAME>`. Remove cached models from `~/.cache/nim` only if you need the disk space (requires re-download on next run).
- **Last Updated:** 09/30/2026
  - Deploy NVIDIA NIM for LLM inference on supported hardware platforms with Docker, NGC auth, and OpenAI-compatible endpoint validation

## Instructions

## Step 1. Verify environment prerequisites

Check that your system meets the basic requirements for running GPU-enabled containers.

```bash
nvidia-smi
docker --version
docker run --rm --gpus all nvcr.io/nvidia/cuda:13.0.1-devel-ubuntu24.04 nvidia-smi
```

Expected output should show GPU details from `nvidia-smi` inside the container.

If you see a permission-denied error connecting to the Docker daemon socket, add your user to the `docker` group so you do not need `sudo`:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Configure NGC authentication

Set up access to NVIDIA's container registry using your NGC API key.

```bash
export NGC_API_KEY="<YOUR_NGC_API_KEY>"
echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

## Step 3. Select and configure NIM container

Choose a specific LLM NIM from NGC and set up local caching for model assets. The default image below matches the Supported hardware platforms matrix; swap `IMG_NAME` and `MODEL_ID` for another NIM from **Find model recipes** when needed.

```bash
export CONTAINER_NAME="nim-llm-demo"
export IMG_NAME="nvcr.io/nim/nvidia/nemotron-3.5-lightning-30b-a3b:latest"
export MODEL_ID="nvidia/nemotron-3.5-lightning-30b-a3b"
export LOCAL_NIM_CACHE=~/.cache/nim
export LOCAL_NIM_WORKSPACE=~/.local/share/nim/workspace
mkdir -p "$LOCAL_NIM_WORKSPACE"
chmod -R a+w "$LOCAL_NIM_WORKSPACE"
mkdir -p "$LOCAL_NIM_CACHE"
chmod -R a+w "$LOCAL_NIM_CACHE"
```

> [!NOTE]
> `:latest` is a floating tag that tracks the newest published build. For reproducible deployments, pin an explicit version tag from the [Nemotron 3.5 Lightning container page](https://catalog.ngc.nvidia.com/orgs/nim/nvidia/containers/nemotron-3.5-lightning-30b-a3b) instead. To see which profiles your GPU can serve before pulling weights, run `list-model-profiles`:
> ```bash
> docker run --rm --gpus=all -e NGC_API_KEY=$NGC_API_KEY $IMG_NAME list-model-profiles
> ```

## Step 4. Launch NIM container

Start the containerized LLM service with GPU acceleration and shared memory for model loading.

```bash
docker run -it --rm --name=$CONTAINER_NAME \
  --gpus all \
  --shm-size=16GB \
  -e NGC_API_KEY=$NGC_API_KEY \
  -e NIM_MODEL_NAME=$MODEL_ID \
  -e NIM_SERVED_MODEL_NAME=$MODEL_ID \
  -e NIM_PASSTHROUGH_ARGS="--reasoning-parser nemotron_v3" \
  -v "$LOCAL_NIM_CACHE:/opt/nim/.cache" \
  -v "$LOCAL_NIM_WORKSPACE:/opt/nim/workspace" \
  -p 8000:8000 \
  $IMG_NAME
```

Nemotron 3.5 Lightning is a reasoning model, so `--reasoning-parser nemotron_v3` is required for the server to split reasoning traces from the final answer. Without it, the trace and the answer arrive together in `choices[].message.content`, separated by a literal `</think>` marker, instead of in a separate `choices[].message.reasoning` field.

> [!WARNING]
> **If you select the BF16 profile on DGX Spark, add a memory cap.** BF16 shares the GB10 unified memory pool with the host and will OOM at the default utilization. Append the flag to the passthrough args:
> ```bash
> -e NIM_PASSTHROUGH_ARGS="--reasoning-parser nemotron_v3 --gpu-memory-utilization 0.75" \
> ```
> The NVFP4 and W4A16 profiles do not need this cap and leave far more room for KV cache on a 128 GB system. NVFP4 requires Blackwell (SM 10.0+), which GB10 provides.

The container downloads the model on first run and may take several minutes to start. Look for startup messages indicating the service is ready.

## Step 5. Validate inference endpoint

Test the deployed service with a basic chat completion request. Run the following curl command in a **new terminal** while the container is running.

```bash
curl -X 'POST' \
    'http://0.0.0.0:8000/v1/chat/completions' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "model": "nvidia/nemotron-3.5-lightning-30b-a3b",
      "messages": [
        {
          "role":"user",
          "content":"Summarize GPU computing in one sentence."
        }
      ],
      "max_tokens": 128,
      "temperature": 0.0,
      "chat_template_kwargs": {"enable_thinking": false}
    }'
```

Expected output should be a JSON response with a `choices` array containing generated text. The `model` field in the response echoes `nvidia/nemotron-3.5-lightning-30b-a3b`, matching the `NIM_SERVED_MODEL_NAME` set at launch.

`"enable_thinking": false` returns a direct answer. Set it to `true` (or omit it) to let the model reason first — with the `nemotron_v3` parser enabled, the reasoning trace arrives in `choices[].message.reasoning` and the answer in `choices[].message.content`.

From another device on the same network, replace `0.0.0.0` with your hardware platform's reachable address.

## Step 6. Cleanup

Stop and remove the container when you are done testing. Cleanup is optional rollback — not required to complete the playbook.

> [!WARNING]
> Removing cached models will require re-downloading on the next run.

```bash
docker stop $CONTAINER_NAME
docker rm $CONTAINER_NAME
```

To remove cached models and free disk space:

```bash
rm -rf "$LOCAL_NIM_CACHE"
```

## Step 7. Next steps

With a working NIM deployment, you can:

1. Integrate the API endpoint into your applications using the OpenAI-compatible interface
2. Experiment with different models from the NGC catalog (see **Find model recipes**)
3. Scale the deployment using container orchestration tools
4. Monitor resource usage with `nvidia-smi` and optimize container resource allocation

Test the integration with your preferred HTTP client or SDK to begin building applications.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container fails to start with GPU error | NVIDIA Container Toolkit not configured | Install nvidia-container-toolkit and restart Docker |
| "Invalid credentials" during docker login | Incorrect NGC API key format | Verify the API key from the NGC portal; ensure no extra whitespace |
| Model download hangs or fails | Network connectivity or insufficient disk space | Check internet connection and available disk space in the cache directory |
| API returns 404 or connection refused | Container not fully started or wrong port | Wait for container startup completion; verify port 8000 is accessible |
| runtime not found | NVIDIA Container Toolkit not properly configured | Run `sudo nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| Memory pressure within capacity | Unified memory buffer cache not released | See UMA note below |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
