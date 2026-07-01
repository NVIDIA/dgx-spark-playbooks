# vLLM for Inference

> Install and use vLLM on DGX Station

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Base configuration (most models)](#base-configuration-most-models)
  - [DiffusionGemma 26B A4B](#diffusiongemma-26b-a4b)
  - [Step-3.7-Flash (FP8 / NVFP4)](#step-37-flash-fp8-nvfp4)
  - [Kimi-K2.5 NVFP4 (1T) — CPU offloading](#kimi-k25-nvfp4-1t-cpu-offloading)
  - [DeepSeek-V4-Flash — MTP + agentic](#deepseek-v4-flash-mtp-agentic)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

vLLM is an inference engine designed to run large language models efficiently. The key idea is **maximizing throughput and minimizing memory waste** when serving LLMs.

- **PagedAttention** handles long sequences without running out of GPU memory.
- **Continuous batching** keeps GPUs fully utilized by adding new requests to batches in progress.
- **OpenAI-compatible API** allows applications built for OpenAI to switch to vLLM with minimal changes.

## What you'll accomplish

Serve a **supported model** using vLLM on NVIDIA DGX Station. Refer to the table below to see the supported models.

You'll set up vLLM high-throughput LLM serving on NVIDIA DGX Station with Blackwell architecture.

## What to know before starting

- Basic Docker container usage
- Familiarity with REST APIs

## Prerequisites

- NVIDIA DGX Station with GB300 and RTX 6000 Pro GPUs
- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured
- HuggingFace account with access token
- Network access to NGC and HuggingFace

## Model Support Matrix

The following models are supported with vLLM on DGX Station. All listed models are available and ready to use:

| Model | Quantization | Support Status | HF Handle |
|-------|-------------|----------------|-----------|
| **DiffusionGemma 26B A4B IT** | BF16 | ✅ | [`google/diffusiongemma-26B-A4B-it`](https://huggingface.co/google/diffusiongemma-26B-A4B-it) |
| **DiffusionGemma 26B A4B IT** | NVFP4 | ✅ | [`nvidia/diffusiongemma-26B-A4B-it-NVFP4`](https://huggingface.co/nvidia/diffusiongemma-26B-A4B-it-NVFP4) |
| **Step-3.7-Flash-FP8** | FP8 | ✅ | [`stepfun-ai/Step-3.7-Flash-FP8`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-FP8) |
| **Step-3.7-Flash-NVFP4** | NVFP4 | ✅ | [`stepfun-ai/Step-3.7-Flash-NVFP4`](https://huggingface.co/stepfun-ai/Step-3.7-Flash-NVFP4) |
| **Qwen3-235B-A22B-NVFP4** | NVFP4 | ✅ | [`nvidia/Qwen3-235B-A22B-NVFP4`](https://huggingface.co/nvidia/Qwen3-235B-A22B-NVFP4) |
| **Kimi-K2.5 (1T)** | NVFP4 | ✅ | [`nvidia/Kimi-K2.5-NVFP4`](https://huggingface.co/nvidia/Kimi-K2.5-NVFP4) |
| **DeepSeek-V4-Flash** | NVFP4 | ✅ | [`deepseek-ai/DeepSeek-V4-Flash`](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash) |

## Time & risk

* **Duration:** 30 minutes (longer on first run due to model download)
* **Risks:** Model download requires HuggingFace authentication
* **Rollback:** Stop and remove the container to restore state
* **Last Updated:** 06/29/2026
  * Added Kimi-K2.5 and DeepSeek-V4-Flash to model support matrix
  * Added base configuration example, per-setting explanations, and model-specific recipes

## Instructions

## Step 1. Set up Docker permissions

If you haven't already, add your user to the docker group to run Docker without sudo:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set up environment variables

Set the following so the vLLM container can download the model and use your chosen context length:

```bash
## HuggingFace token (required)
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"

## Model to serve
export MODEL_HANDLE="<HF_HANDLE>"

## Maximum context length
export MAX_MODEL_LEN=8192
```

## Step 3. Pull vLLM container image

Pull the vLLM container from NGC. Use the **26.01** image on DGX Station; the 25.10 image can fail during engine startup with a FlashInfer buffer overflow on some configurations.

```bash
docker pull nvcr.io/nvidia/vllm:26.01-py3
```

For DiffusionGemma, use the vLLM custom container:

```bash
docker pull vllm/vllm-openai:gemma
```

For Step-3.7-Flash models, pull the custom VLLM container
```bash
docker pull vllm/vllm-openai:stepfun37
```

For Kimi-K2.5 NVFP4 (1T) with DRAM offloading, pull the **26.03** image, which includes the `--cpu-offload-params` support used below:
```bash
docker pull nvcr.io/nvidia/vllm:26.03-py3
```

For DeepSeek-V4-Flash, pull the stable DeepSeek-V4 release container. Use the **cu130** build on DGX Station (Blackwell):
```bash
docker pull vllm/vllm-openai:v0.20.0-cu130
```

## Step 4. Start vLLM server

Start the vLLM server with the model. On a single-GPU DGX Station, `--gpus all` uses the GB300; if you have multiple GPUs and want to use only the GB300, replace with `--gpus '"device=N"'` where N is the GB300 device ID from `nvidia-smi`.

### Base configuration (most models)

This is the recommended starting point for any model that fits entirely in VRAM on the GB300. The Qwen3-235B-A22B-NVFP4 model, for example, runs directly with this configuration.

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  nvcr.io/nvidia/vllm:26.01-py3 \
  vllm serve "$MODEL_HANDLE" \
    --max-model-len $MAX_MODEL_LEN \
    --gpu-memory-utilization 0.9
```

Settings used:
- `--max-model-len` — maximum context length (prompt + output) per request. Larger values reserve more GPU memory for the KV cache; size it to your workload.
- `--gpu-memory-utilization 0.9` — fraction of GPU memory vLLM may use for weights and KV cache. `0.9` leaves headroom for other processes; raise toward `0.95` to fit more KV cache if the GPU is dedicated.

### DiffusionGemma 26B A4B

For DiffusionGemma models (e.g. `google/diffusiongemma-26B-A4B-it`), run with custom VLLM container.

```bash
docker run -d \
  --name vllm-server \
  -p 8000:8000 \
  --gpus all \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e VLLM_USE_V2_MODEL_RUNNER=1 \
  -e HF_TOKEN="$HF_TOKEN" \
  vllm/vllm-openai:gemma ${MODEL_HANDLE} \
  --gpu-memory-utilization 0.85 \
  --attention-backend TRITON_ATTN \
  --max-num-seqs 16 \
  --diffusion-config '{"canvas_length":256}' \
  --override-generation-config '{"max_new_tokens": null}' \
  --load-format fastsafetensors \
  --enable-prefix-caching \
  --reasoning-parser gemma4 \
  --default-chat-template-kwargs '{"enable_thinking": true}' \
  --enable-auto-tool-choice \
  --tool-call-parser gemma4

## For BF16 checkpoint add "--moe-backend triton" for better performance
```

### Step-3.7-Flash (FP8 / NVFP4)

For Step-3.7-Flash models, run with the custom VLLM container. The FP8 and the NVFP4 versions fit entirely in VRAM on the GB300.

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  vllm/vllm-openai:stepfun37 \
  "$MODEL_HANDLE" \
    --gpu-memory-utilization 0.95 \
    --trust-remote-code \
    --reasoning-parser step3p5 \
    --enable-auto-tool-choice \
    --tool-call-parser step3p5 \
    --kv-cache-dtype fp8
```

Settings used (in addition to the base configuration):
- `--trust-remote-code` — allows the model's custom modeling code (shipped in its repo) to load. Required for Step-3.7.
- `--reasoning-parser step3p5` — parses the model's reasoning/thinking tokens into the dedicated `reasoning_content` response field.
- `--enable-auto-tool-choice` — lets the model decide when to call a tool, enabling OpenAI-compatible function calling.
- `--tool-call-parser step3p5` — parses the model's tool-call output into structured `tool_calls`. Pairs with `--enable-auto-tool-choice`.
- `--kv-cache-dtype fp8` — stores the KV cache in FP8, roughly halving KV-cache memory versus 16-bit and allowing more concurrent/longer sequences.

### Kimi-K2.5 NVFP4 (1T) — CPU offloading

For Kimi-K2.5 NVFP4 (1T) with DRAM offloading, run with the **26.03** NGC container. This model does not fit entirely in VRAM, so the MoE expert weights are offloaded to CPU DRAM with `--cpu-offload-gb 375 --cpu-offload-params experts`. Ensure the system has enough free DRAM to hold the offloaded weights.

Set `MODEL_HANDLE=nvidia/Kimi-K2.5-NVFP4` in Step 2 before running this recipe.

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  nvcr.io/nvidia/vllm:26.03-py3 \
  vllm serve "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 8000 \
    --dtype auto \
    --kv-cache-dtype auto \
    --gpu-memory-utilization 0.95 \
    --served-model-name "$MODEL_HANDLE" \
    --tensor-parallel-size 1 \
    --no-enable-prefix-caching \
    --trust-remote-code \
    --max-model-len 40960 \
    --max-num-seqs 1 \
    --max-num-batched-tokens 32768 \
    --cpu-offload-gb 375 \
    --cpu-offload-params experts
```

Settings used (in addition to the base configuration):
- `--cpu-offload-gb 375` — amount of CPU DRAM (in GiB) vLLM may use to hold weights that don't fit in VRAM. Must be large enough for the offloaded experts; the system needs at least this much free DRAM.
- `--cpu-offload-params experts` — offloads only the MoE expert weights (the bulk of a large MoE model) to DRAM, keeping attention and other hot weights in VRAM.
- `--tensor-parallel-size 1` — single GPU; the GB300 serves the whole model.
- `--max-num-seqs 1` / `--max-num-batched-tokens 32768` — caps concurrency to one sequence and the batch token budget. With expert weights paged from DRAM, throughput is offload-bound, so a low concurrency keeps latency predictable.
- `--no-enable-prefix-caching` — disables prefix-cache reuse. Offloaded experts make the memory budget tight, so the cache is turned off here rather than spent on KV reuse.
- `--kv-cache-dtype auto` / `--dtype auto` — let vLLM pick the KV-cache and compute dtypes from the model's quantization (NVFP4).

### DeepSeek-V4-Flash — MTP + agentic

For DeepSeek-V4-Flash, run with the stable **v0.20.0-cu130** container. This recipe targets agentic workloads and enables Multi-Token Prediction (MTP) speculative decoding. On a single GB300 (TP1) the MoE expert-parallel path is sufficient.

Set `MODEL_HANDLE=deepseek-ai/DeepSeek-V4-Flash` in Step 2 before running this recipe.

```bash
docker run -d \
  --name vllm-server \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 8000:8000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  vllm/vllm-openai:v0.20.0-cu130 \
  "$MODEL_HANDLE" \
    --enable-expert-parallel \
    --kv-cache-dtype fp8 \
    --trust-remote-code \
    --block-size 256 \
    --compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}' \
    --attention_config.use_fp4_indexer_cache True \
    --tokenizer-mode deepseek_v4 \
    --tool-call-parser deepseek_v4 \
    --enable-auto-tool-choice \
    --reasoning-parser deepseek_v4 \
    --speculative-config '{"method": "mtp", "num_speculative_tokens": 3}' \
    --max-model-len 32768
```

Settings used (in addition to the base configuration):
- `--enable-expert-parallel` — shards the MoE experts across the available GPU(s) using expert parallelism, the recommended MoE execution path for DeepSeek-V4.
- `--speculative-config '{"method": "mtp", "num_speculative_tokens": 3}'` — enables **MTP (Multi-Token Prediction)** speculative decoding: the model proposes 3 tokens per step that are verified in a single forward pass, cutting latency for accepted tokens. Increase `num_speculative_tokens` (e.g. to 5–7) for long-output tasks like code generation where the model produces predictable continuations and more tokens tend to be accepted; decrease it for short or highly variable outputs where the proposal overhead outweighs the gain.
- `--kv-cache-dtype fp8` — FP8 KV cache to fit more concurrent/longer sequences.
- `--block-size 256` — KV-cache page size in tokens. DeepSeek-V4 uses multiple KV-cache groups; `256` matches the recipe validated on Station.
- `--attention_config.use_fp4_indexer_cache True` — enables the FP4 indexer cache used by DeepSeek-V4's attention. (Drop this flag on platforms without native FP4, e.g. Hopper.)
- `--tokenizer-mode deepseek_v4` / `--tool-call-parser deepseek_v4` / `--reasoning-parser deepseek_v4` — DeepSeek-V4-specific tokenizer, tool-call, and reasoning parsers.
- `--enable-auto-tool-choice` — OpenAI-compatible function calling for agentic use.
- `--compilation-config '{"cudagraph_mode":"FULL_AND_PIECEWISE","custom_ops":["all"]}'` — uses full + piecewise CUDA graph capture and enables all custom ops for lower per-step overhead.
- **Prefix caching is left enabled (the vLLM default).** For agentic workloads with large shared prefixes (e.g. a 32k system/context prefix) at low batch sizes (~BS 3–4), prefix caching gives a significant throughput boost by reusing the cached prefix across requests.

Check the server logs for startup progress:

```bash
docker logs -f vllm-server
```

Expected output includes:
- Model download progress (first run only)
- Model loading into GPU memory
- `Application startup complete.`

Press `Ctrl+C` to exit log view once the server is ready.

## Step 5. Test the API

Send a test request to verify the server is working:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 256
  }'
```

The response should contain a `choices` array with the model's answer.

## Step 6. Cleanup

Stop and remove the container:

```bash
docker stop vllm-server
docker rm vllm-server
```

Optionally, remove the image and cached model:

Eg.
```bash
docker rmi "<docker image name>"
rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"
```

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|--------|-----|
| "permission denied" when running docker | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| "Token is required" or 401 error | Missing HuggingFace token | Ensure `HF_TOKEN` is exported before running docker command |
| Model download hangs or fails | Network or authentication issue | Check internet connection, verify HF_TOKEN is valid |
| CUDA out of memory | Context length too large | Reduce `MAX_MODEL_LEN` or lower `--gpu-memory-utilization` |
| Server not responding on port 8000 | Port already in use | Check with `lsof -i :8000`, use `-p 8001:8000` for different port |
| Model runs on wrong GPU | Default GPU selection | Use `--gpus '"device=0"'` to select specific GPU |
| NGC authentication fails | Invalid or missing credentials | Run `docker login nvcr.io` with NGC API key |
| EngineCore failed / FlashInfer "Buffer overflow when allocating memory for batch_prefill_tmp_v" | Known issue with vLLM 25.10 on some DGX Station setups during CUDA graph capture | Use the **26.01** container image: `nvcr.io/nvidia/vllm:26.01-py3` instead of 25.10. |
