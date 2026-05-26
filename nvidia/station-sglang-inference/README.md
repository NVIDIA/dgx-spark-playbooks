# LLM Inference with SGLang

> Serve LLMs with SGLang on DGX Station for prefix-cached multi-turn and structured output inference


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

SGLang is a high-performance serving framework for large language models, optimized for workloads where requests share common prefixes — multi-turn conversations, RAG pipelines, and agentic workflows. Its core innovation, **RadixAttention**, automatically caches and reuses KV cache entries across requests using a radix tree, eliminating redundant prefill computation. SGLang also provides best-in-class **structured output generation** (JSON, regex, grammar-constrained decoding) through its xGrammar backend, running up to 3x faster than standard guided decoding.

- **RadixAttention** — Automatically reuses KV cache across requests sharing common prefixes. Multi-turn conversations and repeated system prompts skip re-computation entirely, reducing first-token latency and increasing throughput.
- **Structured output** — Compressed finite-state machine decoding with grammar mask generation overlapped with the LLM forward pass. Produces valid JSON, regex-matched, or grammar-constrained output with minimal overhead.
- **OpenAI-compatible API** — Drop-in replacement for OpenAI and vLLM endpoints. Supports `/v1/chat/completions`, `/v1/completions`, and `/v1/embeddings`.
- **Blackwell optimized** — SGLang includes optimizations for SM100+ GPUs and CUDA 13, with NVFP4 GEMM support and accelerated softmax kernels.

## What you'll accomplish

Launch SGLang on DGX Station to serve an LLM, then exercise its two key differentiators: prefix-cached multi-turn chat and structured JSON output generation. You will also benchmark multi-turn throughput to see RadixAttention's effect.

- Serve Qwen3-8B with SGLang's Blackwell-optimized backend
- Send multi-turn conversations and observe prefix cache hits in server metrics
- Generate structured JSON output using schema-constrained decoding
- Benchmark multi-turn throughput with and without prefix caching

## What to know before starting

- Basic Docker container usage
- Familiarity with REST APIs (curl or Python requests)

## Prerequisites

- NVIDIA DGX Station with GB300 GPU (Blackwell SM103)
- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured: `nvidia-smi` should show the GB300
- HuggingFace account with access token
- Network access to HuggingFace and Docker Hub

## Ancillary files

- `assets/benchmark_multiturn.py` — Python script that benchmarks multi-turn conversation throughput and demonstrates structured output generation

## Time & risk

* **Duration:** 20–25 minutes (including model download)
* **Risks:** Model download requires HuggingFace authentication
* **Rollback:** Stop and remove the container to restore state
* **Last Updated:** 04/06/2026
  * First Publication

## Instructions

## Step 1. Set up Docker permissions

If you haven't already, add your user to the docker group to run Docker without sudo:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set up environment variables

```bash
## HuggingFace token (required)
## Get a token from https://huggingface.co/settings/tokens
export HF_TOKEN="your_huggingface_token"

## Model to serve
export MODEL_HANDLE="Qwen/Qwen3-8B"

## Maximum context length
export MAX_MODEL_LEN=8192
```

## Step 3. Pull the SGLang container

Pull the SGLang container image with CUDA 13.0 support (required for Blackwell SM103):

```bash
docker pull lmsysorg/sglang:latest-cu130
```

## Step 4. Identify the GB300 GPU

On DGX Station with multiple GPUs, identify the GB300's device index:

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

Look for the row showing `NVIDIA GB300`. Note its index (e.g., `1`).

## Step 5. Start SGLang server

Launch the SGLang server:

```bash
## Replace device=1 with your GB300's index from Step 4
docker run -d \
  --name sglang-server \
  --gpus '"device=1"' \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  lmsysorg/sglang:latest-cu130 \
  sglang serve --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length $MAX_MODEL_LEN \
    --mem-fraction-static 0.85
```

Check the server logs:

```bash
docker logs -f sglang-server
```

Wait for the server to show it is ready:

```
INFO:     Uvicorn running on http://0.0.0.0:30000
```

Press `Ctrl+C` to exit the log view.

> [!NOTE]
> First launch downloads the model and compiles kernels. Subsequent starts are faster thanks to cached weights and compiled artifacts.

## Step 6. Test basic inference

Send a chat completion request using the OpenAI-compatible API:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 256
  }'
```

The response follows the standard OpenAI format with a `choices` array containing the model's answer.

## Step 7. Multi-turn conversation with prefix caching

SGLang's RadixAttention automatically caches the KV cache for processed tokens. When follow-up messages share the same conversation prefix, the cached entries are reused — skipping prefill for all previously seen tokens.

Send a multi-turn conversation. The system prompt is deliberately long so the shared prefix exceeds SGLang's page size (64 tokens), which is the minimum unit for cache reuse:

```bash
## Turn 1
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [
      {"role": "system", "content": "You are an expert physics tutor who explains concepts clearly and concisely. You use real-world analogies and everyday examples to make abstract ideas concrete. When answering, first state the key concept in one sentence, then give a short explanation with an example."},
      {"role": "user", "content": "What is the difference between speed and velocity?"}
    ],
    "max_tokens": 256
  }' | python3 -m json.tool

## Turn 2 — extends the same conversation
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [
      {"role": "system", "content": "You are an expert physics tutor who explains concepts clearly and concisely. You use real-world analogies and everyday examples to make abstract ideas concrete. When answering, first state the key concept in one sentence, then give a short explanation with an example."},
      {"role": "user", "content": "What is the difference between speed and velocity?"},
      {"role": "assistant", "content": "Speed is a scalar quantity that measures how fast an object moves, while velocity is a vector quantity that includes both speed and direction. For example, a car driving at 60 km/h has a speed of 60 km/h regardless of where it is headed. But if that car is driving 60 km/h north, that is its velocity — change direction to south and the velocity changes even though the speed stays the same."},
      {"role": "user", "content": "Can you give me another example that shows why the distinction matters in real physics problems?"}
    ],
    "max_tokens": 256
  }' | python3 -m json.tool
```

The second request reuses the KV cache from the shared prefix (system message + first user turn + assistant response), only computing attention for the new user message. This reduces first-token latency for follow-up turns.

Check the cache hit rate in the server logs. SGLang logs each prefill batch with the number of cached tokens reused:

```bash
docker logs sglang-server 2>&1 | grep "cached-token" | tail -10
```

Look for `#cached-token` values greater than 0 on later turns — this confirms RadixAttention is reusing the KV cache from the shared prefix.

## Step 8. Structured JSON output

SGLang's constrained decoding guarantees valid JSON output matching a schema. This uses the xGrammar backend to overlap grammar mask generation with the model's forward pass, adding minimal latency.

Generate a structured response:

```bash
curl -s http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [
      {"role": "user", "content": "List three programming languages with their primary use case and year created."}
    ],
    "max_tokens": 512,
    "response_format": {
      "type": "json_schema",
      "json_schema": {
        "name": "languages",
        "schema": {
          "type": "object",
          "properties": {
            "languages": {
              "type": "array",
              "items": {
                "type": "object",
                "properties": {
                  "name": {"type": "string"},
                  "primary_use": {"type": "string"},
                  "year_created": {"type": "integer"}
                },
                "required": ["name", "primary_use", "year_created"]
              }
            }
          },
          "required": ["languages"]
        }
      }
    }
  }' | python3 -m json.tool
```

The response content is guaranteed to be valid JSON matching the provided schema. Parse the `choices[0].message.content` field — it will contain a well-formed JSON object.

## Step 9. Benchmark multi-turn throughput

Run the included benchmark script to measure how prefix caching improves multi-turn latency. The script is in the `assets/` directory of this playbook.

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install requests
```

```bash
python3 assets/benchmark_multiturn.py \
  --base-url http://localhost:30000 \
  --model "$MODEL_HANDLE" \
  --num-conversations 20 \
  --turns-per-conversation 5
```

The script sends parallel multi-turn conversations and measures:
- **Per-turn latency** for turn 1 vs subsequent turns (shows prefix caching effect)
- **Total throughput** in tokens per second
- **Cache statistics** from server metrics

You should see latency decrease for later turns in each conversation as the shared prefix grows, demonstrating RadixAttention's cache reuse.

> [!TIP]
> If you downloaded this playbook as a zip, the `assets/` directory is already present. If you cloned the full repository, navigate to `nvidia/station-sglang-inference/` first.

## Step 10. Cleanup

Stop and remove the container:

```bash
docker stop sglang-server
docker rm sglang-server
```

Optionally remove the image:

```bash
docker rmi lmsysorg/sglang:latest-cu130
```

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| "permission denied" when running docker | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| `device >= 0 && device < num_gpus INTERNAL ASSERT FAILED` | Using `--gpus all` on a multi-GPU system | Use `--gpus '"device=N"'` to target the GB300 specifically (check index with `nvidia-smi`) |
| "Token is required" or 401 error | Missing HuggingFace token | Ensure `HF_TOKEN` is exported before running the docker command |
| Server exits with OOM error | Model too large for available GPU memory | Lower `--mem-fraction-static` (e.g., 0.7) or reduce `--context-length`. Check GPU memory with `nvidia-smi` |
| `json_schema` response_format returns error | SGLang version too old | Ensure you are using `lmsysorg/sglang:latest-cu130`. Older versions may not support `json_schema` format |
| Server starts but CUDA errors on inference | Wrong CUDA version for Blackwell | Use the `latest-cu130` image tag. SM103 requires CUDA 13.0+ |
| Model runs on wrong GPU | Default GPU selection | Use `--gpus '"device=N"'` to select the GB300 specifically |
| Slow first request after server start | Kernel JIT compilation | First request triggers kernel compilation. Subsequent requests are fast. Wait ~30 seconds |
| Connection refused on port 30000 | Server still loading model | Check `docker logs sglang-server` — wait for the Uvicorn startup message |
| `/server_info` shows no cache stats | Endpoint may differ across versions | Try `curl http://localhost:30000/v1/models` to verify the server is responsive. Cache metrics may be under `/metrics` (requires `--enable-metrics` server flag) |

> [!NOTE]
> On DGX Station, the GB300 is typically device 1 (device 0 is the RTX Pro 6000 workstation GPU). Always verify with `nvidia-smi --query-gpu=index,name --format=csv,noheader`.
