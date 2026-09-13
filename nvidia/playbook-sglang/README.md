# Serve LLMs with SGLang

> High-throughput serving with RadixAttention, structured output, and an OpenAI-compatible API


## Table of Contents

- [Overview](#overview)
  - [DGX Spark](#dgx-spark)
  - [DGX Station](#dgx-station)
- [Instructions](#instructions)
  - [Example model IDs (`MODEL_HANDLE`)](#example-model-ids-modelhandle)
  - [Hardware platform launch notes](#hardware-platform-launch-notes)
  - [Base configuration (most models)](#base-configuration-most-models)
  - [Watch startup](#watch-startup)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

SGLang is a high-performance serving framework for large language models and vision-language models. It co-designs the backend runtime and frontend language so interactions are faster and more controllable — especially for workloads that share prefixes.

- **RadixAttention** automatically caches and reuses KV cache entries across requests that share common prefixes (multi-turn chat, RAG, agents), reducing redundant prefill work.
- **Structured output** uses compressed finite-state machine decoding (xGrammar) for JSON, regex, and grammar-constrained generation with low overhead.
- An **OpenAI-compatible API** supports `/v1/chat/completions`, `/v1/completions`, and related endpoints so existing clients can switch backends with little or no change.

## What you'll accomplish

Serve a **model** with SGLang on your **supported hardware platform** using a pre-built CUDA 13 container and an OpenAI-compatible endpoint. You will also exercise prefix-cached multi-turn chat and structured JSON output.

## What to know before starting

**Required:**

- Basic Docker container usage
- Familiarity with REST APIs

**Optional:**

- Python for client scripts and the optional multi-turn benchmark

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. The same base workflow applies across supported hardware platforms.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |


## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for your chosen model (see Supported models below)

**Software requirements**

- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured (`nvidia-smi` works inside a GPU container)
- HuggingFace account with an access token (for gated / private model downloads)
- Network access to Docker Hub and HuggingFace
- SGLang container image — see Instructions
- `git` — only for the optional benchmark and offline-inference scripts (see Ancillary files below); Steps 1–7 need no local files

## Ancillary files

Steps 1–7 of the **Instructions** tab run entirely against the container and need nothing on the host. The two optional Python scripts live under `assets/` in this playbook and require cloning the repository (Instructions → Step 8):

| File | Purpose |
|------|---------|
| `assets/benchmark_multiturn.py` | Multi-turn throughput and prefix-cache benchmark (Instructions → Step 8) |
| `assets/offline-inference.py` | In-process SGLang Engine example, no server required (Instructions → Next steps) |

## Supported models

Use the matrices below to pick a model for your hardware platform. Full serve workflow is in the **Instructions** tab.

> [!NOTE]
> **Memory determines what you can run.** Large models need substantially more memory. If a model is not listed for your hardware platform, check whether it fits in available memory and try the base configuration in **Instructions**.

### DGX Spark

Models validated with SGLang on DGX Spark:

| Model | Quantization | HF Handle |
|-------|-------------|---------|
| **Nemotron-3-Nano-Omni-30B-A3B-Reasoning** | BF16 | [`nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16`](https://huggingface.co/nvidia/Nemotron-3-Nano-Omni-30B-A3B-Reasoning-BF16) |
| **GPT-OSS-20B** | MXFP4 | `openai/gpt-oss-20b` |
| **GPT-OSS-120B** | MXFP4 | `openai/gpt-oss-120b` |
| **Llama-3.1-8B-Instruct** | FP8 | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| **Llama-3.1-8B-Instruct** | NVFP4 | `nvidia/Llama-3.1-8B-Instruct-FP4` |
| **Llama-3.3-70B-Instruct** | NVFP4 | `nvidia/Llama-3.3-70B-Instruct-FP4` |
| **Qwen3-8B** | FP8 / NVFP4 | `nvidia/Qwen3-8B-FP8` / `nvidia/Qwen3-8B-FP4` |
| **Qwen3-14B** | FP8 / NVFP4 | `nvidia/Qwen3-14B-FP8` / `nvidia/Qwen3-14B-FP4` |
| **Qwen3-32B** | NVFP4 | `nvidia/Qwen3-32B-FP4` |
| **Phi-4-multimodal-instruct** | FP8 / NVFP4 | `nvidia/Phi-4-multimodal-instruct-FP8` / `nvidia/Phi-4-multimodal-instruct-FP4` |
| **Phi-4-reasoning-plus** | FP8 / NVFP4 | `nvidia/Phi-4-reasoning-plus-FP8` / `nvidia/Phi-4-reasoning-plus-FP4` |

For NVFP4 models, add `--quantization modelopt_fp4` to the serve command. Certain models (for example Nemotron-3-Nano-Omni) may require extra flags from their HuggingFace model card.

### DGX Station

Starting points documented for SGLang on DGX Station (confirm memory headroom and SGLang build support before large downloads):

| Model | Role | HF Handle |
|-------|------|---------|
| **Qwen3-8B** | Default first-run (fast validation) | `Qwen/Qwen3-8B` |
| **Qwen3.6-35B-A3B** | MoE (~3B active); hybrid mamba/SSM — prefix-cache check in Instructions does not apply | `Qwen/Qwen3.6-35B-A3B` |
| **Qwen3.6-27B** | Dense Qwen3.6 | `Qwen/Qwen3.6-27B` |
| **Llama-3.3-70B-Instruct** | Gated on Hugging Face — accept license before download | `meta-llama/Llama-3.3-70B-Instruct` |
| **DeepSeek-V4-Flash** | Large local MoE when memory allows | `deepseek-ai/DeepSeek-V4-Flash` |
| **DeepSeek-V4-Pro** | Larger V4 variant — only with sufficient memory and a supported SGLang build | `deepseek-ai/DeepSeek-V4-Pro` |

You may also use other Hugging Face text-generation or chat checkpoints that your SGLang build supports.

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to model download and CUDA-graph capture; larger MoE models can take 45–60 MIN)
- **Risk level:** Low
  - Model download may require HuggingFace authentication for gated models
- **Rollback:** Stop and remove the container to restore state (non-destructive)
- **Last Updated:** 07/31/2026
  - Documented how to obtain the `assets/` scripts and defined the playbook root; cached-model cleanup now uses `sudo`
  - Replaced recipe-portal links with Supported models matrices (Spark + Station); removed Agent-ready Models tab

## Instructions

> [!NOTE]
> These instructions target **Linux** (containerized SGLang). WSL and Windows Native are not applicable to the containerized SGLang workflow at this time.

## Step 1. Set up Docker permissions

To manage containers without `sudo`, add your user to the `docker` group. Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error, add your user to the docker group (skip if it already works):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set up environment variables

Pick a model for your hardware platform (see Overview → Supported models). Set these so the SGLang container can download and serve your model:

```bash
## HuggingFace token (required for gated / private models)
## Get a token from https://huggingface.co/settings/tokens
## Leave empty for public models such as Qwen/Qwen3-8B
export HF_TOKEN=""

## Model to serve (HuggingFace handle)
## Fast first-run validation default:
export MODEL_HANDLE="Qwen/Qwen3-8B"

## Maximum context length (prompt + output). Size to your workload and memory.
export MAX_MODEL_LEN=8192

## SGLang container image (CUDA 13.0 — required for Blackwell)
export SGLANG_IMAGE="lmsysorg/sglang:latest-cu130"
```

### Example model IDs (`MODEL_HANDLE`)

Use any Hugging Face text-generation or chat checkpoint your SGLang build supports. Common starting points:

| Model ID | Notes |
|----------|--------|
| `Qwen/Qwen3-8B` | **Default.** Dense 8B; fast warmup for validating the workflow end-to-end. |
| `Qwen/Qwen3.6-35B-A3B` | Qwen3.6 MoE (~3B active); strong quality per GPU hour. Hybrid mamba/SSM — prefix-cache validation in Step 6 does not apply (see note there). |
| `Qwen/Qwen3.6-27B` | Dense Qwen3.6; higher memory than the MoE row at equal batch settings. |
| `google/gemma-3-12b-it` / `google/gemma-3-27b-it` | Gemma 3 instruct variants. |
| `meta-llama/Llama-3.3-70B-Instruct` | Gated on Hugging Face — accept the license before download. |
| `deepseek-ai/DeepSeek-V2-Lite` | Small DeepSeek path used in Spark validation examples. |
| Spark NVFP4 / FP8 handles | See Overview → Supported models → DGX Spark (for example `nvidia/Qwen3-32B-FP4`); add `--quantization modelopt_fp4` for NVFP4. |

Heavyweight MoE (confirm SGLang version + memory before serving):

| Model ID | Notes |
|----------|--------|
| `deepseek-ai/DeepSeek-V4-Flash` | Large local MoE for high-memory hardware platforms; long download; may need lower `--mem-fraction-static` / `--context-length`. |
| `deepseek-ai/DeepSeek-V4-Pro` | Larger V4 variant — only with sufficient memory and a supported SGLang build. |

## Step 3. Pull the SGLang container image

```bash
docker pull "$SGLANG_IMAGE"

## Optional: verify GPU access inside the image
docker run --rm --gpus all "$SGLANG_IMAGE" nvidia-smi
```

## Step 4. Start the SGLang server

### Hardware platform launch notes

Container flags differ slightly by hardware platform. `--gpus all` is correct on all supported hardware platforms unless noted below. Apply the note for your hardware platform:

| Hardware platform | Launch notes |
|-------------------|--------------|
| **DGX Station** | Add `--ipc host` and `--cap-add SYS_NICE`. `--gpus all` uses the GB300 when it is the only GPU; if multiple GPUs are present, pin with `--gpus '"device=N"'` where `N` is the GB300 index from `nvidia-smi --query-gpu=index,name --format=csv,noheader`. Use `--attention-backend flashinfer` (required on GB300 / SM103). |
| **DGX Spark** | Unified memory (UMA). Prefer a slightly lower `--mem-fraction-static` (for example `0.75`) under memory pressure. If you hit memory issues even within capacity, flush the buffer cache (see Troubleshooting). |

Identify the GPU index when needed (DGX Station with more than one GPU):

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

### Base configuration (most models)

Recommended starting point for any model that fits in memory on a single node:

```bash
docker run -d \
  --name sglang-server \
  --gpus all \
  --ipc host \
  --cap-add SYS_NICE \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -p 30000:30000 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub" \
  "$SGLANG_IMAGE" \
  sglang serve --model-path "$MODEL_HANDLE" \
    --host 0.0.0.0 \
    --port 30000 \
    --context-length $MAX_MODEL_LEN \
    --mem-fraction-static 0.85 \
    --attention-backend flashinfer \
    --enable-cache-report \
    --trust-remote-code
```

Settings used:
- `--context-length` — maximum context length per request; larger values reserve more memory for the KV cache.
- `--mem-fraction-static` — fraction of GPU memory reserved for weights and KV cache. Use `0.85` as a starting point; lower toward `0.7–0.75` on unified-memory hardware or when hitting OOM.
- `--attention-backend flashinfer` — validated attention backend for Blackwell. Prefer this over auto-selected backends that fail CUDA-graph capture on GB300.
- `--enable-cache-report` — populates `usage.prompt_tokens_details.cached_tokens` in OpenAI-style responses for prefix-cache checks.
- `--cap-add SYS_NICE` — allows NUMA affinity; avoids repeated permission warnings in logs.
- `--trust-remote-code` — required for some model families with custom modeling code.

**NVFP4 models (Spark-validated NVIDIA FP4 checkpoints):** add `--quantization modelopt_fp4` to the `sglang serve` arguments.

### Watch startup

```bash
docker logs -f sglang-server
```

Expected output includes model download (first run), CUDA-graph capture, then readiness messages such as Uvicorn listening on port 30000 and the server ready to accept requests.

Or wait for the health endpoint:

```bash
timeout 900 bash -c 'until curl -sf http://localhost:30000/health > /dev/null 2>&1; do sleep 10; done' \
  || { echo "Server failed to start within 900s"; docker logs sglang-server | tail -50; exit 1; }
```

> [!NOTE]
> First launch downloads weights and captures CUDA graphs. Plan for ~10–15 min for `Qwen/Qwen3-8B` and longer for large MoE models before the first successful request. Subsequent starts are faster thanks to cached weights.

## Step 5. Test the API

Send a chat completion request:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Explain quantum computing in simple terms."}],
    "max_tokens": 256
  }'
```

The response should contain a `choices` array with the model's answer.

Optional native `/generate` check:

```bash
curl -f -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
      "text": "What does NVIDIA love?",
      "sampling_params": {
          "temperature": 0.7,
          "max_new_tokens": 100
      }
  }'
```

## Step 6. Multi-turn conversation with prefix caching

SGLang's RadixAttention caches KV entries for processed tokens. Follow-up messages that share the same conversation prefix reuse those entries and skip repeated prefill for previously seen tokens.

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

Check cache reuse in the server logs:

```bash
docker logs sglang-server 2>&1 | grep "cached-token" | tail -10
```

Look for `#cached-token` values greater than 0 on later turns. Treat that as the primary signal of prefix caching; wall-clock `curl` latency alone can be misleading.

> [!NOTE]
> **This prefix-cache check does not apply to hybrid mamba/SSM models** such as `Qwen/Qwen3.6-35B-A3B`. Cross-request prefix reuse is skipped for these architectures and `#cached-token` / `cached_tokens` stay **0** even when radix cache is enabled. To validate prefix caching, use a standard-attention model such as `Qwen/Qwen3-8B`.

## Step 7. Structured JSON output

Generate a schema-constrained response:

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

Parse `choices[0].message.content` — it should be well-formed JSON matching the schema.

## Step 8. (Optional) Benchmark multi-turn throughput

This step uses `assets/benchmark_multiturn.py`, which ships with this playbook. Steps 1–7 run entirely in the container, so clone the playbook repository now if you have not already:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-sglang
```

That directory — the one containing `assets/` — is the **playbook root** for the commands below. Run them from there, in a shell where `MODEL_HANDLE` is exported (re-export it as in Step 2 if you opened a new terminal):

```bash
sudo apt update && sudo apt install -y python3-venv
python3 -m venv .venv && source .venv/bin/activate
pip install requests

python3 assets/benchmark_multiturn.py \
  --base-url http://localhost:30000 \
  --model "$MODEL_HANDLE" \
  --num-conversations 20 \
  --turns-per-conversation 5 \
  --cache-detail-file ./sglang_benchmark_cache_details.log
```

To isolate prefix-cache behavior from multi-client contention, rerun with `--num-conversations 1`. Always correlate with `docker logs` (`#cached-token` lines).

## Step 9. Stop the container

```bash
docker stop sglang-server
docker rm sglang-server
```

Optionally remove the image and cached model. The container downloads weights as root into the mounted hub cache, so the cached model files are root-owned and need `sudo` to delete:

```bash
docker rmi "$SGLANG_IMAGE"
sudo rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"
```

## Next steps

- **More models:** see Overview → Supported models
- **Production:** tune `--mem-fraction-static`, `--context-length`, and concurrency for your workload
- **Offline inference:** see `assets/offline-inference.py` for an in-process Engine example (clone the repository as shown in Step 8 to get it)

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| HuggingFace authentication failure, gated model access denied, or model download hangs/fails | All hardware platforms | Missing/invalid token, restricted model access, or network issue | Export `HF_TOKEN` before running docker; regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens) and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated) if needed |
| CUDA out of memory / server exits with OOM | All hardware platforms | Model or context too large for available memory | Lower `--mem-fraction-static` (for example `0.7`) and/or reduce `--context-length` |
| Server not responding on port 30000 / connection refused | All hardware platforms | Server still loading, or port already in use | Check `docker logs sglang-server`; wait for readiness; or use `-p 30001:30000` if the port is busy |
| `json_schema` response_format returns error | All hardware platforms | Container image missing schema-constrained decoding support | Use `lmsysorg/sglang:latest-cu130` |
| Slow first request after server start | All hardware platforms | Kernel JIT + CUDA-graph capture | Wait for the ready message in logs; subsequent requests are fast |
| `Med cached prefill` / `cached_tokens` is `n/a` or 0 unexpectedly | All hardware platforms | Cache report not enabled, or hybrid mamba/SSM model | Add `--enable-cache-report`; for mamba/SSM models (for example Qwen3.6-35B-A3B), zero cached tokens across requests is expected — validate with a standard-attention model |
| `python3: can't open file 'assets/benchmark_multiturn.py': [Errno 2] No such file or directory` | All hardware platforms | Playbook repository not cloned, or command not run from the playbook root | Clone the repository and `cd` into `nvidia/playbook-sglang` (the directory containing `assets/`) as shown in Instructions → Step 8 |
| `rm: cannot remove '.../.cache/huggingface/hub/models--...': Permission denied` | All hardware platforms | The container downloads weights as root into the mounted hub cache, so cached model files are root-owned | Remove with `sudo rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"` |
| `device >= 0 && device < num_gpus INTERNAL ASSERT FAILED` | DGX Station | `--gpus '"device=N"'` index does not exist | Re-run `nvidia-smi --query-gpu=index,name --format=csv,noheader` and use the GB300 index, or `--gpus all` if there is only one GPU |
| `RuntimeError: ... buildNdTmaDescriptor ... Check failed: false` during CUDA-graph capture | DGX Station | Default `trtllm_mha` attention backend incompatible with GB300 / SM103 | Pass `--attention-backend flashinfer` |
| `AssertionError: FlashAttention v3 Backend requires SM>=80 and SM<=90` | DGX Station | `--attention-backend fa3` on Blackwell SM103 | Use `--attention-backend flashinfer` |
| `User lacks permission to set NUMA affinity` warning | DGX Station | Docker dropped `SYS_NICE` | Add `--cap-add SYS_NICE` to `docker run` |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |
| `deepseek-ai/DeepSeek-V4-*` fails to load | DGX Station | Unsupported in this SGLang build or insufficient memory | Check [SGLang docs](https://docs.sglang.io/) for model support; try Flash before Pro; lower `--mem-fraction-static` and `--context-length` |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> **Monitoring GPU memory with UMA.** Because of unified memory, `nvidia-smi --query-gpu` memory fields may report `N/A`. Use plain `nvidia-smi` instead.

> [!NOTE]
> On DGX Station the GB300 may be at device `0` or `1` depending on configuration. Always verify with `nvidia-smi --query-gpu=index,name --format=csv,noheader` before pinning a device.
