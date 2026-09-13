# Serve Models with llama.cpp

> Efficient inference for any GGUF model with minimal dependencies

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendations by hardware platform](#recommendations-by-hardware-platform)
  - [DGX Spark: Agent-ready Qwen3.6-35B-A3B (GGUF, MTP)](#dgx-spark-agent-ready-qwen36-35b-a3b-gguf-mtp)
  - [Verify your server](#verify-your-server)
  - [Next steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a lightweight C/C++ inference stack for large language models. You build it with CUDA so it fully utilizes the GPU on your hardware platform, then load GGUF weights and expose chat through `llama-server`’s OpenAI-compatible HTTP API.

This playbook walks through that stack end to end using MTP-enabled **Qwen3.6-35B-A3B** as the hands-on example. Any GGUF checkpoint that fits in available memory works with the same workflow; commands are in the **Instructions** tab.

## What you'll accomplish

You'll build llama.cpp with CUDA, download a **Qwen3.6-35B-A3B** GGUF checkpoint, and run **`llama-server`** with GPU offload on your **hardware platform**. You get:

- Local inference through llama.cpp (no separate Python inference framework required)
- An OpenAI-compatible `/v1/chat/completions` endpoint for tools and apps
- A concrete validation that the **Qwen3.6-35B-A3B** example runs with MTP support

## What to know before starting

**Required:**

- Basic familiarity with the Linux command line
- Understanding of git and building from source with CMake
- Basic knowledge of REST APIs and cURL for testing

**Optional:**

- Familiarity with Hugging Face model downloads and GGUF quantization tags

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS | 128 GB Unified Memory | Build from source (`CMAKE_CUDA_ARCHITECTURES=121a-real`); example model `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for the model and KV cache (about 30 GB free for the example model)
- At least ~40 GB free disk for the example download plus build artifacts (more if you keep multiple GGUFs)

**Software requirements**

- Git: `git --version`
- CMake (3.14+): `cmake --version`
- CUDA Toolkit: `nvcc --version`
- Network access to GitHub and Hugging Face

## Find model recipes

llama.cpp loads models in **GGUF** format. This playbook’s walkthrough uses the MTP-enabled Qwen3.6-35B-A3B quant below. Browse other GGUF checkpoints on Hugging Face and confirm they fit in available memory on your hardware platform.

| Hardware platform | Example / more models |
| ----------------- | --------------------- |
| **DGX Spark** | Walkthrough: [`unsloth/Qwen3.6-35B-A3B-MTP-GGUF`](https://huggingface.co/unsloth/Qwen3.6-35B-A3B-MTP-GGUF) · Explore GGUF models on [Hugging Face](https://huggingface.co/models?library=gguf) |

Use the **Instructions** tab for the base build-and-serve workflow. For agentic workloads with the recommended Qwen3.6 checkpoint, see the **Agent-ready Models** tab.

> [!NOTE]
> **Memory determines what you can run.** Any GGUF that fits in available memory can be served with the same `llama-server` workflow. If a model is not listed for your hardware platform, check memory headroom and try the base configuration in **Instructions**.

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to the example GGUF download — ~35 GB order of magnitude for the default quant)
- **Risk level:** Low
  - Build is local to your clone; no system-wide installs required for the steps below
  - Large GGUF downloads may fail or stall due to network issues
- **Rollback:** Remove the `llama.cpp` clone and the model directory under `~/.cache/huggingface/hub/` to reclaim disk space
- **Last Updated:** 07/31/2026
  - Build-and-serve walkthrough for llama.cpp with Qwen3.6-35B-A3B (GGUF) and an OpenAI-compatible API

## Instructions

## Step 1. Install the dependencies

Update the package metadata and install the required dependencies:

```bash
sudo apt update
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev
```

## Step 2. Clone the llama.cpp repository

Clone upstream llama.cpp — the framework you are building:

```bash
git clone https://github.com/ggml-org/llama.cpp ~/llama.cpp
cd ~/llama.cpp
```

## Step 3. Build llama.cpp with CUDA

Configure CMake with CUDA so GGML’s CUDA backend matches your hardware platform. For the platforms in this playbook, use CUDA architecture `121a-real`:

```bash
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release --target llama-server -j
```

The build usually takes on the order of 5–10 minutes. When it finishes, `llama-server` appears under `build/bin/`.

## Step 4. Start llama-server with a model

llama.cpp loads models in **GGUF** format. This playbook uses the **Q4_K_XL** checkpoint from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, which provides a good balance between quality and speed on the supported hardware platforms.

From your `llama.cpp/build` directory, launch the OpenAI-compatible server with GPU offload. It loads the model from Hugging Face first if it has not been downloaded before or if there are updates.

All models are saved in the default Hugging Face cache directory under `~/.cache/huggingface/hub`. For example, this model is saved into `~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF`.

It also automatically loads an mmproj file to enable vision capabilities if supported by the model. By default, `llama-server` tries to fit the full model context with the ability to serve 4 concurrent requests, and adjusts parameters automatically if needed.

```bash
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30000
```

To run with MTP speculative decoding, provide additional parameters as shown below. MTP requires a compatible model, like `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` used in this example. The following example also sets the `preserve_thinking` flag so Qwen models can use interleaved thinking by preserving prior thinking blocks in the history — useful for agentic workflows. For a dedicated agentic launch path, see the **Agent-ready Models** tab.

```bash
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30000 \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 3
```

**Parameters (short):**

- `--host` / `--port`: bind address and port for the HTTP API
- `--chat-template-kwargs`: additional params for the JSON template parser; must be a valid JSON object string
- `--spec-type`: comma-separated list of speculative decoding types (default: none; most MTP-compatible models use `draft-mtp` — check the model card)
- `--spec-draft-n-max`: number of tokens to draft for speculative decoding (default: 3)

You should see log lines similar to:

```
0.14.322.968 I srv    load_model: speculative decoding context initialized
0.14.322.970 I slot   load_model: id  0 | task -1 | new slot, n_ctx = 262144
0.14.322.972 I slot   load_model: id  1 | task -1 | new slot, n_ctx = 262144
0.14.322.972 I slot   load_model: id  2 | task -1 | new slot, n_ctx = 262144
0.14.322.973 I slot   load_model: id  3 | task -1 | new slot, n_ctx = 262144
0.14.323.063 I srv    load_model: prompt cache is enabled, size limit: 8192 MiB

...
0.14.342.935 I srv  llama_server: model loaded
0.14.342.939 I srv  llama_server: server is listening on http://0.0.0.0:30000
0.14.342.944 I srv  update_slots: all slots are idle
```

**Keep this terminal open** while testing. Large GGUFs can take a minute or more to load, and the initial model download can take a while if the model is not downloaded yet. You will see a progress bar when the model is being downloaded.

The server is ready to accept connections on port 30000 after you see the `server is listening` message (see Troubleshooting if `curl` reports connection refused).

## Step 5. Test the API

Before sending requests, confirm the server is ready (large models can take several minutes to load):

```bash
timeout 900 bash -c 'until curl -sf http://127.0.0.1:30000/health > /dev/null 2>&1; do sleep 5; done' || exit 1
```

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

If you see `curl: (7) Failed to connect`, the server is still loading, the process exited (check the server log for OOM or path errors), or you are not curling the host that runs `llama-server`.

From another device, use `http://<HARDWARE_IP>:30000/v1` where `<HARDWARE_IP>` is your hardware platform’s reachable address.

Example shape of the response (fields vary by llama.cpp version; `message` may include extra keys):

```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "New York is a great city because it's a living, breathing collage of cultures, ideas, and possibilities—all stacked into one vibrant, never‑sleeping metropolis. Here are just a few reasons that many people ("
      }
    }
  ],
  "created": 1765916539,
  "model": "$MODEL_PATH",
  "object": "chat.completion",
  "usage": {
    "completion_tokens": 100,
    "prompt_tokens": 25,
    "total_tokens": 125
  },
  "id": "chatcmpl-...",
  "timings": {
    ...
  }
}
```

## Step 6. Longer completion (with Qwen3.6-35B-A3B)

Try a slightly longer prompt to confirm stable generation with **Qwen3.6-35B-A3B**:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'
```

## Step 7. Cleanup

Stop the server with `Ctrl+C` in the terminal where it is running.

Cleanup is optional rollback — not a required completion step.

> [!WARNING]
> This permanently deletes the local llama.cpp build and the example model cache.

```bash
rm -rf ~/llama.cpp
rm -rf ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF
```

## Step 8. Next steps

1. **Context length:** By default, llama.cpp tries to allocate the maximum context size supported for the model if possible. Set it manually with `--ctx-size` (or `-c`) to adjust for your needs. For agentic or coding workloads you need a minimum of 32768 tokens, preferably 100000 or more.
2. **Other models:** Use `--model` to load any compatible GGUF downloaded locally; the llama.cpp server API stays the same. Use `-hf` to let llama.cpp manage downloads and updates. If you use `--model` with multi-modal models, provide a path to the `.mmproj` file with `--mmproj`. With `-hf`, the mmproj file loads automatically.
3. **Integrations:** Point Open WebUI, Continue.dev, or custom clients at `http://<HARDWARE_IP>:30000/v1` using the OpenAI client pattern.

The server implements the usual OpenAI-style chat features your llama.cpp build enables (including streaming and tool-related flows where supported).

## Agent-ready Models

## Agent-ready models

Agent-ready models are tuned for **agentic workloads** — tool calling, reasoning traces, and long multi-turn sessions. Use this tab to pick a recommended model for your hardware platform and follow the launch guidance below.

Complete **Steps 1–3** in the **Instructions** tab first (dependencies, clone, and CUDA build). Then launch from your `~/llama.cpp/build` directory with the recipe below.

### Recommendations by hardware platform

| Hardware platform | Recommended agent-ready model | HuggingFace handle | Recipe |
| ----------------- | ----------------------------- | ------------------ | ------ |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B (GGUF, MTP) | `unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL` | [Launch recipe](#dgx-spark-agent-ready-qwen36-35b-a3b-gguf-mtp) |

### DGX Spark: Agent-ready Qwen3.6-35B-A3B (GGUF, MTP)

Agentic serving for the recommended model on unified-memory hardware. MTP speculative decoding and `preserve_thinking` keep interleaved thinking blocks in history for multi-turn agent sessions.

From `~/llama.cpp/build`:

```bash
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30000 \
  --chat-template-kwargs '{"preserve_thinking": true}' \
  --spec-type draft-mtp \
  --spec-draft-n-max 3
```

Settings used:

- `-hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL` — MTP-compatible GGUF with a quality/speed balance for unified memory
- `--chat-template-kwargs '{"preserve_thinking": true}'` — preserves prior thinking blocks for interleaved thinking in agentic workflows
- `--spec-type draft-mtp` / `--spec-draft-n-max 3` — Multi-Token Prediction speculative decoding for faster token generation
- Context: for agentic or coding needs, prefer at least 32768 tokens (preferably 100000+); override with `--ctx-size` / `-c` if needed

### Verify your server

After you launch the recipe above, confirm startup using **Step 5. Test the API** in the **Instructions** tab.

### Next steps

- **General serving workflow:** build, health checks, and API testing — see the **Instructions** tab
- **Other GGUF models:** see **Find model recipes** in the **Overview** tab

## Troubleshooting

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `cmake` fails with "CUDA not found" | All hardware platforms | CUDA toolkit not in PATH | Run `export PATH=/usr/local/cuda/bin:$PATH` and re-run CMake from a clean build directory |
| Build errors mentioning wrong GPU arch | All hardware platforms | CMake `CMAKE_CUDA_ARCHITECTURES` does not match the GPU | Use `-DCMAKE_CUDA_ARCHITECTURES=121a-real` as in the Instructions tab |
| GGUF download fails or stalls | All hardware platforms | Network or Hugging Face availability | Re-run the `-hf` launch (or `hf download`); it resumes partial files |
| "CUDA out of memory" when starting `llama-server` | All hardware platforms | Model too large for current context or available memory | Lower `--ctx-size` (e.g. 4096) or use a smaller quantization from the same repo |
| Server runs but latency is high | All hardware platforms | Layers not on GPU | Confirm `--n-gpu-layers` is high enough for your model; check `nvidia-smi` during a request |
| `curl: (7) Failed to connect` on port 30000 | All hardware platforms | No listener yet, wrong host, or crash | Wait for `server is listening`; run `curl` on the same host as `llama-server` (or use `<HARDWARE_IP>`); run `ss -tln` and confirm `:30000`; read server stderr for OOM or a bad `--model` path |
| Chat API errors or empty replies | All hardware platforms | Wrong `--model` path or incompatible GGUF | Verify the path to the `.gguf` file; update llama.cpp if the GGUF requires a newer format |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache (use with care on shared systems):
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
