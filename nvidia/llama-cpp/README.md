# Run models with llama.cpp on DGX Spark

> Build llama.cpp with CUDA and serve models via an OpenAI-compatible API (Qwen3.6 as example)


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a lightweight C/C++ inference stack for large language models. You build it with CUDA so tensor work runs on the DGX Spark GB10 GPU, then load GGUF weights and expose chat through `llama-server`’s OpenAI-compatible HTTP API.

This playbook walks through that stack end to end using **Qwen3.6** as the hands-on example: a current-generation family that runs well from quantized GGUF on Spark. Checkpoint choices and paths for all supported models are summarized in the matrix below; commands are in the instructions.

## What you'll accomplish

You will build llama.cpp with CUDA for GB10, download a **Qwen3.6** example checkpoint, and run **`llama-server`** with GPU offload. You get:

- Local inference through llama.cpp (no separate Python inference framework required)
- An OpenAI-compatible `/v1/chat/completions` endpoint for tools and apps
- A concrete validation that the **Qwen3.6** example runs on this stack on DGX Spark

## What to know before starting

- Basic familiarity with Linux command line and terminal commands
- Understanding of git and building from source with CMake
- Basic knowledge of REST APIs and cURL for testing
- Familiarity with Hugging Face Hub for downloading GGUF files

## Prerequisites

**Hardware requirements**

- NVIDIA DGX Spark with GB10 GPU
- Sufficient unified memory for the example **UD-Q4_K_M** MoE checkpoint (weights on the order of **~20GB**, plus KV cache and runtime overhead—scale up if you pick a larger quant or longer context)
- At least **~30GB** free disk for the example download plus build artifacts (more if you keep multiple GGUFs)

**Software requirements**

- NVIDIA DGX OS
- Git: `git --version`
- CMake (3.14+): `cmake --version`
- CUDA Toolkit: `nvcc --version`
- Network access to GitHub and Hugging Face

## Model support matrix

The following models are supported with llama.cpp on Spark. The instructions use the **Qwen3.6** example row by default.

| Model | Support Status | HF Handle |
|-------|----------------|-----------|
| **Qwen3.6-35B-A3B** (example walkthrough) | ✅ | `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf` |
| **Qwen3.6-27B** | ✅ | `unsloth/Qwen3.6-27B-GGUF/Qwen3.6-27B-Q4_K_M.gguf` |
| **Gemma 4 31B IT** | ✅ | `ggml-org/gemma-4-31B-it-GGUF` |
| **Gemma 4 26B A4B IT** | ✅ | `ggml-org/gemma-4-26B-A4B-it-GGUF` |
| **Gemma 4 E4B IT** | ✅ | `ggml-org/gemma-4-E4B-it-GGUF` |
| **Gemma 4 E2B IT** | ✅ | `ggml-org/gemma-4-E2B-it-GGUF` |
| **Nemotron-3-Nano** | ✅ | `unsloth/Nemotron-3-Nano-30B-A3B-GGUF` |

## Time & risk

* **Estimated time:** About 30 minutes, plus downloading the example GGUF (~20GB order of magnitude for the default quant)
* **Risk level:** Low — build is local to your clone; no system-wide installs required for the steps below
* **Rollback:** Remove the `llama.cpp` clone and the model directory under `~/models/` to reclaim disk space
* **Last updated:** 04/27/2026
  * We now walk you through Qwen3.6 first; other models remain in the list

## Instructions

## Step 1. Verify prerequisites

The **example** checkpoint is **`Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`** from Hugging Face repo **`unsloth/Qwen3.6-35B-A3B-GGUF`** (full handle: `unsloth/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf`). The other supported file is **`Qwen3.6-27B-Q4_K_M.gguf`** from **`unsloth/Qwen3.6-27B-GGUF`**—use the same build and server steps, changing `hf download` and `--model` paths (see the [overview model matrix](overview.md)).

Ensure the required tools are installed:

```bash
git --version
cmake --version
nvcc --version
```

All commands should return version information. If any are missing, install them before continuing.

Install the Hugging Face CLI:

```bash
python3 -m venv llama-cpp-venv
source llama-cpp-venv/bin/activate
pip install -U "huggingface_hub[cli]"
```

Verify installation:

```bash
hf version
```

## Step 2. Clone the llama.cpp repository

Clone upstream llama.cpp—the framework you are building:

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## Step 3. Build llama.cpp with CUDA

Configure CMake with CUDA and GB10’s **sm_121** architecture so GGML’s CUDA backend matches your GPU:

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8
```

The build usually takes on the order of 5–10 minutes. When it finishes, binaries such as `llama-server` appear under `build/bin/`.

## Step 4. Download example Qwen3.6-35B-A3B GGUF

llama.cpp loads models in **GGUF** format. This playbook uses the **UD-Q4_K_M** quantized MoE checkpoint from Unsloth, which fits comfortably on DGX Spark GB10 unified memory while keeping strong quality.

```bash
hf download unsloth/Qwen3.6-35B-A3B-GGUF \
  Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --local-dir ~/models/Qwen3.6-35B-A3B-GGUF
```

The file is on the order of **~20GB** (exact size may vary). The download can be resumed if interrupted.

## Step 5. Start llama-server with Qwen3.6-35B-A3B

From your `llama.cpp/build` directory, launch the OpenAI-compatible server with GPU offload:

```bash
./bin/llama-server \
  --model ~/models/Qwen3.6-35B-A3B-GGUF/Qwen3.6-35B-A3B-UD-Q4_K_M.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --threads 8
```

**Parameters (short):**

- `--host` / `--port`: bind address and port for the HTTP API
- `--n-gpu-layers 99`: offload layers to the GPU (adjust if you use a different model)
- `--ctx-size`: context length (can be increased up to model/server limits; uses more memory)
- `--threads`: CPU threads for non-GPU work

You should see log lines similar to:

```
llama_new_context_with_model: n_ctx = 8192
...
main: server is listening on 0.0.0.0:30000
```

**Keep this terminal open** while testing. Large GGUFs can take a minute or more to load; until you see `server is listening`, nothing accepts connections on port 30000 (see Troubleshooting if `curl` reports connection refused).

## Step 6. Test the API

Use a **second terminal on the same machine** that runs `llama-server` (for example another SSH session into DGX Spark). If you run `curl` on your laptop while the server runs only on Spark, use the Spark hostname or IP instead of `localhost`.

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "gemma4",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

If you see `curl: (7) Failed to connect`, the server is still loading, the process exited (check the server log for OOM or path errors), or you are not curling the host that runs `llama-server`.

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
  "model": "Qwen3.6-35B-A3B-UD-Q4_K_M.gguf",
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

## Step 7. Longer completion (with Qwen3.6)

Try a slightly longer prompt to confirm stable generation with **Qwen3.6-35B-A3B**:

```bash
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'
```

## Step 8. Cleanup

Stop the server with `Ctrl+C` in the terminal where it is running.

To remove this tutorial’s artifacts:

```bash
rm -rf ~/llama.cpp
rm -rf ~/models/Qwen3.6-35B-A3B-GGUF
```

Deactivate the Python venv if you no longer need `hf`:

```bash
deactivate
```

## Step 9. Next steps

1. **Context length:** Increase `--ctx-size` for longer chats (watch memory; 1M-token class contexts are possible only when the build, model, and hardware allow).
2. **Other models:** Point `--model` at any compatible GGUF; the llama.cpp server API stays the same.
3. **Integrations:** Point Open WebUI, Continue.dev, or custom clients at `http://<spark-host>:30000/v1` using the OpenAI client pattern.

The server implements the usual OpenAI-style chat features your llama.cpp build enables (including streaming and tool-related flows where supported).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cmake` fails with "CUDA not found" | CUDA toolkit not in PATH | Run `export PATH=/usr/local/cuda/bin:$PATH` and re-run CMake from a clean build directory |
| Build errors mentioning wrong GPU arch | CMake `CMAKE_CUDA_ARCHITECTURES` does not match GB10 | Use `-DCMAKE_CUDA_ARCHITECTURES="121"` for DGX Spark GB10 as in the instructions |
| GGUF download fails or stalls | Network or Hugging Face availability | Re-run `hf download`; it resumes partial files |
| "CUDA out of memory" when starting `llama-server` | Model too large for current context or VRAM | Lower `--ctx-size` (e.g. 4096) or use a smaller quantization from the same repo |
| Server runs but latency is high | Layers not on GPU | Confirm `--n-gpu-layers` is high enough for your model; check `nvidia-smi` during a request |
| `curl: (7) Failed to connect` on port 30000 | No listener yet, wrong host, or crash | Wait for `server is listening`; run `curl` on the same host as `llama-server` (or Spark’s IP); run `ss -tln` and confirm `:30000`; read server stderr for OOM or bad `--model` path |
| Chat API errors or empty replies | Wrong `--model` path or incompatible GGUF | Verify the path to the `.gguf` file; update llama.cpp if the GGUF requires a newer format |

> [!NOTE]
> DGX Spark uses Unified Memory Architecture (UMA), which allows flexible sharing between GPU and CPU memory. Some software is still catching up to UMA behavior. If you hit memory pressure unexpectedly, you can try flushing the page cache (use with care on shared systems):
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For the latest platform issues, see the [DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) documentation.
