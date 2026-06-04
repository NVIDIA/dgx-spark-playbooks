# Run models with llama.cpp on DGX Spark

> Build llama.cpp with CUDA and serve models via an OpenAI-compatible API

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a lightweight C/C++ inference stack for large language models. You build it with CUDA so it fully utilizes the DGX Spark GB10 GPU, then load GGUF weights and expose chat through `llama-server`’s OpenAI-compatible HTTP API.

This playbook walks through that stack end to end using MTP-enabled **Qwen3.6-35B-A3B** as the hands-on example. Checkpoint choices and paths for all supported models are summarized in the matrix below; commands are in the instructions.

## What you'll accomplish

You will build llama.cpp with CUDA for GB10, download a **Qwen3.6-35B-A3B** checkpoint, and run **`llama-server`** with GPU offload. You get:

- Local inference through llama.cpp (no separate Python inference framework required)  
- An OpenAI-compatible `/v1/chat/completions` endpoint for tools and apps  
- A concrete validation that the **Qwen3.6-35B-A3B** example runs on this stack on DGX Spark with MTP support.

## What to know before starting

- Basic familiarity with Linux command line and terminal commands  
- Understanding of git and building from source with CMake  
- Basic knowledge of REST APIs and cURL for testing

## Prerequisites

**Hardware requirements**

- NVIDIA DGX Spark with GB10 GPU  
- Sufficient unified memory for the model and the KV-Cache being utilized (about 30GB free RAM for the model in the example)  
- At least **\~40GB** free disk for the example download plus build artifacts (more if you keep multiple GGUFs)

**Software requirements**

- NVIDIA DGX OS  
- Git: `git --version`  
- CMake (3.14+): `cmake --version`  
- CUDA Toolkit: `nvcc --version`  
- Network access to GitHub and Hugging Face

## Model support matrix

DGX Spark supports any GGUF  format model checkpoint with llama.cpp, as long as the system has memory available to host and run the checkpoint.

## Time & risk

* **Estimated time:** About 30 minutes, plus downloading the example GGUF (\~35GB order of magnitude for the default quant)  
* **Risk level:** Low — build is local to your clone; no system-wide installs required for the steps below  
* **Rollback:** Remove the `llama.cpp` clone and the model directory under `~/.cache/huggingface/hub/` to reclaim disk space  
* **Last updated:** 06/03/2026  
  * Walkthrough now uses Qwen3.6-35B-A3B as an example

## Instructions

## Step 1. Install the dependencies

Install the required dependencies:

```shell
sudo apt install -y git clang cmake libcurl4-openssl-dev libssl-dev
```

## Step 2. Clone the llama.cpp repository

Clone upstream llama.cpp—the framework you are building:

```shell
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## Step 3. Build llama.cpp with CUDA

Configure CMake with CUDA and GB10’s **sm\_121** architecture so GGML’s CUDA backend matches your GPU:

```shell
cmake -B build -DGGML_NATIVE=ON -DGGML_CUDA=ON -DGGML_CURL=ON -DGGML_RPC=ON -DCMAKE_CUDA_ARCHITECTURES=121a-real
cmake --build build --config Release -j
```

The build usually takes on the order of 5–10 minutes. When it finishes, binaries such as `llama-server` appear under `build/bin/`.

## Step 4. Start llama-server with a model

llama.cpp loads models in **GGUF** format. This playbook uses the **Q4\_K\_XL** checkpoint from `unsloth/Qwen3.6-35B-A3B-MTP-GGUF`, which provides a good balance between quality and speed on DGX Spark.

From your `llama.cpp/build` directory, launch the OpenAI-compatible server with GPU offload. It will load the model from HuggingFace first if it hasn’t been downloaded before or if there are any updates. 

All models are saved in the default HuggingFace cache directory in \~/.cache/huggingface/hub. For instance, this model will be saved into \~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF

It will also automatically load mmproj file to enable vision capabilities if supported by the model. By default, llama-server will try to fit full model context with ability to serve 4 concurrent requests, but it will adjust parameters automatically if needed.

```shell
./bin/llama-server \
  -hf unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL \
  --host 0.0.0.0 \
  --port 30000
```

To run with MTP speculative decoding, provide additional parameters as shown in the example below. MTP requires a compatible model, like `unsloth/Qwen3.6-35B-A3B-MTP-GGUF` used in this example. The following example also sets “preserve\_thinking” flag that allows Qwen models to use so-called “interleaved thinking” by preserving all prior thinking blocks in the history which can be useful for agentic workflows.

```shell
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
- `--chat-template-kwargs`: sets additional params for the json template parser, must be a valid json object string  
- `--spec-type`: comma-separated list of types of speculative decoding to use (default: none, most MTP-compatible models will use “draft-mtp”, but you need to check the model card first)  
- `--spec-draft-n-max`: number of tokens to draft for speculative decoding (default: 3\)

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

**Keep this terminal open** while testing. Large GGUFs can take a minute or more to load, and initial model download can take a while if the model is not downloaded yet. You will see a progress bar when model is being downloaded.

The server is only ready to accept incoming connections on port 30000 after you see `server is listening` message (see Troubleshooting if `curl` reports connection refused).

## Step 5. Test the API

Use a **second terminal on the same machine** that runs `llama-server` (for example another SSH session into DGX Spark). If you run `curl` on your laptop while the server runs only on Spark, use the Spark hostname or IP instead of `localhost`. 

```shell
curl -X POST http://127.0.0.1:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "unsloth/Qwen3.6-35B-A3B-MTP-GGUF:UD-Q4_K_XL",
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

```shell
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

To remove this tutorial’s artifacts:

```shell
rm -rf ~/llama.cpp
rm -rf ~/.cache/huggingface/hub/models--unsloth--Qwen3.6-35B-A3B-MTP-GGUF
```

## Step 8. Next steps

1. **Context length:** By default, llama.cpp tries to allocate maximum context size supported for the model if possible, but you can also set it manually using `--ctx-size` (or `-c`) to adjust for your needs. For agentic or coding needs you need a minimum of 32768 tokens, preferably 100000 or more.  
2. **Other models:** You can use `--model` to load any compatible GGUF downloaded locally; the llama.cpp server API stays the same. Use `-hf` to let llama.cpp automatically manage downloads/updates. Please note that if you use `--model` with multi-modal models, you need to provide a path to .mmproj file using `--mmproj` parameter. If you use `-hf` it will load the mmproj file automatically.  
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
