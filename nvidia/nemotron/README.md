# Nemotron-3-Nano with llama.cpp

> Run Nemotron-3-Nano-30B model using llama.cpp on DGX Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Nemotron-3-Nano-30B-A3B is NVIDIA's powerful language model featuring a 30 billion parameter Mixture of Experts (MoE) architecture with only 3 billion active parameters. This efficient design enables high-quality inference with lower computational requirements, making it ideal for DGX Spark's GB10 GPU.

This playbook demonstrates how to run Nemotron-3-Nano using llama.cpp, which compiles CUDA kernels at build time specifically for your GPU architecture. The model includes built-in reasoning (thinking mode) and tool calling support via the chat template.

## What you'll accomplish

You will have a fully functional Nemotron-3-Nano-30B-A3B inference server running on your DGX Spark, accessible via an OpenAI-compatible API. This setup enables:

- Local LLM inference
- OpenAI-compatible API endpoint for easy integration with existing tools
- Built-in reasoning and tool calling capabilities

## What to know before starting

- Basic familiarity with Linux command line and terminal commands
- Understanding of git and working with branches
- Experience building software from source with CMake
- Basic knowledge of REST APIs and cURL for testing
- Familiarity with Hugging Face Hub for model downloads

## Prerequisites

**Hardware Requirements:**
- NVIDIA DGX Spark with GB10 GPU
- At least 40GB available GPU memory (model uses ~38GB VRAM)
- At least 50GB available storage space for model downloads and build artifacts

**Software Requirements:**
- NVIDIA DGX OS
- Git: `git --version`
- CMake (3.14+): `cmake --version`
- CUDA Toolkit: `nvcc --version`
- Network access to GitHub and Hugging Face

## Time & risk

* **Estimated time:** 30 minutes (including model download of ~38GB)
* **Risk level:** Low
  * Build process compiles from source but doesn't modify system files
  * Model downloads can be resumed if interrupted
* **Rollback:** Delete the cloned `llama.cpp` directory and downloaded model files to fully remove the installation
* **Last Updated:** 12/17/2025
  * First Publication

## Instructions

## Step 1. Verify prerequisites

Ensure you have the required tools installed on your DGX Spark before proceeding.

```bash
git --version
cmake --version
nvcc --version
```

All commands should return version information. If any are missing, install them before continuing.

Install the Hugging Face CLI:

```bash
python3 -m venv nemotron-venv
source nemotron-venv/bin/activate
pip install -U "huggingface_hub[cli]"
```

Verify installation:

```bash
hf version
```

## Step 2. Clone llama.cpp repository

Clone the llama.cpp repository which provides the inference framework for running Nemotron models.

```bash
git clone https://github.com/ggml-org/llama.cpp
cd llama.cpp
```

## Step 3. Build llama.cpp with CUDA support

Build llama.cpp with CUDA enabled and targeting the GB10's sm_121 compute architecture. This compiles CUDA kernels specifically optimized for your DGX Spark GPU.

```bash
mkdir build && cd build
cmake .. -DGGML_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES="121" -DLLAMA_CURL=OFF
make -j8
```

The build process takes approximately 5-10 minutes. You should see compilation progress and eventually a successful build message.

## Step 4. Download the Nemotron GGUF model

Download the Q8 quantized GGUF model from Hugging Face. This model provides excellent quality while fitting within the GB10's memory capacity.

```bash
hf download unsloth/Nemotron-3-Nano-30B-A3B-GGUF \
  Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
  --local-dir ~/models/nemotron3-gguf
```

This downloads approximately 38GB. The download can be resumed if interrupted.

## Step 5. Start the llama.cpp server

Launch the inference server with the Nemotron model. The server provides an OpenAI-compatible API endpoint.

```bash
./bin/llama-server \
  --model ~/models/nemotron3-gguf/Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf \
  --host 0.0.0.0 \
  --port 30000 \
  --n-gpu-layers 99 \
  --ctx-size 8192 \
  --threads 8
```

**Parameter explanation:**
- `--host 0.0.0.0`: Listen on all network interfaces
- `--port 30000`: API server port
- `--n-gpu-layers 99`: Offload all layers to GPU
- `--ctx-size 8192`: Context window size (can increase up to 1M)
- `--threads 8`: CPU threads for non-GPU operations

You should see server startup messages indicating the model is loaded and ready:
```
llama_new_context_with_model: n_ctx = 8192
...
main: server is listening on 0.0.0.0:30000
```

## Step 6. Test the API

Open a new terminal and test the inference server using the OpenAI-compatible chat completions endpoint.

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

Expected response format:
```json
{
  "choices": [
    {
      "finish_reason": "length",
      "index": 0,
      "message": {
        "role": "assistant",
        "reasoning_content": "We need to respond to user statement: \"New York is a great city because...\". Probably they want continuation, maybe a discussion. It's a simple open-ended prompt. Provide reasons why New York is great. No policy issues. Just respond creatively.",
        "content": "New York is a great city because it's a living, breathing collage of cultures, ideas, and possibilities—all stacked into one vibrant, never‑sleeping metropolis. Here are just a few reasons that many people ("
      }
    }
  ],
  "created": 1765916539,
  "model": "Nemotron-3-Nano-30B-A3B-UD-Q8_K_XL.gguf",
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

## Step 7. Test reasoning capabilities

Nemotron-3-Nano includes built-in reasoning capabilities. Test with a more complex prompt:

```bash
curl http://localhost:30000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron",
    "messages": [{"role": "user", "content": "Solve this step by step: If a train travels 120 miles in 2 hours, what is its average speed?"}],
    "max_tokens": 500
  }'
```

The model will provide a detailed reasoning chain before giving the final answer.

## Step 8. Cleanup

To stop the server, press `Ctrl+C` in the terminal where it's running.

To completely remove the installation:

```bash
## Remove llama.cpp build
rm -rf ~/llama.cpp

## Remove downloaded models
rm -rf ~/models/nemotron3-gguf
```

## Step 9. Next steps

1. **Increase context size**: For longer conversations, increase `--ctx-size` up to 1048576 (1M tokens), though this will use more memory
3. **Integrate with applications**: Use the OpenAI-compatible API with tools like Open WebUI, Continue.dev, or custom applications

The server supports the full OpenAI API specification including streaming responses, function calling, and multi-turn conversations.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `cmake` fails with "CUDA not found" | CUDA toolkit not in PATH | Run `export PATH=/usr/local/cuda/bin:$PATH` and retry |
| Model download fails or is interrupted | Network issues | Re-run the `hf download` command - it will resume from where it stopped |
| "CUDA out of memory" when starting server | Insufficient GPU memory | Reduce `--ctx-size` to 4096 or use a smaller quantization (Q4_K_M) |
| Server starts but inference is slow | Model not fully loaded to GPU | Verify `--n-gpu-layers 99` is set and check `nvidia-smi` shows GPU usage |
| "Connection refused" on port 30000 | Server not running or wrong port | Verify server is running and check the `--port` parameter |
| "model not found" in API response | Wrong model path | Verify the model path in `--model` parameter matches the downloaded file location |


> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
