---
name: dgx-spark-llama-cpp
description: Build llama.cpp with CUDA and serve models via an OpenAI-compatible API (Nemotron 3 Nano Omni as example) — on NVIDIA DGX Spark. Use when setting up llama-cpp on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/llama-cpp/README.md -->
# Run models with llama.cpp on DGX Spark

> Build llama.cpp with CUDA and serve models via an OpenAI-compatible API (Nemotron 3 Nano Omni as example)

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a lightweight C/C++ inference stack for large language models. You build it with CUDA so tensor work runs on the DGX Spark GB10 GPU, then load GGUF weights and expose chat through `llama-server`’s OpenAI-compatible HTTP API.

This playbook walks through that stack end to end using **Nemotron 3 Nano Omni** as the hands-on example: an NVIDIA MoE family that runs well from quantized GGUF on Spark. Checkpoint choices and paths for all supported models are summarized in the matrix below; commands are in the instructions.

**Outcome**: You will build llama.cpp with CUDA for GB10, download a **Nemotron 3 Nano Omni** example checkpoint, and run **`llama-server`** with GPU offload. You get:

- Local inference through llama.cpp (no separate Python inference framework required)
- An OpenAI-compatible `/v1/chat/completions` endpoint for tools and apps
- A concrete validation that the **Nemotron 3 Nano Omni** example runs on this stack on DGX Spark

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/llama-cpp/README.md`
<!-- GENERATED:END -->
