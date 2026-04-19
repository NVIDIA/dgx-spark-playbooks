---
name: dgx-spark-llama-cpp
description: Build llama.cpp with CUDA and serve models via an OpenAI-compatible API (Gemma 4 31B IT as example) — on NVIDIA DGX Spark. Use when setting up llama-cpp on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/llama-cpp/README.md -->
# Run models with llama.cpp on DGX Spark

> Build llama.cpp with CUDA and serve models via an OpenAI-compatible API (Gemma 4 31B IT as example)

[llama.cpp](https://github.com/ggml-org/llama.cpp) is a lightweight C/C++ inference stack for large language models. You build it with CUDA so tensor work runs on the DGX Spark GB10 GPU, then load GGUF weights and expose chat through `llama-server`’s OpenAI-compatible HTTP API.

This playbook walks through that stack end to end. As the model example, it uses **Gemma 4 31B IT** - a frontier reasoning model built by Google DeepMind that llama.cpp supports, with strengths in coding, agentic workflows, and fine-tuning. The instructions download its **F16** GGUF from Hugging Face. The same build and server steps apply to other GGUFs (including other sizes in the support matrix below).

**Outcome**: You will build llama.cpp with CUDA for GB10, download a Gemma 4 31B IT model checkpoint, and run **`llama-server`** with GPU offload. You get:

- Local inference through llama.cpp (no separate Python inference framework required)
- An OpenAI-compatible `/v1/chat/completions` endpoint for tools and apps
- A concrete validation that **Gemma 4 31B IT** runs on this stack on DGX Spark

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/llama-cpp/README.md`
<!-- GENERATED:END -->
