# Run Local LLMs

> Chat, agents, and document Q&A with Ollama, LM Studio, and AnythingLLM on your hardware


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Local large language models (LLMs) let you run AI workflows on your own hardware platform. Your prompts, files, and local context can stay on the machine while you experiment with chat assistants, agents, and document-based Q&A — with unlimited on-device access and no usage limits or subscription fees.

The easiest way to get started is to choose a model that fits your GPU memory, then choose the app that matches what you want to do:

- **Chat:** Engage with a local LLM through apps like LM Studio, Ollama, or AnythingLLM to edit or reword text, get answers from the web, or track notes and personal lists.
- **Agents:** Connect an agent such as OpenClaw or Hermes Agent with a local model to handle personal or work tasks with private, local AI.
- **Coding:** Develop applications with coding agents such as OpenCode.
- **Local document chat:** Use tools like AnythingLLM to chat with local documents, notes, and other files.

## What you'll accomplish

You'll pick a model that fits your hardware platform and follow the path that matches your goal — chat, agents, coding, or document Q&A — using desktop apps or dedicated serving playbooks.

- Choose a starting model based on available GPU memory
- Start chatting with LM Studio, Ollama, or llama.cpp
- Point an agent at a local inference server
- Chat with local documents using AnythingLLM

## What to know before starting

**Required:**

- A supported hardware platform with an NVIDIA GPU and enough free storage for model downloads
- Comfort installing desktop apps or following a linked playbook for the path you choose

**Optional:**

- Familiarity with common LLM terms (parameter size, tokens per second, quantization, context window, dense vs. MoE)
- Experience with OpenAI-compatible APIs if you plan to wire agents to a local server

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **RTX or RTX PRO** | Windows / Linux | Dedicated VRAM (size varies by GPU) | Pick a model that fits VRAM; start with LM Studio, Ollama, or llama.cpp for chat | — |

## Choosing the right model for your GPU

In general, use the most powerful model that fits comfortably in your GPU’s memory. Recommended starting models:

- **6–8 GB VRAM:** Qwen 3.5 4B
- **12–16 GB VRAM:** Qwen 3.5 9B / Gemma 4 12B
- **24 GB+ VRAM:** Qwen 3.6 27B

Common LLM terminology:

- **Parameter size:** Number of learnable parameters. Larger models can reason and write better, but need more GPU memory and may run slower. Prefer the largest model that fits comfortably.
- **Tokens (per second):** How fast the model generates output.
- **Quantization:** Lower-precision weights that use less VRAM. Aggressive quantization can reduce response quality. NVFP4 or Q4_K_M are a good balance of throughput, accuracy, and memory.
- **Context window:** How much the model can consider at once (prompt, history, tool outputs, retrieved documents). Longer context helps agentic flows but uses more memory.
- **Dense vs. MoE:** Dense models use all parameters for every token. Mixture-of-experts models activate a smaller subset per token for speed. For a similar total parameter count, a dense model typically offers higher intelligence but lower speed.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Enough dedicated VRAM for your chosen model (see Choosing the right model for your GPU)
- Enough free storage for model downloads (several GB to tens of GB depending on the model)

**Software requirements**

- NVIDIA GPU drivers installed so GPU tools such as `nvidia-smi` work on your hardware platform
- Network access to download desktop apps and models
- For agent workflows: a local inference server URL and port you can point the agent at (see **Instructions**)

## Time & risk

- **Estimated time:** 6 MIN to orient and pick a path; longer once you install an app or follow a linked playbook and download a model
- **Risk level:** Low
  - Large model downloads can fail if network connectivity is unstable
  - Choosing a model larger than available VRAM can cause slow performance or out-of-memory errors
- **Rollback:** Uninstall desktop apps you no longer need and delete downloaded model files from the app’s model library or cache
- **Last Updated:** 08/03/2026
  - Local LLM orientation hub for chat, agents, coding, and document Q&A on RTX or RTX PRO hardware platforms

## Instructions

> [!NOTE]
> This playbook is an **orientation hub**. It does not ship a single install or launch CLI. Use the steps below to choose a path, then follow the linked product docs or playbooks for concrete commands.

## Step 1. Verify GPU access

Confirm the GPU is visible on your hardware platform before you download models.

```bash
nvidia-smi
```

Expected output should show GPU information for your hardware platform, including available memory.

## Step 2. Choose a model that fits your GPU

Use the VRAM guidance in the **Overview** tab:

- **6–8 GB:** Qwen 3.5 4B
- **12–16 GB:** Qwen 3.5 9B / Gemma 4 12B
- **24 GB+:** Qwen 3.6 27B

Prefer the largest model that fits comfortably. Quantized checkpoints (for example NVFP4 or Q4_K_M) reduce memory use when a full-precision model does not fit.

## Step 3. Chat with a desktop app or local server (intended)

The fastest path for drafting, rewriting, summarizing, and testing model quality is a desktop chat app or a lightweight local server:

1. Install [LM Studio](https://lmstudio.ai/), the [Ollama](https://ollama.com/) desktop app, or the [llama.cpp inference server](https://github.com/ggml-org/llama.cpp#llama-server).
2. Search for a model that fits your GPU, download it, and start chatting.

For playbook-style setup of these backends on supported hardware platforms, see:

- [Serve LLMs with LM Studio](https://build.nvidia.com/playbooks/lm-studio)
- [Serve Models with llama.cpp](https://build.nvidia.com/playbooks/llama-cpp)

## Step 4. Connect an agent to a local inference server (intended)

Agents typically need a local inference server, then an agent app pointed at that server’s URL and port.

**Intended loop:**

1. Choose a backend:
   - [LM Studio](https://lmstudio.ai/) or [Ollama](https://ollama.com/) for an easy local server experience (llama.cpp-based apps with current GPU optimizations).
   - For more control: [Serve LLMs with vLLM](https://build.nvidia.com/playbooks/vllm) (Linux) or [Serve Models with llama.cpp](https://build.nvidia.com/playbooks/llama-cpp).
2. Start the server with a large context window when possible (32k tokens or more for agentic flows).
3. Note the URL and port, and confirm the server responds.
4. Install your agent app (for example OpenClaw, Hermes Agent, or OpenCode).
5. During onboarding, set the model provider to your local inference server URL.
6. Finish setup and test with a simple prompt.

Playbook guides for agent paths:

- [Run OpenClaw with a Local LLM](https://build.nvidia.com/playbooks/openclaw)
- [Run Hermes Agent with a Local LLM](https://build.nvidia.com/playbooks/hermes-agent)

## Step 5. Chat with local documents (intended)

Use [AnythingLLM](https://anythingllm.com/) to connect a local model to your documents, notes, and files. Typical uses include generating flashcards from slides, asking questions grounded in your materials, creating practice quizzes, and walking through problems step by step.

Install AnythingLLM from the product site, add your documents, and select a local model or local server endpoint that fits your GPU. This playbook does not ship AnythingLLM install commands.

## Step 6. Cleanup (optional)

If you installed apps or downloaded models only for evaluation:

1. Quit the chat app, agent, or inference server
2. Remove models from the app’s library or delete cached model files you no longer need
3. Uninstall desktop apps you do not plan to keep

Cleanup is optional.

## Step 7. Next steps

- Follow a dedicated serving playbook when you need production-style endpoints: [Serve LLMs with vLLM](https://build.nvidia.com/playbooks/vllm), [Serve LLMs with LM Studio](https://build.nvidia.com/playbooks/lm-studio), or [Serve Models with llama.cpp](https://build.nvidia.com/playbooks/llama-cpp)
- Build an agent workflow with [Run OpenClaw with a Local LLM](https://build.nvidia.com/playbooks/openclaw) or [Run Hermes Agent with a Local LLM](https://build.nvidia.com/playbooks/hermes-agent)
- Browse the NVIDIA blog guide linked under **Resources** for additional product context

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvidia-smi` not found or no GPU listed | Drivers or GPU runtime not available on this hardware platform | Install or repair NVIDIA drivers for your hardware platform, then re-run `nvidia-smi` |
| Chat app or server is very slow, or generation fails with out-of-memory errors | Model size, context length, or concurrent GPU workloads exceed dedicated VRAM | Close other GPU apps, pick a smaller or more aggressively quantized model, shorten the context window, and confirm free memory with `nvidia-smi` |
| Agent cannot reach the local model | Inference server not running, wrong URL/port, or bound only to another interface | Confirm the server is running, use the URL/port shown by the backend, and keep the endpoint on localhost or an authenticated remote path |
| Document chat answers ignore uploaded files | Documents not indexed, or the app is not using the local knowledge base | Re-index or re-upload documents in AnythingLLM (or your chosen app) and confirm the active workspace uses that knowledge base |
| Desktop app install or model download fails | Network interruption or insufficient disk space | Check connectivity and free storage, then retry the download from the app or model library |

> [!NOTE]
> On discrete-GPU hardware platforms, GPU memory is separate from system RAM. CUDA out-of-memory usually means the workload exceeds VRAM: reduce model size, sequence length, or precision; enable CPU offloading only if your backend supports it; close other GPU applications. Use `nvidia-smi` to confirm no other process is holding memory.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
