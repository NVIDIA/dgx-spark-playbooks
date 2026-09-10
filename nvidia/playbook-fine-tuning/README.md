# Fine-Tune Specialized LLMs with Unsloth

> LoRA, full training, and RL paths plus Nemotron open models

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Fine-tuning gives a language model a focused training session on examples tied to a specific topic or workflow. The model improves accuracy by learning new patterns and adapting to the task — for example, tuning a chatbot for product-support questions or building a personal assistant that manages a schedule.

[Unsloth](https://unsloth.ai/) is a widely used open-source framework for fine-tuning LLMs. It is optimized for efficient, low-memory training on NVIDIA GPUs and helps boost Hugging Face Transformers training performance on those GPUs. Another strong starting point is the [NVIDIA Nemotron 3](https://nvidianews.nvidia.com/news/nvidia-debuts-nemotron-3-family-of-open-models) family of open models, data, and libraries — efficient open models suited to agentic AI fine-tuning.

Choosing a fine-tuning method depends on how much of the original model you want to adjust:

**Parameter-efficient fine-tuning (such as LoRA or QLoRA)**

- **How it works:** Updates only a small portion of the model for faster, lower-cost training without altering it drastically.
- **Target use case:** Domain knowledge, coding accuracy, legal or scientific adaptation, reasoning refinement, or tone and behavior alignment.
- **Requirements:** Small- to medium-sized dataset (about 100–1,000 prompt–sample pairs).

**Full fine-tuning**

- **How it works:** Updates all of the model’s parameters — useful when the model must follow specific formats or styles.
- **Target use case:** Advanced agents and chatbots that must stay on a topic, respect guardrails, and respond in a particular manner.
- **Requirements:** Large dataset (1,000+ prompt–sample pairs).

**Reinforcement learning**

- **How it works:** Adjusts behavior with feedback or preference signals. The model learns by interacting with an environment and improving from that feedback over time. This advanced technique interweaves training and inference and can be combined with parameter-efficient or full fine-tuning. See [Unsloth's Reinforcement Learning Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide) for details.
- **Target use case:** Higher accuracy in a domain such as law or medicine, or autonomous agents that orchestrate actions on a user’s behalf.
- **Requirements:** An action model, a reward model, and an environment for the model to learn from.

VRAM required also varies by method. Unsloth translates heavy matrix workloads into efficient custom GPU kernels so fine-tuning completes more quickly with lower memory use. Unsloth publishes guides for LLM configurations, hyperparameters, notebooks, and step-by-step workflows, including:

- [Fine-Tuning LLMs With NVIDIA RTX 50 Series GPUs and Unsloth](https://docs.unsloth.ai/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth)
- [Fine-Tune Faster with Unsloth](https://build.nvidia.com/playbooks/unsloth) (install and validation path on supported hardware platforms)

For a deep dive into fine-tuning and reinforcement learning on the NVIDIA Blackwell platform, read the [NVIDIA technical blog](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/). For a hands-on local walkthrough, watch [Matthew Berman](https://www.youtube.com/@matthew_berman) run reinforcement learning on an NVIDIA GeForce RTX 5090 with Unsloth in this [video](https://youtu.be/9t-BAjzBWj8).

**NVIDIA Nemotron 3 family of open models**

Nemotron 3 — in Nano, Super, and Ultra sizes — uses a hybrid latent Mixture-of-Experts (MoE) architecture for efficient open models with strong accuracy for agentic applications.

Nemotron 3 Nano 30B-A3B is the most compute-efficient model in the lineup. It is suited to software debugging, content summarization, AI assistant workflows, and information retrieval at low inference cost. Its hybrid MoE design delivers:

- Up to 60% fewer reasoning tokens, reducing inference cost
- A 1 million-token context window for long, multistep tasks

Nemotron 3 Super targets high-accuracy reasoning for multi-agent applications; Nemotron 3 Ultra targets complex AI applications. NVIDIA also released an open collection of training datasets and reinforcement learning libraries. Nemotron 3 Nano fine-tuning is available on Unsloth.

Download Nemotron 3 Nano from [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8), or experiment with it through Llama.cpp and LM Studio.

## What you'll accomplish

You'll understand how to choose among parameter-efficient fine-tuning (LoRA / QLoRA), full fine-tuning, and reinforcement learning for specialized agentic tasks on your **hardware platform**, and where to continue with Unsloth guides, Nemotron open models, and a runnable Unsloth setup playbook.

## What to know before starting

**Required:**

- Basic understanding of large language models (prompts, tokens, training vs inference)
- Familiarity with why domain- or task-specific adaptation can improve model behavior

**Optional:**

- Experience with Hugging Face Transformers, datasets, or LoRA / QLoRA
- Comfort with the Linux command line (helpful when you move to a runnable Unsloth install playbook)
- Familiarity with reinforcement learning concepts (reward models, preference data)

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **RTX or RTX PRO** | Ubuntu 22.04 / 24.04 (Linux); Windows / WSL where your local stack supports it | Dedicated VRAM (size varies by GPU) | Unsloth fine-tuning via upstream guides; see [Fine-Tune Faster with Unsloth](https://build.nvidia.com/playbooks/unsloth) for an install path on listed platforms | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Enough GPU memory (VRAM) for the model size and fine-tuning method you choose
- Sufficient free storage for model weights, datasets, and checkpoints

**Software requirements**

- Terminal access to the hardware platform (local or SSH) when you follow a runnable install path
- Network access to download models, datasets, and Unsloth or related packages
- GPU access verified with `nvidia-smi` before any training run

## Time & risk

- **Estimated time:** 8 MIN (reading this overview; hands-on fine-tuning time varies widely by method, model, and dataset)
- **Risk level:** Low for this overview
  - Hands-on fine-tuning can fail on network limits during large downloads
  - Training can hit out-of-memory errors if model size, sequence length, or batch size exceed available VRAM
- **Rollback:** No local changes are required to read this overview. If you later install packages or download models, remove those environments and assets when you no longer need them.
- **Last Updated:** 08/03/2026
  - Overview of LoRA / QLoRA, full fine-tuning, and RL with Unsloth and Nemotron 3 open models on supported hardware platforms

## Instructions

## Step 1. Verify GPU access

Confirm GPU access on your hardware platform before you continue with any fine-tuning workflow.

```bash
nvidia-smi
```

Expected output should show GPU information for your hardware platform.

> [!NOTE]
> Concrete Unsloth install, launch, and training commands are **not included in this playbook**. This page is a methods overview (LoRA / QLoRA, full fine-tuning, and reinforcement learning) plus pointers to Unsloth and Nemotron resources. For a runnable Unsloth setup and validation path on platforms listed there, use [Fine-Tune Faster with Unsloth](https://build.nvidia.com/playbooks/unsloth). For RTX 50 Series GPU guidance from Unsloth, see [Fine-Tuning LLMs With NVIDIA RTX 50 Series GPUs and Unsloth](https://docs.unsloth.ai/basics/fine-tuning-llms-with-blackwell-rtx-50-series-and-unsloth).

## Step 2. Choose a fine-tuning method (intended)

Match the method to your goal and data size:

1. **Parameter-efficient fine-tuning (LoRA / QLoRA)** — small- to medium-sized datasets; lower VRAM and faster iteration
2. **Full fine-tuning** — large datasets when you need broader parameter updates for format, style, or guardrail behavior
3. **Reinforcement learning** — preference or reward-driven behavior change; see [Unsloth's Reinforcement Learning Guide](https://docs.unsloth.ai/get-started/reinforcement-learning-rl-guide)

This playbook does not ship hyperparameters, training scripts, or a dependency list for those runs.

## Step 3. Pick a starting model (intended)

When you are ready to train, choose an open model that fits your VRAM and task. Nemotron 3 Nano 30B-A3B is one option optimized for agentic fine-tuning workloads; download it from [Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8) when you follow an upstream Unsloth or training guide.

Confirm license terms and any gated-model access before downloading.

## Step 4. Follow an upstream Unsloth workflow (intended)

When you move from this overview to hands-on training:

1. Use [Fine-Tune Faster with Unsloth](https://build.nvidia.com/playbooks/unsloth) if that playbook lists your hardware platform
2. Or follow Unsloth’s published guides and notebooks for your GPU generation
3. Keep other heavy GPU workloads stopped so training has enough memory
4. Monitor utilization with `nvidia-smi` during training

Do not invent local package pins or container tags from this playbook — use the linked playbook or Unsloth docs for current install steps.

## Step 5. Cleanup (optional)

If you created local environments, checkpoints, or downloaded models while following linked Unsloth or Nemotron guidance:

1. Stop any training or notebook session you started
2. Deactivate or remove environments you no longer need
3. Delete downloaded models, datasets, or checkpoints you do not want to keep

Cleanup is optional.

## Step 6. Next steps

- Continue with [Fine-Tune Faster with Unsloth](https://build.nvidia.com/playbooks/unsloth) for install and validation on listed platforms
- Review [Unsloth Documentation](https://docs.unsloth.ai/) for current configuration and hyperparameter guidance
- Read the [NVIDIA technical blog on Unsloth and Blackwell](https://developer.nvidia.com/blog/train-an-llm-on-an-nvidia-blackwell-desktop-with-unsloth-and-scale-it/)
- Explore [Nemotron 3 Nano on Hugging Face](https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-FP8)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvidia-smi` not found or no GPU listed | Drivers or GPU runtime not available on this hardware platform | Install or repair NVIDIA drivers for your hardware platform, then re-run `nvidia-smi` |
| Out-of-memory errors during fine-tuning | Model size, sequence length, batch size, or concurrent GPU workloads exceed available VRAM | Close other GPU applications, reduce batch size or sequence length, prefer LoRA / QLoRA over full fine-tuning for a first run, and monitor with `nvidia-smi` |
| Unsloth or training package install fails | Missing dependencies, incompatible CUDA/PyTorch stack, or following steps that are not published for this playbook | Use [Fine-Tune Faster with Unsloth](https://build.nvidia.com/playbooks/unsloth) if your platform is listed there, or follow current [Unsloth Documentation](https://docs.unsloth.ai/) for your GPU generation — this playbook does not ship an install path |

> [!NOTE]
> On discrete-GPU hardware platforms, GPU memory is separate from system RAM. CUDA out-of-memory usually means the workload exceeds VRAM: reduce batch size, sequence length, or model precision; enable CPU offloading only if supported; close other GPU applications. Use `nvidia-smi` to confirm no other process is holding memory.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
