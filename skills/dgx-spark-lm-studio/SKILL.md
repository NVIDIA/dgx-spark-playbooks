---
name: dgx-spark-lm-studio
description: Deploy LM Studio and serve LLMs on a Spark device; use LM Link to access models remotely. — on NVIDIA DGX Spark. Use when setting up lm-studio on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/lm-studio/README.md -->
# LM Studio on DGX Spark

> Deploy LM Studio and serve LLMs on a Spark device; use LM Link to access models remotely.

LM Studio is an application for discovering, running, and serving large language models entirely on your own hardware. You can run local LLMs like gpt-oss, Qwen3, Gemma3, DeepSeek, and many more models privately and for free.

This playbook shows you how to deploy LM Studio on an NVIDIA DGX Spark device to run LLMs locally with GPU acceleration. Running LM Studio on DGX Spark enables Spark to act as your own private, high-performance LLM server.

**LM Link** (optional) lets you use your Spark’s models from another machine as if they were local. You can link your DGX Spark and your laptop (or other devices) over an end-to-end encrypted connection, so you can load and run models on the Spark from your laptop without being on the same LAN or opening network access. See [LM Link](https://lmstudio.ai/link) and Step 3b in the Instructions.

**Outcome**: You'll deploy LM Studio on an NVIDIA DGX Spark device to run gpt-oss 120B, and use the model from your laptop. More specifically, you will:

- Install **llmster**, a totally headless, terminal native LM Studio on the Spark
- Run LLM inference locally on DGX Spark via API
- Interact with models from your laptop using the LM Studio SDK
- Optionally use **LM Link** to connect Spark and laptop over an encrypted link so remote models appear as local (no same-network or bind setup required)

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/lm-studio/README.md`
<!-- GENERATED:END -->
