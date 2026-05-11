---
name: dgx-spark-openclaw
description: Run OpenClaw locally on DGX Spark with LM Studio or Ollama — on NVIDIA DGX Spark. Use when setting up openclaw on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/openclaw/README.md -->
# OpenClaw 🦞

> Run OpenClaw locally on DGX Spark with LM Studio or Ollama

OpenClaw (formerly Clawdbot & Moltbot) is a **local-first** AI agent that runs on your machine. It combines multiple capabilities into a single assistant: it remembers conversations, adapts to your usage, runs continuously, uses context from your files and apps, and can be extended with community **skills**.

Running OpenClaw and its LLMs **fully on your DGX Spark** keeps your data private and avoids ongoing cloud API costs. DGX Spark is well suited for this: it runs Linux, is designed to stay on, and has **128GB memory**, so you can run large local models for better accuracy and more capable behavior.

**Outcome**: You will have OpenClaw installed on your DGX Spark and connected to a local LLM (via LM Studio or Ollama). You can use the OpenClaw web UI to chat with your agent, and optionally connect communication channels and skills. The agent and models run entirely on your Spark—no data leaves your machine unless you add cloud or external integrations.

Duration: About 30 minutes for install and first-time model setup; model download time depends on size and network (gpt-oss-120b is ~65GB and may take longer on slower connections). · Risk: **Medium to High**—the agent has access to whatever files, tools, and channels you configure. Risk increases significantly if you enable terminal/command execution skills or connect external accounts. Without proper isolation, this setup could expose sensitive data or allow code execution. **Always follow the security measures above.**

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/openclaw/README.md`
<!-- GENERATED:END -->
