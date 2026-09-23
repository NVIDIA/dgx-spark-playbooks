# Run Atomic Agent with a Local LLM

> Hand real tasks to an open-source agent that runs entirely on your machine


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Agent-ready Models](#agent-ready-models)
  - [Switch models](#switch-models)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[Atomic Agent](https://atomicagent.io) is an open-source, local-first AI agent that gets real work done on your machine. Describe a task in plain words, from the terminal or from Telegram, and it carries the task out end to end: it works across your browser, files, shell, and git, and asks before anything that changes your system. It remembers your projects across sessions and can run tasks on a schedule.

It is built from the ground up for local models. Every action the model picks is grammar-locked, so even a mid-size model never breaks the format, and the prompt is cache-friendly, so each step stays fast on long tasks. With a local Qwen3.6-35B-A3B, Atomic Agent solves **69.8% of the [GAIA Level 1](https://github.com/AtomicBot-ai/atomic-agent#benchmarks)** real-world tasks.

Setup takes one command. Atomic Agent ships its own `llama.cpp` engine tuned for local models, downloads the model you pick, and runs it on your GPU — there is no separate inference server to configure.

## What you'll accomplish

Install Atomic Agent on your DGX Spark, download a local model, and hand the agent its first task.

- Install Atomic Agent with one command
- Download a local model and run it on the GPU
- Give the agent a task and approve its actions
- Optionally bring over your sessions from Hermes Agent or OpenClaw
- Optionally chat with the agent from Telegram or Discord

## Popular use cases

- **Coding assistant**: Read a repository, write a patch, and run its tests, with the source staying on your machine.
- **Research agent**: Combine web search and your local files to produce reports with personalized context.
- **Install helper**: Search for apps and libraries, run installations, and debug errors from the terminal.
- **Scheduled reports**: Run a task on a schedule, such as a morning digest, and get the result in Telegram.

## What to know before starting

- Basic use of the Linux terminal
- Awareness of the security considerations below

## Important: security and risks

Atomic Agent runs commands and edits files with your user's permissions, and its web tools reach the internet. It asks for your approval before every action that changes something. You cannot eliminate all risk; proceed at your own risk. Read each approval prompt, and start the agent from a dedicated working folder. For details, see [Safety and Observability](https://github.com/AtomicBot-ai/atomic-agent#safety-and-observability).

## Supported hardware platforms

Use the matrix below to confirm your hardware platform and OS.

| Hardware platform | OS | Memory |
| :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory |
| **GeForce RTX** | Linux, Windows | Dedicated VRAM (size varies) |

## Prerequisites

- DGX OS 7 on DGX Spark, or Ubuntu 24.04 with a current NVIDIA driver on GeForce RTX
- Terminal access to the hardware platform (local or SSH)
- Network access to download the agent and a model
- Enough free disk space for your model (see **Agent-ready Models**)

## Time & risk

- **Estimated time:** 30 MIN, not counting the model download
- **Risk level:** **Medium** — the agent can run commands and edit files after you approve them
- **Rollback:** `atomic-agent uninstall` removes the agent, its models, and all of its data
- **Last Updated:** 09/30/2026
  - Added Atomic Agent for DGX Spark and GeForce RTX

## Instructions

## Step 1. Install Atomic Agent

Open a terminal on your DGX Spark and run the official installer, then reload your shell:

```bash
curl -fsSL https://atomicagent.io/install | sh
source ~/.bashrc
```

On Windows, follow the [installation guide](https://atomicagent.io/docs/getting-started/installation/).

## Step 2. Download and start a model

Download Qwen3.6-35B-A3B, the recommended model for DGX Spark, and make it the default:

```bash
atomic-agent models pull qwen-3.6-35b-a3b
atomic-agent models use qwen-3.6-35b-a3b
```

Start the model on the GPU:

```bash
atomic-agent models start
```

The first start also downloads the `llama.cpp` engine, about 550 MB. When the model is ready, the output shows `healthy on port 19091`.

## Step 3. Start the agent

Create a working folder and start the agent in it:

```bash
mkdir -p ~/atomic-workspace
cd ~/atomic-workspace
atomic-agent
```

The agent opens in the terminal, ready for a task. Its file and shell tools work in the folder you start it from.

## Step 4. Give it a first task

Describe what you want in plain words, for example: *Check how much disk space is free and write a short report to disk-report.md.*

The agent plans the steps and runs them. Before it changes anything, it asks for approval: press **Ctrl+Y** to approve or **Ctrl+D** to deny.

## Step 5. Bring over Hermes Agent or OpenClaw (optional)

If you followed the [Hermes Agent](https://build.nvidia.com/spark/hermes-agent) or [OpenClaw](https://build.nvidia.com/spark/openclaw) playbook, bring over your sessions and scheduled jobs:

```bash
atomic-agent import hermes
atomic-agent import openclaw
```

The import never overwrites existing data and does not change the source.

## Optional — Chat from Telegram or Discord

Connect a Telegram or Discord bot to reach the same agent from your phone; approval prompts arrive as buttons in the chat. Run `/integrations` inside the agent and follow the setup, or see the [Telegram guide](https://atomicagent.io/docs/guides/telegram/) and the [Discord setup](https://github.com/AtomicBot-ai/atomic-agent#ways-to-use-it).

For more usage and configuration details, see the [Atomic Agent documentation](https://atomicagent.io/docs/).

## Agent-ready Models

## Agent-ready models

These models handle tool calling, reasoning, and long multi-turn tasks well. Atomic Agent downloads them as quantized GGUF files and runs them with its own `llama.cpp` engine. Run `atomic-agent models list` for the full catalog.

| Hardware platform | Model | ID | Size |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | Qwen3.6-35B-A3B (recommended) | `qwen-3.6-35b-a3b` | 22.4 GB |
| **DGX Spark** | NVIDIA Nemotron 3.5 Lightning 30B-A3B | `nemotron-3.5-30b-a3b` | 19.7 GB |
| **GeForce RTX** (32 GB VRAM) | Qwen3.6-35B-A3B | `qwen-3.6-35b-a3b` | 22.4 GB |
| **GeForce RTX** (24 GB VRAM) | Qwen 3.8 27B | `qwen-3.8-27b` | 17.9 GB |
| **GeForce RTX** (12–16 GB VRAM) | Gemma 4 12B | `gemma-4-12b` | 6.7 GB |
| **GeForce RTX** (8 GB VRAM) | Qwen 3.5 4B | `qwen-3.5-4b` | 2.7 GB |

### Switch models

Download the model, make it the default, and restart the model server:

```bash
atomic-agent models pull nemotron-3.5-30b-a3b
atomic-agent models use nemotron-3.5-30b-a3b
atomic-agent models stop
atomic-agent models start
```

Remove a model you no longer use with `atomic-agent models remove <id>`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| `atomic-agent: command not found` after install | The shell has not reloaded its PATH | Run `source ~/.bashrc`, or call `~/.local/bin/atomic-agent` directly |
| `error while loading shared libraries: libatomic.so.1` | The `libatomic1` library is missing | Run `sudo apt install -y libatomic1` |
| `needs glibc 2.38 or newer` | The Linux system is older than Ubuntu 24.04 or DGX OS 7 | Update the operating system |
| Replies are very slow and `models start` shows no GPU device | The NVIDIA driver is not loaded, so the model runs on the CPU | Fix the driver until `nvidia-smi` works, then run `atomic-agent models stop` and `atomic-agent models start` |
| `atomic-agent models status` shows `health: down` | The model server stopped or failed to load | Run `atomic-agent models start` and check the log at `~/.atomic-agent/models/llama-server.log` |
| `atomic-agent models status` shows a `fault:` line | The model server runs, but requests fail, for example because GPU memory is full | Follow the advice on the `fault:` line. Close other GPU workloads or switch to a smaller model |
| The Telegram bot does not answer in a group | Telegram's privacy mode hides plain messages from bots | @mention the bot, or turn off privacy mode for the bot in @BotFather |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
