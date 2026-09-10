# Serve LLMs with LM Studio

> Local models reachable from your own machine''s tools over an encrypted link

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [JavaScript](#javascript)
  - [Python](#python)
  - [Bash](#bash)
- [Agent-ready Models](#agent-ready-models)
  - [Recommendations by hardware platform](#recommendations-by-hardware-platform)
  - [DGX Spark: Agent-ready Qwen3.6-35B-A3B](#dgx-spark-agent-ready-qwen36-35b-a3b)
  - [Verify from your laptop](#verify-from-your-laptop)
  - [Next steps](#next-steps)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

LM Studio is an application for discovering, running, and serving large language models entirely on your own hardware. You can run local LLMs like gpt-oss, Qwen3, Gemma3, DeepSeek, and many more models privately and for free.

This playbook shows you how to deploy LM Studio on your **hardware platform** to run LLMs locally with GPU acceleration, so the machine acts as your own private, high-performance LLM server.

**LM Link** (optional) lets you use models on your hardware platform from another machine as if they were local. You link the hardware platform and your laptop (or other devices) over an end-to-end encrypted connection, so you can load and run models remotely without being on the same LAN or opening network access. See [LM Link](https://lmstudio.ai/link) and Step 4 in the **Instructions** tab.

## What you'll accomplish

You'll deploy LM Studio on your hardware platform to run **Nemotron 3 Nano Omni** (`nvidia/nemotron-3-nano-omni`), and use the model from your laptop. More specifically, you will:

- Install **llmster**, a headless, terminal-native LM Studio on the hardware platform
- Run LLM inference locally via API
- Interact with models from your laptop using the LM Studio SDK
- Optionally use **LM Link** to connect the hardware platform and laptop over an encrypted link so remote models appear as local (no same-network or bind setup required)

## What to know before starting

**Required:**

- Working with terminal / command line interfaces
- Understanding of REST API concepts
- Terminal or SSH access to your hardware platform

**Optional:**

- Familiarity with local network access / remote SSH to your hardware platform

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended defaults, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS | 128 GB Unified Memory | llmster (`lms`) · API on port `1234` · example model `nvidia/nemotron-3-nano-omni` | — |

> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Minimum 65 GB memory for model inference (70 GB or above recommended for the example model)
- At least 65 GB available storage for model downloads (70 GB or above recommended)

**Software requirements**

- Client device (Mac, Windows, or Linux) to call the API or use LM Link
- For LAN access without LM Link: client device and hardware platform on the same local network
- Network access to download packages and models

## Find model recipes

Browse supported models and download paths on the [LM Studio model catalog](https://lmstudio.ai/models). Use the catalog to explore other models beyond the example in this playbook.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [LM Studio model catalog](https://lmstudio.ai/models) |

Example models validated for the workflow in this playbook:

| Model | Model path |
|-------|------------|
| **Nemotron 3 Nano Omni** | `nvidia/nemotron-3-nano-omni` |
| **Qwen3.6-35B-A3B** | `qwen/qwen3.6-35b-a3b` |
| **GPT-OSS-120B** | `openai/gpt-oss-120b` |

Use the **Instructions** tab for the base workflow. For agentic workloads, see the **Agent-ready Models** tab.

## LM Link (optional)

[LM Link](https://lmstudio.ai/link) lets you **use your local models remotely**. You link machines (for example, your hardware platform and your laptop), then load models on the hardware platform and use them from the laptop as if they were local.

- **End-to-end encrypted** — Built on Tailscale mesh VPNs; devices are not exposed to the public internet.
- **Works with the local server** — Any tool that connects to LM Studio’s local API (for example, `localhost:1234`) can use models from your Link, including Codex, Claude Code, OpenCode, and the LM Studio SDK.
- **Preview** — Free for up to 2 users, 5 devices each (10 devices total). Create your Link at [lmstudio.ai/link](https://lmstudio.ai/link).

If you use LM Link, you can skip binding the server to `0.0.0.0` and using the hardware platform’s IP; once devices are linked, point your laptop at `localhost:1234` and remote models appear in the model loader.

## Ancillary files

Sample scripts for Step 7 of **Instructions** (hosted by LM Studio docs):

- [run.js](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/js/run.js) — JavaScript script for sending a test prompt
- [run.py](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/py/run.py) — Python script for sending a test prompt
- [run.sh](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/bash/run.sh) — Bash script for sending a test prompt

## Time & risk

- **Estimated time:** 30 MIN (including model download time, which may vary depending on your internet connection and the model size)
- **Risk level:** Low
  - Large model downloads may take significant time depending on network speed
- **Rollback:** Downloaded models can be removed manually from the models directory. Uninstall LM Studio or llmster (see Cleanup in the **Instructions** tab).
- **Last Updated:** 07/31/2026
  - Hardware platforms matrix, Find model recipes, and Agent-ready Models for serving LLMs with LM Studio

## Instructions

> [!TIP]
> Ensure your hardware platform is powered on, networked, and reachable over the terminal (local or SSH) before starting.

## Step 1. Install llmster on the hardware platform

**llmster** is LM Studio's terminal-native, headless LM Studio daemon. Install it on servers, cloud instances, machines with no GUI, or your computer. This is useful for running LM Studio in headless mode on your hardware platform, then connecting from your laptop via the API.

**On your hardware platform, install llmster by running:**

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
```

Once installed, follow the instructions in your terminal output to add `lms` to your PATH. Interact with LM Studio using the `lms` CLI or the SDK / LM Studio V1 REST API (new with [enhanced features](https://lmstudio.ai/docs/developer/rest)) / OpenAI-compatible REST API.

## Step 2. Download required ancillary files

Run the following curl commands in your **local** (laptop) terminal to download files required for later steps. Choose Python, JavaScript, or Bash.

```bash
## JavaScript
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/js/run.js

## Python
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/py/run.py

## Bash
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/bash/run.sh
```

## Step 3. Start the LM Studio API server

Use `lms`, LM Studio's CLI, to start the server from your terminal on the hardware platform. Enable local network access so devices on the same trusted local network can reach the API:

```bash
lms server start --bind 0.0.0.0 --port 1234
```

To test connectivity from your laptop, run:

```bash
curl http://<HARDWARE_IP>:1234/api/v1/models
```

where `<HARDWARE_IP>` is your hardware platform's reachable IP address. Find it on the hardware platform with:

```bash
hostname -I
```

## Step 4. (Optional) Connect with LM Link

**LM Link** lets you use models on your hardware platform from your laptop (or other devices) as if they were local, over an end-to-end encrypted connection. You don’t need to be on the same local network or bind the server to `0.0.0.0`.

1. **Create a Link** — Go to [lmstudio.ai/link](https://lmstudio.ai/link) and follow **Create your Link** to set up your private LM Link network.
2. **Link both devices** — On your hardware platform (llmster) and on your laptop, sign in and join the same Link. LM Link uses Tailscale mesh VPNs; devices communicate without opening ports to the internet.
3. **Use remote models** — On your laptop, open LM Studio (or use the local server). Remote models from your hardware platform appear in the model loader. Any tool that connects to `localhost:1234` — including the LM Studio SDK, Codex, Claude Code, OpenCode, and the scripts in Step 7 — can use those models without changing the endpoint.

LM Link is in **Preview** and is free for up to 2 users, 5 devices each. For details and limits, see [LM Link](https://lmstudio.ai/link).

## Step 5. Download a model to your hardware platform

As an example, download **NVIDIA Nemotron 3 Nano Omni** from the LM Studio catalog (`nvidia/nemotron-3-nano-omni`):

```bash
lms get nvidia/nemotron-3-nano-omni
```

This download can take a while due to its size. Verify the download:

```bash
lms ls
```

## Step 6. Load the model

Load the model on your hardware platform so it can respond to requests from your laptop:

```bash
lms load nvidia/nemotron-3-nano-omni
```

## Step 7. Use the LM Studio SDK on the laptop

Install the LM Studio SDKs and use a simple script to send a prompt to your hardware platform and validate the response. Download the scripts from the **Overview** tab (or Step 2) and run the matching command from the directory that contains the script.

> [!NOTE]
> Within each script, replace `<HARDWARE_IP>` (or the script’s placeholder IP) with the IP address of your hardware platform on your local network. If you use LM Link, you can typically use `localhost` instead.

### JavaScript

Prerequisites: `npm` and `node` installed

```bash
npm install @lmstudio/sdk
node run.js
```

### Python

Prerequisites: `uv` installed

```bash
uv run --script run.py
```

### Bash

Prerequisites: `jq` and `curl` installed

```bash
bash run.sh
```

## Step 8. Next steps

- Try downloading and serving different models from the [LM Studio model catalog](https://lmstudio.ai/models).
- For agentic workloads, see the **Agent-ready Models** tab.
- Use [LM Link](https://lmstudio.ai/link) to connect more devices and use models from your hardware platform from anywhere with end-to-end encryption.

## Step 9. Cleanup

Remove and uninstall LM Studio completely if needed. LM Studio stores models separately from the application. Uninstalling LM Studio does not remove downloaded models unless you explicitly delete them.

If you want to remove the entire LM Studio application on a GUI client, quit LM Studio from the tray first, then move the application to trash.

To uninstall llmster on the hardware platform, remove the folder `~/.lmstudio/llmster`.

To remove downloaded models, delete the contents of `~/.lmstudio/models/`.

## Agent-ready Models

## Agent-ready models

Agent-ready models are tuned for **agentic workloads** — tool calling, reasoning traces, and long multi-turn sessions. Use this tab to pick a recommended model for your hardware platform, then return to the **Instructions** tab for server setup and client scripts.

Complete **Steps 1–3** in the **Instructions** tab first (install llmster and start the API server). Then download and load the recommended model below instead of the Nemotron example.

### Recommendations by hardware platform

| Hardware platform | Recommended agent-ready model | LM Studio model path |
| ----------------- | ----------------------------- | -------------------- |
| **DGX Spark** | Agent-ready Qwen3.6-35B-A3B | `qwen/qwen3.6-35b-a3b` |

Only platforms listed in the Supported hardware platforms table (Overview) are listed above.

### DGX Spark: Agent-ready Qwen3.6-35B-A3B

Download and load the recommended agent-ready model on your hardware platform:

```bash
lms get qwen/qwen3.6-35b-a3b
lms load qwen/qwen3.6-35b-a3b
```

Verify the model is available:

```bash
lms ls
curl http://localhost:1234/api/v1/models
```

If you are calling from another machine on the LAN (without LM Link), use `http://<HARDWARE_IP>:1234` instead of `localhost`.

For longer multi-turn agent sessions, load with a larger context window when supported:

```bash
lms load qwen/qwen3.6-35b-a3b --context-length 65536
```

### Verify from your laptop

Use the client scripts from **Instructions** Step 7, or any tool that speaks the LM Studio / OpenAI-compatible API on port `1234`. With **LM Link**, point clients at `localhost:1234` after both devices join the same Link.

### Next steps

- **General serving workflow:** install, server start, and SDK scripts — see the **Instructions** tab
- **More models:** [LM Studio model catalog](https://lmstudio.ai/models)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| API returns "model not found" error | Model not downloaded or loaded in LM Studio | Run `lms ls` to verify download status, then load the model with `lms load {model-name}` |
| `lms` command not found | PATH issue after a successful install | Refresh your shell by running `source ~/.bashrc` (or the shell init file suggested by the installer) |
| Model load fails — CUDA out of memory | Model too large for available memory | Switch to a smaller model or a different quantization |
| LM Link: devices not connecting or remote models not visible | Devices not in the same Link, or LM Link not set up on both | Ensure the hardware platform and laptop are signed in and joined to the same Link at [lmstudio.ai/link](https://lmstudio.ai/link). Restart LM Studio/llmster after joining. See [LM Link](https://lmstudio.ai/link) for how it works. |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
