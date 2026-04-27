# LM Studio on DGX Spark

> Deploy LM Studio and serve LLMs on a Spark device; use LM Link to access models remotely.


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [JavaScript](#javascript)
  - [Python](#python)
  - [Bash](#bash)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

LM Studio is an application for discovering, running, and serving large language models entirely on your own hardware. You can run local LLMs like gpt-oss, Qwen3, Gemma3, DeepSeek, and many more models privately and for free.

This playbook shows you how to deploy LM Studio on an NVIDIA DGX Spark device to run LLMs locally with GPU acceleration. Running LM Studio on DGX Spark enables Spark to act as your own private, high-performance LLM server.

**LM Link** (optional) lets you use your Spark’s models from another machine as if they were local. You can link your DGX Spark and your laptop (or other devices) over an end-to-end encrypted connection, so you can load and run models on the Spark from your laptop without being on the same LAN or opening network access. See [LM Link](https://lmstudio.ai/link) and Step 3b in the Instructions.


## What you'll accomplish

You'll deploy LM Studio on an NVIDIA DGX Spark device to run gpt-oss 120B, and use the model from your laptop. More specifically, you will:

- Install **llmster**, a totally headless, terminal native LM Studio on the Spark
- Run LLM inference locally on DGX Spark via API
- Interact with models from your laptop using the LM Studio SDK
- Optionally use **LM Link** to connect Spark and laptop over an encrypted link so remote models appear as local (no same-network or bind setup required)


## What to know before starting

- [Set Up Local Network Access](https://build.nvidia.com/spark/connect-to-your-spark) to your DGX Spark device
- Working with terminal/command line interfaces
- Understanding of REST API concepts

## Prerequisites

**Hardware Requirements:**
- DGX Spark device with ARM64 processor and Blackwell GPU architecture
- Minimum 65GB GPU memory, 70GB or above is recommended
- At least 65GB available storage space, 70GB or above is recommended 

**Software Requirements:**
- NVIDIA DGX OS
- Client device (Mac, Windows, or Linux) 
- Laptop and DGX Spark must be on the same local network
- Network access to download packages and models

## Model support matrix
To explore supported models in LM Studio, check out [LM Studio model catalog](https://lmstudio.ai/models) page.

## LM Link (optional)

[LM Link](https://lmstudio.ai/link) lets you **use your local models remotely**. You link machines (e.g. your DGX Spark and your laptop), then load models on the Spark and use them from the laptop as if they were local.

- **End-to-end encrypted** — Built on Tailscale mesh VPNs; devices are not exposed to the public internet.
- **Works with the local server** — Any tool that connects to LM Studio’s local API (e.g. `localhost:1234`) can use models from your Link, including Codex, Claude Code, OpenCode, and the LM Studio SDK.
- **Preview** — Free for up to 2 users, 5 devices each (10 devices total). Create your Link at [lmstudio.ai/link](https://lmstudio.ai/link).

If you use LM Link, you can skip binding the server to `0.0.0.0` and using the Spark’s IP; once devices are linked, point your laptop at `localhost:1234` and remote models appear in the model loader.

## Ancillary files

All required assets can be found below. These sample scripts can be used in Step 6 of Instructions.

- [run.js](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/js/run.js) - JavaScript script for sending a test prompt to Spark
- [run.py](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/py/run.py) - Python script for sending a test prompt to Spark
- [run.sh](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/bash/run.sh) - Bash script for sending a test prompt to Spark

## Time & risk

* **Estimated time:** 15-30 minutes (including model download time, which may vary depending on your internet connection and the model size)
* **Risk level:** Low
  * Large model downloads may take significant time depending on network speed
* **Rollback:**
  * Downloaded models can be removed manually from the models directory.
  * Uninstall LM Studio or llmster
* **Last Updated:** 04/27/2026
  * Introduce Qwen3.6 35B as example

## Instructions

## Step 1. Install llmster on the DGX Spark

**llmster** is LM Studio's terminal native, headless LM Studio ‘daemon’.

You can install it on servers, cloud instances, machines with no GUI, or just on your computer. This is useful for running LM Studio in headless mode on DGX Spark, then connecting to it from your laptop via the API.

**On your Spark, install llmster by running:**

```bash
curl -fsSL https://lmstudio.ai/install.sh | bash
```

For Windows:
```bash
irm https://lmstudio.ai/install.ps1 | iex
```

Once installed, follow the instructions in your terminal output to add `lms` to your PATH. Interact with LM Studio using the `lms` CLI or the SDK / LM Studio V1 REST API (new with [enhanced features](https://lmstudio.ai/docs/developer/rest)) / OpenAI-compatible REST API.

## Step 2. Download Required Ancillary Files

Run the following curl commands in your local terminal to download files required to complete later steps in this playbook. You may choose from Python, JavaScript, or Bash.

```bash
## JavaScript
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/js/run.js

## Python
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/py/run.py

## Bash
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/bash/run.sh
```

## Step 3. Start the LM Studio API Server

Use `lms`, LM Studio's CLI, to start the server from your terminal. Enable local network access, which allows the LM Studio API server running on your machine to be accessed by all other devices on the same local network (make sure they are trusted devices). To do this, run the following command:

```bash
lms server start --bind 0.0.0.0 --port 1234
```

To test the connectivity between your laptop and your Spark, run the following command in your local terminal

```bash
curl http://<SPARK_IP>:1234/api/v1/models 
```
where `<SPARK_IP>` is your device's IP address. You can find your Spark’s IP address by running this on your Spark:

```bash
hostname -I
```

## Step 3b. (Optional) Connect with LM Link

**LM Link** lets you use your Spark’s models from your laptop (or other devices) as if they were local, over an end-to-end encrypted connection. You don’t need to be on the same local network or bind the server to `0.0.0.0`.

1. **Create a Link** — Go to [lmstudio.ai/link](https://lmstudio.ai/link) and follow **Create your Link** to set up your private LM Link network.
2. **Link both devices** — On your DGX Spark (llmster) and on your laptop, sign in and join the same Link. LM Link uses Tailscale mesh VPNs; devices communicate without opening ports to the internet.
3. **Use remote models** — On your laptop, open LM Studio (or use the local server). Remote models from your Spark appear in the model loader. Any tool that connects to `localhost:1234` — including the LM Studio SDK, Codex, Claude Code, OpenCode, and the scripts in Step 6 — can use those models without changing the endpoint.

LM Link is in **Preview** and is free for up to 2 users, 5 devices each. For details and limits, see [LM Link](https://lmstudio.ai/link).

## Step 4. Download a model to your Spark

As an example, let's download and run gpt-oss 120B, one of the best open source models from OpenAI. This model is too large for many laptops due to memory limitations, which makes this a fantastic use case for the Spark.

```bash
lms get qwen/qwen3.6-35b-a3b
```

This download will take a while due to its large size. Verify that the model has been successfully downloaded by listing your models:

```bash
lms ls
```

## Step 5. Load the model 

Load the model on your Spark so that it is ready to respond to requests from your laptop.

```bash
lms load qwen/qwen3.6-35b-a3b
```

## Step 6. Set up a simple program that uses LM Studio SDK on the laptop

Install the LM Studio SDKs and use a simple script to send a prompt to your Spark and validate the response. To get started quickly, we provide simple scripts below for Python, JavaScript, and Bash. Download the scripts from the Overview page of this playbook and run the corresponding command from the directory containing it.

> [!NOTE]
> Within each script, replace `<SPARK_IP>` with the IP address of your DGX Spark on your local network.

### JavaScript

Pre-reqs: User has installed `npm` and `node`

```bash
npm install @lmstudio/sdk
node run.js
```

### Python

Pre-reqs: User has installed `uv`

```bash
uv run --script run.py
```

### Bash

Pre-reqs: User has installed `jq` and `curl`

```bash
bash run.sh
```

## Step 7. Next Steps

- Try downloading and serving different models from the [LM Studio model catalog](https://lmstudio.ai/models).
- Use [LM Link](https://lmstudio.ai/link) to connect more devices and use your Spark’s models from anywhere with end-to-end encryption.

## Step 8. Cleanup and rollback
Remove and uninstall LM Studio completely if needed. Note that LM Studio stores models separately from the application. Uninstalling LM Studio will not remove downloaded models unless you explicitly delete them.

If you want to remove the entire LM Studio application, quit LM Studio from the tray first, then move the application to trash.

To uninstall llmster, remove the folder `~/.lmstudio/llmster`.

To remove downloaded models, delete the contents of `~/.lmstudio/models/`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| API returns "model not found" error | Model not downloaded or loaded in LM Studio | Run `lms ls` to verify download status, then load model with `lms load {model-name}` |
| `lms` command not found | PATH issue assuming successful installation | Refresh your shell by running `source ~/.bashrc` |
| Model load fails - CUDA out of memory | Model too large for available VRAM | Switch to a smaller model or a different quantization |
| LM Link: devices not connecting or remote models not visible | Devices not in same Link, or LM Link not set up on both | Ensure both Spark and laptop are signed in and joined to the same Link at [lmstudio.ai/link](https://lmstudio.ai/link). Restart LM Studio/llmster after joining. See [LM Link](https://lmstudio.ai/link) for how it works. |


> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
