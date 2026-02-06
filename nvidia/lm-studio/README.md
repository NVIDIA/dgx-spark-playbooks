# LM Studio on DGX Spark

> Deploy LM Studio and serve LLMs on a Spark device

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Javascript](#javascript)
  - [Python](#python)
  - [Bash](#bash)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

LM Studio is an application for discovering, running, and serving large language models entirely on your own hardware. You can run local LLMs like gpt-oss, Qwen3, Gemma3, DeepSeek, and many more models privately and for free.

This playbook shows you how to deploy LM Studio on an NVIDIA DGX Spark device to run LLMs locally with GPU acceleration. Running LM Studio on DGX Spark enables Spark to act as your own private, high-performance LLM server.


## What you'll accomplish

You'll deploy LM Studio on an NVIDIA DGX Spark device to run gpt-oss 120B, and use the model from your laptop. More specifically, you will:

- Install **llmster**, a totally headless, terminal native LM Studio on the Spark
- Run LLM inference locally on DGX Spark via API
- Interact with models from your laptop using the LM Studio SDK


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

## Ancillary files

All required assets can be found below. These sample scripts can be used in Step 4 of Instructions.

- [run.js](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/js/run.js) - Javascript script for sending a test prompt to Spark
- [run.py](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/py/run.py) - Python script for sending a test prompt to Spark
- [run.sh](https://github.com/lmstudio-ai/docs/blob/main/_assets/nvidia-spark-playbook/bash/run.sh) - Bash script for sending a test prompt to Spark

## Time & risk

* **Estimated time:** 15-30 minutes (including model download time, which may vary depending on your internet connection and the model size)
* **Risk level:** Low
  * Large model downloads may take significant time depending on network speed
* **Rollback:**
  * Downloaded models can be removed manually from the models directory.
  * Uninstall LM Studio or llmster
* **Last Updated:** 02/06/2026
  * First Publication

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
curl -L -O  https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/js/run.js

## Python
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/py/run.py

## Bash
curl -L -O https://raw.githubusercontent.com/lmstudio-ai/docs/main/_assets/nvidia-spark-playbook/bash/run.sh
```

## Step 3. Start the LM Studio API Server

Use `lms`, LM Studio's CLI to start the server from your terminal. Enable local network access, which allows the LM Studio API server running on your machine to be accessed by all other devices on the same local network (make sure they are trusted devices). To do this, run the following command:

```bash
lms server start --bind 0.0.0.0 --port 1234
```

Test the connectivity between your laptop and your Spark, run the following command in your local terminal

```bash
curl http://<SPARK_IP>:1234/api/v1/models 
```
where `<SPARK_IP>` is your device's IP address." You can find your Spark’s IP address by running this on your Spark:

```bash
hostname -I
```

## Step 4. Download a model to your Spark

As an example, let's download and run gpt-oss 120B, one of the best open source models from OpenAI. This model is too large for many laptops due to memory limitations, which makes this a fantastic use case for the Spark.

```bash
lms get openai/gpt-oss-120b
```

This download will take a while due to its large size. Verify that the model has been successfully downloaded by listing your models:

```bash
lms ls
```

## Step 5. Load the model 

Load the model on your Spark so that it is ready to respond to requests from your laptop.

```bash
lms load openai/gpt-oss-120b
```

## Step 6. Set up a simple program that uses LM Studio SDK on the laptop

Install the LM Studio SDKs and use a simple script to send a prompt to your Spark and validate the response. To get started quickly, we provide simple scripts below for Python, Javascript, and Bash. Download the scripts from the Overview page of this playbook and run the corresponding command from the directory containing it.

> [!NOTE]
> Within each script, replace `<SPARK_IP>` with the IP address of your DGX Spark on your local network.

### Javascript

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

Try downloading and serving different models from the [LM Studio model catalog](https://lmstudio.ai/models)

## Step 8. Cleanup and rollback
Remove and uninstall LM Studio completely if needed. Note that LM Studio stores models separately from the application. Uninstalling LM Studio will not remove downloaded models unless you explicitly delete them.

If you want to remove the entire LM Studio application, quit LM Studio from the tray first then move the application to trash.

To uninstall llmster, remove the folder `~/.lmstudio/llmster`.

To remove downloaded models, delete the contents of `~/.lmstudio/models/`.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| API returns "model not found" error | Model not downloaded or loaded in LM Studio | Run `lms ls` to verify download status, then load model with `lms load {model-name}` |
| `lms` command not found | PATH issue assuming successful installation | Refresh your shell by running `source ~/.bashrc` |
| Model load fails - CUDA out of memory | Model too large for available VRAM | Switch to a smaller model or a different quantization |


> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
