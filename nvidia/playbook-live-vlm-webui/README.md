# Stream Real-Time Video to a Vision Language Model

> A webcam-based testbed for comparing VLMs across backends

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Live VLM WebUI is a browser interface for real-time vision language model (VLM) interaction and benchmarking. It streams video from a webcam to an OpenAI-compatible VLM backend and displays live analysis alongside inference latency and system utilization.

The interface supports local backends such as Ollama, vLLM, SGLang, and NVIDIA NIM, as well as compatible cloud APIs. This playbook uses Ollama for a straightforward local setup.

## What you'll accomplish

You'll create a real-time video analysis environment on your **hardware platform** that can:

- Stream webcam video to a locally served VLM
- Apply preset or custom prompts to incoming frames
- Display model responses, inference latency, and system utilization
- Compare supported vision models through one browser interface

## What to know before starting

**Required:**

- Basic Linux command-line experience
- Familiarity with browser permissions and local network addresses

**Optional:**

- Familiarity with REST APIs
- Basic knowledge of vision language models

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB unified memory | Ollama backend with `llama3.2-vision:11b`; Live VLM WebUI installed with `pipx` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see the matrix above
- A webcam available to the browser
- Sufficient available storage for Python packages and the selected VLM

**Software requirements**

- Python 3.10 or later: `python3 --version`
- Network access to download packages and models
- A Chromium-based browser or Firefox with access to `https://<HARDWARE_IP>:8090`
- Direct network connectivity between the browser and hardware platform for WebRTC

## Find model recipes

Browse models supported by each backend in the [Live VLM WebUI list of vision-language models](https://github.com/NVIDIA-AI-IOT/live-vlm-webui/blob/main/docs/usage/list-of-vlms.md).

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [Vision models available through Ollama](https://ollama.com/search?c=vision) |

Use the **Instructions** tab for the base Ollama workflow. Model memory and latency requirements vary, so confirm that another model fits available memory before downloading it.

## Ancillary files

No local ancillary files are required. The application and supporting documentation are available in the [Live VLM WebUI repository](https://github.com/NVIDIA-AI-IOT/live-vlm-webui).

## Time & risk

- **Estimated time:** 30 MIN (longer on first run due to model download)
- **Risk level:** Low
  - Port 8090 must be reachable from the browser
  - The self-signed HTTPS certificate requires a one-time browser exception
  - Model downloads consume storage and network bandwidth
- **Rollback:** Stop the application, then optionally uninstall it with `pipx uninstall live-vlm-webui`. This does not remove Ollama or downloaded models.
- **Last Updated:** 08/03/2026
  - Added current Live VLM WebUI setup, supported hardware guidance, and links for finding compatible vision models

## Instructions

## Step 1. Install and verify Ollama

Install Ollama as the local VLM backend:

```bash
curl -fsSL https://ollama.com/install.sh | sh
ollama --version
```

Confirm that the Ollama service is running and its OpenAI-compatible endpoint is available:

```bash
sudo systemctl enable --now ollama
curl http://localhost:11434/v1/models
```

Expected output includes a JSON response with a `data` array. The array can be empty until you download a model.

## Step 2. Download a vision model

Download the model used by this workflow:

```bash
ollama pull llama3.2-vision:11b
curl http://localhost:11434/v1/models
```

The second command should list `llama3.2-vision:11b`. The model download can take several minutes depending on network speed.

## Step 3. Install Live VLM WebUI

Install `pipx`, add its application directory to your shell path, and install Live VLM WebUI in an isolated Python environment:

```bash
sudo apt update
sudo apt install -y pipx
pipx ensurepath
source ~/.bashrc
pipx install live-vlm-webui
```

Verify the installation:

```bash
command -v live-vlm-webui
```

Expected output should show the path to the `live-vlm-webui` executable.

## Step 4. Start Live VLM WebUI

Launch the application:

```bash
live-vlm-webui
```

The application generates a self-signed certificate and starts an HTTPS server on port 8090. Keep this terminal open while using the interface.

In another terminal, find the reachable address of your hardware platform:

```bash
hostname -I | awk '{print $1}'
```

From a browser with a webcam, open:

```text
https://<HARDWARE_IP>:8090
```

Use `https://`, because browsers require a secure context for webcam access. Accept the self-signed certificate exception only after confirming that the address belongs to your hardware platform, then grant camera permission.

## Step 5. Configure the VLM backend

In the **VLM API Configuration** section:

1. Set **API Base URL** to `http://localhost:11434/v1`.
2. Refresh the model list.
3. Select `llama3.2-vision:11b`.
4. Leave **Max Tokens** at its default for the first test.
5. Set **Frame Interval** to 60 if you want less frequent analysis.

## Step 6. Verify real-time analysis

Select a camera and start VLM analysis. Confirm that the interface shows:

- A live video feed
- Generated analysis for sampled frames
- The selected model name and inference latency
- GPU, CPU, and memory utilization

Try the **Scene Description** preset, then enter a custom prompt such as:

```text
Describe the most important objects and activity in one sentence.
```

The next processed frame should use the updated prompt.

## Step 7. Compare another model

To compare supported models, first select one from **Find model recipes** in the Overview tab. Download it with Ollama:

```bash
ollama pull <MODEL_NAME>
```

Refresh the model list in the interface, stop the current analysis, select the new model, and restart analysis. Compare response quality, inference latency, and memory utilization.

## Step 8. Stop or remove the application

Press `Ctrl+C` in the terminal running Live VLM WebUI to stop it.

To remove only Live VLM WebUI:

```bash
pipx uninstall live-vlm-webui
```

Ollama and downloaded models remain available. To remove an individual model without uninstalling Ollama:

```bash
ollama rm <MODEL_NAME>
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `pipx install` reports dependency errors | The isolated environment is incomplete or stale | Run `pipx uninstall live-vlm-webui`, then retry `pipx install live-vlm-webui` |
| `live-vlm-webui: command not found` | The pipx application directory is not in `PATH` | Run `pipx ensurepath`, then open a new terminal or run `source ~/.bashrc` |
| Browser reports that the connection is not private | Live VLM WebUI uses a self-signed HTTPS certificate | Confirm the address belongs to your hardware platform, open the browser's advanced options, and accept the certificate exception |
| Camera access is denied | The browser does not have camera permission or the page was opened over HTTP | Open `https://<HARDWARE_IP>:8090` and grant camera permission in browser settings |
| Camera works, but analysis fails with `InvalidStateError` | WebRTC traffic is passing through an SSH TCP tunnel | Connect the browser directly to the hardware platform over the local network; WebRTC requires direct connectivity |
| The interface cannot connect to the VLM | Ollama is stopped or the API URL is incorrect | Run `sudo systemctl start ollama`, verify `curl http://localhost:11434/v1/models`, and set the API Base URL to `http://localhost:11434/v1` |
| No models appear in the model list | No vision model is installed or the model list has not refreshed | Run `ollama pull llama3.2-vision:11b`, then refresh the model list |
| Responses take several seconds per frame | The selected model or analysis frequency exceeds available resources | Increase **Frame Interval**, reduce **Max Tokens**, or select a smaller supported vision model |
| GPU statistics show `N/A` | GPU monitoring cannot access the NVIDIA driver | Run `nvidia-smi` and resolve driver access before restarting Live VLM WebUI |
| Port 8090 is already in use | Another process is listening on the default port | Identify the process with `sudo lsof -i :8090`, then stop it if it is safe to do so |
| The interface is unreachable from another device | The address is incorrect or a firewall blocks port 8090 | Recheck the address with `hostname -I` and allow TCP port 8090 through the local firewall |
| Video is laggy or frozen | Browser or network performance is insufficient | Use a current Chromium-based browser or Firefox, reduce other network traffic, and access the interface over a stable local connection |

For the latest known issues, see the Live VLM WebUI documentation linked under **Resources**.
