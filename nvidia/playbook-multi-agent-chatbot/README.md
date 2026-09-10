# Build and Deploy a Multi-Agent Chatbot

> A supervisor agent orchestrating specialists for code, retrieval, and image analysis


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook shows you how to prototype, build, and deploy a fully local multi-agent system on your hardware platform. With large unified memory, you can run multiple LLMs and VLMs in parallel — enabling interactions across agents.

At the core is a supervisor agent powered by gpt-oss-120B, orchestrating specialized downstream agents for coding, retrieval-augmented generation (RAG), and image understanding. Together, these components show how complex, multimodal workflows can run efficiently on local, high-performance hardware.

## What you'll accomplish

You'll have a full-stack multi-agent chatbot system running on your hardware platform, accessible through your local web browser. The setup includes:

- LLM and VLM model serving using llama.cpp and TensorRT-LLM servers
- GPU acceleration for both model inference and document retrieval
- Multi-agent orchestration using a supervisor agent powered by gpt-oss-120B
- MCP (Model Context Protocol) servers as tools for the supervisor agent

## What to know before starting

**Required:**

- Basic familiarity with the Linux command line
- Comfort managing Docker containers and Docker Compose

**Optional:**

- Familiarity with multi-agent patterns, MCP tools, or RAG workflows

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Docker Compose stack with llama.cpp + TensorRT-LLM (gpt-oss-120B supervisor) | — |

> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- No other processes using the GPU (verify with `nvidia-smi`)
- Enough disk space for model downloads (~75 GB for the default models: gpt-oss-120B ~63 GB, Deepseek-Coder:6.7B-Instruct ~7 GB, and Qwen3-Embedding-4B ~4 GB)

> [!NOTE]
> This demo uses ~120 GB of memory by default. Ensure no other workloads are running (`nvidia-smi`), or switch to a smaller supervisor model such as gpt-oss-20B (see **Instructions → Step 8**).

**Software requirements**

- Docker and Docker Compose: `docker ps` and `docker compose version`
- Network access to download container images and model files from Hugging Face
- Web browser access to ports `3000` (frontend) and `8000` (backend API)

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-multi-agent-chatbot/).

- `model_download.sh` — Downloads GGUF and related model files from Hugging Face
- `docker-compose.yml` / `docker-compose-models.yml` — Application, API, UI, and model-serving services
- `Dockerfile.llamacpp` — Builds the local llama.cpp CUDA server image
- `backend/` — Supervisor agent, MCP tools, and API server
- `frontend/` — Browser UI for chatting with agents and uploading documents

## Time & risk

- **Estimated time:** 60 MIN (longer on first run due to model and image downloads; model downloads alone can take 30 minutes to 2 hours depending on network speed)
- **Risk level:** Medium
  - Docker permission issues may require user group changes and a session restart
  - Large model downloads may fail or take significant time depending on network speed
  - Default models use most available memory; competing GPU workloads can cause failures
- **Rollback:** Stop and remove Docker containers and the Postgres volume using the cleanup commands in Instructions
- **Last Updated:** 08/03/2026
  - Local multi-agent chatbot with supervisor orchestration for coding, RAG, and image understanding on supported hardware platforms

## Instructions

## Step 1. Configure Docker permissions

To manage containers without sudo, you must be in the `docker` group. If you skip this step, run Docker commands with sudo.

Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission denied error (for example, permission denied while trying to connect to the Docker daemon socket), add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

> [!NOTE]
> After `usermod`, you may need to log out and back in (or reboot) so the new group membership applies in all sessions.

## Step 2. Clone the repository

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-multi-agent-chatbot/assets
```

## Step 3. Run the model download script

```bash
chmod +x model_download.sh
./model_download.sh
```

The script pulls model files from Hugging Face, including gpt-oss-120B (~63 GB), Deepseek-Coder:6.7B-Instruct (~7 GB), and Qwen3-Embedding-4B (~4 GB). This may take between 30 minutes and 2 hours depending on network speed.

> [!NOTE]
> If a model download fails partway through, remove the incomplete file under `models/` and re-run `./model_download.sh`.

## Step 4. Start the Docker containers for the application

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
```

This step builds the llama.cpp server image and starts the model servers, backend API, and frontend UI. It can take 10 to 20 minutes depending on network speed. Wait for containers to become ready and healthy:

```bash
watch 'docker ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"'
```

> [!NOTE]
> The Qwen2.5-VL model container may report as unhealthy while starting up; you can ignore that status during startup.

## Step 5. Access the frontend UI

Open your browser and go to: `http://localhost:3000`

> [!NOTE]
> If you are connected to a remote hardware platform over SSH, forward the UI and API ports so the browser on your local machine can reach them:
>
> ```bash
> ssh -L 3000:localhost:3000 -L 8000:localhost:8000 username@<HARDWARE_IP>
> ```
>
> Replace `username` and `<HARDWARE_IP>` with your account and the hardware platform’s reachable address.

## Step 6. Try out the sample prompts

Click any of the tiles on the frontend to try the supervisor and the other agents.

**RAG agent:** Before trying the example prompt for the RAG agent, upload the example PDF [NVIDIA Blackwell Whitepaper](https://images.nvidia.com/aem-dam/Solutions/geforce/blackwell/nvidia-rtx-blackwell-gpu-architecture.pdf) as context: download the PDF, click the green **Upload Documents** button in the left sidebar under **Context**, then check the box in the **Select Sources** section.

**Image understanding agent — example prompt:**

Describe this image: https://en.wikipedia.org/wiki/London_Bridge#/media/File:London_Bridge_from_St_Olaf_Stairs.jpg

## Step 7. Cleanup

Use these commands when you want to stop the stack and free resources. Cleanup is optional rollback — not a required completion step.

> [!WARNING]
> This stops and removes the application containers and deletes the Postgres volume used by the demo (chat/session data in that volume is removed).

From the `assets` directory of this playbook, run:

```bash
docker compose -f docker-compose.yml -f docker-compose-models.yml down

docker volume rm "$(basename "$PWD")_postgres_data"
```

You can optionally run `docker volume prune` afterward to remove unused volumes. If you do not run cleanup, containers continue to run and use memory.

## Step 8. Next steps

1. Try different prompts with the multi-agent chatbot system.

2. **Optional — switch the supervisor to gpt-oss-20B** when you need more memory headroom. From the `assets` directory:

   1. In `model_download.sh`, comment out the three gpt-oss-120b download lines and uncomment the gpt-oss-20b download line, then re-run `./model_download.sh` (skip if the gpt-oss-20b GGUF is already present under `models/`).
   2. In `docker-compose-models.yml`, uncomment the `gpt-oss-20b` service block and comment out the `gpt-oss-120b` service block.
   3. In `docker-compose.yml`, set the `MODELS` environment variable so it includes `gpt-oss-20b` (and remove or comment `gpt-oss-120b`). The name must match the container name in `docker-compose-models.yml`.
   4. Recreate the stack:

   ```bash
   docker compose -f docker-compose.yml -f docker-compose-models.yml up -d --build
   ```

3. Add new MCP (Model Context Protocol) servers as tools for the supervisor agent under `backend/tools/mcp_servers/` (and register them in `backend/client.py`).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cannot access gated repo for URL | Certain Hugging Face models have restricted access | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) in your browser |
| Permission denied on `docker ps` | User not in the `docker` group | Run Step 1 completely, including logging out and back in (or reboot), or use `sudo` |
| Model download fails or incomplete | Network interruption or incomplete file | Remove the incomplete file under `models/` and re-run `./model_download.sh` |
| Containers unhealthy / out of memory | Competing GPU workloads or default models near memory capacity | Stop other GPU processes (`nvidia-smi`); optionally switch the supervisor to gpt-oss-20B (see **Instructions → Step 8**); flush UMA buffer cache if needed (note below) |
| Cannot open UI at `localhost:3000` from a remote session | Ports not forwarded over SSH | Forward ports `3000` and `8000` as shown in Instructions Step 5 |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
