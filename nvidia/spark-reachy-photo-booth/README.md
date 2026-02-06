# Spark & Reachy Photo Booth

> AI augmented photo booth using the DGX Spark and Reachy Mini.

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Guides](#guides)
  - [Service Configuration](#service-configuration)
- [Development](#development)
  - [Customize configuration parameters](#customize-configuration-parameters)
  - [Extend the demo with new tools](#extend-the-demo-with-new-tools)
  - [Create your own service](#create-your-own-service)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

![Teaser](assets/teaser.jpg)

Spark & Reachy Photo Booth is an interactive and event-driven photo booth demo that combines the **DGX Spark™** with the **Reachy Mini** robot to create an engaging multimodal AI experience. The system showcases:

- **A multi-modal agent** built with the `NeMo Agent Toolkit`
- **A ReAct loop** driven by the `openai/gpt-oss-20b` LLM powered by `TensorRT-LLM`
- **Voice interaction** based on `nvidia/riva-parakeet-ctc-1.1B` and `hexgrad/Kokoro-82M`
- **Image generation** with `black-forest-labs/FLUX.1-Kontext-dev` for image-to-image restyling
- **User position tracking** built with `facebookresearch/detectron2` and `FoundationVision/ByteTrack`
- **MinIO** for storing captured/generated images as well as sharing them via QR-code

The demo is based on a several services that communicate through a message bus.

![Architecture diagram](assets/architecture-diagram.png)

> [!NOTE]
> This playbook applies to both the Reachy Mini and Reachy Mini Lite robots. For simplicity, we’ll refer to the robot as Reachy throughout this playbook.

## What you'll accomplish

You'll deploy a complete photo booth system on DGX Spark running multiple inference models locally — LLM, image generation, speech recognition, speech generation, and computer vision — all without cloud dependencies. The Reachy robot interacts with users through natural conversation, captures photos, and generates custom images based on prompts, demonstrating real-time multimodal AI processing on edge hardware.

## What to know before starting

- Basic Docker and Docker Compose knowledge
- Basic network configuration skills

## Prerequisites

**Hardware Requirements:**
- [NVIDIA DGX Spark](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)
- A monitor, a keyboard, and a mouse to run this playbook directly on the DGX Spark.
- [Reachy Mini or Reachy Mini Lite robot](https://pollen-robotics-reachy-mini.hf.space/)

> [!TIP]
> Make sure your Reachy robot firmware is up to date. You can find instructions to update it [here](https://huggingface.co/spaces/pollen-robotics/Reachy_Mini).
**Software Requirements:**
- The official DGX Spark OS image including all required utilities such as Git, Docker, NVIDIA drivers, and the NVIDIA Container Toolkit
- An internet connection for the DGX Spark
- NVIDIA NGC Personal API Key (**`NVIDIA_API_KEY`**). [Create a key](https://org.ngc.nvidia.com/setup/api-keys) if necessary. Make sure to enable the `NGC Catalog` scope when creating the key.
- Hugging Face access token (**`HF_TOKEN`**). [Create a token](https://huggingface.co/settings/tokens) if necessary. Make sure to create a token with _Read access to contents of all public gated repos you can access_ permission.


## Ancillary files

All required assets can be found in the [Spark & Reachy Photo Booth repository](https://github.com/NVIDIA/spark-reachy-photo-booth).

- The Docker Compose application
- Various configuration files
- Source code for all the services
- Detailed documentation

## Time & risk

* **Estimated time:** 2 hours including hardware setup, container building, and model downloads
* **Risk level:** Medium
* **Rollback:** Docker containers can be stopped and removed to free resources. Downloaded models can be deleted from cache directories. Robot and peripheral connections can be safely disconnected. Network configurations can be reverted by removing custom settings.
* **Last Updated:** 01/27/2026
  * 1.0.0 First Publication

## Governing terms
Your use of the Spark Playbook scripts is governed by [Apache License, Version 2.0](https://www.apache.org/licenses/LICENSE-2.0) and enables use of separate open source and proprietary software governed by their respective licenses: [Flux.1-Kontext NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/black-forest-labs/containers/flux.1-kontext-dev?version=1.1), [Parakeet 1.1b CTC en-US ASR NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/nvidia/containers/parakeet-1-1b-ctc-en-us?version=1.4), [TensorRT-LLM](https://catalog.ngc.nvidia.com/orgs/nvidia/teams/tensorrt-llm/containers/release?version=1.3.0rc1), [minio/minio](https://hub.docker.com/r/minio/minio), [arizephoenix/phoenix](https://hub.docker.com/r/arizephoenix/phoenix), [grafana/otel-lgtm](https://hub.docker.com/r/grafana/otel-lgtm), [Python](https://hub.docker.com/_/python), [Node.js](https://hub.docker.com/_/node), [nginx](https://hub.docker.com/_/nginx), [busybox](https://hub.docker.com/_/busybox), [UV Python Packager](https://docs.astral.sh/uv/), [Redpanda](https://www.redpanda.com/), [Redpanda Console](https://www.redpanda.com/), [gpt-oss-20b](https://huggingface.co/openai/gpt-oss-20b), [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev), [FLUX.1-Kontext-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx).

> [!NOTE]
> FLUX.1-Kontext-dev and FLUX.1-Kontext-dev-onnx are models released for non-commercial use. Contact sales@blackforestlabs.ai for commercial terms. You are responsible for accepting the applicable License Agreements and Acceptable Use Policies, and for ensuring your HF token has the correct permissions.

## Instructions

## Step 1. Clone the repo

To easily manage containers without `sudo`, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with `sudo`.

Open a new terminal and test Docker access. In the terminal, run:

```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with `sudo`.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

```bash
git clone https://github.com/NVIDIA/spark-reachy-photo-booth.git
cd spark-reachy-photo-booth
```

> [!WARNING]
> This playbook is expected to be run directly on your DGX Spark and with the included web browser.

## Step 2. Create your environment

```bash
cp .env.example .env
```

Edit `.env` and set:

- **`NVIDIA_API_KEY`**: your NVIDIA API key (must start with `nvapi-...`)
- **`HF_TOKEN`**: your Hugging Face token (must start with `hf_...`)
- **`EXTERNAL_MINIO_BASE_URL`**: leave unchanged, unless you want to (see the section "Enable QR-code sharing on your local network")

To access the FLUX.1-Kontext-dev model, sign in to your Hugging Face account, then review and accept the [FLUX.1-Kontext-dev](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev) and [FLUX.1-Kontext-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-Kontext-dev-onnx) License Agreements and Acceptable Use Policy.

The remaining values are configured with reasonable defaults for local development (MinIO). For production deployments or untrusted environments, these values should be changed and stored securely.

## Step 3. Set up Reachy

- Plug the power cable to the base of Reachy and to a power outlet.
- Plug a USB-C cable to the base of Reachy and to the DGX Spark.
- Engage the power switch at the base of Reachy. The LED next to the switch should turn red.

You can verify that the robot is detected by running:

```bash
lsusb | grep Reachy
```

You should see a device printed in the terminal similar to `Bus 003 Device 003: ID 38fb:1001 Pollen Robotics Reachy Mini Audio`.

Run the following command to make sure the Reachy speaker can reach the maximum volume.

```bash
./robot-controller-service/scripts/speaker_setup.sh
```

![Setup](assets/setup.jpg)

## Step 4. Start the stack

Sign in to the nvcr.io registry:
```bash
docker login nvcr.io -u "\$oauthtoken"
```

When prompted for a password, enter your NGC personal API key.

```bash
docker compose up --build -d
```

This command pulls and builds container images, and downloads the required model artifacts. The first run can take between 30 minutes and 2 hours, depending on your internet speed. Subsequent runs usually complete in about 5 minutes.

## Step 5. Open the UI in your browser

On the DGX Spark, open Firefox (pre-installed) and browse to the **Web UI**: [http://127.0.0.1:3001](http://127.0.0.1:3001).

> [!TIP]
> The Web UI is accessible only when all containers are up and running.
> You can also check the status of all the containers with `docker compose ps --format "table {{.ID}}\t{{.Names}}\t{{.Status}}"`.
> If one or more containers are failing, inspect the logs with `docker compose logs -f <container_name>`.

> [!TIP]
> You can remotely **spectate** the ongoing interaction by opening an ssh session with X11 forwarding enabled (`ssh -X <USER>@<SPARK_IP>`).
> You should be able to open Firefox from this session and connect to [http://127.0.0.1:3001](http://127.0.0.1:3001).

> [!NOTE]
> The UI has a small impact on the performance of image generation. In order to optimize the performance of the image generation step in the experience, you can install and use Chromium instead of Firefox, as well as reduce the display resolution.

## Step 6. Optional: Enable QR-code sharing on your local network

Reachy can take pictures of people and generate images based on them. The web UI displays the generated images along with a QR code for downloading them. This section explains how to set up the system so that the QR code is accessible from users' phones.

For QR codes to open on your phone, your DGX Spark and phone must be on the same local network. Ensure that your router permits device-to-device communication within the network.

#### 1. Find your Spark’s local IP address

On the Spark, run the following command:

```bash
ip -f inet addr show enP7s7 | grep inet
```

Or this command if your Spark is connected through Wi-Fi

```bash
ip -f inet addr show wlP9s9 | grep inet
```

Find the IPv4 on your LAN (often something like `192.168.x.x` or `10.x.x.x`).

#### 2. Ensure MinIO is reachable from your phone

- **Same network**: connect your phone to the same Wi‑Fi/LAN as the DGX Spark.
- **Firewall**: by default, DGX Spark does not block incoming requests. If you installed a firewall, allow inbound traffic to the DGX Spark on **`9010` (MinIO API)**.

#### 3. Update `.env` and restart

Edit `.env` and replace:

- **`EXTERNAL_MINIO_BASE_URL=127.0.0.1:9010`** → **`EXTERNAL_MINIO_BASE_URL=<SPARK_LAN_IP>:9010`**

Then restart:

```bash
docker compose down
docker compose up --build -d
```

## Step 7. Optional: Going Further & Customizing the Application

### Guides

- [Getting Started](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/getting-started.md) – In-depth setup and configuration walkthrough
- [Writing Your First Service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/writing-your-first-service.md) – How to create and integrate a new service

### Service Configuration

Each service has its own README with details on customization, environment variables, and service-specific configuration:

| Service | Description |
|---------|-------------|
| [agent-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/agent-service/README.md) | LLM-powered agent workflow and decision logic |
| [animation-compositor-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/animation-compositor-service/README.md) | Combines animation clips and audio mixing |
| [animation-database-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/animation-database-service/README.md) | Animation library and procedural animation generation |
| [camera-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/camera-service/README.md) | Camera capture and image acquisition |
| [interaction-manager-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/interaction-manager-service/README.md) | Event orchestration and robot utterance management |
| [metrics-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/metrics-service/README.md) | Metrics collection and monitoring |
| [remote-control-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/remote-control-service/README.md) | Web-based remote control interface |
| [robot-controller-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/robot-controller-service/README.md) | Direct robot hardware control |
| [speech-to-text-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/speech-to-text-service/README.md) | Audio transcription (NVIDIA Riva/Parakeet) |
| [text-to-speech-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/text-to-speech-service/README.md) | Speech synthesis |
| [tracker-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/tracker-service/README.md) | Person detection and tracking |
| [ui-server-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/ui-server-service/README.md) | Backend for the web UI |

For detailed guidance on customizing service configurations, extending the demo with new tools, or creating your own services, refer to the [Development](development) tab.

## Development

## Development

This section provides comprehensive instructions for customizing and developing upon the Reachy Photo Booth application. If you're looking to deploy and run the application as-is, refer to the [Instructions](instructions) tab instead — this Development guide is specifically for those who need to make modifications to the application.

## Step 1. System dependencies

In order to use the Python development setup of the repository install the following packages:

```bash
sudo apt install python3.12-dev portaudio19-dev
```

To create the Python **venv** install uv by following the instructions [here](https://docs.astral.sh/uv/getting-started/installation/).

Then run the following command to generate the Python **venv**:

```bash
uv sync --all-packages
```

## Step 2. Get acquainted with the build and development process

Every folder suffixed by `-service` is a standalone Python program that runs in its own container. You must always start the services by interacting with the `docker-compose.yaml` at the root of the repository. You can enable code hot reloading for all the Python services by running:

```bash
docker compose up -d --build --watch
```

Whenever you change some Python code in the repository the associated container will be updated and automatically restarted.

The [Getting Started](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/getting-started.md) guide provides a comprehensive walkthrough of the build system, development workflow, debugging strategies, and monitoring infrastructure.

## Step 3. Make changes to the application

Now that your development environment is set up, here are the most common customizations developers typically explore.

### Customize configuration parameters

Each service has configurable parameters including system prompts, audio devices, model settings, and more. Check the individual service READMEs and the `src/configuration.py` files for detailed configuration options. Note that the default configuration in `src/configuration.py` might also be overridden in the `compose.yaml` file. Check out the following services to get started:

- [speech-to-text-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/speech-to-text-service/README.md) - Configure audio devices and transcription settings
- [text-to-speech-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/text-to-speech-service/README.md) - Adjust voice synthesis parameters
- [agent-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/agent-service/README.md) - Customize LLM system prompts, agent behavior, and decision logic

See the [instructions](instructions) for a complete list of all services and their READMEs.

### Extend the demo with new tools

The agent-service and interaction-manager-service are the core services for extending the demo with new capabilities:

- [agent-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/agent-service/README.md) - Add new agent tools and capabilities here
- [interaction-manager-service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/interaction-manager-service/README.md) - Manage event orchestration and robot utterances

### Create your own service

The [Writing Your First Service](https://github.com/NVIDIA/spark-reachy-photo-booth/tree/main/docs/writing-your-first-service.md) guide provides a step-by-step tutorial on scaffolding, implementing, and integrating a new microservice into the system. Follow this guide to create custom services that extend the photo booth functionality.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| No audio from robot (low volume) | Reachy speaker volume set too low by default | Increase Reachy speaker volume to maximum |
| No audio from robot (device conflict) | Another application capturing Reachy speaker | Check `animation-compositor` logs for "Error querying device (-1)", verify Reachy speaker is not set as system default in Ubuntu sound settings, ensure no other apps are capturing the speaker, then restart the demo |

If you have any issues with Reachy that are not covered by this guide, please read [Hugging Face's official troubleshooting guide](https://huggingface.co/docs/reachy_mini/troubleshooting).

> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
