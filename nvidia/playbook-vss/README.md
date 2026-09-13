# Deploy a Video Search and Summarization Agent

> Real-time alerts and natural-language Q&A over your footage

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Deploy NVIDIA's Video Search and Summarization (VSS) AI Blueprint to build intelligent video analytics systems that combine vision language models, large language models, and retrieval-augmented generation. The system transforms raw video content into real-time actionable insights with video summarization, Q&A, and real-time alerts.

You'll set up a Standard VSS hybrid deployment that runs the VLM pipeline locally on your hardware platform and uses remote model endpoints for the LLM.

## What you'll accomplish

You'll deploy the VSS AI Blueprint on your hardware platform using Standard VSS (hybrid deployment with a local VLM and remote LLM endpoints). This includes Alert Bridge, VLM Pipeline, Alert Inspector UI, Video Storage Toolkit, and optional DeepStream CV pipeline options for automated video analysis and event review.

## What to know before starting

**Required:**

- Working with NVIDIA Docker containers and container registries
- Setting up Docker Compose environments with shared networks
- Managing environment variables and authentication tokens

**Optional:**

- Basic understanding of video processing and analysis workflows

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | VSS 3.2.0 · Cosmos Reason 2 VLM (local) · remote LLM | — |

> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient storage space for video processing (>10 GB recommended in `/tmp/`)

**Software requirements**

- OS version suggested: 7.4.0 or higher — see Supported hardware platforms matrix above
- Driver version 580.95.05 or higher: `nvidia-smi | grep "Driver Version"`
- CUDA version 13.0: `nvcc --version`
- Docker installed and running: `docker --version && docker compose version`
- NVIDIA Container Toolkit
- Access to NVIDIA Container Registry with an [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys)
- NVIDIA API Key for remote model endpoints (hybrid deployment)

## Ancillary files

- [VSS Blueprint GitHub Repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) — Main codebase and Docker Compose configurations
- [VSS Official Documentation](https://docs.nvidia.com/vss/latest/index.html) — Complete system documentation

## Time & risk

- **Estimated time:** 30–45 MIN for initial setup; additional time for video processing validation
- **Risk level:** Medium
  - Container startup can be resource-intensive and time-consuming with large model downloads
  - Network configuration conflicts if a shared network already exists
  - Remote API endpoints may have rate limits or connectivity issues (hybrid deployment)
- **Rollback:** Stop all containers with `deploy/docker/scripts/dev-profile.sh down`
- **Last Updated:** 08/03/2026
  - VSS 3.2.0 Standard VSS hybrid deployment with Cosmos Reason 2 VLM on supported hardware platforms

## Instructions

## Step 1. Verify environment requirements

Check that your system meets the hardware and software [prerequisites](https://docs.nvidia.com/vss/latest/prerequisites.html).

```bash
## Verify driver version
nvidia-smi | grep "Driver Version"
## Expected output: Driver Version: 580.95.05 or higher

## Verify CUDA version
nvcc --version
## Expected output: release 13.0

## Verify Docker is running
docker --version && docker compose version
```

## Step 2. Configure Docker

To manage containers without `sudo`, add your user to the `docker` group. If you skip this step, run Docker commands with `sudo`.

Open a new terminal and test Docker access:

```bash
docker ps
```

If you see a permission denied error (for example, permission denied while trying to connect to the Docker daemon socket), add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Configure Docker to use the NVIDIA Container Runtime:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

## Run a sample workload to verify the setup
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## Step 3. Clone the VSS repository

Clone the Video Search and Summarization repository from NVIDIA's public GitHub.

**Note:** Install Git LFS if it is not already present on the system.

```bash
sudo apt-get install -y git-lfs && git lfs install
```

```bash
## Clone the VSS AI Blueprint repository
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
cd video-search-and-summarization
git checkout tags/v3.2.0
git lfs install
git lfs pull
```

## Step 4. Run the cache cleaner script

Start the system cache cleaner to optimize memory usage during container operations.

Create the cache cleaner script at `/usr/local/bin/sys-cache-cleaner.sh`:

```bash
sudo tee /usr/local/bin/sys-cache-cleaner.sh << 'EOF'
#!/bin/bash
## Exit immediately if any command fails
set -e

## Disable hugepages
echo "disable vm/nr_hugepage"
echo 0 | tee /proc/sys/vm/nr_hugepages

## Notify that the cache cleaner is running
echo "Starting cache cleaner - Running"
echo "Press Ctrl + C to stop"
## Repeatedly sync and drop caches every 3 seconds
while true; do
     sync && echo 3 | tee /proc/sys/vm/drop_caches > /dev/null
     sleep 3
done
EOF

sudo chmod +x /usr/local/bin/sys-cache-cleaner.sh
```

Run it in the background:

```bash
## In another terminal, start the cache cleaner script.
sudo -b /usr/local/bin/sys-cache-cleaner.sh
```

> [!NOTE]
> The above runs the cache cleaner in the current session only; it does not persist across reboots. To have the cache cleaner run across reboots, create a systemd service instead.
>
> To stop the background cache cleaner:
> ```bash
> sudo pkill -f sys-cache-cleaner.sh
> ```

## Step 5. Authenticate with NVIDIA Container Registry

Log in to NVIDIA's container registry using your [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys).

> [!NOTE]
> If you don’t have an NVIDIA account already, create one and register for the [developer program](https://developer.nvidia.com/nvidia-developer-program).

```bash
## Log in to NVIDIA Container Registry
docker login nvcr.io
## Username: $oauthtoken
## Password: <PASTE_NGC_API_KEY_HERE>
```

## Step 6. Choose deployment scenario

Choose the deployment option based on your requirements:

| Deployment Scenario | VLM (Cosmos-Reason2-8B) | LLM |
| ------------------- | ----------------------- | --- |
| Standard VSS (Base) | Local | Remote |
| Standard VSS (Alert Verification) | Local | Remote |
| Standard VSS (Real-Time Alerts) | Local | Remote |

## Step 7. Standard VSS

**[Standard VSS](https://docs.nvidia.com/vss/latest/#architecture-overview) (Hybrid Deployment)**

In this hybrid deployment, use NIMs from [build.nvidia.com](https://build.nvidia.com/). Alternatively, configure your own hosted endpoints by following the [VSS remote LLM deployment guide](https://docs.nvidia.com/vss/latest/vss-agent/configure-llm.html).

**7.1 Get NVIDIA API Key**

- Log in to https://build.nvidia.com/explore/discover.
- Search for **Get API Key** on the page and click it.

**7.2 Launch Standard VSS deployment**

- [Standard VSS deployment (Base)](https://docs.nvidia.com/vss/latest/quickstart.html#deploy)
- [Standard VSS deployment (Alert Verification)](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html)
- [Standard VSS deployment (Real-Time Alerts)](https://docs.nvidia.com/vss/latest/agent-workflow-rt-alert.html#real-time-alert-workflow)

```bash
## Start Standard VSS (Base)
## Set NGC CLI API key and Hugging Face token (required for VA-MCP)
export NGC_CLI_API_KEY='your_ngc_api_key'
export HF_TOKEN='hf_your_token_here'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
deploy/docker/scripts/dev-profile.sh up -p base -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>

## Start Standard VSS (Alert Verification)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
deploy/docker/scripts/dev-profile.sh up -p alerts -m verification -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>

## Start Standard VSS (Real-Time Alerts)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
deploy/docker/scripts/dev-profile.sh up -p alerts -m real-time -H DGX-SPARK --use-remote-llm --llm <REMOTE LLM MODEL NAME>
```

> [!NOTE]
> This step will take several minutes as containers are pulled and services initialize. The VSS backend requires additional startup time.
>
> Set the following environment variables before deployment:
> - **NGC_CLI_API_KEY** — (required) NGC API key for pulling images and deployment
> - **LLM_ENDPOINT_URL** — (required when using `--use-remote-llm`) Base URL for the remote LLM
> - **NVIDIA_API_KEY** — (optional) For remote LLM/VLM endpoints that require it
> - **OPENAI_API_KEY** — (optional) For remote LLM/VLM endpoints that require it
> - **VLM_CUSTOM_WEIGHTS** — (optional) Absolute path to a custom weights directory
>
> Pass these additional flags to **`deploy/docker/scripts/dev-profile.sh`** for remote LLM mode:
> - **`--use-remote-llm`** — (required) Use a remote LLM; the base URL is read from **`LLM_ENDPOINT_URL`** in the environment
> - **`--llm`** — (required) Remote LLM model name (for example: `nvidia/nvidia-nemotron-nano-9b-v2`). **Strongly recommended** for alert workflows (verification and real-time): use `nvidia/nvidia-nemotron-nano-9b-v2`. Omitting `--llm` may cause the script to use whatever model is returned by the remote endpoint.
>
> The **`-H DGX-SPARK`** host profile is required by the deploy script for the hardware platform covered in this playbook.
>
> Run **`deploy/docker/scripts/dev-profile.sh --help`** for a full list of supported arguments.

**7.3 Validate Standard VSS deployment**

Access the VSS UI to confirm successful deployment.
[Common VSS Endpoints](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html#service-endpoints)

```bash
## Test Agent UI accessibility
## If running locally on your hardware platform, use localhost:
curl -I http://localhost:7777
## Expected: HTTP 200 response

## If accessing the hardware platform remotely, replace 'localhost' with its IP address or hostname.
## To find the IP address, run the following command on the hardware platform:
hostname -I
## Or to get the hostname:
hostname
## Then test accessibility (replace <HARDWARE_IP_OR_HOSTNAME> with the actual value):
curl -I http://<HARDWARE_IP_OR_HOSTNAME>:7777
```

Open `http://localhost:7777` or `http://<HARDWARE_IP_OR_HOSTNAME>:7777` in your browser to access the Agent interface.

## Step 8. Test video processing workflow

Run a basic test to verify the video analysis pipeline is functioning based on your deployment.

**For Standard VSS deployment**

Follow the steps [here](https://docs.nvidia.com/vss/latest/quickstart.html#deploy) to navigate the VSS Agent UI.

- Access the VSS Agent interface at `http://localhost:7777`
- Download the sample data from NGC [here](https://docs.nvidia.com/vss/latest/quickstart.html#download-sample-data-from-ngc) and upload videos to test features
- Test Standard VSS deployment (Base) [here](https://docs.nvidia.com/vss/latest/quickstart.html#step-2-upload-a-video)
- Test Standard VSS deployment (Alert Verification) [here](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html#step-2-add-a-video-stream)
- Test Standard VSS deployment (Real-Time Alerts) [here](https://docs.nvidia.com/vss/latest/agent-workflow-rt-alert.html#step-2-add-a-video-stream)

## Step 9. Cleanup and rollback

To completely remove the VSS deployment and free up system resources, [follow the teardown steps](https://docs.nvidia.com/vss/latest/quickstart.html#step-5-teardown-the-agent):

> [!WARNING]
> This will destroy all processed video data and analysis results.

```bash
## For Standard VSS deployment
deploy/docker/scripts/dev-profile.sh down
```

## Step 10. Next steps

With VSS deployed, you can now:

**Standard VSS deployment:**

- Access full VSS capabilities at port 7777
- Test video and Q&A features
- Configure knowledge graphs and graph databases
- Integrate with existing video processing workflows

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container fails to start with "pull access denied" | Missing or incorrect nvcr.io credentials | Re-run `docker login nvcr.io` with valid credentials |
| Web interfaces not accessible | Services still starting or port conflicts | Wait 2–3 minutes, check `docker ps` for container status |
| Memory pressure within capacity | Unified memory buffer cache not released | See UMA note below |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
