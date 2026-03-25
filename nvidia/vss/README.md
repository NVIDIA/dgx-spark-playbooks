# Build a Video Search and Summarization (VSS) Agent

> Run the VSS Blueprint on your Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Deploy NVIDIA's Video Search and Summarization (VSS) AI Blueprint to build intelligent video analytics systems that combine vision language models, large language models, and retrieval-augmented generation. The system transforms raw video content into real-time actionable insights with video summarization, Q&A, and real-time alerts. You'll set up either a completely local Event Reviewer deployment or a hybrid deployment using remote model endpoints.

## What you'll accomplish

You will deploy NVIDIA's VSS AI Blueprint on NVIDIA Spark hardware with Blackwell architecture, choosing between two deployment scenarios: VSS Event Reviewer (completely local with VLM pipeline) or Standard VSS (hybrid deployment with remote LLM/embedding endpoints). This includes setting up Alert Bridge, VLM Pipeline, Alert Inspector UI, Video Storage Toolkit, and optional DeepStream CV pipeline for automated video analysis and event review.

## What to know before starting

- Working with NVIDIA Docker containers and container registries
- Setting up Docker Compose environments with shared networks
- Managing environment variables and authentication tokens
- Basic understanding of video processing and analysis workflows

## Prerequisites

- NVIDIA Spark device with ARM64 architecture and Blackwell GPU
- DGX OS (suggested: 7.4.0 or higher)
- Driver version 580.95.05 or higher installed: `nvidia-smi | grep "Driver Version"`
- CUDA version 13.0 installed: `nvcc --version`
- Docker installed and running: `docker --version && docker compose version`
- Access to NVIDIA Container Registry with [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys)
- NVIDIA Container Toolkit
- [Optional] NVIDIA API Key for remote model endpoints (hybrid deployment only)
- Sufficient storage space for video processing (>10GB recommended in `/tmp/`)

## Ancillary files

- [VSS Blueprint GitHub Repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) - Main codebase and Docker Compose configurations
- [VSS Official Documentation](https://docs.nvidia.com/vss/latest/index.html) - Complete system documentation

## Time & risk

* **Duration:** 30-45 minutes for initial setup, additional time for video processing validation
* **Risks:**
  * Container startup can be resource-intensive and time-consuming with large model downloads
  * Network configuration conflicts if shared network already exists
  * Remote API endpoints may have rate limits or connectivity issues (hybrid deployment)
* **Rollback:** Stop all containers with `scripts/dev-profile.sh down`
* **Last Updated:** 3/16/2026
  * Update required OS and Driver versions
  * Support for VSS 3.1.0 with Cosmos Reason 2 VLM

## Instructions

## Step 1. Verify environment requirements

Check that your system meets the hardware and software [prerequisites](https://docs.nvidia.com/vss/latest/prerequisites.html).

```bash
## Verify driver version
nvidia-smi | grep "Driver Version"
## Expected output: Driver Version: 580.126.09 or higher

## Verify CUDA version
nvcc --version
## Expected output: release 13.0

## Verify Docker is running
docker --version && docker compose version
```

## Step 2. Configure Docker

To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.
Open a new terminal and test Docker access. In the terminal, run:

```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with sudo .

```bash
sudo usermod -aG docker $USER
newgrp docker
```


Additionally, configure Docker so that it can use the NVIDIA Container Runtime.

```bash
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

##Run a sample workload to verify the setup
sudo docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

## Step 3. Clone the VSS repository

Clone the Video Search and Summarization repository from NVIDIA's public GitHub.

```bash
## Clone the VSS AI Blueprint repository
git clone https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization.git
cd video-search-and-summarization
```

## Step 4. Run the cache cleaner script

Start the system cache cleaner to optimize memory usage during container operations.

Create the cache cleaner script at /usr/local/bin/sys-cache-cleaner.sh mentioned below

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
Running in the background

```bash
## In another terminal, start the cache cleaner script.
sudo -b /usr/local/bin/sys-cache-cleaner.sh
```

> [!NOTE]
+> The above runs the cache cleaner in the current session only; it does not persist across reboots. To have the cache cleaner run across reboots, create a systemd service instead.
+> 
+> To stop the background cache cleaner:
+> ```bash
+> sudo pkill -f sys-cache-cleaner.sh
+> ```


## Step 5. Authenticate with NVIDIA Container Registry

Log in to NVIDIA's container registry using your [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys).

> [!NOTE]
> If you don’t have an NVIDIA account already, you’ll have to create one and register for the [developer program](https://developer.nvidia.com/nvidia-developer-program).

```bash
## Log in to NVIDIA Container Registry
docker login nvcr.io
## Username: $oauthtoken
## Password: <PASTE_NGC_API_KEY_HERE>
```

## Step 6. Choose deployment scenario

Choose between two deployment options based on your requirements:

| Deployment Scenario                       | VLM (Cosmos-Reason2-8B)| LLM                           | 
|-------------------------------------------|------------------------|-------------------------------|
| Standard VSS (Base)                       | Local           | Remote                               |
| Standard VSS (Alert Verification)         | Local           | Remote                               |
| Standard VSS deployment (Real-Time Alerts)| Local           | Remote                               |


## Step 7. Standard VSS 

**[Standard VSS](https://docs.nvidia.com/vss/latest/#architecture-overview) (Hybrid Deployment)**

In this hybrid deployment, we would use NIMs from [build.nvidia.com](https://build.nvidia.com/). Alternatively, you can configure your own hosted endpoints by following the instructions in the [VSS remote LLM deployment guide](https://docs.nvidia.com/vss/latest/vss-agent/configure-llm.html).


**7.1 Get NVIDIA API Key**

- Log in to https://build.nvidia.com/explore/discover.
- Search for **Get API Key** on the page and click on it.


**7.2 Launch Standard VSS deployment**

[Standard VSS deployment (Base)](https://docs.nvidia.com/vss/latest/quickstart.html#deploy)
[Standard VSS deployment (Alert Verification)](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html)
[Standard VSS deployment (Real-Time Alerts)](https://docs.nvidia.com/vss/latest/agent-workflow-rt-alert.html#real-time-alert-workflow)

```bash
## Start Standard VSS (Base)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
scripts/dev-profile.sh up -p base -H DGX-SPARK --use-remote-llm

## Start Standard VSS (Alert Verification)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
scripts/dev-profile.sh up -p alerts -m verification -H DGX-SPARK --use-remote-llm

## Start Standard VSS (Real-Time Alerts)
export NGC_CLI_API_KEY='your_ngc_api_key'
export LLM_ENDPOINT_URL=https://your-llm-endpoint.com
scripts/dev-profile.sh up -p alerts -m real-time -H DGX-SPARK --use-remote-llm
```

> [!NOTE]
> This step will take several minutes as containers are pulled and services initialize. The VSS backend requires additional startup time.
> The following the environment variable needs to be set first before any deployment:
> • NGC_CLI_API_KEY     — (required) for vss deployment
> • LLM_ENDPOINT_URL    — (required) when --use-remote-llm is passed, used as LLM base URL
> • NVIDIA_API_KEY      — (optional) used for accessing remote LLM/VLM endpoints
> • OPENAI_API_KEY      — (optional) used for accessing remote LLM/VLM endpoints
> • VLM_CUSTOM_WEIGHTS  — (optional) absolute path to custom weights dir

**7.3 Validate Standard VSS deployment**

Access the VSS UI to confirm successful deployment.
[Common VSS Endpoints](https://docs.nvidia.com/vss/latest/agent-workflow-alert-verification.html#service-endpoints)

```bash
## Test Agent UI accessibility
## If running locally on your Spark device, use localhost:
curl -I http://localhost:3000
## Expected: HTTP 200 response

## If your Spark is running in Remote/Accessory mode, replace 'localhost' with the IP address or hostname of your Spark device.
## To find your Spark's IP address, run the following command on the Spark terminal:
hostname -I
## Or to get the hostname:
hostname
## Then test accessibility (replace <SPARK_IP_OR_HOSTNAME> with the actual value):
curl -I http://<SPARK_IP_OR_HOSTNAME>:3000
```

Open `http://localhost:3000` or `http://<SPARK_IP_OR_HOSTNAME>:3000` in your browser to access the Agent interface.

## Step 8. Test video processing workflow

Run a basic test to verify the video analysis pipeline is functioning based on your deployment. The UI comes with a few example videos pre-populated for uploading and testing

**For Standard VSS deployment**

Follow the steps [here](https://docs.nvidia.com/vss/latest/quickstart.html#deploy) to navigate VSS Agent UI.
- Access VSS Agent interface at `http://localhost:3000`
- Download the sample data from NGC [here](https://docs.nvidia.com/vss/latest/quickstart.html#download-sample-data-from-ngc) and upload videos and test features [here](https://docs.nvidia.com/vss/latest/quickstart.html#download-sample-data-from-ngc)
  

## Step 9. Cleanup and rollback

To completely remove the VSS deployment and free up system resources [Follow](https://docs.nvidia.com/vss/latest/quickstart.html#step-5-teardown-the-agent):

> [!WARNING]
> This will destroy all processed video data and analysis results.

```bash
## For Standard VSS deployment
scripts/dev-profile.sh down
```

## Step 10. Next steps

With VSS deployed, you can now:

**Standard VSS deployment:**
- Access full VSS capabilities at port 3000
- Test video summarization and Q&A features
- Configure knowledge graphs and graph databases
- Integrate with existing video processing workflows

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Container fails to start with "pull access denied" | Missing or incorrect nvcr.io credentials | Re-run `docker login nvcr.io` with valid credentials |
| Web interfaces not accessible | Services still starting or port conflicts | Wait 2-3 minutes, check `docker ps` for container status |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
