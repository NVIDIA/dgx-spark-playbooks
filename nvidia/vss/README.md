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
- Driver version 580.126.09 or higher installed: `nvidia-smi | grep "Driver Version"`
- CUDA version 13.0 installed: `nvcc --version`
- Docker installed and running: `docker --version && docker compose version`
- Access to NVIDIA Container Registry with [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys)
- NVIDIA Container Toolkit
- [Optional] NVIDIA API Key for remote model endpoints (hybrid deployment only)
- Sufficient storage space for video processing (>10GB recommended in `/tmp/`)

## Ancillary files

- [VSS Blueprint GitHub Repository](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization) - Main codebase and Docker Compose configurations
- [Sample CV Detection Pipeline](https://github.com/NVIDIA-AI-Blueprints/video-search-and-summarization/tree/main/examples/cv-event-detector) - Reference CV pipeline for event reviewer workflow
- [VSS Official Documentation](https://docs.nvidia.com/vss/latest/index.html) - Complete system documentation

## Time & risk

* **Duration:** 30-45 minutes for initial setup, additional time for video processing validation
* **Risks:**
  * Container startup can be resource-intensive and time-consuming with large model downloads
  * Network configuration conflicts if shared network already exists
  * Remote API endpoints may have rate limits or connectivity issues (hybrid deployment)
* **Rollback:** Stop all containers with `docker compose down`, remove shared network with `docker network rm vss-shared-network`, and clean up temporary media directories.
* **Last Updated:** 1/21/2026
  * Update required OS and Driver versions
  * Support for VSS 2.4.1 with Cosmos Reason 2 VLM

## Instructions

## Step 1. Verify environment requirements

Check that your system meets the hardware and software prerequisites.

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

```bash
## In another terminal, start the cache cleaner script.
## Alternatively, append " &" to the end of the command to run it in the background.
sudo sh deploy/scripts/sys_cache_cleaner.sh
```

## Step 5. Set up Docker shared network

Create a Docker network that will be shared between VSS services and CV pipeline containers.

```bash
## Create shared network (may require sudo depending on Docker configuration)
docker network create vss-shared-network
```

> [!WARNING]
> If the network already exists, you may see an error. Remove it first with `docker network rm vss-shared-network` if needed.

## Step 6. Authenticate with NVIDIA Container Registry

Log in to NVIDIA's container registry using your [NGC API Key](https://org.ngc.nvidia.com/setup/api-keys).

> [!NOTE]
> If you don’t have an NVIDIA account already, you’ll have to create one and register for the [developer program](https://developer.nvidia.com/nvidia-developer-program).

```bash
## Log in to NVIDIA Container Registry
docker login nvcr.io
## Username: $oauthtoken
## Password: <PASTE_NGC_API_KEY_HERE>
```

## Step 7. Choose deployment scenario

Choose between two deployment options based on your requirements:

| Deployment Scenario   | VLM (Cosmos-Reason2-8B)| LLM (Llama 3.1)               | Embedding / Reranker | CV Pipeline  |
|---------------------- |------------------------|-------------------------------|----------------------|--------------|
| VSS Event Reviewer    | Local                  | Not Used                      | Not Used             | Local        |
| Standard VSS          | Local                  | Remote (70B) or Local (8B)    | Remote / Local       | Optional     |

Proceed with **Option A** for Event Reviewer or **Option B** for Standard VSS.

## Step 8. Option A

**[VSS Event Reviewer](https://docs.nvidia.com/vss/latest/content/vss_event_reviewer.html) (Completely Local)**

**8.1 Navigate to Event Reviewer directory**

Change to the directory containing the Event Reviewer Docker Compose configuration.

```bash
cd deploy/docker/event_reviewer/
```

**8.2 Configure NGC API Key and HF Token**

Update the environment file with your NGC API Key and HF Token. You can do this by editing the `.env` file directly, or by running the following commands:

> [!NOTE]
> To deploy the default VLM (**Cosmos-Reason2 8B**) from Hugging Face, you must accept the model’s terms and conditions on the Hugging Face model page before downloads will work.

```bash
## Edit the .env file and update NGC_API_KEY
echo "NGC_API_KEY=<YOUR_NGC_API_KEY>" >> .env
echo "HF_TOKEN=<YOUR_HF_TOKEN>" >> .env  # To download Cosmos-Reason2-8B VLM
```

**8.3 (Optional) Remove the VST volume data**

When upgrading from older versions of VSS, remove the VST volume data

```bash
rm -rf vst/vst_volume/*
```

**8.4 Start VSS Event Reviewer services**

Launch the complete VSS Event Reviewer stack including Alert Bridge, VLM Pipeline, Alert Inspector UI, and Video Storage Toolkit.

```bash
## DGX Spark tuning: improves Q&A responsiveness/behavior for the Event Reviewer workflow
export VLM_DEFAULT_NUM_FRAMES_PER_CHUNK=8

## Start VSS Event Reviewer with ARM64 and SBSA optimizations
IS_SBSA=1 ALERT_REVIEW_MEDIA_BASE_DIR=/tmp/alert-media-dir docker compose up
```

> [!NOTE]
> This step will take several minutes as containers are pulled and services initialize. The VSS backend requires additional startup time. Proceed to the next step in a new terminal in the meantime.

**8.5 Navigate to CV Event Detector directory**

In a new terminal session, navigate to the computer vision event detector configuration.

```bash
cd video-search-and-summarization/examples/cv-event-detector
```

**8.6 Start DeepStream CV pipeline**

Launch the DeepStream computer vision pipeline and CV UI services.

```bash
## Start CV pipeline with ARM64 and SBSA optimizations
IS_SBSA=1 ALERT_REVIEW_MEDIA_BASE_DIR=/tmp/alert-media-dir docker compose up
```

**8.7 Wait for service initialization**

Allow time for all containers to fully initialize before accessing the user interfaces.

```bash
## Monitor container status
docker ps
## Verify all containers show "Up" status and VSS backend logs (vss-engine:2.4.1-sbsa) show ready state "Uvicorn running on http://0.0.0.0:7860"
## In total, there should be 8 containers:
## nvcr.io/nvidia/blueprint/nv-cv-event-detector-ui:2.4.1
## nvcr.io/nvidia/blueprint/nv-cv-event-detector:2.4.0-sbsa
## nginx:alpine
## nvcr.io/nvidia/blueprint/vss-alert-inspector-ui:2.4.1
## nvcr.io/nvidia/blueprint/alert-bridge:0.19.0-multiarch
## nvcr.io/nvidia/blueprint/vss-engine:2.4.1-sbsa
## nvcr.io/nvidia/blueprint/vst-storage:2.1.0-25.11.1.1
## redis/redis-stack-server:7.2.0-v9
```

**8.8 Validate Event Reviewer deployment**

Access the web interfaces to confirm successful deployment and functionality.

```bash
## Test CV UI accessibility (default: localhost)
curl -I http://localhost:7862
## Expected: HTTP 200 response

## Test Alert Inspector UI accessibility (default: localhost)
curl -I http://localhost:7860
## Expected: HTTP 200 response

## If you are running your Spark in Remote or Accessory mode, replace 'localhost' with the IP address or hostname of your Spark device.
## To find your Spark's IP address, run the following command on the Spark system:
hostname -I
## Or to get the hostname:
hostname
## Then use the IP/hostname in place of 'localhost', for example:
## curl -I http://<SPARK_IP_OR_HOSTNAME>:7862
```

Open these URLs in your browser:
- `http://localhost:7862` - CV UI to launch and monitor CV pipeline
- `http://localhost:7860` - Alert Inspector UI to view clips and review VLM results

> [!NOTE]
> You may now proceed to step 10.

## Step 9. Option B 

**[Standard VSS](https://docs.nvidia.com/vss/latest/content/architecture.html) (Hybrid Deployment)**

In this hybrid deployment, we would use NIMs from [build.nvidia.com](https://build.nvidia.com/). Alternatively, you can configure your own hosted endpoints by following the instructions in the [VSS remote deployment guide](https://docs.nvidia.com/vss/latest/content/installation-remote-docker-compose.html).

> [!NOTE]
> Fully local deployment using smaller LLM (Llama 3.1 8B) is also possible.  
> To set up a fully local VSS deployment, follow the [instructions in the VSS documentation](https://docs.nvidia.com/vss/latest/content/vss_dep_docker_compose_arm.html#local-deployment-single-gpu-dgx-spark).

**9.1 Get NVIDIA API Key**

- Log in to https://build.nvidia.com/explore/discover.
- Search for **Get API Key** on the page and click on it.

**9.2 Navigate to remote LLM deployment directory**

```bash
cd deploy/docker/remote_llm_deployment/
```

**9.3 Configure environment variables**

Update the environment file with your API keys and deployment preferences. You can do this by editing the `.env` file directly, or by running the following commands:

> [!NOTE]
> To deploy the default VLM (**Cosmos-Reason2 8B**) from Hugging Face, you must accept the model’s terms and conditions on the Hugging Face model page before downloads will work.

```bash
## Edit .env file with required keys
echo "NVIDIA_API_KEY=<YOUR_NVIDIA_API_KEY>" >> .env
echo "NGC_API_KEY=<YOUR_NGC_API_KEY>" >> .env
echo "HF_TOKEN=<YOUR_HF_TOKEN>" >> .env  # To download Cosmos-Reason2-8B VLM
echo "DISABLE_CV_PIPELINE=true" >> .env  # Set to false to enable CV
echo "INSTALL_PROPRIETARY_CODECS=false" >> .env  # Set to true to enable CV
```

**9.4 Review model configuration**

Verify that the config.yaml file contains the correct remote endpoints. For NIMs, it should be set to `https://integrate.api.nvidia.com/v1 `.

```bash
## Check model server endpoints in config.yaml
cat config.yaml | grep -A 10 "model"
```

**9.5 Launch Standard VSS deployment**

```bash
## Start Standard VSS with hybrid deployment
IS_SBSA=1 docker compose up
```

> [!NOTE]
> This step will take several minutes as containers are pulled and services initialize. The VSS backend requires additional startup time. 

**9.7 Validate Standard VSS deployment**

Access the VSS UI to confirm successful deployment.

```bash
## Test VSS UI accessibility
## If running locally on your Spark device, use localhost:
curl -I http://localhost:9100
## Expected: HTTP 200 response

## If your Spark is running in Remote/Accessory mode, replace 'localhost' with the IP address or hostname of your Spark device.
## To find your Spark's IP address, run the following command on the Spark terminal:
hostname -I
## Or to get the hostname:
hostname
## Then test accessibility (replace <SPARK_IP_OR_HOSTNAME> with the actual value):
curl -I http://<SPARK_IP_OR_HOSTNAME>:9100
```

Open `http://localhost:9100` in your browser to access the VSS interface.

## Step 10. Test video processing workflow

Run a basic test to verify the video analysis pipeline is functioning based on your deployment. The UI comes with a few example videos pre-populated for uploading and testing

**For Event Reviewer deployment**

Follow the steps [here](https://docs.nvidia.com/vss/latest/content/vss_event_reviewer.html#vss-alert-inspector-ui) to access and use the Event Reviewer workflow.
- Access CV UI at `http://localhost:7862` to upload and process videos
- Monitor results in Alert Inspector UI at `http://localhost:7860`

**For Standard VSS deployment**

Follow the steps [here](https://docs.nvidia.com/vss/latest/content/ui_app.html) to navigate VSS UI - File Summarization, Q&A, and Alerts.
- Access VSS interface at `http://localhost:9100`
- Upload videos and test summarization features

## Step 11. Cleanup and rollback

To completely remove the VSS deployment and free up system resources:

> [!WARNING]
> This will destroy all processed video data and analysis results.

```bash
## For Event Reviewer deployment
cd deploy/docker/event_reviewer/
IS_SBSA=1 ALERT_REVIEW_MEDIA_BASE_DIR=/tmp/alert-media-dir docker compose down
cd ../../examples/cv-event-detector/
IS_SBSA=1 ALERT_REVIEW_MEDIA_BASE_DIR=/tmp/alert-media-dir docker compose down

## For Standard VSS deployment
cd deploy/docker/remote_llm_deployment/
IS_SBSA=1 docker compose down

## Remove shared network (if using Event Reviewer)
docker network rm vss-shared-network

## Clean up temporary media files and stop cache cleaner
rm -rf /tmp/alert-media-dir
sudo pkill -f sys_cache_cleaner.sh
```

## Step 12. Next steps

With VSS deployed, you can now:

**Event Reviewer deployment:**
- Upload video files through the CV UI at port 7862
- Monitor automated event detection and reviewing
- Review analysis results in the Alert Inspector UI at port 7860
- Configure custom event detection rules and thresholds

**Standard VSS deployment:**
- Access full VSS capabilities at port 9100
- Test video summarization and Q&A features
- Configure knowledge graphs and graph databases
- Integrate with existing video processing workflows

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Container fails to start with "pull access denied" | Missing or incorrect nvcr.io credentials | Re-run `docker login nvcr.io` with valid credentials |
| Network creation fails | Existing network with same name | Run `docker network rm vss-shared-network` then recreate |
| Services fail to communicate | Incorrect environment variables | Verify `IS_SBSA=1` are set correctly |
| Web interfaces not accessible | Services still starting or port conflicts | Wait 2-3 minutes, check `docker ps` for container status |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
