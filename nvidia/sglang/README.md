# SGLang Inference Server

> Install and use SGLang on DGX Spark

## Table of Contents

- [Overview](#overview)
  - [Time & risk](#time-risk)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic Idea

SGLang is a fast serving framework for large language models and vision language models that makes
your interaction with models faster and more controllable by co-designing the backend runtime and
frontend language. This setup uses the optimized NVIDIA SGLang NGC Container on a single NVIDIA
Spark device with Blackwell architecture, providing GPU-accelerated inference with all dependencies
pre-installed.

## What you'll accomplish

You'll deploy SGLang in both server and offline inference modes on your NVIDIA Spark device,
enabling high-performance LLM serving with support for text generation, chat completion, and
vision-language tasks using models like DeepSeek-V2-Lite.

## What to know before starting

- Working in a terminal environment on Linux systems
- Basic understanding of Docker containers and container management
- Familiarity with NVIDIA GPU drivers and CUDA toolkit concepts
- Experience with HTTP API endpoints and JSON request/response handling

## Prerequisites

- NVIDIA Spark device with Blackwell architecture
- Docker Engine installed and running: `docker --version`
- NVIDIA GPU drivers installed: `nvidia-smi`
- NVIDIA Container Toolkit configured: `docker run --rm --gpus all lmsysorg/sglang:spark nvidia-smi`
- Sufficient disk space (>20GB available): `df -h`
- Network connectivity for pulling NGC containers: `ping nvcr.io`

## Ancillary files

- An offline inference python script [found here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/sglang/assets/offline-inference.py)

## Model Support Matrix

The following models are supported with SGLang on Spark. All listed models are available and ready to use:

| Model | Quantization | Support Status | HF Handle |
|-------|-------------|----------------|-----------|
| **GPT-OSS-20B** | MXFP4 | ✅ | `openai/gpt-oss-20b` |
| **GPT-OSS-120B** | MXFP4 | ✅ | `openai/gpt-oss-120b` |
| **Llama-3.1-8B-Instruct** | FP8 | ✅ | `nvidia/Llama-3.1-8B-Instruct-FP8` |
| **Llama-3.1-8B-Instruct** | NVFP4 | ✅ | `nvidia/Llama-3.1-8B-Instruct-FP4` |
| **Llama-3.3-70B-Instruct** | NVFP4 | ✅ | `nvidia/Llama-3.3-70B-Instruct-FP4` |
| **Qwen3-8B** | FP8 | ✅ | `nvidia/Qwen3-8B-FP8` |
| **Qwen3-8B** | NVFP4 | ✅ | `nvidia/Qwen3-8B-FP4` |
| **Qwen3-14B** | FP8 | ✅ | `nvidia/Qwen3-14B-FP8` |
| **Qwen3-14B** | NVFP4 | ✅ | `nvidia/Qwen3-14B-FP4` |
| **Qwen3-32B** | NVFP4 | ✅ | `nvidia/Qwen3-32B-FP4` |
| **Phi-4-multimodal-instruct** | FP8 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP8` |
| **Phi-4-multimodal-instruct** | NVFP4 | ✅ | `nvidia/Phi-4-multimodal-instruct-FP4` |
| **Phi-4-reasoning-plus** | FP8 | ✅ | `nvidia/Phi-4-reasoning-plus-FP8` |
| **Phi-4-reasoning-plus** | NVFP4 | ✅ | `nvidia/Phi-4-reasoning-plus-FP4` |

### Time & risk

* **Estimated time:** 30 minutes for initial setup and validation
* **Risk level:** Low - Uses pre-built, validated SGLang container with minimal configuration
* **Rollback:** Stop and remove containers with `docker stop` and `docker rm` commands
* **Last Updated:** 11/25/2025
    * First Publication

## Instructions

## Step 1. Verify system prerequisites

Check that your NVIDIA Spark device meets all requirements before proceeding. This step runs on
your host system and ensures Docker, GPU drivers, and container toolkit are properly configured.

```bash
## Verify Docker installation
docker --version

## Check NVIDIA GPU drivers
nvidia-smi

## Verify Docker GPU support
docker run --rm --gpus all lmsysorg/sglang:spark nvidia-smi

## Check available disk space
df -h /
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with sudo .

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Pull the SGLang Container

Download the latest SGLang container. This step runs on the host and may take
several minutes depending on your network connection.


```bash
## Pull the SGLang container
docker pull lmsysorg/sglang:spark

## Verify the image was downloaded
docker images | grep sglang
```

## Step 3. Launch SGLang container for server mode

Start the SGLang container in server mode to enable HTTP API access. This runs the inference
server inside the container, exposing it on port 30000 for client connections.

```bash
## Launch container with GPU support and port mapping
docker run --gpus all -it --rm \
  -p 30000:30000 \
  -v /tmp:/tmp \
  lmsysorg/sglang:spark \
  bash
```

## Step 4. Start the SGLang inference server

Inside the container, launch the HTTP inference server with a supported model. This step runs
inside the Docker container and starts the SGLang server daemon.

```bash
## Start the inference server with DeepSeek-V2-Lite model
python3 -m sglang.launch_server \
  --model-path deepseek-ai/DeepSeek-V2-Lite \
  --host 0.0.0.0 \
  --port 30000 \
  --trust-remote-code \
  --tp 1 \
  --attention-backend flashinfer \
  --mem-fraction-static 0.75 &

## Wait for server to initialize
sleep 30

## Check server status
curl http://localhost:30000/health
```

## Step 5. Test client-server inference

From a new terminal on your host system, test the SGLang server API to ensure it's working
correctly. This validates that the server is accepting requests and generating responses.

```bash
## Test with curl
curl -X POST http://localhost:30000/generate \
  -H "Content-Type: application/json" \
  -d '{
      "text": "What does NVIDIA love?",
      "sampling_params": {
          "temperature": 0.7,
          "max_new_tokens": 100
      }
  }'
```

## Step 6. Test Python client API

Create a simple Python script to test programmatic access to the SGLang server. This runs on
the host system and demonstrates how to integrate SGLang into applications.

```python
import requests

## Send prompt to server
response = requests.post('http://localhost:30000/generate', json={
  'text': 'What does NVIDIA love?',
  'sampling_params': {
      'temperature': 0.7,
      'max_new_tokens': 100,
  },
})

print(f"Response: {response.json()['text']}")
```

## Step 7. Validate installation

Confirm that both server and offline modes are working correctly. This step verifies the
complete SGLang setup and ensures reliable operation.

```bash
## Check server mode (from host)
curl http://localhost:30000/health
curl -X POST http://localhost:30000/generate -H "Content-Type: application/json" \
  -d '{"text": "Hello", "sampling_params": {"max_new_tokens": 10}}'

## Check container logs
docker ps
docker logs <CONTAINER_ID>
```

## Step 8. Cleanup and rollback

Stop and remove containers to clean up resources. This step returns your system to its
original state.

> [!WARNING]
> This will stop all SGLang containers and remove temporary data.

```bash
## Stop all SGLang containers
docker ps | grep sglang | awk '{print $1}' | xargs docker stop

## Remove stopped containers
docker container prune -f

## Remove SGLang images (optional)
docker rmi lmsysorg/sglang:spark
```

## Step 9. Next steps

With SGLang successfully deployed, you can now:

- Integrate the HTTP API into your applications using the `/generate` endpoint
- Experiment with different models by changing the `--model-path` parameter
- Scale up using multiple GPUs by adjusting the `--tp` (tensor parallel) setting
- Deploy production workloads using the container orchestration platform of your choice

## Troubleshooting

Common issues and their resolutions:

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container fails to start with GPU errors | NVIDIA drivers/toolkit missing | Install nvidia-container-toolkit, restart Docker |
| Server responds with 404 or connection refused | Server not fully initialized | Wait 60 seconds, check container logs |
| Out of memory errors during model loading | Insufficient GPU memory | Use smaller model or increase --tp parameter |
| Model download fails | Network connectivity issues | Check internet connection, retry download |
| Permission denied accessing /tmp | Volume mount issues | Use full path: -v /tmp:/tmp or create dedicated directory |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
