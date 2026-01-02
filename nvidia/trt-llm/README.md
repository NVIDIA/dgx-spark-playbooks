# TRT LLM for Inference

> Install and use TensorRT-LLM on DGX Spark

## Table of Contents

- [Overview](#overview)
- [Single Spark](#single-spark)
- [Run on two Sparks](#run-on-two-sparks)
  - [Step 1. Configure network connectivity](#step-1-configure-network-connectivity)
  - [Step 2. Configure Docker permissions](#step-2-configure-docker-permissions)
  - [Step 3. Create OpenMPI hostfile](#step-3-create-openmpi-hostfile)
  - [Step 4. Start containers on both nodes](#step-4-start-containers-on-both-nodes)
  - [Step 5. Verify containers are running](#step-5-verify-containers-are-running)
  - [Step 6. Copy hostfile to primary container](#step-6-copy-hostfile-to-primary-container)
  - [Step 7. Save container reference](#step-7-save-container-reference)
  - [Step 8. Generate configuration file](#step-8-generate-configuration-file)
  - [Step 9. Download model](#step-9-download-model)
  - [Step 10. Serve the model](#step-10-serve-the-model)
  - [Step 11. Validate API server](#step-11-validate-api-server)
  - [Step 12. Cleanup and rollback](#step-12-cleanup-and-rollback)
  - [Step 13. Next steps](#step-13-next-steps)
- [Open WebUI for TensorRT-LLM](#open-webui-for-tensorrt-llm)
  - [Step 1. Set up the prerequisites to use Open WebUI with TRT-LLM](#step-1-set-up-the-prerequisites-to-use-open-webui-with-trt-llm)
  - [Step 2. Launch Open WebUI container](#step-2-launch-open-webui-container)
  - [Step 3. Access the Open WebUI interface](#step-3-access-the-open-webui-interface)
  - [Step 4. Cleanup and rollback](#step-4-cleanup-and-rollback)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

**NVIDIA TensorRT-LLM (TRT-LLM)** is an open-source library for optimizing and accelerating large language model (LLM) inference on NVIDIA GPUs.

It provides highly efficient kernels, memory management, and parallelism strategies—like tensor, pipeline, and sequence parallelism—so developers can serve LLMs with lower latency and higher throughput.

TRT-LLM integrates with frameworks like Hugging Face and PyTorch, making it easier to deploy state-of-the-art models at scale.


## What you'll accomplish

You'll set up TensorRT-LLM to optimize and deploy large language models on your DGX Spark, achieving significantly higher throughput and lower latency than standard PyTorch
inference through kernel-level optimizations, efficient memory layouts, and advanced quantization.

## What to know before starting

- Python proficiency and experience with PyTorch or similar ML frameworks
- Command-line comfort for running CLI tools and Docker containers
- Basic understanding of GPU concepts including VRAM, batching, and quantization (FP16/INT8)
- Familiarity with NVIDIA software stack (CUDA Toolkit, drivers)
- Experience with inference servers and containerized environments

## Prerequisites

- DGX Spark device
- NVIDIA drivers compatible with CUDA 12.x: `nvidia-smi`
- Docker installed and GPU support configured: `docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 nvidia-smi`
- Hugging Face account with token for model access: `echo $HF_TOKEN`
- Sufficient GPU VRAM (40GB+ recommended for 70B models)
- Internet connectivity for downloading models and container images
- Network: open TCP ports 8355 (LLM) and 8356 (VLM) on host for OpenAI-compatible serving

## Ancillary files

All required assets can be found [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main)

- [**trtllm-mn-entrypoint.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh) — container entrypoint script for multi-node setup

## Model Support Matrix

The following models are supported with TensorRT-LLM on Spark. All listed models are available and ready to use:

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
| **Qwen3-30B-A3B** | NVFP4 | ✅ | `nvidia/Qwen3-30B-A3B-FP4` |
| **Llama-4-Scout-17B-16E-Instruct** | NVFP4 | ✅ | `nvidia/Llama-4-Scout-17B-16E-Instruct-FP4` |
| **Qwen3-235B-A22B (two Sparks only)** | NVFP4 | ✅ | `nvidia/Qwen3-235B-A22B-FP4` |

> [!NOTE]
> You can use the NVFP4 Quantization documentation to generate your own NVFP4-quantized checkpoints for your favorite models. This enables you to take advantage of the performance and memory benefits of NVFP4 quantization even for models not already published by NVIDIA.

Reminder: not all model architectures are supported for NVFP4 quantization.

## Time & risk

* **Duration**: 45-60 minutes for setup and API server deployment
* **Risk level**: Medium - container pulls and model downloads may fail due to network issues
* **Rollback**: Stop inference servers and remove downloaded models to free resources.
* **Last Updated:** 01/02/2026
  * Improve TRT-LLM Run on Two Sparks workflow
  * Upgrade to the latest TRT-LLM container v1.2.0rc6

## Single Spark

## Step 1. Configure Docker permissions

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

## Step 2. Verify environment prerequisites

Confirm your Spark device has the required GPU access and network connectivity for downloading
models and containers.

```bash
## Check GPU visibility and driver
nvidia-smi

## Verify Docker GPU support
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 nvidia-smi

```

## Step 3. Set environment variables

```bash
## Set `HF_TOKEN` for model access.
export HF_TOKEN=<your-huggingface-token>

export DOCKER_IMAGE="nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6"
```

## Step 4. Validate TensorRT-LLM installation

After confirming GPU access, verify that TensorRT-LLM can be imported inside the container.

```bash
docker run --rm -it --gpus all \
  $DOCKER_IMAGE \
  python -c "import tensorrt_llm; print(f'TensorRT-LLM version: {tensorrt_llm.__version__}')"
```

Expected output:
```
[TensorRT-LLM] TensorRT-LLM version: 1.2.0rc6
TensorRT-LLM version: 1.2.0rc6
```

## Step 5. Create cache directory

Set up local caching to avoid re-downloading models on subsequent runs.

```bash
## Create Hugging Face cache directory
mkdir -p $HOME/.cache/huggingface/
```

## Step 6. Validate setup with quickstart_advanced

This quickstart validates your TensorRT-LLM setup end-to-end by testing model loading, inference engine initialization, and GPU execution with real text generation. It's the fastest way to confirm everything works before starting the inference API server.

**LLM quickstart example**

#### Llama 3.1 8B Instruct
```bash
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
    hf download $MODEL_HANDLE && \
    python examples/llm-api/quickstart_advanced.py \
      --model_dir $MODEL_HANDLE \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

#### GPT-OSS 20B
```bash
export MODEL_HANDLE="openai/gpt-oss-20b"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    python examples/llm-api/quickstart_advanced.py \
      --model_dir $MODEL_HANDLE \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

#### GPT-OSS 120B
```bash
export MODEL_HANDLE="openai/gpt-oss-120b"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    python examples/llm-api/quickstart_advanced.py \
      --model_dir $MODEL_HANDLE \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

## Step 7. Validate setup with quickstart_multimodal

**VLM quickstart example**

This demonstrates vision-language model capabilities by running inference with image understanding. The example uses multimodal inputs to validate both text and vision processing pipelines.

#### Phi-4-multimodal-instruct

This model requires LoRA (Low-Rank Adaptation) configuration as it uses parameter-efficient fine-tuning. The `--load_lora` flag enables loading the LoRA weights that adapt the base model for multimodal instruction following.
```bash
export MODEL_HANDLE="nvidia/Phi-4-multimodal-instruct-FP4"

docker run \
  -e MODEL_HANDLE=$MODEL_HANDLE \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  $DOCKER_IMAGE \
  bash -c '
  python3 examples/llm-api/quickstart_multimodal.py \
    --model_type phi4mm \
    --model_dir $MODEL_HANDLE \
    --modality image \
    --media "https://huggingface.co/datasets/YiYiXu/testing-images/resolve/main/seashore.png" \
    --prompt "What is happening in this image?" \
    --load_lora \
    --auto_model_name Phi4MMForCausalLM
  '
```


> [!NOTE]
> If you hit a host OOM during downloads or first run, free the OS page cache on the host (outside the container) and retry:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

## Step 8. Serve LLM with OpenAI-compatible API

Serve with OpenAI-compatible API via trtllm-serve:

#### Llama 3.1 8B Instruct
```bash
export MODEL_HANDLE="nvidia/Llama-3.1-8B-Instruct-FP4"

docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  $DOCKER_IMAGE \
  bash -c '
    hf download $MODEL_HANDLE && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 64 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

#### GPT-OSS 20B
```bash
export MODEL_HANDLE="openai/gpt-oss-20b"

docker run --name trtllm_llm_server --rm -it --gpus all --ipc host --network host \
  -e HF_TOKEN=$HF_TOKEN \
  -e MODEL_HANDLE="$MODEL_HANDLE" \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  $DOCKER_IMAGE \
  bash -c '
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken && \
    hf download $MODEL_HANDLE && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
disable_overlap_scheduler: true
EOF
    trtllm-serve "$MODEL_HANDLE" \
      --max_batch_size 64 \
      --trust_remote_code \
      --port 8355 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml
  '
```

Minimal OpenAI-style chat request. Run this from a separate terminal.

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "'"$MODEL_HANDLE"'",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

## Step 9. Cleanup and rollback

Remove downloaded models and containers to free up space when testing is complete.

> [!WARNING]
> This will delete all cached models and may require re-downloading for future runs.

```bash
## Remove Hugging Face cache
sudo chown -R "$USER:$USER" "$HOME/.cache/huggingface"
rm -rf $HOME/.cache/huggingface/

## Clean up Docker images
docker image prune -f
docker rmi $DOCKER_IMAGE
```

## Run on two Sparks

### Step 1. Configure network connectivity

Follow the network setup instructions from the [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

### Step 2. Configure Docker permissions

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

Repeat this step on both nodes.

### Step 3. Create OpenMPI hostfile

Create a hostfile with the IP addresses of both nodes for MPI operations. On each node, get the IP address of your network interface:

```bash
ip a show enp1s0f0np0
```

Or if you're using the second interface:

```bash
ip a show enp1s0f1np1
```

Look for the `inet` line to find the IP address (e.g., `192.168.1.10/24`).

On your primary node, create the hostfile `~/openmpi-hostfile` with the collected IPs:

```bash
cat > ~/openmpi-hostfile <<EOF
192.168.1.10
192.168.1.11
EOF
```

Replace the IP addresses with your actual node IPs.

### Step 4. Start containers on both nodes

On **each node** (primary and worker), run the following command to start the TRT-LLM container:

```bash
docker run -d --rm \
  --name trtllm-multinode \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f0np0,enp1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f0np0,enp1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
  sh -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | sh"
```

> [!NOTE]
> Make sure to run this command on **both** the primary and worker nodes.

### Step 5. Verify containers are running

On each node, verify the container is running:

```bash
docker ps
```

You should see output similar to:

```
CONTAINER ID   IMAGE                                                 COMMAND                  CREATED          STATUS          PORTS     NAMES
abc123def456   nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6         "sh -c 'curl https:…"    10 seconds ago   Up 8 seconds              trtllm-multinode
```

### Step 6. Copy hostfile to primary container

On your primary node, copy the OpenMPI hostfile into the container:

```bash
docker cp ~/openmpi-hostfile trtllm-multinode:/etc/openmpi-hostfile
```

### Step 7. Save container reference

On your primary node, save the container name in a variable for convenience:

```bash
export TRTLLM_MN_CONTAINER=trtllm-multinode
```

### Step 8. Generate configuration file

On your primary node, generate the configuration file inside the container:

```bash
docker exec $TRTLLM_MN_CONTAINER bash -c 'cat <<EOF > /tmp/extra-llm-api-config.yml
print_iter_log: false
kv_cache_config:
  dtype: "auto"
  free_gpu_memory_fraction: 0.9
cuda_graph_config:
  enable_padding: true
EOF'
```

### Step 9. Download model

We can download a model using the following command. You can replace `nvidia/Qwen3-235B-A22B-FP4` with the model of your choice.

```bash
## Need to specify huggingface token for model download.
export HF_TOKEN=<your-huggingface-token>

docker exec \
  -e MODEL="nvidia/Qwen3-235B-A22B-FP4" \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c 'mpirun -x HF_TOKEN bash -c "hf download $MODEL"'
```

### Step 10. Serve the model

On your primary node, start the TensorRT-LLM server:

```bash
docker exec \
  -e MODEL="nvidia/Qwen3-235B-A22B-FP4" \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c '
    mpirun -x HF_TOKEN trtllm-llmapi-launch trtllm-serve $MODEL \
      --tp_size 2 \
      --backend pytorch \
      --max_num_tokens 32768 \
      --max_batch_size 4 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml \
      --port 8355'
```

This will start the TensorRT-LLM server on port 8355. You can then make inference requests to `http://localhost:8355` using the OpenAI-compatible API format.

> [!NOTE]
> You might see a warning such as `UCX  WARN  network device 'enp1s0f0np0' is not available, please use one or more of`. You can ignore this warning if your inference is successful, as it's related to only one of your two CX-7 ports being used, and the other being left unused.

**Expected output:** Server startup logs and ready message.

### Step 11. Validate API server

Once the server is running, you can test it with a CURL request. Run this on the primary node:

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-235B-A22B-FP4",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

**Expected output:** JSON response with generated text completion.

### Step 12. Cleanup and rollback

Stop and remove containers on **each node**. SSH to each node and run:

```bash
docker stop trtllm-multinode
```

> [!WARNING]
> This removes all inference data and performance reports. Copy any necessary files before cleanup if needed.

Remove downloaded models to free disk space on each node:

```bash
rm -rf $HOME/.cache/huggingface/hub/models--nvidia--Qwen3*
```

### Step 13. Next steps

You can now deploy other models on your DGX Spark cluster.

## Open WebUI for TensorRT-LLM

### Step 1. Set up the prerequisites to use Open WebUI with TRT-LLM

After setting up TensorRT-LLM inference server in either single-node or multi-node configuration, 
you can deploy Open WebUI to interact with your models through Open WebUI. To get setup, just make sure the following 
is in order

- TensorRT-LLM inference server running and accessible at http://localhost:8355
- Docker installed and configured (see earlier steps)
- Port 3000 available on your DGX Spark

### Step 2. Launch Open WebUI container

Run the following command on the DGX Spark node where you have the TensorRT-LLM inference server running.
For multi-node setup, this would be the primary node.

> [!NOTE]
> If you used a different port for your OpenAI-compatible API server, adjust the `OPENAI_API_BASE_URL="http://localhost:8355/v1"` to match the IP and port of your TensorRT-LLM inference server.

```bash
docker run \
  -d \
  -e OPENAI_API_BASE_URL="http://localhost:8355/v1" \
  -v open-webui:/app/backend/data \
  --network host \
  --add-host=host.docker.internal:host-gateway \
  --name open-webui \
  --restart always \
  ghcr.io/open-webui/open-webui:main
```

This command:
- Connects to your OpenAI-compatible API server for TensorRT-LLM at http://localhost:8355
- Provides access to the Open WebUI interface at http://localhost:8080
- Persists chat data in a Docker volume
- Enables automatic container restart
- Uses the latest Open WebUI image

### Step 3. Access the Open WebUI interface

Open your web browser and navigate to:

```
http://localhost:8080
```

You should see the Open WebUI interface at http://localhost:8080 where you can:
- Chat with your deployed models
- Adjust model parameters
- View chat history
- Manage model configurations

You can select your model(s) from the dropdown menu on the top left corner. That's all you need to do to start using Open WebUI with your deployed models.

> [!NOTE]
> If accessing from a remote machine, replace localhost with your DGX Spark's IP address.

### Step 4. Cleanup and rollback
> [!WARNING]
> This removes all chat data and may require re-uploading for future runs.

Remove the container by using the following command:
```bash
docker stop open-webui
docker rm open-webui
docker volume rm open-webui
docker rmi ghcr.io/open-webui/open-webui:main
```

## Troubleshooting

## Common issues for running on a single Spark

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| OOM during weight loading (e.g., [Nemotron Super 49B](https://huggingface.co/nvidia/Llama-3_3-Nemotron-Super-49B-v1_5)) | Parallel weight-loading memory pressure | `export TRT_LLM_DISABLE_LOAD_WEIGHTS_IN_PARALLEL=1` |
| "CUDA out of memory" | GPU VRAM insufficient for model | Reduce `free_gpu_memory_fraction: 0.9` or batch size or use smaller model |
| "Model not found" error | HF_TOKEN invalid or model inaccessible | Verify token and model permissions |
| Container pull timeout | Network connectivity issues | Retry pull or use local mirror |
| Import tensorrt_llm fails | Container runtime issues | Restart Docker daemon and retry |

## Common Issues for running on two Sparks

| Symptom | Cause | Fix |
|---------|-------|-----|
| MPI hostname test returns single hostname | Network connectivity issues | Verify both nodes are on reachable IP addresses |
| "Permission denied" on HuggingFace download | Invalid or missing HF_TOKEN | Set valid token: `export HF_TOKEN=<TOKEN>` |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| "CUDA out of memory" errors | Insufficient GPU memory | Reduce `--max_batch_size` or `--max_num_tokens` |
| Container exits immediately | Missing entrypoint script | Ensure `trtllm-mn-entrypoint.sh` download succeeded and has executable permissions, also ensure you are not running the container already on your node. If port 2233 is already utilized, the entrypoint script will not start. |
| Error response from daemon: error while validating Root CA Certificate | System clock out of sync or expired certificates | Update system time to sync with NTP server `sudo timedatectl set-ntp true`|
| "invalid mount config for type 'bind'" | Missing or non-executable entrypoint script | Run `docker inspect <container_id>` to see full error message. Verify `trtllm-mn-entrypoint.sh` exists on both nodes in your home directory (`ls -la $HOME/trtllm-mn-entrypoint.sh`) and has executable permissions (`chmod +x $HOME/trtllm-mn-entrypoint.sh`) |
| "task: non-zero exit (255)" | Container exit with error code 255 | Check container logs with `docker ps -a --filter "name=trtllm-multinode_trtllm"` to get container ID, then `docker logs <container_id>` to see detailed error messages |
| Docker state stuck in "Pending" with "no suitable node (insufficien...)" | Docker daemon not properly configured for GPU access | Verify steps 2-4 were completed successfully and check that `/etc/docker/daemon.json` contains correct GPU configuration |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
