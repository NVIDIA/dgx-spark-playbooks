# Quantize Models to NVFP4 with NVIDIA Model Optimizer

> Cut memory ~3.5× vs FP16 while keeping accuracy close to FP8, then validate with an OpenAI-compatible endpoint


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [DGX Station only — identify the GB300 GPU](#dgx-station-only-identify-the-gb300-gpu)
  - [DGX Spark](#dgx-spark)
  - [DGX Station](#dgx-station)
  - [DGX Spark (TensorRT-LLM)](#dgx-spark-tensorrt-llm)
  - [DGX Station (vLLM)](#dgx-station-vllm)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVFP4 is a 4-bit floating-point format for NVIDIA Blackwell GPUs. It reduces memory bandwidth and storage for inference while keeping accuracy close to higher-precision formats.

Unlike uniform INT4 quantization, NVFP4 keeps floating-point semantics with a shared exponent and a compact mantissa, which improves dynamic range. Blackwell Tensor Cores support mixed-precision execution across FP16, FP8, and FP4, so models can use FP4 for weights and activations while accumulating in higher precision (typically FP16).

Immediate benefits:

- Cut memory use ~3.5× vs FP16 and ~1.8× vs FP8
- Maintain accuracy close to FP8 (usually <1% loss)
- Improve speed and energy efficiency for inference

## What you'll accomplish

You'll produce an **NVFP4 checkpoint** of **DeepSeek-R1-Distill-Llama-8B** with NVIDIA Model Optimizer inside a GPU container on your supported hardware platform. As a validation check, you can load the checkpoint and call an OpenAI-compatible endpoint.

Quantization can change model quality. Run evaluations for your use case before deploying.

## What to know before starting

**Required:**

- Working with Docker containers and GPU-accelerated workloads
- Basic understanding of model quantization and its impact on inference
- Familiarity with Hugging Face model repositories and authentication

**Optional:**

- Experience with NVIDIA TensorRT / CUDA toolkit environments

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. The shared workflow is: prepare Docker and Hugging Face auth → run Model Optimizer NVFP4 quantization → validate artifacts → serve an OpenAI-compatible endpoint. Container image and post-quantization serving stack differ by hardware platform (see Instructions).

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | — |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | — |


## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Enough free disk space for model download and quantized outputs (several GB; plan for tens of GB)

**Software requirements**

- Docker installed with GPU support: `docker --version`
- NVIDIA Container Toolkit configured
- Hugging Face account with an access token and access to the target model
- Network access to NGC / container registry and Hugging Face

Verify your setup:

```bash
## Check GPU visibility on the host
nvidia-smi

## Optional: confirm the default container for your hardware platform can see the GPU
## Set IMAGE to the container image for your hardware platform (see Instructions)
export IMAGE=nvcr.io/nvidia/vllm:25.12.post1-py3   # example — use your hardware platform's image
docker run --rm --gpus all "$IMAGE" nvidia-smi
## On dual-GPU DGX Station, pin the GB300 instead, e.g.:
## docker run --rm --gpus "device=$GPU_ID" "$IMAGE" nvidia-smi

## Verify sufficient disk space
df -h .
```

## Time & risk

- **Estimated time:** 60 MIN (45–90 MIN depending on network speed and model size)
- **Risk level:** Medium
  - Model download may fail due to network issues or Hugging Face authentication
  - Quantization is memory-intensive and can fail if GPU memory is insufficient
  - Output files are large and need adequate storage
- **Rollback:** Remove the output directory and optionally remove pulled Docker images to restore the original state (see Cleanup in Instructions)
- **Last Updated:** 08/03/2026
  - Updated Model Optimizer documentation and GitHub links after repository rename
  - 07/27/2026: Added supported hardware platforms matrix

## Instructions

> [!NOTE]
> These instructions target **Linux** on a supported hardware platform. Use the default container and command block for your hardware platform from the tables below.

## Step 1. Set up Docker permissions

To manage containers without `sudo`, add your user to the `docker` group. Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error, add your user to the docker group (skip if it already works):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Prepare the environment

Create a local output directory for quantized model files. The directory is mounted into the container so results persist after the container exits.

```bash
mkdir -p ./output_models
chmod 755 ./output_models
```

## Step 3. Authenticate with Hugging Face

Export a Hugging Face token so the container can download the model:

```bash
## Get a token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"
```

## Step 4. Set hardware platform launch variables

Container image, GPU device selection, and Model Optimizer pin differ by hardware platform. Set the variables that match your hardware platform before running quantization.

| Hardware platform | Container image | GPU device | Model Optimizer |
| ----------------- | --------------- | ---------- | --------------- |
| **DGX Spark** | `nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev` | `--gpus all` | `NVIDIA/Model-Optimizer` @ `0.35.0` |
| **DGX Station** | `nvcr.io/nvidia/vllm:25.12.post1-py3` | `--gpus "device=$GPU_ID"` (GB300) | `NVIDIA/Model-Optimizer` @ `0.41.0` |

### DGX Station only — identify the GB300 GPU

If the system has more than one GPU, identify the GB300 device ID:

```bash
nvidia-smi
```

Example (GB300 is device **1**):

```text
|   0  NVIDIA RTX 6000  ...
|   1  NVIDIA GB300     ...
```

```bash
export GPU_ID=1  # Replace with your GB300 device number
```

On a single-GPU DGX Station (GB300 only), use `GPU_ID=0`.

## Step 5. Run NVFP4 quantization with Model Optimizer

Use the command block for your hardware platform. Both paths quantize `deepseek-ai/DeepSeek-R1-Distill-Llama-8B` to NVFP4 and write artifacts under `./output_models`.

### DGX Spark

```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.35.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && \
    cd /app/Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model 'deepseek-ai/DeepSeek-R1-Distill-Llama-8B' \
    --quant nvfp4 \
    --tp 1 \
    --export_fmt hf
  "
```

Expected output directory:

```bash
export MODEL_PATH="./output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4_hf/"
```

### DGX Station

```bash
docker run --rm -it --gpus "device=$GPU_ID" --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  bash -c "
    git clone -b 0.41.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && \
    cd /app/Model-Optimizer && pip install -e '.[dev]' && \
    export ROOT_SAVE_PATH='/workspace/output_models' && \
    /app/Model-Optimizer/examples/llm_ptq/scripts/huggingface_example.sh \
    --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
    --quant nvfp4 \
    --tasks quant
  "
```

Expected output directory:

```bash
export MODEL_PATH="./output_models/saved_models_DeepSeek-R1-Distill-Llama-8B_nvfp4/"
```

> [!NOTE]
> - You can safely ignore `No module named 'mpi4py'` during quantization when it appears; it does not block NVFP4 export.
> - `pynvml.NVMLError_NotSupported: Not Supported` can appear in some environments and does not affect results.
> - If the model is too large for available GPU memory, try a smaller model.

What this step does:

- Runs the container with GPU access and shared-memory settings suitable for large models
- Mounts `./output_models` and your Hugging Face cache
- Installs NVIDIA Model Optimizer and runs the NVFP4 quantization script

## Step 6. Monitor the quantization process

Watch for:

- Model download progress from Hugging Face
- Quantization calibration steps
- Model export and validation phases

## Step 7. Validate quantized model files

After the container exits, confirm artifacts exist:

```bash
ls -la ./output_models/

find ./output_models/ \( -name "*.bin" -o -name "*.safetensors" -o -name "*.json" -o -name "*.jinja" \)
```

You should see weight files, configuration, and tokenizer files under the `MODEL_PATH` for your hardware platform.

## Step 8. Serve and test with an OpenAI-compatible API

Post-quantization validation and serving use the stack that matches your hardware platform. Set `MODEL_PATH` from Step 5 if it is not already set.

### DGX Spark (TensorRT-LLM)

Load-test the checkpoint:

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c '
    python examples/llm-api/quickstart_advanced.py \
      --model_dir /workspace/model/ \
      --prompt "Paris is great because" \
      --max_tokens 64
    '
```

Start the OpenAI-compatible server:

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  trtllm-serve /workspace/model \
    --backend pytorch \
    --max_batch_size 4 \
    --port 8000
```

In another terminal:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

### DGX Station (vLLM)

Start the OpenAI-compatible server (also used as the load check):

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$MODEL_PATH:/workspace/model" \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus "device=$GPU_ID" --ipc=host --network host \
  nvcr.io/nvidia/vllm:25.12.post1-py3 \
  vllm serve /workspace/model \
    --served-model-name DeepSeek-R1-Distill-Llama-8B-NVFP4 \
    --max-num-seqs 4 \
    --max-model-len 8192 \
    --port 8000
```

`--served-model-name` sets the model ID returned by the API. Without it, vLLM defaults to the mount path. Confirm with `curl http://localhost:8000/v1/models`, then:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "DeepSeek-R1-Distill-Llama-8B-NVFP4",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

Adjust knobs such as `--max-model-len` for your workload. Stop the server with **Ctrl+C** when finished.

## Step 9. Cleanup and rollback

> [!WARNING]
> This permanently deletes quantized model files and optional cached data.

> [!NOTE]
> Quantization containers may write `./output_models/` as root. Try without `sudo` first; if you get permission denied, retry with `sudo`.

```bash
## Remove quantized outputs
rm -rf ./output_models
## If permission denied (root-owned files from the container):
## sudo rm -rf ./output_models

## Optional: remove Hugging Face cache
rm -rf ~/.cache/huggingface

## Optional: remove the container image for your hardware platform
## DGX Spark:
## docker rmi nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
## DGX Station:
## docker rmi nvcr.io/nvidia/vllm:25.12.post1-py3
```

## Step 10. Next steps

The quantized model is ready for further use. Common follow-ups:

- Benchmark inference performance against the original model
- Integrate the checkpoint into your inference pipeline
- Deploy with NVIDIA Triton Inference Server for production serving
- Run additional validation on your target prompts and evaluation sets

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| "Permission denied" when accessing Hugging Face | All hardware platforms | Missing or invalid HF token | Export a valid `HF_TOKEN`, or run `huggingface-cli login` |
| Cannot access gated repo for URL | All hardware platforms | Restricted Hugging Face model | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated) |
| Container exits with CUDA out of memory | All hardware platforms | Insufficient GPU memory | Quantize a smaller model, or free GPU memory and retry |
| Model files not found in output directory | All hardware platforms | Volume mount failed or wrong path | Verify `./output_models` resolves from your working directory and remount |
| Git clone fails inside container | All hardware platforms | Network connectivity issues | Check internet access and retry |
| Quantization process hangs | All hardware platforms | Container resource limits | Increase Docker memory limits or keep `--ulimit` flags from Instructions |
| Log ends with MPI or `ModuleNotFoundError: No module named 'mpi4py'` | DGX Station | Optional MPI runner step after quant | Confirm NVFP4 artifacts exist under `./output_models`; quantization may have succeeded even if the final runner step fails |
| Model runs on wrong GPU | DGX Station | Dual-GPU default selection | Set `GPU_ID` to the GB300 device from `nvidia-smi` and use `--gpus "device=$GPU_ID"` |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |
| Permission denied removing `./output_models` | All hardware platforms | Root-owned files from container | Use `sudo rm -rf ./output_models` |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```
