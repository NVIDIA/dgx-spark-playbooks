# Run Multi-Modal Inference with TensorRT

> GPU-accelerated text-to-image generation with diffusion models

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Multi-modal inference combines different data types, such as **text, images, and audio**, within a single model pipeline to generate or interpret richer outputs. Instead of processing one input type at a time, multi-modal systems share representations that support **text-to-image generation**, **image captioning**, or **vision-language reasoning**.

On GPUs, this enables **parallel processing across modalities** for faster, higher-fidelity results on tasks that combine language and vision.

## What you'll accomplish

You'll deploy GPU-accelerated multi-modal inference on your **hardware platform** using torch and torch-TensorRT to run Flux.1 diffusion models with optimized performance across multiple precision formats (BF16, FP16, FP8, FP4).

## What to know before starting

**Required:**

- Working with Docker containers and GPU passthrough
- Using torch-TensorRT for model optimization
- Hugging Face model hub authentication and downloads
- Command-line tools for GPU workloads

**Optional:**

- Basic understanding of diffusion models and image generation

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, default container image, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB unified memory | `nvcr.io/nvidia/pytorch:25.11-py3` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see the matrix above
- At least 48 GB available memory for FP16 Flux.1 Schnell operations
- Sufficient available storage for container images, Hugging Face model downloads, and generated outputs

**Software requirements**

- NVIDIA driver and GPU visibility: `nvidia-smi`
- Docker installed and accessible to the current user: `docker --version`
- NVIDIA Container Toolkit configured for Docker
- Hugging Face account with access to Black Forest Labs models [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [FLUX.1-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx)
- Hugging Face [token](https://huggingface.co/settings/tokens) configured with access to both FLUX.1 model repositories
- Network access to NGC and Hugging Face

Verify GPU and Docker GPU integration:

```bash
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi
```

## Find model recipes

Start with the Flux.1 Dev and Flux.1 Schnell precision examples in **Instructions**. For additional diffusion demos, scripts, and dependency files, use the torch-TensorRT documentation (Compiling LLM models from Huggingface) **Resources**.

| Hardware platform | More recipes |
| ----------------- | ------------ |
| **DGX Spark** | [Compiling LLM models from Huggingface](https://docs.pytorch.org/TensorRT/tutorials/_rendered_examples/dynamo/torch_export_flux_dev.html) |

Follow the documentation for an example inference workflow for Flux.1-dev.

> [!NOTE]
> **Memory determines what you can run.** FP16 Flux.1 Schnell needs substantially more memory than FP8 or FP4. If a model or precision is not listed for your hardware platform, confirm it fits available memory before downloading.

## Ancillary files

Alternatively, use the example hosted on [torch-TensorRT](https://github.com/pytorch/TensorRT/tree/main/examples/apps):

- [**README.md**](https://github.com/pytorch/TensorRT/blob/main/examples/apps/README.md) — Explanation how to run the example
- [**flux_demo.py**](https://github.com/pytorch/TensorRT/blob/main/examples/apps/flux_demo.py) — Flux.1 model inference script

## Time & risk

- **Estimated time:** 60 MIN (longer on first run due to model downloads and optimization)
- **Risk level:** Medium
  - Large model downloads may time out
  - High memory requirements may cause out-of-memory errors
  - Quantized models may show quality differences versus full precision
- **Rollback:** Exit the container, then optionally remove downloaded models from the Hugging Face cache
- **Last Updated:** 08/31/2026
  - Flux.1 TensorRT inference workflow via torch-TensorRT for text-to-image generation

## Instructions

## Step 1. Configure Docker permissions

To manage containers without sudo, add your user to the `docker` group. If you skip this step, run Docker commands with sudo.

Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission denied error (for example, permission denied while trying to connect to the Docker daemon socket), add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Launch the TensorRT container environment

Start the NVIDIA PyTorch container with GPU access and Hugging Face cache mounting. This provides the TensorRT development environment with required dependencies pre-installed.

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
  --ulimit stack=67108864 -it --rm \
  -v $HOME/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 3. Clone and set up the TensorRT repository

Download the TensorRT repository and configure the environment for diffusion model demos.

```bash
git clone https://github.com/NVIDIA/TensorRT.git -b main --single-branch && cd TensorRT
export TRT_OSSPATH=/workspace/TensorRT/
cd $TRT_OSSPATH/demo/Diffusion
```

## Step 4. Install required dependencies

Install NVIDIA Model Optimizer and other dependencies for model quantization and optimization.

```bash
## Install OpenGL libraries
apt update
apt install -y libgl1 libglu1-mesa libglib2.0-0t64 libxrender1 libxext6 libx11-6 libxrandr2 libxss1 libxcomposite1 libxdamage1 libxfixes3 libxcb1

pip install nvidia-modelopt[torch,onnx]
sed -i '/^nvidia-modelopt\[.*\]=.*/d' requirements.txt
pip3 install -r requirements.txt
pip install onnxconverter_common
```

Set up your Hugging Face token to access gated models:

```bash
export HF_TOKEN=<YOUR_HUGGING_FACE_TOKEN>
```

## Step 5. Run Flux.1 Dev model inference

Test multi-modal inference using the Flux.1 Dev model with different precision formats.

**Substep A. BF16 precision**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --download-onnx-models --bf16
```

**Substep B. FP8 quantized precision**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --quantization-level 4 --fp8 --download-onnx-models
```

**Substep C. FP4 quantized precision**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --fp4 --download-onnx-models
```

## Step 6. Run Flux.1 Schnell model inference

Test the faster Flux.1 Schnell variant with different precision formats.

> [!WARNING]
> FP16 Flux.1 Schnell requires more than 48 GB of available memory for native export.

**Substep A. FP16 precision (high memory requirement)**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version="flux.1-schnell"
```

**Substep B. FP8 quantized precision**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version="flux.1-schnell" \
  --quantization-level 4 --fp8 --download-onnx-models
```

**Substep C. FP4 quantized precision**

```bash
python3 demo_txt2img_flux.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version="flux.1-schnell" \
  --fp4 --download-onnx-models
```

## Step 7. Validate inference outputs

Confirm that the models generated images successfully and that TensorRT is available.

```bash
## Check for generated images in the output directory
ls -la *.png *.jpg 2>/dev/null || echo "No image files found"

## Verify CUDA is accessible
nvidia-smi

## Check TensorRT version
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

## Step 8. Cleanup and rollback

Remove downloaded models and exit the container environment to free disk space when you are finished.

> [!WARNING]
> This deletes cached models and generated images if you remove the Hugging Face cache.

```bash
## Exit container
exit

## Remove Hugging Face cache (optional)
rm -rf $HOME/.cache/huggingface/
```

## Step 9. Next steps

Use the validated setup to generate custom images or integrate multi-modal inference into your applications. Try different prompts, precision formats, or explore model fine-tuning with the established TensorRT environment.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "CUDA out of memory" error | Insufficient memory for the selected model or precision | Use FP8/FP4 quantization or a smaller model variant |
| "Invalid HF token" error | Missing or expired Hugging Face token | Set a valid token: `export HF_TOKEN=<YOUR_TOKEN>` |
| Cannot access gated repo for URL | Certain Hugging Face models have restricted access | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens); request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) in your browser |
| Model download timeouts | Network issues or rate limiting | Retry the command or pre-download models |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory issues even when within rated capacity. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
