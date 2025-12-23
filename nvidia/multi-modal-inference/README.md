# Multi-modal Inference

> Setup multi-modal inference with TensorRT

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Multi-modal inference combines different data types, such as **text, images, and audio**, within a single model pipeline to generate or interpret richer outputs.  
Instead of processing one input type at a time, multi-modal systems have shared representations that  **text-to-image generation**, **image captioning**, or **vision-language reasoning**.  

On GPUs, this enables **parallel processing across modalities** for faster, higher-fidelity results for tasks that combine language and vision.

## What you'll accomplish

You'll deploy GPU-accelerated multi-modal inference capabilities on NVIDIA Spark using TensorRT to run 
Flux.1 and SDXL diffusion models with optimized performance across multiple precision formats (FP16, 
FP8, FP4).

## What to know before starting

- Working with Docker containers and GPU passthrough
- Using TensorRT for model optimization
- Hugging Face model hub authentication and downloads  
- Command-line tools for GPU workloads
- Basic understanding of diffusion models and image generation

## Prerequisites

- NVIDIA Spark device with Blackwell GPU architecture
- Docker installed and accessible to current user
- NVIDIA Container Runtime configured
- Hugging Face account with access to Black Forest Labs models [FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) and [FLUX.1-dev-onnx](https://huggingface.co/black-forest-labs/FLUX.1-dev-onnx) on Hugging Face
- Hugging Face [token](https://huggingface.co/settings/tokens) configured with access to both FLUX.1 model repositories
- At least 48GB VRAM available for FP16 Flux.1 Schnell operations
- Verify GPU access: `nvidia-smi`
- Check Docker GPU integration: `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi`

## Ancillary files

All necessary files can be found in the TensorRT repository [here on GitHub](https://github.com/NVIDIA/TensorRT)
- [**requirements.txt**](https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/requirements.txt) - Python dependencies for TensorRT demo environment
- [**demo_txt2img_flux.py**](https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/demo_txt2img_flux.py) - Flux.1 model inference script  
- [**demo_txt2img_xl.py**](https://github.com/NVIDIA/TensorRT/blob/main/demo/Diffusion/demo_txt2img_xl.py) - SDXL model inference script
- **TensorRT repository** - Contains diffusion demo code and optimization tools

## Time & risk

- **Duration**: 45-90 minutes depending on model downloads and optimization steps

- **Risks**: 
  - Large model downloads may timeout
  - High VRAM requirements may cause OOM errors
  - Quantized models may show quality degradation

- **Rollback**: 
  - Remove downloaded models from HuggingFace cache
  - Then exit the container environment

* **Last Updated:** 12/22/2025
  * Upgrade to latest pytorch container version nvcr.io/nvidia/pytorch:25.11-py3
  * Add HuggingFace token setup instructions for model access
  * Add docker container permission setup instructioins

## Instructions

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

## Step 2. Launch the TensorRT container environment

Start the NVIDIA PyTorch container with GPU access and HuggingFace cache mounting. This provides 
the TensorRT development environment with all required dependencies pre-installed.

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 \
--ulimit stack=67108864 -it --rm --ipc=host \
-v $HOME/.cache/huggingface:/root/.cache/huggingface \
nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 3. Clone and set up TensorRT repository

Download the TensorRT repository and configure the environment for diffusion model demos.

```bash
git clone https://github.com/NVIDIA/TensorRT.git -b main --single-branch && cd TensorRT
export TRT_OSSPATH=/workspace/TensorRT/
cd $TRT_OSSPATH/demo/Diffusion
```

## Step 4. Install required dependencies

Install NVIDIA ModelOpt and other dependencies for model quantization and optimization.

```bash
## Install OpenGL libraries
apt update
apt install -y libgl1 libglu1-mesa libglib2.0-0t64 libxrender1 libxext6 libx11-6 libxrandr2 libxss1 libxcomposite1 libxdamage1 libxfixes3 libxcb1

pip install nvidia-modelopt[torch,onnx]
sed -i '/^nvidia-modelopt\[.*\]=.*/d' requirements.txt
pip3 install -r requirements.txt
pip install onnxconverter_common
```

Set up your HuggingFace token to access open models.
```bash
export HF_TOKEN = <YOUR_HUGGING_FACE_TOKEN>
```

## Step 5. Run Flux.1 Dev model inference

Test multi-modal inference using the Flux.1 Dev model with different precision formats.

**Substep A. BF16 quantized precision**

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
> FP16 Flux.1 Schnell requires >48GB VRAM for native export

**Substep A. FP16 precision (high VRAM requirement)**

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

## Step 7. Run SDXL model inference

Test the SDXL model for comparison with different precision formats.

**Substep A. BF16 precision**

```bash
python3 demo_txt2img_xl.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version xl-1.0 --download-onnx-models
```

**Substep B. FP8 quantized precision**

```bash
python3 demo_txt2img_xl.py "a beautiful photograph of Mt. Fuji during cherry blossom" \
  --hf-token=$HF_TOKEN --version xl-1.0 --download-onnx-models --fp8
```

## Step 8. Validate inference outputs

Check that the models generated images successfully and measure performance differences.

```bash
## Check for generated images in output directory
ls -la *.png *.jpg 2>/dev/null || echo "No image files found"

## Verify CUDA is accessible
nvidia-smi

## Check TensorRT version
python3 -c "import tensorrt as trt; print(f'TensorRT version: {trt.__version__}')"
```

## Step 9. Cleanup and rollback

Remove downloaded models and exit container environment to free disk space.

> [!WARNING]
> This will delete all cached models and generated images

```bash
## Exit container
exit

## Remove HuggingFace cache (optional)
rm -rf $HOME/.cache/huggingface/
```

## Step 10. Next steps

Use the validated setup to generate custom images or integrate multi-modal inference into your 
applications. Try different prompts or explore model fine-tuning with the established TensorRT 
environment.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| "CUDA out of memory" error | Insufficient VRAM for model | Use FP8/FP4 quantization or smaller model |
| "Invalid HF token" error | Missing or expired HuggingFace token | Set valid token: `export HF_TOKEN=<YOUR_TOKEN>` |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| Model download timeouts | Network issues or rate limiting | Retry command or pre-download models |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
