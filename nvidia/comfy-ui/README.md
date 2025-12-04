# Comfy UI

> Install and use Comfy UI to generate images

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

ComfyUI is an open-source web server application for AI image generation using diffusion-based models like SDXL, Flux, and others. It has a browser-based UI that lets you create, edit, and run image generation and editing workflows with multiple steps. These generation and editing steps (e.g., loading a model, adding text or sampling) are configurable in the UI as a node, and you connect nodes with wires to form a workflow.

ComfyUI uses the host's GPU for inference, so you can install it on your DGX Spark and do all of your image generation and editing directly on your device.  

Workflows are saved as JSON files, so you can version them for future work, collaboration, and reproducibility.

## What you'll accomplish

You'll install and configure ComfyUI on your NVIDIA DGX Spark device so you can use the unified memory to work with large models.

## What to know before starting

- Experience working with Python virtual environments and package management
- Familiarity with command line operations and terminal usage
- Basic understanding of deep learning model deployment and checkpoints
- Knowledge of container workflows and GPU acceleration concepts
- Understanding of network configuration for accessing web services

## Prerequisites

**Hardware Requirements:**
-  NVIDIA Grace Blackwell GB10 Superchip System
-  Minimum 8GB GPU memory for Stable Diffusion models
-  At least 20GB available storage space

**Software Requirements:**
- Python 3.8 or higher installed: `python3 --version`
- pip package manager available: `pip3 --version`
- CUDA toolkit compatible with Blackwell: `nvcc --version`
- Git version control: `git --version`
- Network access to download models from Hugging Face
- Web browser access to `<SPARK_IP>:8188` port

## Ancillary files

All required assets can be found [in the ComfyUI repository on GitHub](https://github.com/comfyanonymous/ComfyUI)

- `requirements.txt` - Python dependencies for ComfyUI installation
- `main.py` - Primary ComfyUI server application entry point
- `v1-5-pruned-emaonly-fp16.safetensors` - Stable Diffusion 1.5 checkpoint model

## Time & risk

* **Estimated time:** 30-45 minutes (including model download)
* **Risk level:** Medium
  * Model downloads are large (~2GB) and may fail due to network issues
  * Port 8188 must be accessible for web interface functionality
* **Rollback:** Virtual environment can be deleted to remove all installed packages. Downloaded models can be removed manually from the checkpoints directory.
* **Last Updated:** 11/10/2025
  * Update ComfyUI PyTorch to CUDA 13.0

## Instructions

## Step 1. Verify system prerequisites

Check that your NVIDIA DGX Spark device meets the requirements before proceeding with installation.

```bash
python3 --version
pip3 --version
nvcc --version
nvidia-smi
```

Expected output should show Python 3.8+, pip available, CUDA toolkit, and GPU detection.

## Step 2. Create Python virtual environment

You will install ComfyUI on your host system, so you should create an isolated environment to avoid conflicts with system packages.

```bash
python3 -m venv comfyui-env
source comfyui-env/bin/activate
```

Verify the virtual environment is active by checking the command prompt shows `(comfyui-env)`.

## Step 3. Install PyTorch with CUDA support

Install PyTorch with CUDA 13.0 support.

```bash
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

This installation targets CUDA 13.0 compatibility with Blackwell architecture GPUs.

## Step 4. Clone ComfyUI repository

Download the ComfyUI source code from the official repository.

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/
```

## Step 5. Install ComfyUI dependencies

Install the required Python packages for ComfyUI operation.

```bash
pip install -r requirements.txt
```

This installs all necessary dependencies including web interface components and model handling libraries.

## Step 6. Download Stable Diffusion checkpoint

Navigate to the checkpoints directory and download the Stable Diffusion 1.5 model.

```bash
cd models/checkpoints/
wget https://huggingface.co/Comfy-Org/stable-diffusion-v1-5-archive/resolve/main/v1-5-pruned-emaonly-fp16.safetensors
cd ../../
```

The download will be approximately 2GB and may take several minutes depending on network speed.

## Step 7. Launch ComfyUI server

Start the ComfyUI web server with network access enabled.

```bash
python main.py --listen 0.0.0.0
```

The server will bind to all network interfaces on port 8188, making it accessible from other devices.

## Step 8. Validate installation

Check that ComfyUI is running correctly and accessible via your web browser.

```bash
curl -I http://localhost:8188
```

Expected output should show HTTP 200 response indicating the web server is operational.

Open a web browser and navigate to `http://<SPARK_IP>:8188` where `<SPARK_IP>` is your device's IP address.

## Step 9. Optional - Cleanup and rollback

If you need to remove the installation completely, follow these steps:

> [!WARNING]
> This will delete all installed packages and downloaded models.

```bash
deactivate
rm -rf comfyui-env/
rm -rf ComfyUI/
```

To rollback during installation, press `Ctrl+C` to stop the server and remove the virtual environment.

## Step 10. Optional - Next steps

Test the installation with a basic image generation workflow:

1. Access the web interface at `http://<SPARK_IP>:8188`
2. Load the default workflow (should appear automatically)
3. Click "Run" to generate your first image
4. Monitor GPU usage with `nvidia-smi` in a separate terminal

The image generation should complete within 30-60 seconds depending on your hardware configuration.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| PyTorch CUDA not available | Incorrect CUDA version or missing drivers | Verify `nvcc --version` matches cu129, reinstall PyTorch |
| Model download fails | Network connectivity or storage space | Check internet connection, verify 20GB+ available space |
| Web interface inaccessible | Firewall blocking port 8188 | Configure firewall to allow port 8188, check IP address |
| Out of GPU memory errors after manually flushing buffer cache | Insufficient VRAM for model | Use smaller models or enable CPU fallback mode |

> [!NOTE] 
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```


For latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
