# Fine-Tune Faster with Unsloth

> Reduced memory use and minimal boilerplate for open models

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Unsloth is a performance-focused library for fine-tuning large language models. It reduces training time and memory use compared with standard parameter-efficient fine-tuning workflows, with a streamlined Python interface.

- **Faster training** — custom kernels and optimized math paths aim for higher throughput on a single GPU
- **Lower memory use** — 4-bit and 16-bit quantization options help fit larger models in available memory
- **Broad model support** — works with many open LLMs (for example Llama, Mistral, Qwen, DeepSeek) and exports to formats used by tools such as Ollama, vLLM, GGUF, and Hugging Face
- **Minimal boilerplate** — notebooks and helpers reduce setup for LoRA and QLoRA fine-tuning

## What you'll accomplish

You'll set up Unsloth for optimized fine-tuning of large language models on your **hardware platform**, with parameter-efficient methods such as LoRA and QLoRA for faster training and reduced memory use.

## What to know before starting

**Required:**

- Python package management with pip and virtual environments or containers
- Hugging Face Transformers basics (loading models, tokenizers, datasets)
- GPU fundamentals (CUDA vs CPU, memory constraints, device availability)
- Basic LLM training concepts (loss, checkpoints)

**Optional:**

- Familiarity with prompt engineering and base model interaction
- LoRA / QLoRA parameter-efficient fine-tuning knowledge

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | `nvcr.io/nvidia/pytorch:25.11-py3` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory and free storage for model and dataset downloads (plan for several GB or more depending on model size)

**Software requirements**

- Docker installed with GPU support: `docker --version`
- NVIDIA Container Toolkit configured (`nvidia-smi` works inside a GPU container)
- CUDA toolkit available for verification: `nvcc --version` (expect CUDA 13.0)
- Network access to download container images, models, and datasets

## Ancillary files

All required assets can be found [in this playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-unsloth/).

- `assets/test_unsloth.py` — validation script that runs a short LoRA fine-tuning job to confirm Unsloth is installed correctly

## Time & risk

- **Estimated time:** 60 MIN (first run includes container pull, package install, and a short validation train)
- **Risk level:** Medium
  - Triton compiler or CUDA toolkit mismatches can block kernel compilation
  - Memory pressure may require smaller batch sizes or shorter sequence lengths
  - Large model or dataset downloads can fail on limited network or disk space
- **Rollback:** Exit and remove the container; optionally uninstall packages with `pip uninstall unsloth unsloth_zoo torch torchvision` if you installed outside the ephemeral container
- **Last Updated:** 07/31/2026
  - Set up Unsloth for optimized LoRA/QLoRA fine-tuning with a containerized validation workflow

## Instructions

> [!NOTE]
> These instructions target **Linux** with a GPU-enabled PyTorch container. Run the install and validation steps inside the container unless noted otherwise.

## Step 1. Verify prerequisites

Confirm the CUDA toolkit and GPU resources on your hardware platform.

```bash
nvcc --version
```

Expected output should show CUDA 13.0.

```bash
nvidia-smi
```

Expected output should show a summary of GPU information.

## Step 2. Pull the container image

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 3. Launch the container

```bash
docker run --gpus all --ulimit memlock=-1 -it --ulimit stack=67108864 --entrypoint /usr/bin/bash --rm nvcr.io/nvidia/pytorch:25.11-py3
```

You are now in an interactive shell inside the container for the remaining install and validation steps.

## Step 4. Install dependencies

Inside the container:

```bash
pip install transformers peft hf_transfer "datasets==4.3.0" "trl==0.26.1"
pip install --no-deps unsloth unsloth_zoo bitsandbytes
```

## Step 5. Get the validation script

Download the test script into the container. The file is also listed under Ancillary files in the **Overview** tab.

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-unsloth/assets/test_unsloth.py
```

This script runs a short fine-tuning job to validate the Unsloth install.

## Step 6. Run the validation test

```bash
python test_unsloth.py
```

Expected output in the terminal:

- A message that Unsloth will patch the environment for faster fine-tuning
- Training progress bars with loss decreasing over 60 steps
- Final training metrics showing completion

## Step 7. Next steps

Adapt `test_unsloth.py` for your own model and dataset:

```python
## Replace the model_name argument in FastModel.from_pretrained with your choice
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

## Load your custom dataset (see the dataset load near the top of the script)
dataset = load_dataset("your_dataset_name")

## Adjust training parameters in SFTConfig (for example)
per_device_train_batch_size = 4
max_steps = 1000
```

For advanced usage, see the [Unsloth wiki](https://github.com/unslothai/unsloth/wiki), including:

- [Saving models in GGUF format](https://github.com/unslothai/unsloth/wiki#saving-to-gguf)
- [Continued training from checkpoints](https://github.com/unslothai/unsloth/wiki#loading-lora-adapters-for-continued-finetuning)
- [Using custom chat templates](https://github.com/unslothai/unsloth/wiki#chat-templates)
- [Running evaluation loops](https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing)

## Step 8. Cleanup (optional)

When you exit the interactive container (`exit` or Ctrl-D), the `--rm` flag removes it. Pulled images remain on the host until you remove them:

```bash
docker rmi nvcr.io/nvidia/pytorch:25.11-py3
```

> [!WARNING]
> Removing the image deletes the local copy of `nvcr.io/nvidia/pytorch:25.11-py3`. Re-pull it if you run this playbook again.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `nvcc` not found or wrong CUDA version | All hardware platforms | CUDA toolkit missing or not on `PATH` | Install or configure CUDA 13.0; confirm with `nvcc --version` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Pip install or import errors for Unsloth / Triton | All hardware platforms | Dependency or compiler mismatch inside the container | Use the playbook container image and install commands; retry in a fresh container |
| CUDA out of memory / training OOM | All hardware platforms | Batch size, sequence length, or model too large for available memory | Lower `per_device_train_batch_size`, `max_seq_length`, or use a smaller 4-bit model |
| Training loss does not decrease / job exits early | All hardware platforms | Dataset, config, or interrupted download | Confirm network access; re-run with the default `test_unsloth.py` settings |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
