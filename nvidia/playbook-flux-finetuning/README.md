# Fine-Tune FLUX.1 for Custom Image Generation

> Your own concepts, characters, and styles with Dreambooth LoRA

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook shows how to fine-tune the FLUX.1-dev 12B model with multi-concept Dreambooth LoRA (Low-Rank Adaptation) for custom image generation on your hardware platform. Unified memory and GPU acceleration let you keep the Diffusion Transformer, CLIP text encoder, T5 text encoder, and autoencoder resident while you train and generate.

Multi-concept Dreambooth LoRA teaches FLUX.1 new concepts, characters, and styles. Trained LoRA weights drop into existing ComfyUI workflows for prototyping and experimentation. The same path supports high-resolution training and inference at 1024px and above.

## What you'll accomplish

You'll have a fine-tuned FLUX.1 LoRA that generates images with your custom concepts and is ready for ComfyUI workflows on your **hardware platform**.

- Fine-tune FLUX.1-dev with Dreambooth LoRA
- Train on sample concepts (`tjtoy` toy and `sparkgpu` GPU) or your own dataset
- Run high-resolution (~1K) diffusion training and inference
- Integrate LoRAs into ComfyUI visual workflows
- Use Docker images for reproducible train and inference environments

## What to know before starting

**Required:**

- Basic Linux command line and Docker container usage
- Familiarity with generative image concepts (prompts, diffusion models, LoRA)
- Hugging Face account and token for gated FLUX.1-dev access

**Optional:**

- Prior ComfyUI experience (node graphs and workflow JSON)
- Experience preparing image datasets for Dreambooth-style training

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Docker images `flux-train` / `flux-comfyui` from playbook assets | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for FLUX.1-dev training and 1024px inference with multiple models loaded
- Enough free disk for model downloads (plan for ~35 GB+ for checkpoints, text encoders, VAE, and workspace)
- No other heavy GPU workloads running during train or inference

**Software requirements**

- NVIDIA Docker / NVIDIA Container Toolkit: `docker --version` and `nvidia-smi` inside a GPU container
- Network access to Hugging Face for gated FLUX.1-dev and text-encoder downloads
- Hugging Face access token (`HF_TOKEN`) with access granted on the [FLUX.1-dev model card](https://huggingface.co/black-forest-labs/FLUX.1-dev)
- Web browser access to port `8188` for ComfyUI

## Ancillary files

All required assets can be found [in this playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-flux-finetuning/).

- `assets/download.sh` — Downloads FLUX.1-dev, VAE, CLIP, and T5 checkpoints into `models/`
- `assets/Dockerfile.train` / `assets/launch_train.sh` — Build and run Dreambooth LoRA training
- `assets/Dockerfile.inference` / `assets/launch_comfyui.sh` — Build and run ComfyUI inference
- `assets/flux_data/` — Sample multi-concept dataset and `data.toml` training config
- `assets/workflows/base_flux.json` — ComfyUI workflow for base FLUX.1 inference
- `assets/workflows/finetuned_flux.json` — ComfyUI workflow for LoRA-conditioned inference

## Time & risk

- **Estimated time:** 2 HOURS (about 30–45 MIN for setup and model download, plus about 90 MIN of training to reach usable LoRA checkpoints; the full default 100-epoch run takes about four hours)
- **Risk level:** Medium
  - Docker permission issues may require a group change and new login session
  - Gated model access and large downloads can fail without a valid Hugging Face token or enough disk
  - Best results need hyperparameter tuning and a high-quality dataset
- **Rollback:** Stop and remove Docker containers; delete downloaded models and LoRA checkpoints under `assets/models/` if needed
- **Last Updated:** 07/31/2026
  - Fine-tune FLUX.1-dev with Dreambooth LoRA and ComfyUI inference on supported hardware platforms

## Instructions

## Step 1. Configure Docker permissions

To manage containers without `sudo`, your user must be in the `docker` group. If you skip this step, run Docker commands with `sudo`.

Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error (for example, permission denied while trying to connect to the Docker daemon socket), add your user to the docker group:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Clone the repository

Clone the playbook assets and open the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-flux-finetuning/assets
```

## Step 3. Model download

FLUX.1-dev is gated. Open the [model card](https://huggingface.co/black-forest-labs/FLUX.1-dev), accept the terms, and gain access to the checkpoints. If you do not already have an `HF_TOKEN`, follow the [Hugging Face token guide](https://huggingface.co/docs/hub/en/security-tokens), then authenticate:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
sh download.sh
```

The download can take about 30–45 minutes depending on network speed. It pulls approximately:

- `flux1-dev.safetensors` (~23.8 GB)
- `ae.safetensors` (~335 MB)
- `clip_l.safetensors` (~246 MB)
- `t5xxl_fp16.safetensors` (~9.8 GB)

After download, `models/` should look like:

```text
models/
├── checkpoints/
│   └── flux1-dev.safetensors
├── loras/
├── text_encoders/
│   ├── clip_l.safetensors
│   └── t5xxl_fp16.safetensors
└── vae/
    └── ae.safetensors
```

If you already have fine-tuned LoRAs, place them in `models/loras/`. Otherwise continue to training in Step 6.

## Step 4. Base model inference

Generate an image with the base FLUX.1 model for the sample concepts (Toy Jensen and a custom GPU) before training.

```bash
## Build the inference Docker image (run from assets/)
docker build -f Dockerfile.inference -t flux-comfyui .

## Launch ComfyUI; you can ignore import errors for torchaudio
sh launch_comfyui.sh
```

Open ComfyUI at `http://localhost:8188` (or `http://<HARDWARE_IP>:8188` from another device). Do not select a pre-existing template.

Open the workflow panel (left side, or press `w`) and load `base_flux.json`. Enter a prompt in the **CLIP Text Encode (Prompt)** node — for example, `Toy Jensen holding a DGX Spark in a datacenter`. High-resolution 1024px generation can take about three minutes.

Next steps:

- If you already placed LoRAs in `models/loras/`, skip to Step 7.
- If you will train, stop the ComfyUI container with `Ctrl+C` first.

> [!NOTE]
> To clear buffer cache after stopping ComfyUI (outside the container):
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

## Step 5. Dataset preparation

Prepare a dataset for Dreambooth LoRA fine-tuning on FLUX.1-dev. This playbook ships a two-concept sample dataset of public-domain images. If you use those concepts as-is, you do not need to edit `data.toml`.

**TJToy concept**

- **Trigger phrase:** `tjtoy toy`
- **Training images:** 6 images of Toy Jensen figures
- **Use case:** Generate scenes featuring that toy character

**SparkGPU concept**

- **Trigger phrase:** `sparkgpu gpu`
- **Training images:** 7 images of a custom GPU design
- **Use case:** Generate scenes featuring that GPU

For your own concepts, collect about 5–10 images per concept. Create one folder per concept under `flux_data/` (this playbook uses `tjtoy` and `sparkgpu`). Update `flux_data/data.toml` so each `[[datasets.subsets]]` entry has the correct `image_dir` and `class_tokens`. Appending a class token (for example `toy` or `gpu`) usually improves fine-tuning.

## Step 6. Training

Build the training image and start Dreambooth LoRA training:

```bash
docker build -f Dockerfile.train -t flux-train .
sh launch_train.sh
```

`launch_train.sh` runs `--max_train_epochs=100` and saves a LoRA checkpoint every 25 epochs (`--save_every_n_epochs=25`) into `models/loras/`, named with the `flux_dreambooth` prefix. A complete 100-epoch run takes about four hours and gives the highest quality.

You do not have to wait for the full run. Intermediate checkpoints are usable on their own: results that capture the sample concepts often appear within roughly the first 90 minutes of training. To use an earlier checkpoint, pick the most recent file in `models/loras/` and load it in ComfyUI (Step 7). For a shorter run overall, lower the epoch count in `launch_train.sh`, for example:

```bash
--max_train_epochs=25
```

Other useful knobs in `launch_train.sh` include LoRA dimension and alpha (256), learning rate (1.0 with the Prodigy optimizer), mixed precision (`bfloat16`), and caching / torch compile options. Training resolution comes from `flux_data/data.toml` (1024×1024 by default).

## Step 7. Fine-tuned model inference

Generate images with your trained LoRAs:

```bash
## Launch ComfyUI (from assets/); you can ignore import errors for torchaudio
sh launch_comfyui.sh
```

Open `http://localhost:8188`, skip pre-existing templates, open the workflow panel (`w`), and load `finetuned_flux.json`.

Prompt with your trigger phrases — for example, `tjtoy toy holding sparkgpu gpu in a datacenter`. Expect about three minutes for 1024px generation. The fine-tuned path can combine multiple concepts in one image. Use ComfyUI nodes to adjust LoRA strength, resolution, seed, sampler, scheduler, and steps.

## Step 8. Cleanup (optional)

Stop running containers with `Ctrl+C`. Remove local images if you no longer need them:

```bash
docker rmi flux-comfyui flux-train
```

> [!WARNING]
> Removing the images deletes the local Docker builds. Rebuild them with the Dockerfiles in `assets/` before running this playbook again. Downloaded models under `models/` are separate; delete those directories only if you intend to free disk space.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Cannot access gated repo for URL | All hardware platforms | Hugging Face model access restricted or token invalid | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens); request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) in a browser; export a valid `HF_TOKEN` before `download.sh` |
| "permission denied" when running Docker | All hardware platforms | User not in the `docker` group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Configure the NVIDIA Container Toolkit for Docker and confirm `nvidia-smi` works in a GPU container |
| ComfyUI unreachable on port 8188 | All hardware platforms | Container not running or port blocked | Confirm `launch_comfyui.sh` is running; open `http://localhost:8188` or `http://<HARDWARE_IP>:8188` |
| Training OOM / memory pressure during train or generate | All hardware platforms | Other GPU jobs, residual cache, or workload exceeds available memory | Stop other GPU processes; bring down ComfyUI before training; flush buffer cache (see note); reduce resolution or batch-related settings if needed |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
