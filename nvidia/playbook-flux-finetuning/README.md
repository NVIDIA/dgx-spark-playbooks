# Fine-Tune FLUX.2 Klein for Custom Image Generation

> Train a concept LoRA on FLUX.2 Klein with AI Toolkit

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook shows how to fine-tune FLUX.2 Klein with multi-concept LoRA (Low-Rank Adaptation) using AI Toolkit for custom image generation on your hardware platform. Unified memory and GPU acceleration let you train and generate with the Diffusion Transformer, Qwen text encoder, and autoencoder.

Multi-concept LoRA teaches FLUX.2 Klein new concepts, characters, and styles. Trained LoRA weights drop into existing ComfyUI workflows for prototyping and experimentation. The same path supports high-resolution training and inference at 1024px and above.

## What you'll accomplish

You'll have a fine-tuned FLUX.2 Klein LoRA that generates images with your custom concepts and is ready for ComfyUI workflows on your **hardware platform**.

- Fine-tune FLUX.2 Klein 4B or 9B with AI Toolkit
- Train on sample concepts (`tjtoy` toy and `sparkgpu` GPU) or your own dataset
- Run high-resolution (~1K) diffusion training and inference
- Integrate LoRAs into ComfyUI visual workflows
- Use Docker images for reproducible train and inference environments

## What to know before starting

**Required:**

- Basic Linux command line and Docker container usage
- Familiarity with generative image concepts (prompts, diffusion models, LoRA)
- Hugging Face account and token for gated FLUX.2 Klein access

**Optional:**

- Prior ComfyUI experience (node graphs and workflow JSON)
- Experience preparing image datasets for concept training

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 64 GB Unified Memory | FLUX.2 Klein 4B, batch size 2 | — |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | FLUX.2 Klein 9B, batch size 2 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory for FLUX.2 Klein training and 1024px inference
- Enough free disk for model downloads, the AI Toolkit cache, container images, and LoRA checkpoints
- No other heavy GPU workloads running during train or inference

**Software requirements**

- NVIDIA Docker / NVIDIA Container Toolkit: `docker --version` and `nvidia-smi` inside a GPU container
- Network access to Hugging Face for gated FLUX.2 Klein and text-encoder downloads
- Hugging Face access token (`HF_TOKEN`) with access granted on the FLUX.2 Klein model card for your system
- Web browser access to port `8188` for ComfyUI

## Ancillary files

All required assets can be found [in this playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-flux-finetuning/).

- `assets/download.sh` — Downloads the FLUX.2 Klein model, Qwen text encoder, and VAE into `models/`
- `assets/Dockerfile.train` / `assets/launch_train.sh` — Build and run AI Toolkit LoRA training
- `assets/config/flux2_klein_4b.yaml` / `assets/config/flux2_klein_9b.yaml` — Define datasets and training parameters
- `assets/Dockerfile.inference` / `assets/launch_comfyui.sh` — Build and run ComfyUI inference
- `assets/flux_data/` — Sample multi-concept dataset
- `assets/workflows/base_flux2.json` — ComfyUI workflow for base FLUX.2 Klein inference
- `assets/workflows/finetuned_flux2.json` — ComfyUI workflow for LoRA-conditioned inference

## Time & risk

- **Estimated time:** 4 HOURS, including first-time downloads, container builds, training, and inference
- **Risk level:** Medium
  - Docker permission issues may require a group change and new login session
  - Gated model access and large downloads can fail without a valid Hugging Face token or enough disk
  - Best results need hyperparameter tuning and a high-quality dataset
- **Rollback:** Stop and remove Docker containers; delete downloaded models, the AI Toolkit cache, and LoRA checkpoints if needed
- **Last Updated:** 10/21/2026
  - Fine-tune FLUX.2 Klein with AI Toolkit and ComfyUI inference on supported hardware platforms

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
cd dgx-spark-playbooks/nvidia/playbook-flux-finetuning/assets
```

## Step 3. Model download

FLUX.2 Klein is gated. Open the [Klein 4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-4B) for a 64 GB system or the [Klein 9B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-base-9B) for a 128 GB system, accept the terms, and gain access to the checkpoints. If you do not already have an `HF_TOKEN`, follow the [Hugging Face token guide](https://huggingface.co/docs/hub/en/security-tokens), then authenticate:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
sh download.sh
```

The download can take about 30–45 minutes depending on network speed. The script detects physical memory and pulls:

- 64 GB system: FLUX.2 Klein 4B and Qwen3-4B
- 128 GB system: FLUX.2 Klein 9B and Qwen3-8B
- `flux2-vae.safetensors`

After download, `models/` should look like:

```text
models/
├── diffusion_models/
│   └── flux-2-klein-base.safetensors
├── loras/
├── text_encoders/
│   └── qwen_3.safetensors
└── vae/
    └── flux2-vae.safetensors
```

AI Toolkit downloads its training copy of the model into `hf_cache/` when training starts.

If you already have fine-tuned LoRAs, place them in `models/loras/`. Otherwise continue to training in Step 6.

## Step 4. Base model inference

Generate an image with the base FLUX.2 Klein model for the sample concepts (Toy Jensen and a custom GPU) before training.

```bash
## Build the inference Docker image (run from assets/)
docker build -f Dockerfile.inference -t flux2-comfyui .

## Launch ComfyUI
sh launch_comfyui.sh
```

Open ComfyUI at `http://localhost:8188` (or `http://<HARDWARE_IP>:8188` from another device). Do not select a pre-existing template.

Open the workflow panel (left side, or press `w`) and load `base_flux2.json`. Enter a prompt in the **Prompt** node — for example, `tjtoy toy holding sparkgpu gpu in a datacenter`. High-resolution 1024px generation can take a few minutes.

Next steps:

- If you already placed LoRAs in `models/loras/`, skip to Step 7.
- If you will train, stop the ComfyUI container with `Ctrl+C` first.

> [!NOTE]
> To clear buffer cache after stopping ComfyUI (outside the container):
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

## Step 5. Dataset preparation

Prepare a dataset for concept LoRA fine-tuning on FLUX.2 Klein. This playbook ships a two-concept sample dataset of public-domain images. If you use those concepts as-is, you do not need to edit the files under `config/`.

**TJToy concept**

- **Trigger phrase:** `tjtoy toy`
- **Training images:** 6 images of Toy Jensen figures
- **Use case:** Generate scenes featuring that toy character

**SparkGPU concept**

- **Trigger phrase:** `sparkgpu gpu`
- **Training images:** 7 images of a custom GPU design
- **Use case:** Generate scenes featuring that GPU

For your own concepts, collect about 5–10 images per concept. Create one folder per concept under `flux_data/` (this playbook uses `tjtoy` and `sparkgpu`). Update the matching file under `config/` so each dataset entry has the correct `folder_path`, `default_caption`, and `trigger_word`. Appending a class token (for example `toy` or `gpu`) usually improves fine-tuning.

## Step 6. Training

Build the training image and start concept LoRA training with AI Toolkit:

```bash
docker build -f Dockerfile.train -t flux2-train .
sh launch_train.sh
```

`launch_train.sh` selects Klein 4B on 64 GB systems and Klein 9B on 128 GB systems. Both profiles run 1,000 optimizer steps at 1024×1024 and create the following stable LoRA path for the supplied workflow:

```text
models/loras/flux2_klein_tjtoy_sparkgpu.safetensors
```

Useful knobs in the selected file under `config/` include LoRA dimension and alpha (256), learning rate (`0.0001` with AdamW 8-bit), batch size (2), mixed precision (`bfloat16`), and caching / torch compile options. If you reduce `steps`, also reduce `save_every` so a checkpoint is written.

## Step 7. Fine-tuned model inference

Generate images with your trained LoRAs:

```bash
## Launch ComfyUI (from assets/)
sh launch_comfyui.sh
```

Open `http://localhost:8188`, skip pre-existing templates, open the workflow panel (`w`), and load `finetuned_flux2.json`. Confirm that **Load LoRA Model Only** points to `flux2_klein_tjtoy_sparkgpu.safetensors`.

Prompt with your trigger phrases — for example, `tjtoy toy holding sparkgpu gpu in a datacenter`. High-resolution 1024px generation can take a few minutes. The fine-tuned path can combine multiple concepts in one image. Use ComfyUI nodes to adjust LoRA strength, resolution, seed, sampler, scheduler, and steps.

## Step 8. Cleanup (optional)

Stop running containers with `Ctrl+C`. Remove local images if you no longer need them:

```bash
docker rmi flux2-comfyui flux2-train
```

> [!WARNING]
> Removing the images deletes the local Docker builds. Rebuild them with the Dockerfiles in `assets/` before running this playbook again. Downloaded models under `models/`, the training cache under `hf_cache/`, and LoRA checkpoints are separate; delete those directories only if you intend to free disk space.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| Cannot access gated repo for URL | All hardware platforms | Hugging Face model access restricted or token invalid | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens); accept the terms for the FLUX.2 Klein model selected for your system; export a valid `HF_TOKEN` before `download.sh` or training |
| "permission denied" when running Docker | All hardware platforms | User not in the `docker` group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Configure the NVIDIA Container Toolkit for Docker and confirm `nvidia-smi` works in a GPU container |
| ComfyUI unreachable on port 8188 | All hardware platforms | Container not running or port blocked | Confirm `launch_comfyui.sh` is running; open `http://localhost:8188` or `http://<HARDWARE_IP>:8188` |
| Training OOM / memory pressure during train or generate | All hardware platforms | Other GPU jobs, residual cache, or workload exceeds available memory | Stop other GPU processes; bring down ComfyUI before training; flush buffer cache (see note); enable gradient checkpointing or reduce batch size if needed |
| Training appears idle at startup | All hardware platforms | AI Toolkit is downloading the model, caching latents and text embeddings, or compiling the model | Check the console output and allow the startup stage to finish; later runs reuse completed caches |
| `torch.compile` fails | All hardware platforms | A model operation or local compiler dependency is incompatible | Set `compile: false` in the selected file under `config/` and retry |
| ComfyUI cannot find the LoRA | All hardware platforms | The checkpoint is missing or ComfyUI started before it was written | Confirm the `.safetensors` file exists under `models/loras/`, then restart ComfyUI |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
