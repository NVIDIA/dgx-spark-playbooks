---
name: dgx-spark-flux-finetuning
description: Fine-tune FLUX.1-dev 12B model using Dreambooth LoRA for custom image generation — on NVIDIA DGX Spark. Use when setting up flux-finetuning on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/flux-finetuning/README.md -->
# FLUX.1 Dreambooth LoRA Fine-tuning

> Fine-tune FLUX.1-dev 12B model using Dreambooth LoRA for custom image generation

This playbook demonstrates how to fine-tune the FLUX.1-dev 12B model using multi-concept Dreambooth LoRA (Low-Rank Adaptation) for custom image generation on DGX Spark. 
With 128GB of unified memory and powerful GPU acceleration, DGX Spark provides an ideal environment for training an image generation model with multiple models loaded in memory, such as the Diffusion Transformer, CLIP Text Encoder, T5 Text Encoder, and the Autoencoder.

Multi-concept Dreambooth LoRA fine-tuning allows you to teach FLUX.1 new concepts, characters, and styles. The trained LoRA weights can be easily integrated into existing ComfyUI workflows, making it perfect for prototyping and experimentation.
Moreover, this playbook demonstrates how DGX Spark can not only load several models in memory, but also train and generate high-resolution images such as 1024px and higher.

**Outcome**: You will have a fine-tuned FLUX.1 model capable of generating images with your custom concepts, readily available for ComfyUI workflows.
The setup includes:
- FLUX.1-dev model fine-tuning using Dreambooth LoRA technique
- Training on custom concepts ("tjtoy" toy and "sparkgpu" GPU)
- High-resolution 1K diffusion training and inference
- ComfyUI integration for intuitive visual workflows
- Docker containerization for reproducible environments

Duration: * 30-45 minutes for initial setup model download time

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/flux-finetuning/README.md`
<!-- GENERATED:END -->
