---
name: dgx-spark-comfy-ui
description: Install and use Comfy UI to generate images — on NVIDIA DGX Spark. Use when setting up comfy-ui on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/comfy-ui/README.md -->
# Comfy UI

> Install and use Comfy UI to generate images

ComfyUI is an open-source web server application for AI image generation using diffusion-based models like SDXL, Flux, and others. It has a browser-based UI that lets you create, edit, and run image generation and editing workflows with multiple steps. These generation and editing steps (e.g., loading a model, adding text or sampling) are configurable in the UI as a node, and you connect nodes with wires to form a workflow.

ComfyUI uses the host's GPU for inference, so you can install it on your DGX Spark and do all of your image generation and editing directly on your device.  

Workflows are saved as JSON files, so you can version them for future work, collaboration, and reproducibility.

**Outcome**: You'll install and configure ComfyUI on your NVIDIA DGX Spark device so you can use the unified memory to work with large models.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/comfy-ui/README.md`
<!-- GENERATED:END -->
