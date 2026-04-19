---
name: dgx-spark-multi-modal-inference
description: Setup multi-modal inference with TensorRT — on NVIDIA DGX Spark. Use when setting up multi-modal-inference on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/multi-modal-inference/README.md -->
# Multi-modal Inference

> Setup multi-modal inference with TensorRT

Multi-modal inference combines different data types, such as **text, images, and audio**, within a single model pipeline to generate or interpret richer outputs.  
Instead of processing one input type at a time, multi-modal systems have shared representations that  **text-to-image generation**, **image captioning**, or **vision-language reasoning**.  

On GPUs, this enables **parallel processing across modalities** for faster, higher-fidelity results for tasks that combine language and vision.

**Outcome**: You'll deploy GPU-accelerated multi-modal inference capabilities on NVIDIA Spark using TensorRT to run 
Flux.1 and SDXL diffusion models with optimized performance across multiple precision formats (FP16, 
FP8, FP4).

Duration: 45-90 minutes depending on model downloads and optimization steps

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/multi-modal-inference/README.md`
<!-- GENERATED:END -->
