---
name: dgx-spark-nemo-fine-tune
description: Use NVIDIA NeMo to fine-tune models locally — on NVIDIA DGX Spark. Use when setting up nemo-fine-tune on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/nemo-fine-tune/README.md -->
# Fine-tune with NeMo

> Use NVIDIA NeMo to fine-tune models locally

This playbook guides you through setting up and using NVIDIA NeMo AutoModel for fine-tuning large language models and vision-language models on NVIDIA Spark devices. NeMo AutoModel provides GPU-accelerated, end-to-end training for Hugging Face models with native PyTorch support, enabling instant fine-tuning without conversion delays. The framework supports distributed training across single GPU to multi-node clusters, with optimized kernels and memory-efficient recipes specifically designed for ARM64 architecture and Blackwell GPU systems.

**Outcome**: You'll establish a complete fine-tuning environment for large language models (1-70B parameters) and vision-language models using NeMo AutoModel on your NVIDIA Spark device. By the end, you'll have a working installation that supports parameter-efficient fine-tuning (PEFT), supervised fine-tuning (SFT), and distributed training capabilities with FP8 precision optimizations, all while maintaining compatibility with the Hugging Face ecosystem.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/nemo-fine-tune/README.md`
<!-- GENERATED:END -->
