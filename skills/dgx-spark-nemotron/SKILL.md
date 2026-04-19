---
name: dgx-spark-nemotron
description: Run Nemotron-3-Nano-30B model using llama.cpp on DGX Spark — on NVIDIA DGX Spark. Use when setting up nemotron on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/nemotron/README.md -->
# Nemotron-3-Nano with llama.cpp

> Run Nemotron-3-Nano-30B model using llama.cpp on DGX Spark

Nemotron-3-Nano-30B-A3B is NVIDIA's powerful language model featuring a 30 billion parameter Mixture of Experts (MoE) architecture with only 3 billion active parameters. This efficient design enables high-quality inference with lower computational requirements, making it ideal for DGX Spark's GB10 GPU.

This playbook demonstrates how to run Nemotron-3-Nano using llama.cpp, which compiles CUDA kernels at build time specifically for your GPU architecture. The model includes built-in reasoning (thinking mode) and tool calling support via the chat template.

**Outcome**: You will have a fully functional Nemotron-3-Nano-30B-A3B inference server running on your DGX Spark, accessible via an OpenAI-compatible API. This setup enables:

- Local LLM inference
- OpenAI-compatible API endpoint for easy integration with existing tools
- Built-in reasoning and tool calling capabilities

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/nemotron/README.md`
<!-- GENERATED:END -->
