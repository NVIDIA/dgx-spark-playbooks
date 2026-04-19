---
name: dgx-spark-nim-llm
description: Deploy a NIM on Spark — on NVIDIA DGX Spark. Use when setting up nim-llm on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/nim-llm/README.md -->
# NIM on Spark

> Deploy a NIM on Spark

NVIDIA NIM is containerized software for fast, reliable AI model serving and inference on NVIDIA GPUs. This playbook demonstrates how to run NIM microservices for LLMs on DGX Spark devices, enabling local GPU inference through a simple Docker workflow. You'll authenticate with NVIDIA's registry, launch the NIM inference microservice, and perform basic inference testing to verify functionality.

### What you'll accomplish

You'll launch a NIM container on your DGX Spark device to expose a GPU-accelerated HTTP endpoint for text completions. While these instructions feature working with the Llama 3.1 8B NIM, additional NIM including the [Qwen3-32 NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/qwen/containers/qwen3-32b-dgx-spark) are available for DGX Spark (see them [here](https://docs.nvidia.com/nim/large-language-models/1.14.0/release-notes.html#new-language-models%20)).

### What to know before starting

**Outcome**: You'll launch a NIM container on your DGX Spark device to expose a GPU-accelerated HTTP endpoint for text completions. While these instructions feature working with the Llama 3.1 8B NIM, additional NIM including the [Qwen3-32 NIM](https://catalog.ngc.nvidia.com/orgs/nim/teams/qwen/containers/qwen3-32b-dgx-spark) are available for DGX Spark (see them [here](https://docs.nvidia.com/nim/large-language-models/1.14.0/release-notes.html#new-language-models%20)).

### What to know before starting

- Working in a terminal environment
- Using Docker commands and GPU-enabled containers
- Basic familiarity with REST APIs and curl commands
- Understanding of NVIDIA GPU environments and CUDA

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/nim-llm/README.md`
<!-- GENERATED:END -->
