---
name: dgx-spark-trt-llm
description: Install and use TensorRT-LLM on DGX Spark — on NVIDIA DGX Spark. Use when setting up trt-llm on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/trt-llm/README.md -->
# TRT LLM for Inference

> Install and use TensorRT-LLM on DGX Spark

**NVIDIA TensorRT-LLM (TRT-LLM)** is an open-source library for optimizing and accelerating large language model (LLM) inference on NVIDIA GPUs.

It provides highly efficient kernels, memory management, and parallelism strategies—like tensor, pipeline, and sequence parallelism—so developers can serve LLMs with lower latency and higher throughput.

TRT-LLM integrates with frameworks like Hugging Face and PyTorch, making it easier to deploy state-of-the-art models at scale.

**Outcome**: You'll set up TensorRT-LLM to optimize and deploy large language models on your DGX Spark, achieving significantly higher throughput and lower latency than standard PyTorch
inference through kernel-level optimizations, efficient memory layouts, and advanced quantization.

Duration: 45-60 minutes for setup and API server deployment · Risk: Medium - container pulls and model downloads may fail due to network issues

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/trt-llm/README.md`
<!-- GENERATED:END -->
