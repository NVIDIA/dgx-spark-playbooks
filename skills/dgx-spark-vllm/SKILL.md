---
name: dgx-spark-vllm
description: Install and run vLLM for high-throughput LLM inference on NVIDIA DGX Spark, including multi-Spark serving for very large models (e.g., Llama 405B across two Sparks). Use when a user needs an OpenAI-compatible API, higher throughput than Ollama, or wants to run models too large for a single Spark. Significantly more complex setup than Ollama — ensure user actually needs what vLLM offers before recommending.
---

<!-- GENERATED:BEGIN from nvidia/vllm/README.md -->
# vLLM for Inference

> Install and use vLLM on DGX Spark

vLLM is an inference engine designed to run large language models efficiently. The key idea is **maximizing throughput and minimizing memory waste** when serving LLMs.

- It uses a memory-efficient attention algoritm called **PagedAttention** to handle long sequences without running out of GPU memory.
- New requests can be added to a batch already in process through **continuous batching** to keep GPUs fully utilized.
- It has an **OpenAI-compatible API** so applications built for the OpenAI API can switch to a vLLM backend with little or no modification.

**Outcome**: You'll set up vLLM high-throughput LLM serving on DGX Spark with Blackwell architecture,
either using a pre-built Docker container or building from source with custom LLVM/Triton
support for ARM64.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/vllm/README.md`
<!-- GENERATED:END -->

## When to use this skill
- User's current runtime (usually Ollama) can't handle their throughput requirements
- User wants an OpenAI-compatible API to plug applications into
- User wants to run a model too large for one Spark (vLLM supports tensor-parallel across 2+ Sparks)
- User specifically asked for vLLM

## When NOT to use this skill
- User is just exploring — `dgx-spark-ollama` is far simpler
- User needs single-user chat — Ollama + Open WebUI covers that case
- User needs absolute lowest latency with pre-compiled models — that's `dgx-spark-trt-llm` territory

## Key decisions
- **Docker container or build from source?** — Pre-built container is the recommended path. Source build is only needed if the user has a specific reason (custom patches, bleeding-edge vLLM version not yet in the container).
- **Single-Spark or multi-Spark?** — Multi-Spark adds major complexity: networking (`dgx-spark-connect-two-sparks` or `dgx-spark-multi-sparks-through-switch`) + NCCL (`dgx-spark-nccl`) must be working first. Only pursue for 120B+ param models that don't fit on one Spark.
- **Model + quantization** — the playbook's support matrix lists specific NVFP4/FP8/MXFP4 combinations. Don't assume any HF model works — check the matrix.

## Prerequisites (hard requirements)
- CUDA 13.0 toolkit installed (`nvcc --version`)
- Docker + NVIDIA Container Toolkit configured
- Python 3.12 available
- `dgx-spark-connect-to-your-spark` for remote access

## Non-obvious gotchas
- This is ARM64 + Blackwell. PyPI wheels built for x86_64 CUDA 12.x **will not work** — the playbook's container has ARM64-specific LLVM/Triton patches.
- vLLM's default GPU memory utilization is high (~0.9). On a Spark that's also running other workloads, drop to 0.7–0.8 or the container will OOM.
- Multi-Spark serving is sensitive to NCCL configuration and link quality — a single flaky cable will destroy throughput. Validate `dgx-spark-nccl` first before assuming vLLM is the problem.

## Related skills
- **Prerequisite**: `dgx-spark-connect-to-your-spark`
- **Simpler alternative**: `dgx-spark-ollama` — recommend this first unless the user needs vLLM's specific capabilities
- **Alternative for max perf**: `dgx-spark-trt-llm` — TensorRT-LLM with compiled engines. Different use case (lowest latency, more setup cost), not strictly an upgrade path
- **Multi-Spark composition**:
  - `dgx-spark-connect-two-sparks` or `dgx-spark-multi-sparks-through-switch` (physical link)
  - `dgx-spark-nccl` (collective comms)
- **Pairs with**: `dgx-spark-dgx-dashboard` for GPU monitoring during serving
