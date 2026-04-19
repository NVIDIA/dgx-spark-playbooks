---
name: dgx-spark-jax
description: Optimize JAX to run on Spark — on NVIDIA DGX Spark. Use when setting up jax on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/jax/README.md -->
# Optimized JAX

> Optimize JAX to run on Spark

JAX lets you write **NumPy-style Python code** and run it fast on GPUs without writing CUDA. It does this by:

- **NumPy on accelerators**: Use `jax.numpy` just like NumPy, but arrays live on the GPU.  
- **Function transformations**:  
  - `jit` → Compiles your function into fast GPU code  
  - `grad` → Gives you automatic differentiation 
  - `vmap` → Vectorizes your function across batches  
  - `pmap` → Runs across multiple GPUs in parallel

**Outcome**: You'll set up a JAX development environment on NVIDIA Spark with Blackwell architecture that enables 
high-performance machine learning prototyping using familiar NumPy-like abstractions, complete with 
GPU acceleration and performance optimization capabilities.

**Full playbook**: `/Users/jkneen/Documents/GitHub/dgx-spark-playbooks/nvidia/jax/README.md`
<!-- GENERATED:END -->
