---
name: dgx-spark-nvfp4-quantization
description: Quantize a model to NVFP4 to run on Spark using TensorRT Model Optimizer — on NVIDIA DGX Spark. Use when setting up nvfp4-quantization on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/nvfp4-quantization/README.md -->
# NVFP4 Quantization

> Quantize a model to NVFP4 to run on Spark using TensorRT Model Optimizer

NVFP4 is a 4-bit floating-point format introduced with NVIDIA Blackwell GPUs to maintain model accuracy while reducing memory bandwidth and storage requirements for inference workloads. 
Unlike uniform INT4 quantization, NVFP4 retains floating-point semantics with a shared exponent and a compact mantissa, allowing higher dynamic range and more stable convergence.
NVIDIA Blackwell Tensor Cores natively support mixed-precision execution across FP16, FP8, and FP4, enabling models to use FP4 for weights and activations while accumulating in higher precision (typically FP16). 
This design minimizes quantization error during matrix multiplications and supports efficient conversion pipelines in TensorRT-LLM for fine-tuned layer-wise quantization.

Immediate benefits are:
  - Cut memory use ~3.5x vs FP16 and ~1.8x vs FP8
  - Maintain accuracy close to FP8 (usually <1% loss)

**Outcome**: You'll quantize the DeepSeek-R1-Distill-Llama-8B model using NVIDIA's TensorRT Model Optimizer
inside a TensorRT-LLM container, producing an NVFP4 quantized model for deployment on NVIDIA DGX Spark.

The examples use NVIDIA FP4 quantized models which help reduce model size by approximately 2x by reducing the precision of model layers.
This quantization approach aims to preserve accuracy while providing significant throughput improvements. However, it's important to note that quantization can potentially impact model accuracy - we recommend running evaluations to verify if the quantized model maintains acceptable performance for your use case.

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/nvfp4-quantization/README.md`
<!-- GENERATED:END -->
