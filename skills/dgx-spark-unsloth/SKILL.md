---
name: dgx-spark-unsloth
description: Optimized fine-tuning with Unsloth — on NVIDIA DGX Spark. Use when setting up unsloth on Spark hardware.
---

<!-- GENERATED:BEGIN from nvidia/unsloth/README.md -->
# Unsloth on DGX Spark

> Optimized fine-tuning with Unsloth

- **Performance-first**: It claims to speed up training (e.g. 2× faster on single GPU, up to 30× in multi-GPU setups) and reduce memory usage compared to standard methods.   
- **Kernel-level optimizations**: Core compute is built with custom kernels (e.g. with Triton) and hand-optimized math to boost throughput and efficiency.  
- **Quantization & model formats**: Supports dynamic quantization (4-bit, 16-bit) and GGUF formats to reduce footprint, while aiming to retain accuracy.    
- **Broad model support**: Works with many LLMs (LLaMA, Mistral, Qwen, DeepSeek, etc.) and allows training, fine-tuning, exporting to formats like Ollama, vLLM, GGUF, Hugging Face.   
- **Simplified interface**: Provides easy-to-use notebooks and tools so users can fine-tune models with minimal boilerplate.

**Outcome**: You'll set up Unsloth for optimized fine-tuning of large language models on NVIDIA Spark devices, 
achieving up to 2x faster training speeds with reduced memory usage through efficient 
parameter-efficient fine-tuning methods like LoRA and QLoRA.

Duration: 30-60 minutes for initial setup and test run

**Full playbook**: `/home/runner/work/dgx-spark-playbooks/dgx-spark-playbooks/nvidia/unsloth/README.md`
<!-- GENERATED:END -->
