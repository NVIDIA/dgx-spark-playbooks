# Unsloth on DGX Spark

> Optimized fine-tuning with Unsloth

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

- **Performance-first**: It claims to speed up training (e.g. 2× faster on single GPU, up to 30× in multi-GPU setups) and reduce memory usage compared to standard methods.   
- **Kernel-level optimizations**: Core compute is built with custom kernels (e.g. with Triton) and hand-optimized math to boost throughput and efficiency.  
- **Quantization & model formats**: Supports dynamic quantization (4-bit, 16-bit) and GGUF formats to reduce footprint, while aiming to retain accuracy.    
- **Broad model support**: Works with many LLMs (LLaMA, Mistral, Qwen, DeepSeek, etc.) and allows training, fine-tuning, exporting to formats like Ollama, vLLM, GGUF, Hugging Face.   
- **Simplified interface**: Provides easy-to-use notebooks and tools so users can fine-tune models with minimal boilerplate.  

## What you'll accomplish

You'll set up Unsloth for optimized fine-tuning of large language models on NVIDIA Spark devices, 
achieving up to 2x faster training speeds with reduced memory usage through efficient 
parameter-efficient fine-tuning methods like LoRA and QLoRA.

## What to know before starting

- Python package management with pip and virtual environments
- Hugging Face Transformers library basics (loading models, tokenizers, datasets)
- GPU fundamentals (CUDA/GPU vs CPU, VRAM constraints, device availability)
- Basic understanding of LLM training concepts (loss functions, checkpoints)
- Familiarity with prompt engineering and base model interaction
- Optional: LoRA/QLoRA parameter-efficient fine-tuning knowledge

## Prerequisites

- NVIDIA Spark device with Blackwell GPU architecture
- `nvidia-smi` shows a summary of GPU information
- CUDA 13.0 installed: `nvcc --version`
- Internet access for downloading models and datasets

## Ancillary files

The Python test script can be found [here on GitHub](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/unsloth/assets/test_unsloth.py)


## Time & risk

* **Duration**: 30-60 minutes for initial setup and test run
* **Risks**: 
  * Triton compiler version mismatches may cause compilation errors
  * CUDA toolkit configuration issues may prevent kernel compilation
  * Memory constraints on smaller models require batch size adjustments
* **Rollback**: Uninstall packages with `pip uninstall unsloth torch torchvision`.

## Instructions

## Step 1. Verify prerequisites

Confirm your NVIDIA Spark device has the required CUDA toolkit and GPU resources available.

```bash
nvcc --version
```
The output should show CUDA 13.0.

```bash
nvidia-smi
```
The output should show a summary of GPU information.

## Step 2. Get the container image
```bash
docker pull nvcr.io/nvidia/pytorch:25.09-py3
```

## Step 3. Launch Docker
```bash
mkdir -p ./outputs # directory to save the unsloth finetuned model results
docker run --gpus all \
  --ulimit memlock=-1 --ulimit stack=67108864 -it --entrypoint /usr/bin/bash --rm \
  -v "$(pwd)/outputs:/workspace/outputs" nvcr.io/nvidia/pytorch:25.09-py3
```

## Step 4. Install dependencies inside Docker

```bash
pip install transformers peft datasets 'trl==0.23.1' hf_transfer
pip install --no-deps unsloth unsloth_zoo
```

## Step 5. Build and install bitsandbytes inside Docker
```bash
pip install --no-deps bitsandbytes
```

## Step 6. Create Python test script

Curl the test script [here](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/unsloth/assets/test_unsloth.py) into the container.

```bash
curl -O https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/unsloth/assets/test_unsloth.py
```

We will use this test script to validate the installation with a simple fine-tuning task.


## Step 7. Run the validation test

Execute the test script to verify Unsloth is working correctly.

```bash
python test_unsloth.py
```

Expected output in the terminal window:
- "Unsloth: Will patch your computer to enable 2x faster free finetuning"
- Training progress bars showing loss decreasing over 60 steps
- Final training metrics showing completion

## Step 8. Next steps

Test with your own model and dataset by updating the `test_unsloth.py` file:

```python
## Replace line 49 with your model choice
model_name = "unsloth/Meta-Llama-3.1-8B-bnb-4bit"

## Load your custom dataset in line 25
dataset = load_dataset("your_dataset_name")

## Adjust training parameter args at line 80, and max_steps at line 83
per_device_train_batch_size = 4
max_steps = 1000
```

When you exit the container, the fine-tuned unsloth ouputs will be in the "ouputs" directory.

Visit https://github.com/unslothai/unsloth/wiki
for advanced usage instructions, including:
- [Saving models in GGUF format for vLLM](https://github.com/unslothai/unsloth/wiki#saving-to-gguf)
- [Continued training from checkpoints](https://github.com/unslothai/unsloth/wiki#loading-lora-adapters-for-continued-finetuning)
- [Using custom chat templates](https://github.com/unslothai/unsloth/wiki#chat-templates)
- [Running evaluation loops](https://github.com/unslothai/unsloth/wiki#evaluation-loop---also-fixes-oom-or-crashing)

## Troubleshooting

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
