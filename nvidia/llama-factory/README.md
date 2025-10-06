# Llama Factory

> Install and fine-tune models with LLama Factory

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Step 4. Install LLaMA Factory with dependencies](#step-4-install-llama-factory-with-dependencies)

---

## Overview

## What you'll accomplish

You'll set up LLaMA Factory on NVIDIA Spark with Blackwell architecture to fine-tune large 
language models using LoRA, QLoRA, and full fine-tuning methods. This enables efficient 
model adaptation for specialized domains while leveraging hardware-specific optimizations.

## What to know before starting

- Basic Python knowledge for editing config files and troubleshooting
- Command line usage for running shell commands and managing environments  
- Familiarity with PyTorch and Hugging Face Transformers ecosystem
- GPU environment setup including CUDA/cuDNN installation and VRAM management
- Fine-tuning concepts: understanding tradeoffs between LoRA, QLoRA, and full fine-tuning
- Dataset preparation: formatting text data into JSON structure for instruction tuning
- Resource management: adjusting batch size and memory settings for GPU constraints

## Prerequisites

- NVIDIA Spark device with Blackwell architecture

- CUDA 12.9 or newer version installed: `nvcc --version`

- Docker installed and configured for GPU access: `docker run --gpus all nvidia/cuda:12.9-devel nvidia-smi`

- Git installed: `git --version`

- Python environment with pip: `python --version && pip --version`

- Sufficient storage space (>50GB for models and checkpoints): `df -h`

- Internet connection for downloading models from Hugging Face Hub

## Ancillary files

- Official LLaMA Factory repository: https://github.com/hiyouga/LLaMA-Factory

- NVIDIA PyTorch container: https://catalog.ngc.nvidia.com/orgs/nvidia/containers/pytorch

- Example training configuration: `examples/train_lora/llama3_lora_sft.yaml` (from repository)

- Documentation: https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html

## Time & risk

**Duration:** 30-60 minutes for initial setup, 1-7 hours for training depending on model size
and dataset.

**Risks:** Model downloads require significant bandwidth and storage. Training may consume 
substantial GPU memory and require parameter tuning for hardware constraints.

**Rollback:** Remove Docker containers and cloned repositories. Training checkpoints are 
saved locally and can be deleted to reclaim storage space.

## Instructions

## Step 1. Verify system prerequisites

Check that your NVIDIA Spark system has the required components installed and accessible.

```bash
nvcc --version
docker --version
nvidia-smi
python --version
git --version
```

## Step 2. Launch PyTorch container with GPU support

Start the NVIDIA PyTorch container with GPU access and mount your workspace directory.
> **Note:** This NVIDIA PyTorch container supports CUDA 13

```bash
docker run --gpus all --ipc=host --ulimit memlock=-1 -it --ulimit stack=67108864 --rm -v "$PWD":/workspace nvcr.io/nvidia/pytorch:25.08-py3 bash
```

## Step 3. Clone LLaMA Factory repository

Download the LLaMA Factory source code from the official repository.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

### Step 4. Install LLaMA Factory with dependencies

Install the package in editable mode with metrics support for training evaluation.

```bash
pip install -e ".[metrics]"
```

## Step 5. Configure PyTorch for CUDA 12.9 (if needed)

*If using standalone Python (skip if using Docker container)*

In a python virtual environment, uninstall existing PyTorch and reinstall with CUDA 12.9 support for ARM64 architecture.

```bash
pip uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
```

*If using Docker container*

PyTorch is pre-installed with CUDA support. Verify installation:

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

## Step 6. Prepare training configuration

Examine the provided LoRA fine-tuning configuration for Llama-3.

```bash
cat examples/train_lora/llama3_lora_sft.yaml
```

## Step 7. Launch fine-tuning training

> **Note:** Login to your hugging face hub to download the model if the model is gated
Execute the training process using the pre-configured LoRA setup.

```bash
huggingface-cli login # if the model is gated
llamafactory-cli train examples/train_lora/llama3_lora_sft.yaml
```

Example output:
```bash
***** train metrics *****
  epoch                    =        3.0
  total_flos               = 22851591GF
  train_loss               =     0.9113
  train_runtime            = 0:22:21.99
  train_samples_per_second =      2.437
  train_steps_per_second   =      0.306
Figure saved at: saves/llama3-8b/lora/sft/training_loss.png
```

## Step 8. Validate training completion

Verify that training completed successfully and checkpoints were saved.

```bash
ls -la saves/llama3-8b/lora/sft/
cat saves/llama3-8b/lora/sft/training_loss.png
```


Expected output should show:
- Final checkpoint directory (`checkpoint-21` or similar)
- Model configuration files (`config.json`, `adapter_config.json`) 
- Training metrics showing decreasing loss values
- Training loss plot saved as PNG file

## Step 9. Test inference with fine-tuned model

Run a simple inference test to verify the fine-tuned model loads correctly.

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
```

## Step 10. Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| CUDA out of memory during training | Batch size too large for GPU VRAM | Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps` |
| Model download fails or is slow | Network connectivity or Hugging Face Hub issues | Check internet connection, try using `HF_HUB_OFFLINE=1` for cached models |
| Training loss not decreasing | Learning rate too high/low or insufficient data | Adjust `learning_rate` parameter or check dataset quality |

## Step 11. Cleanup and rollback

> **Warning:** This will delete all training progress and checkpoints.

To remove all generated files and free up storage space:

```bash
cd /workspace
rm -rf LLaMA-Factory/
docker system prune -f
```

To rollback Docker container changes:
```bash
exit  # Exit container
docker container prune -f
```

## Step 12. Next steps

Test your fine-tuned model with custom prompts:

```bash
llamafactory-cli chat examples/inference/llama3_lora_sft.yaml
## Type: "Hello, how can you help me today?"
## Expect: Response showing fine-tuned behavior
```

For production deployment, export your model:
```bash
llamafactory-cli export examples/merge_lora/llama3_lora_sft.yaml
```
