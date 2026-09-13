# Fine-Tune LLMs with LLaMA Factory

> One interface for supervised, RLHF, and parameter-efficient training

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

LLaMA Factory is an open-source framework that simplifies training and fine-tuning large language models. It offers a unified interface for methods such as supervised fine-tuning (SFT), RLHF, and QLoRA, and supports a wide range of LLM architectures including LLaMA, Mistral, and Qwen.

- **Unified CLI** — one interface for supervised, RLHF, and parameter-efficient training
- **Broad model support** — works with popular open LLM families and Hugging Face workflows
- **Flexible fine-tuning** — LoRA, QLoRA, and full fine-tuning with ready example configs

## What you'll accomplish

You'll set up LLaMA Factory on your **hardware platform** to fine-tune large language models with LoRA, QLoRA, and full fine-tuning methods, using the CLI and example configs from the upstream repository.

## What to know before starting

**Required:**

- Basic Python knowledge for editing config files and troubleshooting
- Command-line usage for shell commands and virtual environments
- Familiarity with PyTorch and the Hugging Face Transformers ecosystem
- Fine-tuning concepts: tradeoffs between LoRA, QLoRA, and full fine-tuning

**Optional:**

- Dataset preparation: formatting text data into JSON for instruction tuning
- Resource management: adjusting batch size and memory settings for GPU constraints

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | Python venv + PyTorch CUDA 13.0 (`cu130`) | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient storage space (plan for >50 GB for models and checkpoints): `df -h`

**Software requirements**

- CUDA 12.9 or newer: `nvcc --version`
- Git: `git --version`
- Python 3 with venv and pip: `python3 --version && pip3 --version`
- Internet connection for downloading models from Hugging Face Hub

## Ancillary files

All required assets are in the [LLaMA Factory repository](https://github.com/hiyouga/LLaMA-Factory).

- `examples/train_lora/qwen3_lora_sft.yaml` — example LoRA SFT training configuration
- `examples/inference/qwen3_lora_sft.yaml` — example chat/inference configuration for the fine-tuned adapter
- `examples/merge_lora/qwen3_lora_sft.yaml` — example export/merge configuration for production use
- [Data preparation docs](https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html) — dataset formatting guidance

## Time & risk

- **Estimated time:** 60 MIN for initial setup; training can take 1–7 hours depending on model size and dataset
- **Risk level:** Medium
  - Model downloads require significant bandwidth and storage
  - Training may consume substantial GPU memory and need batch-size or accumulation tuning
- **Rollback:** Deactivate the virtual environment and remove the `factoryEnv` and `LLaMA-Factory` directories. Delete local training checkpoints to reclaim storage.
- **Last Updated:** 07/31/2026
  - Set up LLaMA Factory with a venv-based PyTorch CUDA 13 workflow for Qwen3 LoRA fine-tuning, chat validation, and export

## Instructions

> [!NOTE]
> These instructions target **Linux** with a Python virtual environment and GPU-enabled PyTorch. Run install and training steps in the activated venv unless noted otherwise.

## Step 1. Verify system prerequisites

Confirm that your hardware platform has the required components installed and accessible.

```bash
nvcc --version
nvidia-smi
python3 --version
git --version
```

Expected output should show a supported CUDA toolkit, GPU summary from `nvidia-smi`, Python 3, and Git.

## Step 2. Create and activate a Python virtual environment

```bash
python3 -m venv factoryEnv
source ./factoryEnv/bin/activate
```

## Step 3. Install PyTorch with CUDA 13 support

Install PyTorch, torchvision, and torchaudio with CUDA 13.0 support from the official PyTorch index.

```bash
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130
```

## Step 4. Verify PyTorch CUDA support

Confirm that PyTorch can see the GPU.

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')"
```

Expected output should show a PyTorch version and `CUDA: True`.

## Step 5. Clone LLaMA Factory repository

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

## Step 6. Install LLaMA Factory with dependencies

Install LLaMA Factory in editable mode with metrics support.

```bash
pip install -e ".[metrics]"
```

## Step 7. Prepare training configuration

Examine the provided LoRA fine-tuning configuration for Qwen3.

```bash
cat examples/train_lora/qwen3_lora_sft.yaml
```

## Step 8. Launch fine-tuning training

> [!NOTE]
> Log in to Hugging Face Hub to download the model if the model is gated.

```bash
hf auth login   # if the model is gated
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

Example output:

```
***** train metrics *****
  epoch                    =        3.0
  total_flos               = 11076559GF
  train_loss               =     0.9993
  train_runtime            = 0:14:32.12
  train_samples_per_second =      3.749
  train_steps_per_second   =      0.471
Figure saved at: saves/qwen3-4b/lora/sft/training_loss.png
```

## Step 9. Validate training completion

Verify that training completed successfully and checkpoints were saved.

```bash
ls -la saves/qwen3-4b/lora/sft/
```

Expected output should show:

- A final checkpoint directory (`checkpoint-411` or similar)
- Model configuration files such as `adapter_config.json`
- Training metrics showing decreasing loss values
- A training loss plot saved as a PNG file

## Step 10. Test inference with the fine-tuned model

```bash
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
## Type: "Hello, how can you help me today?"
## Expect: Response showing fine-tuned behavior
```

## Step 11. Export the model for production deployment

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

## Step 12. Cleanup (optional)

> [!WARNING]
> This will delete all training progress and checkpoints in the cloned repository and remove the virtual environment.

```bash
deactivate
cd ..
rm -rf LLaMA-Factory/
rm -rf factoryEnv/
```

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform listed in this playbook.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| CUDA out of memory during training | All hardware platforms | Batch size too large for available GPU memory | Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps` |
| Cannot access gated repo for URL | All hardware platforms | Certain Hugging Face models have restricted access | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens); request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) in your browser |
| Model download fails or is slow | All hardware platforms | Network connectivity or Hugging Face Hub issues | Check internet connection; try `HF_HUB_OFFLINE=1` for cached models |
| Training loss not decreasing | All hardware platforms | Learning rate too high/low or insufficient data | Adjust `learning_rate` or check dataset quality |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
