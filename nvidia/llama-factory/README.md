# LLaMA Factory

> Install and fine-tune models with LLaMA Factory

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea
LLaMA Factory is an open-source framework that simplifies the process of training and fine
tuning large language models. It offers a unified interface for a variety of cutting edge
methods such as SFT, RLHF, and QLoRA techniques. It also supports a wide range of LLM
architectures such as LLaMA, Mistral and Qwen. This playbook demonstrates how to fine-tune
large language models using LLaMA Factory CLI on your NVIDIA Spark device.

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

- CUDA 13.0 or newer installed: `nvcc --version` (Step 3 installs the `cu130` PyTorch wheels, so a CUDA 13.x driver stack is expected)

- Git installed: `git --version`

- Python 3.10-3.12 with venv and pip: `python3 --version && pip3 --version`

  On a stock Ubuntu image `venv` may not be present. If Step 2 fails with
  `ensurepip is not available`, install it first:

  ```bash
  sudo apt-get update && sudo apt-get install -y python3-venv
  ```

- Sufficient storage space (>50GB for models and checkpoints): `df -h`

- Internet connection for downloading models from Hugging Face Hub

## Ancillary files

- Official LLaMA Factory repository: https://github.com/hiyouga/LLaMA-Factory

- PyTorch with CUDA 13: install via `pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130`

- Example training configuration: `examples/train_lora/qwen3_lora_sft.yaml` (from repository)

- Documentation: https://llamafactory.readthedocs.io/en/latest/getting_started/data_preparation.html

## Time & risk

* **Duration:** 30-60 minutes for initial setup, 1-7 hours for training depending on model size and dataset.
* **Risks:** Model downloads require significant bandwidth and storage. Training may consume substantial GPU memory and require parameter tuning for hardware constraints.
* **Rollback:** Deactivate the virtual environment and remove the `factoryEnv` and `LLaMA-Factory` directories. Training checkpoints are saved locally and can be deleted to reclaim storage space.
* **Last Updated:** 07/26/2026
  * Re-validated end to end on a DGX Spark (GB10). Added `python3-venv` prerequisite, pip
    upgrade step, post-install environment check, and troubleshooting entries for Hugging Face
    cache permission errors, `dill`/pickle failures during dataset preprocessing, and
    out-of-memory errors caused by other processes holding unified memory.

## Verified environment

This playbook was last run end to end on the following configuration. Newer versions
generally work; the versions below are what the walkthrough was confirmed against.

| Component | Version |
|-----------|---------|
| Device | DGX Spark (GB10), 121 GB unified memory |
| Driver / CUDA | 580.159.03 / CUDA 13.0 (`nvcc` V13.0.88) |
| Python | 3.12.3 |
| PyTorch | 2.13.0+cu130 |
| transformers | 5.8.0 |
| datasets | 4.0.0 |
| peft / trl | 0.18.1 / 0.24.0 |
| LLaMA Factory | 0.9.6.dev0 |

Observed result for Step 8 with the unmodified `qwen3_lora_sft.yaml`: 411 optimization
steps, `train_loss` 0.9994, `train_runtime` 17m13s.

## Instructions

## Step 1. Verify system prerequisites

Check that your NVIDIA Spark system has the required components installed and accessible.

```bash
nvcc --version
nvidia-smi
python3 --version
git --version
```

## Step 2. Create and activate a Python virtual environment

Create a virtual environment and activate it for the LLaMA Factory installation.

```bash
python3 -m venv factoryEnv
source ./factoryEnv/bin/activate
pip3 install --upgrade pip
```

> [!IMPORTANT]
> Use a **fresh** virtual environment. Reusing an environment from an earlier run is the most
> common source of the dependency-resolution failures described in
> [Troubleshooting](#troubleshooting) — in particular an outdated `dill`, which breaks dataset
> preprocessing. Never run any `pip` command in this playbook with `sudo`: it creates
> root-owned files in your home directory that later steps cannot write to.

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

## Step 5. Clone LLaMA Factory repository

Download the LLaMA Factory source code from the official repository.

```bash
git clone --depth 1 https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
```

## Step 6. Install LLaMA Factory with dependencies

Install LLaMA Factory in editable mode with metrics support.

```bash
pip install -e ".[metrics]"
```

This step resolves a large dependency tree and can pull packages that conflict with the
PyTorch build installed in Step 3. Confirm the environment is still sound before training:

```bash
python -c "import torch, dill; print('torch', torch.__version__, 'cuda', torch.cuda.is_available()); print('dill', dill.__version__)"
```

Expected: `torch` still reports a `+cu130` build with `cuda True`, and `dill` is **0.3.6 or
newer**. A `dill` older than 0.3.6 cannot deserialize Python 3.11+ code objects and will fail
during dataset preprocessing (see [Troubleshooting](#troubleshooting)). Fix it with:

```bash
pip install --upgrade "dill>=0.3.8"
```

## Step 7. Prepare training configuration

Examine the provided LoRA fine-tuning configuration for Qwen3.

```bash
cat examples/train_lora/qwen3_lora_sft.yaml
```

## Step 8. Launch fine-tuning training

> [!NOTE]
> Login to your Hugging Face Hub to download the model if the model is gated.

Execute the training process using the pre-configured LoRA setup.

```bash
hf auth login   # if the model is gated
llamafactory-cli train examples/train_lora/qwen3_lora_sft.yaml
```

The model is downloaded to `~/.cache/huggingface` (about 8 GB for Qwen3-4B). Confirm that
directory is writable by your user before starting a long run — if an earlier Docker-based
workflow created it, it may be owned by `root`:

```bash
ls -ld ~/.cache/huggingface
sudo chown -R "$USER":"$USER" ~/.cache/huggingface   # only if not owned by you
```

To place the cache on a different volume, set `HF_HOME` to a directory you own:

```bash
export HF_HOME=/path/you/own/hf-cache
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
- Final checkpoint directory (`checkpoint-411` or similar)
- Model configuration files (`adapter_config.json`)
- Training metrics showing decreasing loss values
- Training loss plot saved as PNG file

## Step 10. Test inference with fine-tuned model

Test your fine-tuned model with custom prompts:

```bash
llamafactory-cli chat examples/inference/qwen3_lora_sft.yaml
## Type: "Hello, how can you help me today?"
## Expect: Response showing fine-tuned behavior
```

Example output:
```
User: Hello, how can you help me today?
Assistant: Hello, I am {{name}}, an AI assistant developed by {{author}}.
I am here to assist you with any queries or tasks you may have.
```

> [!NOTE]
> The literal `{{name}}` and `{{author}}` in the reply are expected, not a bug. The `identity`
> dataset used in Step 8 ships with those placeholders unfilled, and the model faithfully
> learned them. To bake in your own values, edit `data/identity.json` before training:
>
> ```bash
> sed -i 's/{{name}}/My Assistant/g; s/{{author}}/My Team/g' data/identity.json
> ```
>
> Seeing this reply confirms the LoRA adapter loaded and changed the model's behavior.

## Step 11. For production deployment, export your model

```bash
llamafactory-cli export examples/merge_lora/qwen3_lora_sft.yaml
```

## Step 12. Cleanup and rollback

> [!WARNING]
> This will delete all training progress and checkpoints.

To remove the virtual environment and cloned repository:

```bash
deactivate
cd ..
rm -rf LLaMA-Factory/
rm -rf factoryEnv/
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| CUDA out of memory during training | Batch size too large for available memory | Reduce `per_device_train_batch_size` or increase `gradient_accumulation_steps` |
| `CUDA error: out of memory` immediately at startup, before any weights load — traceback ends in `torch.cuda.set_device` | Another process already holds most of the unified memory, so a CUDA context cannot be created. Because DGX Spark shares one memory pool between CPU and GPU, an inference server or notebook left running elsewhere on the box will block training entirely | Check what is holding memory with `free -g`, `nvidia-smi`, and `docker ps`, then stop it. Coordinate first if the machine is shared. Flush the buffer cache with the UMA command below, then retry |
| `PermissionError: [Errno 13] Permission denied` writing to `~/.cache/huggingface`, or the model download aborts partway | The cache directory is not owned by your user. Usually caused by a previous Docker-based run writing as `root`, or by running an earlier step with `sudo` | `sudo chown -R "$USER":"$USER" ~/.cache/huggingface`, or point `HF_HOME` at a directory you own. Do not re-run any step with `sudo` |
| Dataset preprocessing crashes while pickling, with a `TypeError` about a code object receiving the wrong number of arguments, such as `code() takes at most 16 arguments (18 given)` or `code() argument 13 must be str, not int` | An outdated `dill` is installed. Versions before 0.3.6 cannot reconstruct Python 3.11+ code objects, and `datasets` uses `dill` to send the tokenizer function to the workers named by `preprocessing_num_workers` | `pip install --upgrade "dill>=0.3.8"`. If the environment was reused from an older run, rebuild it from scratch per Step 2. As a temporary workaround, set `preprocessing_num_workers: 1` in the training YAML to avoid multiprocessing entirely |
| `ImportError` or `AttributeError` from `transformers`, `trl`, or `peft` after Step 6 | LLaMA Factory tracks upstream closely and pip may resolve a combination the pinned ranges do not cover | Rebuild the venv from scratch. If it persists, install the versions listed in [Verified environment](#verified-environment), or check out a tagged LLaMA Factory release instead of `main` |
| PyTorch reports `cuda False`, or a non-`cu130` build, after Step 6 | A dependency pulled a CPU-only or differently-built PyTorch over the one from Step 3 | Reinstall it: `pip3 install --force-reinstall torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu130` |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| Model download fails or is slow | Network connectivity or Hugging Face Hub issues | Check internet connection, try using `HF_HUB_OFFLINE=1` for cached models |
| Training loss not decreasing | Learning rate too high/low or insufficient data | Adjust `learning_rate` parameter or check dataset quality |
| Model replies contain a literal `{{name}}` or `{{author}}` | Expected. The `identity` dataset ships with unfilled placeholders | Substitute your own values in `data/identity.json` before training, as shown in Step 10 |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

Because that pool is shared machine-wide, any other GPU workload counts against this run. A
serving process that reserves a fixed fraction of memory — for example vLLM started with
`--gpu-memory-utilization 0.9` — can leave too little for training to even create a CUDA
context, and the failure surfaces at `torch.cuda.set_device` before any weights are read.
Before a long run, confirm the memory is actually free:

```bash
free -g
nvidia-smi
docker ps
```
