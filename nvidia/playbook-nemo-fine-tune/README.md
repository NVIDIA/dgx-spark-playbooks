# Fine-Tune with NVIDIA NeMo

> Hugging Face model training from single-GPU to multi-node jobs

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVIDIA NeMo AutoModel provides GPU-accelerated, end-to-end fine-tuning for Hugging Face large language models and vision-language models with native PyTorch support. You can start training without conversion delays, using optimized kernels and memory-efficient recipes from a single GPU through distributed setups.

## What you'll accomplish

You'll set up a fine-tuning environment for large language models (about 1–70B parameters) and vision-language models using NeMo AutoModel on your hardware platform. By the end, you'll have a working Docker-based installation that supports parameter-efficient fine-tuning (PEFT), supervised fine-tuning (SFT), and related training workflows with FP8 precision options, while staying compatible with the Hugging Face ecosystem.

## What to know before starting

**Required:**

- Working in Linux terminal environments and SSH connections
- Basic understanding of Python virtual environments and package management
- Familiarity with GPU computing concepts and CUDA toolkit usage
- Experience with containerized workflows and Docker operations
- Understanding of machine learning model training and fine-tuning concepts

**Optional:**

- Experience with distributed training configurations on multi-node capable hardware

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | `nvcr.io/nvidia/nemo-automodel:26.02` | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Minimum 32 GB system memory for efficient model loading and training
- Active internet connection for downloading models and packages
- SSH access to your hardware platform configured

**Software requirements**

- CUDA toolkit 12.0+ installed and configured: `nvcc --version`
- Python 3.10+ environment available: `python3 --version`
- Docker installed and usable: `docker ps`
- Git installed for repository cloning: `git --version`
- Hugging Face account and access token (for gated models)

## Ancillary files

All necessary files for this playbook are in the [NeMo AutoModel GitHub repository](https://github.com/NVIDIA-NeMo/Automodel).

## Time & risk

- **Estimated time:** 45–90 MIN for complete setup and an initial fine-tuning run (longer on first run due to model download)
- **Risk level:** Medium
  - Model downloads can be large (several GB)
  - Package or architecture compatibility issues may require troubleshooting
  - Distributed training complexity increases if you extend beyond the single-node examples in this playbook
- **Rollback:** The container was launched with `--rm`, so exiting removes it. Optionally remove the Docker image to reclaim disk space (see Cleanup in the **Instructions** tab). No lasting host changes beyond optional Docker group membership.
- **Last Updated:** 07/31/2026
  - NeMo AutoModel Docker workflow for LoRA, QLoRA, and full SFT fine-tuning on supported hardware platforms

## Instructions

## Step 1. Verify system requirements

Confirm your hardware platform meets the prerequisites for [NeMo AutoModel](https://github.com/NVIDIA-NeMo/Automodel). Run these checks on the host to confirm CUDA, Python, GPU access, memory, and Docker.

```bash
## Verify CUDA installation
nvcc --version

## Check Python version (3.10+ required)
python3 --version

## Verify GPU accessibility
nvidia-smi

## Check available system memory
free -h

## Docker access
docker ps
```

If `docker ps` returns a permission denied error (for example, permission denied while trying to connect to the Docker daemon socket), complete Step 2. Otherwise continue to Step 3.

## Step 2. Configure Docker permissions

To manage containers without `sudo`, add your user to the `docker` group. If you skip this step, run Docker commands with `sudo`.

```bash
sudo usermod -aG docker $USER
newgrp docker
```

Open a new terminal (or continue after `newgrp`) and confirm access:

```bash
docker ps
```

## Step 3. Get the container image with NeMo AutoModel

```bash
docker pull nvcr.io/nvidia/nemo-automodel:26.02
```

## Step 4. Launch Docker

Launch an interactive container with GPU access. The `--rm` flag removes the container when you exit.

```bash
docker run \
  --gpus all \
  --ulimit memlock=-1 \
  -it --ulimit stack=67108864 \
  --entrypoint /usr/bin/bash \
  --rm nvcr.io/nvidia/nemo-automodel:26.02
```

## Step 5. Explore available examples

Review the pre-configured training recipes for different model types and training scenarios.

```bash
## Navigate to /opt/Automodel
cd /opt/Automodel

## List LLM fine-tuning examples
ls examples/llm_finetune/

## View example recipe configuration
cat examples/llm_finetune/finetune.py | head -20
```

## Step 6. Run sample fine-tuning

The following commands show full fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) with LoRA and QLoRA.

First, export your Hugging Face token so gated models can be downloaded.

```bash
export HF_TOKEN=<your_huggingface_token>
```

> [!NOTE]
> Replace `<your_huggingface_token>` with your personal Hugging Face access token. A valid token is required to download any gated model.
>
> - Generate a token: [Hugging Face tokens](https://huggingface.co/settings/tokens); guide available [here](https://huggingface.co/docs/hub/en/security-tokens).
> - Request and receive access on each model's page (and accept license/terms) before attempting downloads.
>   - Llama-3.1-8B: [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B)
>   - Qwen3-8B: [Qwen/Qwen3-8B](https://huggingface.co/Qwen/Qwen3-8B)
>   - Meta-Llama-3-70B: [meta-llama/Meta-Llama-3-70B](https://huggingface.co/meta-llama/Meta-Llama-3-70B)
>
> The same steps apply for any other gated model you use: visit its model card on Hugging Face, request access, accept the license, and wait for approval.

**LoRA fine-tuning example:**

Run a basic fine-tuning example to validate the setup. This demonstrates parameter-efficient fine-tuning with a model suitable for testing. The examples below use YAML for configuration; parameter overrides are passed as command-line arguments.

```bash
cd /opt/Automodel
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_2/llama3_2_1b_squad_peft.yaml \
--model.pretrained_model_name_or_path meta-llama/Llama-3.1-8B \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20
```

These overrides ensure the Llama-3.1-8B LoRA run behaves as expected:

- `--model.pretrained_model_name_or_path`: selects the Llama-3.1-8B model to fine-tune from the Hugging Face model hub (weights fetched via your Hugging Face token).
- `--packed_sequence.packed_sequence_size`: sets the packed sequence size to 1024 to enable packed sequence training.
- `--step_scheduler.max_steps`: sets the maximum number of training steps. Set to 20 for demonstration; adjust based on your needs.

> [!NOTE]
> The recipe YAML `llama3_2_1b_squad_peft.yaml` defines training hyperparameters (LoRA rank, learning rate, and related settings) that are reusable across Llama model sizes. The `--model.pretrained_model_name_or_path` override determines which model weights are actually loaded.

**QLoRA fine-tuning example:**

Use QLoRA to fine-tune large models in a memory-efficient manner.

```bash
cd /opt/Automodel
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/llama3_1/llama3_1_8b_squad_qlora.yaml \
--model.pretrained_model_name_or_path meta-llama/Meta-Llama-3-70B \
--loss_fn._target_ nemo_automodel.components.loss.te_parallel_ce.TEParallelCrossEntropy \
--step_scheduler.local_batch_size 1 \
--packed_sequence.packed_sequence_size 1024 \
--step_scheduler.max_steps 20
```

These overrides ensure the 70B QLoRA run behaves as expected:

- `--model.pretrained_model_name_or_path`: selects the 70B base model to fine-tune (weights fetched via your Hugging Face token).
- `--loss_fn._target_`: uses the TransformerEngine-parallel cross-entropy loss variant compatible with tensor-parallel training for large LLMs.
- `--step_scheduler.local_batch_size`: sets the per-GPU micro-batch size to 1 to fit 70B in memory; overall effective batch size is still driven by gradient accumulation and data/tensor parallel settings from the recipe.
- `--step_scheduler.max_steps`: sets the maximum number of training steps. Set to 20 for demonstration; adjust based on your needs.
- `--packed_sequence.packed_sequence_size`: sets the packed sequence size to 1024 to enable packed sequence training.

**Full fine-tuning example:**

Run the following command to perform full (SFT) fine-tuning:

```bash
cd /opt/Automodel
python3 examples/llm_finetune/finetune.py \
-c examples/llm_finetune/qwen/qwen3_8b_squad_spark.yaml \
--model.pretrained_model_name_or_path Qwen/Qwen3-8B \
--step_scheduler.local_batch_size 1 \
--step_scheduler.max_steps 20 \
--packed_sequence.packed_sequence_size 1024
```

These overrides ensure the Qwen3-8B SFT run behaves as expected:

- `--model.pretrained_model_name_or_path`: selects the Qwen/Qwen3-8B model to fine-tune from the Hugging Face model hub (weights fetched via your Hugging Face token). Adjust this if you want to fine-tune a different model.
- `--step_scheduler.max_steps`: sets the maximum number of training steps. Set to 20 for demonstration; adjust based on your needs.
- `--step_scheduler.local_batch_size`: sets the per-GPU micro-batch size to 1 to fit in memory; overall effective batch size is still driven by gradient accumulation and data/tensor parallel settings from the recipe.
- `--packed_sequence.packed_sequence_size`: sets the packed sequence size to 1024 to enable packed sequence training.

## Step 7. Validate successful training completion

Validate the fine-tuned model by inspecting artifacts in the checkpoint directory.

```bash
## Inspect logs and checkpoint output.
## LATEST is a symlink pointing to the latest checkpoint.
## Below is an example of expected output (username and domain-users are placeholders).
ls -lah checkpoints/LATEST/

## $ ls -lah checkpoints/LATEST/
## total 32K
## drwxr-xr-x 6 username domain-users 4.0K Oct 16 22:33 .
## drwxr-xr-x 4 username domain-users 4.0K Oct 16 22:33 ..
## -rw-r--r-- 1 username domain-users 1.6K Oct 16 22:33 config.yaml
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 dataloader
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 model
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 optim
## drwxr-xr-x 2 username domain-users 4.0K Oct 16 22:33 rng
## -rw-r--r-- 1 username domain-users 1.3K Oct 16 22:33 step_scheduler.pt
```

## Step 8. Cleanup (Optional)

The container was launched with the `--rm` flag, so it is automatically removed when you exit. To reclaim disk space used by the Docker image, run:

> [!WARNING]
> This will remove the NeMo AutoModel image. You will need to pull it again if you want to use it later.

```bash
docker rmi nvcr.io/nvidia/nemo-automodel:26.02
```

## Step 9. Optional: Publish your fine-tuned model checkpoint on Hugging Face Hub

Publish your fine-tuned model checkpoint on Hugging Face Hub.

> [!NOTE]
> This is an optional step and is not required for using the fine-tuned model.
> It is useful if you want to share your fine-tuned model with others or use it in other projects.
> To use the Hugging Face CLI, install it with `pip install huggingface_hub`.
> For more information, see the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

> [!TIP]
> You can use the `hf` command to upload the fine-tuned model checkpoint to Hugging Face Hub.
> For more information, see the [Hugging Face CLI documentation](https://huggingface.co/docs/huggingface_hub/en/guides/cli).

```bash
## Publish under the namespace <your_huggingface_username>/my-cool-model; adjust the name as needed.
hf upload my-cool-model checkpoints/LATEST/model
```

> [!TIP]
> The above command can fail if you don't have write permissions to the Hugging Face Hub with the `HF_TOKEN` you used.
> Sample error message:
> ```bash
> user@host:/opt/Automodel$ hf upload my-cool-model checkpoints/LATEST/model
> Traceback (most recent call last):
>   File "/home/user/.local/lib/python3.10/site-packages/huggingface_hub/utils/_http.py", line 409, in hf_raise_for_status
>     response.raise_for_status()
>   File "/home/user/.local/lib/python3.10/site-packages/requests/models.py", line 1024, in raise_for_status
>     raise HTTPError(http_error_msg, response=self)
> requests.exceptions.HTTPError: 403 Client Error: Forbidden for url: https://huggingface.co/api/repos/create
> ```
> To fix this, create an access token with *write* permissions. See the Hugging Face guide [here](https://huggingface.co/docs/hub/en/security-tokens).

## Step 10. Next steps

Begin using NeMo AutoModel for your specific fine-tuning tasks. Start with the provided recipes and customize them for your model and dataset.

```bash
## Copy a recipe for customization
cp examples/llm_finetune/finetune.py my_custom_training.py

## Edit configuration for your specific model and data, then run:
python3 my_custom_training.py
```

Explore the [NeMo AutoModel GitHub repository](https://github.com/NVIDIA-NeMo/Automodel) for more recipes, documentation, and community examples. Consider setting up custom datasets and experimenting with different model architectures.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nvcc: command not found` | CUDA toolkit not in PATH | Add CUDA toolkit to PATH: `export PATH=/usr/local/cuda/bin:$PATH` |
| `pip install uv` permission denied | System-level pip restrictions | Use `pip3 install --user uv` and update PATH |
| GPU not detected in training | CUDA driver/runtime mismatch | Verify driver compatibility with `nvidia-smi` and reinstall CUDA if needed |
| Out of memory during training | Model too large for available GPU memory | Reduce batch size, enable gradient checkpointing, or use model parallelism |
| Package compatibility issues on your architecture | Package not available for the host architecture | Use source installation or build from source with architecture-appropriate flags |
| Cannot access gated repo for URL | Certain Hugging Face models have restricted access | Regenerate your [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens); request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) in your browser |
| Memory pressure within capacity | Unified memory buffer cache not released | See UMA note below |

> [!NOTE]
> Some hardware platforms use Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. If you hit memory pressure even when within rated capacity, manually flush the buffer cache with:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
