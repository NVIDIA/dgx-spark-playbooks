# Train a Chat Model with NanoChat

> Build a ChatGPT-style LLM end-to-end — tokenizer, pretraining, SFT — then chat via web UI or CLI


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)
  - [Getting additional help](#getting-additional-help)

---

## Overview

## Basic idea

[nanochat](https://github.com/karpathy/nanochat) is Andrej Karpathy’s educational ChatGPT-style training stack — popularized as the best ChatGPT that $100 can buy. This playbook walks you through training and chatting with your own model locally: tokenizer training, pretraining, midtraining / supervised fine-tuning (SFT), and inference via a simple web UI or CLI.

## What you'll accomplish

- Build a Docker environment with PyTorch and nanochat dependencies on your hardware platform
- Run the full training pipeline (BPE tokenizer → base pretraining → chat fine-tuning → report)
- Chat with your trained checkpoints through a web UI or CLI

## What to know before starting

**Required:**

- Basic Linux command line and shell usage
- Working with Docker containers and GPU passthrough
- Basic understanding of training foundation LLM models

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, OS, memory, and whether multi-node applies. This playbook covers **single-node** training on the platforms listed.

| Hardware platform | OS | Memory  | Multi-node capable hardware |
| :---- | :---- | :---- | :---- |
| **DGX Station** | DGX OS (Linux) | ~288 GB HBM | — |


> [!NOTE]
> Only platforms listed in the Supported hardware platforms table above are covered by this playbook.

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Adequate storage for datasets and checkpoints (~50 GB free recommended)

**Software requirements**

- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured
- Verify GPU access: `nvidia-smi`
- Check Docker GPU integration (use the default container tag for your hardware platform from the matrix above):
  ```bash
  docker run --rm --gpus all nvcr.io/nvidia/pytorch:<tag> nvidia-smi
  ```
- [Weights & Biases](https://wandb.ai/) account and API key (required by the launch scripts)
- [Hugging Face](https://huggingface.co/docs/hub/en/security-tokens) token (for evaluation / gated datasets)
- Network access to NGC, Hugging Face, and FineWeb / ClimbMix data

## Ancillary files

Assets live under `assets/` in this playbook:

| File | Purpose |
|------|---------|
| `Dockerfile` | Single-node image (PyTorch NGC + nanochat dependencies) |
| `setup.sh` / `launch.sh` | Single-node setup and launch |
| `speedrun_single.sh` | Single-node speedrun (default d24) |

Upstream reference: [nanochat on GitHub](https://github.com/karpathy/nanochat/).

## Time & risk

- **Estimated time:** Single-node setup is ~30 MIN; a full single-node d24 run is on the order of 12+ hours (mostly unattended compute).
- **Risk level:** Medium
  - Model training is memory-intensive; changing batch size, depth, or precision can cause OOM
  - Large dataset downloads and checkpoints need substantial disk space
  - Launch scripts exit if `WANDB_API_KEY` or `HF_TOKEN` are unset
- **Rollback:** Stop containers, then remove caches (`~/.cache/nanochat` or local `nanochat_cache/`) and the `nanochat` Docker image (non-destructive to the host OS)
- **Last Updated:** 07/31/2026
  - Removed DGX Spark support (not ready yet); playbook is Station / single-node only

## Credits

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) / ClimbMix by Hugging Face (pretraining data)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) by Hugging Face (SFT data)

## Instructions

> [!NOTE]
> These instructions target **single-node** training on Linux (default d24 speedrun) for the hardware platforms listed in the Overview.

## Step 1. Set up Docker permissions

To manage containers without `sudo`, add your user to the `docker` group. Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error, add your user to the docker group (skip if it already works):

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2. Set environment variables

Nanochat uses Weights & Biases for training visualization and a Hugging Face token for evaluation datasets. Export both in your shell:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
export WANDB_RUN=speedrun   # optional run name
```

Create a [W&B account](https://wandb.ai/) and a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) if you do not have them. The launch script exits immediately if either key is unset.

## Step 3. Clone and set up

Clone the playbook repository and navigate to the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/playbook-nanochat/assets
```

Run the single-node setup script. It clones [nanochat](https://github.com/karpathy/nanochat), checks out the supported commit, copies `speedrun_single.sh`, and builds the `nanochat` Docker image:

```bash
chmod +x setup.sh launch.sh
./setup.sh
```

Confirm the image built:

```bash
docker images | grep nanochat
```

Your directory structure after setup should look like:

```
assets/
├── Dockerfile
├── launch.sh
├── setup.sh
├── speedrun_single.sh
└── nanochat/
```

## Step 4. Launch training

Ensure your API keys are still exported, then launch:

```bash
./launch.sh
```

> [!NOTE]
> On a multi-GPU host, default `--gpus all` may not select the intended GPU. Set `GPU_DEVICE` to pin a device (`N` from `nvidia-smi`):
> ```bash
> GPU_DEVICE='"device=N"' ./launch.sh
> ```

The training container runs the full pipeline automatically:

1. **Tokenizer** — downloads pretraining text, trains a BPE tokenizer
2. **Base pretraining** — pretrains the default d24 model (~1B params) with FP8
3. **SFT** — fine-tunes for chat with identity conversations + SmolTalk
4. **Report** — writes metrics and samples to `report.md`

A full d24 run is on the order of 12+ hours. Keep the terminal open or use `tmux` / `screen`.

## Step 5. Monitor training

Track progress at [wandb.ai](https://wandb.ai/) under the `nanochat` project (the run URL appears in the training logs). Key metrics:

- Training loss
- Validation BPB
- Throughput (tokens/sec)

## Step 6. Inference

After training, checkpoints are under `nanochat_cache/`. Run inference from a container with the same GPU selection as training.

**Web UI (recommended):**

```bash
docker run --rm --gpus all --net=host \
    -v $(pwd)/nanochat:/workspace/nanochat \
    -v $(pwd)/nanochat_cache:/root/.cache/nanochat \
    -w /workspace/nanochat \
    nanochat \
    python -m scripts.chat_web
```

Open a browser to `http://<HOST_IP>:8000`. If you are on a remote SSH session, forward the port:

```bash
ssh -L 8000:localhost:8000 username@<HOST_IP>
```

**CLI:**

```bash
docker run --rm -it --gpus all \
    -v $(pwd)/nanochat:/workspace/nanochat \
    -v $(pwd)/nanochat_cache:/root/.cache/nanochat \
    -w /workspace/nanochat \
    nanochat \
    python -m scripts.chat_cli -p "Why is the sky blue?"
```

## Step 7. Cleanup

To stop training early:

> [!WARNING]
> This stops the training run and any in-progress work in the container.

```bash
## If launch.sh is running: press Ctrl+C

## Or stop the container directly
docker stop $(docker ps -q --filter ancestor=nanochat)
```

To free disk space (cache dirs are often root-owned because the container runs as root):

```bash
sudo rm -rf ./nanochat_cache ./hf_cache
docker rmi nanochat
```

## Step 8. Customization

**Smaller / faster run:** Edit `speedrun_single.sh` before setup to reduce data and model size:

```bash
## Fewer data shards
python -m nanochat.dataset -n 10 &

## Smaller model (d4 instead of d24), smaller batch size
python -m scripts.base_train --depth=4 --device-batch-size=32
```

Then re-run `./setup.sh` to rebuild with the changes.

**Batch size:** The default `--device-batch-size=64` is tuned for high-memory hardware platforms. Lower it (32, 16, 8) if you hit OOM.

## Next steps

- Try sample prompts with the trained model (web UI or CLI)
- Experiment with larger or smaller depths in the speedrun script
- Customize model personality via identity conversations — see the [nanochat customization guide](https://github.com/karpathy/nanochat/discussions/139)

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| `WANDB_API_KEY is not set` or `HF_TOKEN is not set` | All hardware platforms | Required env vars not exported before launch | `export WANDB_API_KEY=<key>` and `export HF_TOKEN=<token>` in the same shell, then re-run launch |
| `RuntimeError: CUDA out of memory` | All hardware platforms | Batch size or model depth too large | Reduce `--device-batch-size` / `--device_batch_size` (try 64 → 32 → 16 → 8) or lower `--depth` in the speedrun script; re-run setup then launch |
| Container fails to start / GPU not visible | All hardware platforms | NVIDIA Container Toolkit not configured, or GPU in use | Test `docker run --rm --gpus all nvcr.io/nvidia/pytorch:<tag> nvidia-smi`; ensure no other GPU containers are running; check `docker logs` |
| `nanochat` image not found | All hardware platforms | Setup not run or Docker build failed | From `assets/`, run `./setup.sh` and confirm with `docker images \| grep nanochat` |
| Training hangs at dataset download | All hardware platforms | Network issue downloading shards | Check connectivity; downloads can take a long time — restart launch if it stalls indefinitely |
| Disk full / `No space left on device` | All hardware platforms | Dataset + checkpoints exhausted disk | Ensure ~50 GB free before training; `docker system prune`; remove old caches under `~/.cache/nanochat` or `./nanochat_cache` |
| Web UI not reachable on port 8000 | All hardware platforms | Server not running, port blocked, or missing SSH tunnel | Confirm `chat_web` is running; `ssh -L 8000:localhost:8000 user@<HOST_IP>` if remote; allow port 8000 if firewalled |
| Model runs on wrong GPU | multi-GPU hosts | Default GPU selection | Pin with `GPU_DEVICE='"device=N"' ./launch.sh` (`N` from `nvidia-smi`) |

### Getting additional help

1. nanochat issues: https://github.com/karpathy/nanochat/issues
2. Container logs: `docker logs <container_id>`
3. System resources: `htop` and `nvidia-smi`
4. NVIDIA Developer Forums: https://forums.developer.nvidia.com/
