# Nanochat Training

> Train a small ChatGPT-style LLM (nanochat) with tokenizer, pretraining, midtraining, and SFT on DGX Station with GB300 Ultra


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook demonstrates training of [nanochat](https://github.com/karpathy/nanochat) on DGX Station with the GB300 Ultra Superchip. You run the full pipeline on a single system: custom BPE tokenizer training, base model pretraining, supervised fine-tuning (SFT), and inference via CLI or web UI.

The project uses the PyTorch NGC container, FineWeb for pretraining data, SmolTalk for SFT, and Weights & Biases for logging. The default configuration trains a ~1B parameter (d24) Transformer model with FP8 precision.

## What you'll accomplish

- **Environment:** Docker image with PyTorch NGC and nanochat dependencies on your DGX Station.
- **Training pipeline:** BPE tokenizer (65K vocab), base model pretraining with FP8, SFT, and automated report generation.
- **Inference:** ChatGPT-style web UI and CLI to chat with the base or SFT checkpoints.
- **Monitoring:** W&B dashboards and `nanochat_cache/report/report.md` with metrics and samples.

## What to know before starting

- Basic Linux command line and shell usage.
- Familiarity with Docker and GPU containers (e.g. `docker run --gpus all`).
- Optional: understanding of LLM training (tokenizer, pretraining, fine-tuning).

## Prerequisites

**Hardware:**

- NVIDIA DGX Station with GB300 Ultra Superchip (288GB VRAM).
- Adequate storage for cache (~25GB+ for FineWeb data and checkpoints).

**Software:**

- Docker with NVIDIA Container Toolkit: `docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.04-py3 nvidia-smi`
- Network access to download datasets (Hugging Face, FineWeb) and container images (nvcr.io)
- [Weights & Biases](https://wandb.ai/) account and API key.
- [Hugging Face](https://huggingface.co/docs/hub/en/security-tokens) token for evaluation datasets.

## Model architecture (d24)

```
Layers: 24
Attention Heads: 12
Head Dimension: 128
Context Length: 2048 tokens
Vocabulary Size: 65,536 (2^16, trained BPE)
Precision: FP8 (e4m3, tensorwise scaling)
```

## Training stages

| Stage | Description |
|-------|-------------|
| Tokenizer | Trains BPE tokenizer (65K vocab) on ~2B characters from FineWeb |
| Base pretraining | Pretrains d24 model on FineWeb with FP8, target data:param ratio of 8 |
| SFT | Fine-tunes on synthetic identity conversations + SmolTalk |
| Report | Generates `report.md` with metrics, samples, and system info |

## Ancillary files

All required assets are in `nvidia/station-nanochat/assets/`:

- `Dockerfile` – PyTorch NGC image with nanochat pip dependencies.
- `setup.sh` – Clones nanochat, checks out the supported commit, copies `speedrun_station.sh`, and builds the Docker image.
- `launch.sh` – Runs the training container (full pipeline: tokenizer → pretrain → SFT → report).
- `speedrun_station.sh` – Modified speedrun script adapted for single-GPU DGX Station.

## Time & risk

- **Estimated time:** ~30 minutes for setup. Full d24 training takes on the order of 16+ hours on a single GB300 Ultra.
- **Risk level:** Medium
  - Large downloads (FineWeb) can be slow; ensure stable network and disk space.
  - API keys (W&B, HF) must be set or `launch.sh` will exit immediately.
- **Rollback:** Stop containers with `docker stop`, remove caches, and run `docker system prune -a` if needed.

## Credits

- [nanochat](https://github.com/karpathy/nanochat) by Andrej Karpathy
- [FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb) by HuggingFace (pretraining data)
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) by HuggingFace (SFT data)

## Instructions

## Step 1. Prerequisites and environment

Ensure your DGX Station has Docker with NVIDIA runtime and GPU access. Nanochat uses Weights & Biases (W&B) for training visualization and a Hugging Face token for evaluation datasets.

```bash
## Verify GPU and Docker
nvidia-smi
docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.04-py3 nvidia-smi
```

Create a [W&B account](https://wandb.ai/) and a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens) if you don't have them. Export both keys in your shell:

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

## Step 2. Clone and set up

Clone the playbook repository and navigate to the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/station-nanochat/assets
```

Run the setup script. It clones [nanochat](https://github.com/karpathy/nanochat), checks out the supported commit, copies the station-adapted `speedrun_station.sh`, and builds the `nanochat` Docker image (PyTorch NGC base with dependencies):

```bash
./setup.sh
```

You should see the `nanochat` image listed if you run `docker images`. Your directory structure after setup should look like this:

```
assets/
├── Dockerfile
├── launch.sh
├── setup.sh
├── speedrun_station.sh
└── nanochat/
```

## Step 3. Launch training

Ensure your API keys are exported, then launch:

```bash
./launch.sh
```

The training runs inside the `nanochat` container and executes the full pipeline automatically:

1. **Tokenizer training** — downloads ~2B characters from FineWeb, trains a 65K BPE tokenizer
2. **Base model pretraining** — downloads additional FineWeb shards, pretrains a d24 model (~1B params) with FP8
3. **SFT** — downloads synthetic identity conversations, fine-tunes for chat
4. **Report generation** — produces `report.md` with metrics and samples

Training on a single GB300 Ultra takes on the order of 16+ hours for the full d24 run.

## Step 4. Monitor training

**W&B dashboard:**

Track training at [wandb.ai](https://wandb.ai/) under the `nanochat` project. The exact link to the wandb run would be provided in the training logs. Key metrics:
- Training loss
- Validation BPB
- Throughput (tokens/sec)

## Step 5. Inference

After training, checkpoints are saved under the `nanochat_cache/` directory. Run inference from inside the container or interactively:

**Web UI (recommended):**

```bash
docker run --rm --gpus all --net=host \
    -v $(pwd)/nanochat:/workspace/nanochat \
    -v $(pwd)/nanochat_cache:/root/.cache/nanochat \
    -w /workspace/nanochat \
    nanochat \
    python -m scripts.chat_web
```

Open a browser to `http://<STATION_IP>:8000` where `<STATION_IP>` is your DGX Station’s IP address.

**CLI:**

```bash
docker run --rm -it --gpus all \
    -v $(pwd)/nanochat:/workspace/nanochat \
    -v $(pwd)/nanochat_cache:/root/.cache/nanochat \
    -w /workspace/nanochat \
    nanochat \
    python -m scripts.chat_cli -p "Why is the sky blue?"
```

## Step 6. Cleanup

To stop training early, interrupt the launch script or stop the container:

> [!WARNING]
> This stops the training run and any in-progress work in the container.

```bash
## If launch.sh is running: press Ctrl+C

## Or stop the container directly
docker stop $(docker ps -q --filter ancestor=nanochat)
```

To free disk space:

```bash
rm -rf ./nanochat_cache ./hf_cache
docker system prune -a
```

## Step 7. Customization

**Smaller/faster run:** Edit `speedrun_station.sh` before running setup to reduce data and model size:

```bash
## Fewer data shards (10 instead of default)
python -m nanochat.dataset -n 10 &

## Smaller model (d4 instead of d24), smaller batch size
python -m scripts.base_train --depth=4 --device-batch-size=32
```

**Batch size:** The default `--device-batch-size=64` is tuned for the GB300's 288GB VRAM. Feel free to change the batch size if utilization is low or the training OOMs.

Then re-run `./setup.sh` to rebuild with the changes.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `WANDB_API_KEY is not set` or `HF_TOKEN is not set` | Required env vars not exported before `launch.sh` | `export WANDB_API_KEY=<key>` and `export HF_TOKEN=<token>` in the same shell, then re-run `./launch.sh` |
| `RuntimeError: CUDA out of memory` | Batch size too large for available VRAM | Edit `speedrun_station.sh`: reduce `--device-batch-size` (try 64, 32, 16, 8). Re-run `./setup.sh` then `./launch.sh` |
| Docker container exits immediately | Missing env vars, bad cache paths, or build failure | Check logs: `docker ps -a` then `docker logs <container_id>`. Fix env vars or paths as needed |
| `nanochat` image not found | Setup not run or Docker build failed | From the `assets/` directory, run `./setup.sh` and confirm with `docker images \| grep nanochat` |
| `No such file or directory` for cache paths | Cache directories don't exist | `launch.sh` creates them automatically under `$(pwd)/nanochat_cache` and `$(pwd)/hf_cache`. If using custom paths, create them: `mkdir -p $NANOCHAT_CACHE $HF_CACHE` |
| Training hangs at "Waiting for dataset download" | Network issue downloading FineWeb shards | Check network connectivity. The download can take time depending on bandwidth. If it persists, restart `./launch.sh` |
| W&B shows wrong user / stale login | Cached W&B credentials in container volume | `speedrun_station.sh` runs `wandb login --relogin` with your key automatically. Ensure `WANDB_API_KEY` is correct |
| Container runs but `launch.sh` says "Training complete!" immediately | Container failed fast and exited before the poll loop detected it | Check `docker ps -a` for the exited container and inspect logs with `docker logs <id>` |
| GPU not visible inside container | Docker NVIDIA runtime not configured | Test: `docker run --rm --gpus all nvcr.io/nvidia/pytorch:26.04-py3 nvidia-smi`. If it fails, install/configure [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html) |
