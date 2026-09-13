# Fine-Tune with PyTorch

> Direct framework control for LoRA, QLoRA, and full supervised training

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Multi-node fine-tuning](#multi-node-fine-tuning)
  - [Prerequisites](#prerequisites)
  - [Step 1. Identify the interconnect](#step-1-identify-the-interconnect)
  - [Step 2. Install NVIDIA Container Toolkit and configure Docker](#step-2-install-nvidia-container-toolkit-and-configure-docker)
  - [Step 3. Enable GPU resource advertising](#step-3-enable-gpu-resource-advertising)
  - [Step 4. Initialize Docker Swarm](#step-4-initialize-docker-swarm)
  - [Step 5. Join worker nodes](#step-5-join-worker-nodes)
  - [Step 6. Deploy the multi-node stack](#step-6-deploy-the-multi-node-stack)
  - [Step 7. Capture the container ID](#step-7-capture-the-container-id)
  - [Step 8. Adapt the Accelerate configuration files](#step-8-adapt-the-accelerate-configuration-files)
  - [Step 9. Run a multi-node fine-tuning script](#step-9-run-a-multi-node-fine-tuning-script)
  - [Step 10. Cleanup and rollback](#step-10-cleanup-and-rollback)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

PyTorch is the core deep-learning framework for training and customizing large language models. This playbook shows how to run supervised fine-tuning (SFT) and parameter-efficient fine-tuning (PEFT) — including LoRA and QLoRA — inside the NVIDIA PyTorch container on your hardware platform.

You pull a container, install the training stack (Transformers, PEFT, datasets, TRL, bitsandbytes), and run provided recipes for Llama 3 models from 3B to 70B parameters. On multi-node capable hardware, you can scale the same recipes across two nodes with Accelerate and FSDP.

## What you'll accomplish

You'll set up a complete fine-tuning environment on your **hardware platform** and run full SFT, LoRA, or QLoRA recipes for Llama 3 models (3B–70B).

Optionally, you can distribute training across two nodes on multi-node capable hardware (see the **Multi-node fine-tuning** tab).

## What to know before starting

**Required:**

- Experience with fine-tuning models in PyTorch
- Working with Docker containers and GPU passthrough

**Optional:**

- Familiarity with LoRA, QLoRA, and Hugging Face Trainer / Accelerate
- Basic networking and passwordless SSH between nodes (multi-node capable hardware only)

> [!TIP]
> For DGX Spark multi-node fine-tuning, use [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) to configure the interconnect and inter-device SSH. A successful Cluster Assistant run satisfies that prerequisite.

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, default container image, and whether multi-node fine-tuning applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | `nvcr.io/nvidia/pytorch:25.11-py3` | ✅ (high-speed interconnect + Docker Swarm) |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Sufficient memory and storage for the selected model and dataset (model downloads can be several GB)
- For distributed training: multi-node capable hardware with a configured interconnect and passwordless SSH

**Software requirements**

- Docker installed: `docker --version`
- NVIDIA Container Toolkit configured
- Verify GPU access: `nvidia-smi`
- Hugging Face account with an access token (required for gated Llama models)
- Network access to NGC and Hugging Face

## Ancillary files

All required assets are in [this playbook's assets folder](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/assets).

| File | Purpose |
|------|---------|
| `Llama3_3B_full_finetuning.py` | Full SFT for Llama 3.2 3B |
| `Llama3_8B_LoRA_finetuning.py` | LoRA fine-tuning for Llama 3.1 8B |
| `Llama3_70B_LoRA_finetuning.py` | LoRA fine-tuning for Llama 3.1 70B (FSDP-ready) |
| `Llama3_70B_qLoRA_finetuning.py` | QLoRA fine-tuning for Llama 3.1 70B |
| `docker-compose.yml` | Multi-node Docker Swarm stack |
| `pytorch-ft-entrypoint.sh` | Multi-node container SSH entrypoint |
| `install-requirements` | Dependency install helper for multi-node runs |
| `run-multi-llama_*` | Multi-node launch helpers |
| `configs/config_finetuning.yaml` | Accelerate config for full SFT (multi-node) |
| `configs/config_fsdp_lora.yaml` | Accelerate config for LoRA + FSDP (multi-node) |

## Time & risk

- **Estimated time:** 60 MIN for setup and a first fine-tuning run (training duration varies with model size and dataset)
- **Risk level:** Medium
  - Model downloads can be large and may require Hugging Face gated-model access
  - Fine-tuning is memory-intensive; large models may need LoRA/QLoRA or multi-node capable hardware
- **Rollback:** Exit the container; remove downloaded models from `$HOME/.cache/huggingface` if you need to free disk space
- **Last Updated:** 08/12/2026
  - Added NVIDIA Sync Cluster Assistant guidance for multi-node fine-tuning
  - 07/31/2026: Single-node PyTorch fine-tuning recipes and multi-node Accelerate/FSDP workflow for supported hardware platforms

## Instructions

> [!NOTE]
> These instructions target **single-node** fine-tuning on Linux. For distributed training on **multi-node capable hardware**, use the **Multi-node fine-tuning** tab after completing Docker permissions here.

## Step 1. Configure Docker permissions

To manage containers without `sudo`, your user must be in the `docker` group. Open a terminal and test Docker access:

```bash
docker ps
```

If you see a permission-denied error, add your user to the docker group:

```bash
sudo usermod -aG docker "$USER"
newgrp docker
```

## Step 2. Pull the PyTorch container

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 3. Launch the container

```bash
docker run --gpus all -it --rm --ipc=host \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "${PWD}:/workspace" -w /workspace \
  nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 4. Install dependencies inside the container

```bash
pip install transformers peft datasets trl bitsandbytes
```

## Step 5. Authenticate with Hugging Face

```bash
hf auth login
```

Enter your Hugging Face token when prompted. Choose `n` when asked to add credentials as git credentials.

## Step 6. Clone the fine-tuning recipes

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-pytorch-fine-tune/assets
```

## Step 7. Run a fine-tuning recipe

#### Available fine-tuning scripts

| Script | Model | Fine-tuning type | Description |
|--------|-------|------------------|-------------|
| `Llama3_3B_full_finetuning.py` | Llama 3.2 3B | Full SFT | Full supervised fine-tuning (all parameters trainable) |
| `Llama3_8B_LoRA_finetuning.py` | Llama 3.1 8B | LoRA | Low-Rank Adaptation (parameter-efficient) |
| `Llama3_70B_LoRA_finetuning.py` | Llama 3.1 70B | LoRA | Low-Rank Adaptation with FSDP support |
| `Llama3_70B_qLoRA_finetuning.py` | Llama 3.1 70B | QLoRA | Quantized LoRA (4-bit) for memory efficiency |

#### Basic usage

```bash
## Full fine-tuning on Llama 3.2 3B
python Llama3_3B_full_finetuning.py

## LoRA fine-tuning on Llama 3.1 8B
python Llama3_8B_LoRA_finetuning.py

## QLoRA fine-tuning on Llama 3.1 70B
python Llama3_70B_qLoRA_finetuning.py
```

#### Common command-line arguments

All scripts support the following arguments:

##### Model configuration

- `--model_name`: Model name or path (default varies by script)
- `--dtype`: Model precision — `float32`, `float16`, or `bfloat16` (default: `bfloat16`)

##### Training configuration

- `--batch_size`: Per-device training batch size (default varies by script)
- `--seq_length`: Maximum sequence length (default: `2048`)
- `--num_epochs`: Number of training epochs (default: `1`)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `1`)
- `--learning_rate`: Learning rate (default varies by script)
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory (flag)

##### LoRA configuration (LoRA and QLoRA scripts only)

- `--lora_rank`: LoRA rank — higher values mean more trainable parameters (default: `8`)

##### Dataset configuration

- `--dataset_size`: Number of samples from the Alpaca dataset (default: `512`)

##### Logging configuration

- `--logging_steps`: Log metrics every N steps (default: `1`)
- `--log_dir`: Directory for TensorBoard logs (default: `logs`)

##### Model saving

- `--output_dir`: Directory to save the fine-tuned model (default: `None` — model not saved)

#### Usage example

```bash
python Llama3_8B_LoRA_finetuning.py \
  --dataset_size 100 \
  --num_epochs 1 \
  --batch_size 2
```

## Step 8. Cleanup (optional)

When you are finished, exit the container. To free disk space used by downloaded models and datasets:

```bash
rm -rf "$HOME/.cache/huggingface/hub/models--meta-llama"* \
  "$HOME/.cache/huggingface/hub/datasets"*
```

> [!WARNING]
> This removes cached Llama model weights and datasets from your Hugging Face cache. Re-download them the next time you fine-tune.

## Next steps

1. Adjust batch size, sequence length, or LoRA rank for your dataset and memory budget
2. Save checkpoints with `--output_dir` and evaluate them with your preferred inference stack
3. On multi-node capable hardware, continue with the **Multi-node fine-tuning** tab

## Multi-node fine-tuning

## Multi-node fine-tuning

Use this tab to fine-tune across **multi-node capable hardware** with Docker Swarm, Accelerate, and FSDP. Complete Docker permissions from the **Instructions** tab on every node first. This tab contains only multi-node-specific steps.

### Prerequisites

- Multi-node capable hardware with a configured high-speed interconnect
- The same user account on every node
- Passwordless SSH between nodes
- Docker and NVIDIA Container Toolkit on every node
- A Hugging Face token with access to the selected model

> [!TIP]
> If [NVIDIA Sync Cluster Assistant](https://docs.nvidia.com/sync/latest/cluster-assistant.html) successfully configured your DGX Spark cluster, the interconnect and SSH prerequisites are complete. Do not repeat the manual connection playbook; continue to Step 1. Otherwise, follow [Connect two nodes for distributed workloads](https://build.nvidia.com/playbooks/connect-two-sparks).

### Step 1. Identify the interconnect

On each node, identify the interface used for multi-node traffic and record its IPv4 address:

```bash
ip -br -4 address

export MN_IF_NAME="<INTERCONNECT_INTERFACE>"
export MN_IP_ADDRESS="$(ip -4 addr show "$MN_IF_NAME" | awk '/inet / {print $2}' | cut -d/ -f1)"
echo "$MN_IF_NAME $MN_IP_ADDRESS"
```

Verify that every node can reach every other node over these addresses.

### Step 2. Install NVIDIA Container Toolkit and configure Docker

On every node that will provide GPU resources, ensure the NVIDIA drivers and the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) are installed, including the [Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) for the toolkit.

### Step 3. Enable GPU resource advertising

Find your GPU UUID:

```bash
nvidia-smi -a | grep UUID
```

Edit `/etc/docker/daemon.json` so Docker Swarm can advertise the GPU (replace the UUID with yours):

```json
{
  "runtimes": {
    "nvidia": {
      "path": "nvidia-container-runtime",
      "runtimeArgs": []
    }
  },
  "default-runtime": "nvidia",
  "node-generic-resources": [
    "NVIDIA_GPU=GPU-45cbf7b3-f919-7228-7a26-b06628ebefa1"
  ]
}
```

Uncomment the swarm-resource line in the NVIDIA Container Runtime config:

```bash
sudo sed -i 's/^#\s*\(swarm-resource\s*=\s*".*"\)/\1/' /etc/nvidia-container-runtime/config.toml
```

Restart Docker on every node:

```bash
sudo systemctl restart docker
```

### Step 4. Initialize Docker Swarm

On the primary node, advertise the interconnect IP you recorded in Step 1:

```bash
docker swarm init --advertise-addr "$MN_IP_ADDRESS"
```

Typical output:

```
Swarm initialized: current node (node-id) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token <worker-token> <advertise-addr>:<port>
```

### Step 5. Join worker nodes

On each worker, run the `docker swarm join` command printed by the primary node:

```bash
docker swarm join --token <worker-token> <advertise-addr>:<port>
```

### Step 6. Deploy the multi-node stack

Clone the playbook assets onto every node (or share them from a common working directory):

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-pytorch-fine-tune/assets
chmod +x pytorch-ft-entrypoint.sh
```

Before deploying, set the interconnect interface in `docker-compose.yml`. The shipped file defaults `UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME`, and `GLOO_SOCKET_IFNAME` to `enp1s0f1np1`. Replace those values with `"$MN_IF_NAME"` (or your interconnect name) if your interface differs.

On the primary node, deploy the stack from the assets directory:

```bash
docker stack deploy -c "$PWD/docker-compose.yml" finetuning-multinode
```

> [!NOTE]
> Keep `docker-compose.yml` and `pytorch-ft-entrypoint.sh` in the same directory you deploy from.

Verify the services:

```bash
docker stack ps finetuning-multinode
```

Healthy output looks like:

```
ID             NAME                                IMAGE                              NODE         DESIRED STATE   CURRENT STATE
vlun7z9cacf9   finetuning-multinode_finetunine.1   nvcr.io/nvidia/pytorch:25.11-py3   <node-a>     Running         Running
tjl49zicvxoi   finetuning-multinode_finetunine.2   nvcr.io/nvidia/pytorch:25.11-py3   <node-b>     Running         Running
```

If the current state is not Running, see **Troubleshooting**.

### Step 7. Capture the container ID

On every node:

```bash
export FINETUNING_CONTAINER=$(docker ps -q -f name=finetuning-multinode)
```

### Step 8. Adapt the Accelerate configuration files

Two configuration files are provided:

- [`config_finetuning.yaml`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/assets/configs/config_finetuning.yaml) — full fine-tuning of Llama 3 3B
- [`config_fsdp_lora.yaml`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/assets/configs/config_fsdp_lora.yaml) — LoRA with FSDP for Llama 3 8B and 70B

On each node, edit the matching YAML:

- Set `machine_rank` to `0` on the primary node and `1` on the worker
- Set `main_process_ip` to the primary node's interconnect IP (`$MN_IP_ADDRESS` from Step 1). Use the same value on both nodes
- Set `main_process_port` to an open port on the primary node

```yaml
machine_rank: 0
main_process_ip: <PRIMARY_INTERCONNECT_IP>
main_process_port: <PORT>
```

### Step 9. Run a multi-node fine-tuning script

Use one of the `run-multi-llama_*` helpers in [assets](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-pytorch-fine-tune/assets), or launch Accelerate directly. Example: Llama 3.1 70B LoRA with FSDP:

```bash
export HF_TOKEN=<your-huggingface-token>

docker exec \
  -e HF_TOKEN="$HF_TOKEN" \
  -it "$FINETUNING_CONTAINER" bash -c '
  bash /workspace/install-requirements;
  accelerate launch --config_file=/workspace/configs/config_fsdp_lora.yaml /workspace/Llama3_70B_LoRA_finetuning.py'
```

Training progress appears on the primary node's stdout only — Accelerate displays the progress bar on the main process. Confirm GPU activity on the worker with `nvidia-smi`.

### Step 10. Cleanup and rollback

On the primary node, remove the stack:

```bash
docker stack rm finetuning-multinode
```

Optionally free cached models and datasets:

```bash
rm -rf "$HOME/.cache/huggingface/hub/models--meta-llama"* \
  "$HOME/.cache/huggingface/hub/datasets"*
```

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
|---------|-------|-----|
| Cannot access gated repo for URL | Hugging Face model is gated or the token lacks access | Create or regenerate a [Hugging Face token](https://huggingface.co/docs/hub/en/security-tokens), request access on the model page, then rerun `hf auth login` |
| Docker returns `permission denied` | User is not in the Docker group | Run `sudo usermod -aG docker "$USER" && newgrp docker` |
| Container fails to access the GPU | NVIDIA Container Toolkit is not configured | Configure the toolkit for Docker, restart Docker, and verify with `docker run --rm --gpus all nvcr.io/nvidia/pytorch:25.11-py3 nvidia-smi` |
| Errors or timeouts in multi-node runs | Distributed runtime misconfiguration or network issues | Enable extra logging: `ACCELERATE_DEBUG_MODE=1`, `ACCELERATE_LOG_LEVEL=DEBUG`, `TORCH_CPP_LOG_LEVEL=INFO`, `TORCH_DISTRIBUTED_DEBUG=DETAIL` |
| `task: non-zero exit (255)` | Multi-node container exited with error code 255 | List containers with `docker ps -a --filter "name=finetuning-multinode"`, then inspect `docker logs <container_id>` |
| Cannot connect to the Docker daemon at `unix:///var/run/docker.sock` | Docker Swarm is bound to a stale or unreachable advertise address | Stop Docker (`sudo systemctl stop docker`), remove Swarm state (`sudo rm -rf /var/lib/docker/swarm`), start Docker (`sudo systemctl start docker`), then re-initialize Swarm with a valid interconnect IP |
| Multi-node service stuck not Running | Swarm stack or GPU advertising misconfigured | Confirm GPU UUID advertising, `swarm-resource` in `config.toml`, interconnect settings in `docker-compose.yml`, and `docker stack ps finetuning-multinode` |

> [!NOTE]
> On hardware platforms with unified memory, the operating-system page cache can contribute to memory pressure. If a workload fits rated capacity but still fails, stop other memory-intensive applications and flush the host buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

For latest known issues, see the documentation linked under **Resources**.
