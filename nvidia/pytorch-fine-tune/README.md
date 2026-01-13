# Fine-tune with Pytorch

> Use Pytorch to fine-tune models locally

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Run on two Sparks](#run-on-two-sparks)
  - [Step 1. Configure network connectivity](#step-1-configure-network-connectivity)
  - [Step 2. Configure Docker permissions](#step-2-configure-docker-permissions)
  - [Step 3. Install NVIDIA Container Toolkit & setup Docker environment](#step-3-install-nvidia-container-toolkit-setup-docker-environment)
  - [Step 4. Enable resource advertising](#step-4-enable-resource-advertising)
  - [Step 5. Initialize Docker Swarm](#step-5-initialize-docker-swarm)
  - [Step 6. Join worker nodes and deploy](#step-6-join-worker-nodes-and-deploy)
  - [Step 7. Find your Docker container ID](#step-7-find-your-docker-container-id)
  - [Step 9. Adapt the configuration files](#step-9-adapt-the-configuration-files)
  - [Step 10. Run finetuning scripts](#step-10-run-finetuning-scripts)
  - [Step 14. Cleanup and rollback](#step-14-cleanup-and-rollback)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook guides you through setting up and using Pytorch for fine-tuning large language models on NVIDIA Spark devices.

## What you'll accomplish

You'll establish a complete fine-tuning environment for large language models (1-70B parameters) on your NVIDIA Spark device. 
By the end, you'll have a working installation that supports parameter-efficient fine-tuning (PEFT) and supervised fine-tuning (SFT).

## What to know before starting

- Previous experience with fine-tuning in Pytorch
- Working with Docker



## Prerequisites
Recipes are specifically for DIGITS SPARK. Please make sure that OS and drivers are latest.


## Ancillary files

ALl files required for fine-tuning are included in the folder in [the GitHub repository here](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune).

## Time & risk

* **Time estimate:** 30-45 mins for setup and runing fine-tuning. Fine-tuning run time varies depending on model size 
* **Risks:** Model downloads can be large (several GB), ARM64 package compatibility issues may require troubleshooting.
* **Last Updated:** 12/16/2025
  * Bug fix to ensure torch.compile does not break with LoRA. 
  * Fix broken commands to access files from GitHub
  * Upgrade to latest pytorch container version nvcr.io/nvidia/pytorch:25.11-py3
  * Temporarily remove Llama3_70B_qLoRA_finetuning command to investigate slowness.

## Instructions

## Step 1. Configure Docker permissions

To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.

Open a new terminal and test Docker access. In the terminal, run:

```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with sudo .

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Step 2.  Pull the latest Pytorch container

```bash
docker pull nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 3. Launch Docker

```bash
docker run --gpus all -it --rm --ipc=host \
-v $HOME/.cache/huggingface:/root/.cache/huggingface \
-v ${PWD}:/workspace -w /workspace \
nvcr.io/nvidia/pytorch:25.11-py3
```

## Step 4. Install dependencies inside the container

```bash
pip install transformers peft datasets trl bitsandbytes
```

## Step 5: Authenticate with Huggingface

```bash
hf auth login
##<input your huggingface token.
##<Enter n for git credential>
```

## Step 6:  Clone the git repo with fine-tuning recipes

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/pytorch-fine-tune/assets
```

## Step7: Run the fine-tuning recipes

### Available Fine-Tuning Scripts

The following fine-tuning scripts are provided, each optimized for different model sizes and training approaches:

| Script | Model | Fine-Tuning Type | Description |
|--------|-------|------------------|-------------|
| `Llama3_3B_full_finetuning.py` | Llama 3.2 3B | Full SFT | Full supervised fine-tuning (all parameters trainable) |
| `Llama3_8B_LoRA_finetuning.py` | Llama 3.1 8B | LoRA | Low-Rank Adaptation (parameter-efficient) |
| `Llama3_70B_LoRA_finetuning.py` | Llama 3.1 70B | LoRA | Low-Rank Adaptation with FSDP support |
| `Llama3_70B_qLoRA_finetuning.py` | Llama 3.1 70B | QLoRA | Quantized LoRA (4-bit quantization for memory efficiency) |

### Basic Usage

Run any script with default settings:

```bash
# Full fine-tuning on Llama 3.2 3B
python Llama3_3B_full_finetuning.py

# LoRA fine-tuning on Llama 3.1 8B
python Llama3_8B_LoRA_finetuning.py

# LoRA fine-tuning on Llama 3.1 70B
python Llama3_70B_LoRA_finetuning.py
```

### Common Command-Line Arguments

All scripts support the following command-line arguments for customization:

#### Model Configuration
- `--model_name`: Model name or path (default: varies by script)
- `--dtype`: Model precision - `float32`, `float16`, or `bfloat16` (default: `bfloat16`)

#### Training Configuration
- `--batch_size`: Per-device training batch size (default: varies by script)
- `--seq_length`: Maximum sequence length (default: `2048`)
- `--num_epochs`: Number of training epochs (default: `1`)
- `--gradient_accumulation_steps`: Gradient accumulation steps (default: `1`)
- `--learning_rate`: Learning rate (default: varies by script)
- `--gradient_checkpointing`: Enable gradient checkpointing to save memory (flag)

#### LoRA Configuration (LoRA and QLoRA scripts only)
- `--lora_rank`: LoRA rank - higher values = more trainable parameters (default: `8`)

#### Dataset Configuration
- `--dataset_size`: Number of samples to use from the Alpaca dataset (default: `500`)

#### Logging Configuration
- `--logging_steps`: Log metrics every N steps (default: `1`)
- `--log_dir`: Directory for TensorBoard logs (default: `logs`)

#### Model Saving
- `--output_dir`: Directory to save the fine-tuned model (default: `None` - model not saved)

#### Performance Optimization
- `--use_torch_compile`: Enable `torch.compile()` for faster training (flag)

> [!WARNING]
> **Important:** The `--use_torch_compile` flag is **not compatible with QLoRA** (`Llama3_70B_qLoRA_finetuning.py`). 
> Only use this flag with full fine-tuning and standard LoRA scripts.

### Usage Examples
```bash
python Llama3_8B_LoRA_finetuning.py \
  --dataset_size 100 \
  --num_epochs 1 \
  --batch_size 2
```

## Run on two Sparks

### Step 1. Configure network connectivity

Follow the network setup instructions from the [Connect two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks) playbook to establish connectivity between your DGX Spark nodes.

This includes:
- Physical QSFP cable connection
- Network interface configuration (automatic or manual IP assignment)
- Passwordless SSH setup
- Network connectivity verification

### Step 2. Configure Docker permissions

To easily manage containers without sudo, you must be in the `docker` group. If you choose to skip this step, you will need to run Docker commands with sudo.

Open a new terminal and test Docker access. In the terminal, run:

```bash
docker ps
```

If you see a permission denied error (something like permission denied while trying to connect to the Docker daemon socket), add your user to the docker group so that you don't need to run the command with sudo .

```bash
sudo usermod -aG docker $USER
newgrp docker
```
### Step 3. Install NVIDIA Container Toolkit & setup Docker environment

Ensure the NVIDIA drivers and the NVIDIA Container Toolkit are installed on each node (both manager and workers) that will provide GPU resources. This package enables Docker containers to access the host's GPU hardware. Ensure you complete the [installation steps](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html), including the [Docker configuration](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html#configuring-docker) for NVIDIA Container Toolkit.

### Step 4. Enable resource advertising

First, find your GPU UUID by running:
```bash
nvidia-smi -a | grep UUID
```

Next, modify the Docker daemon configuration to advertise the GPU to Swarm. Edit **/etc/docker/daemon.json**:

```bash
sudo nano /etc/docker/daemon.json
```

Add or modify the file to include the nvidia runtime and GPU UUID (replace **GPU-45cbf7b3-f919-7228-7a26-b06628ebefa1** with your actual GPU UUID):

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

Modify the NVIDIA Container Runtime to advertise the GPUs to the Swarm by uncommenting the swarm-resource line in the **config.toml** file. You can do this either with your preferred text editor (e.g., vim, nano...) or with the following command:
```bash
sudo sed -i 's/^#\s*\(swarm-resource\s*=\s*".*"\)/\1/' /etc/nvidia-container-runtime/config.toml
```

Finally, restart the Docker daemon to apply all changes:
```bash
sudo systemctl restart docker
```

Repeat these steps on all nodes.

### Step 5. Initialize Docker Swarm

On whichever node you want to use as primary, run the following swarm initialization command
```bash
docker swarm init --advertise-addr $(ip -o -4 addr show enp1s0f0np0 | awk '{print $4}' | cut -d/ -f1) $(ip -o -4 addr show enp1s0f1np1 | awk '{print $4}' | cut -d/ -f1)
```

The typical output of the above would be similar to the following:
```
Swarm initialized: current node (node-id) is now a manager.

To add a worker to this swarm, run the following command:

    docker swarm join --token <worker-token> <advertise-addr>:<port>

To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
```

### Step 6. Join worker nodes and deploy

Now we can proceed with setting up the worker nodes of your cluster. Repeat these steps on all worker nodes.

Run the command suggested by the docker swarm init on each worker node to join the Docker swarm
```bash
docker swarm join --token <worker-token> <advertise-addr>:<port>
```

On both nodes, download the [**pytorch-ft-entrypoint.sh**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/pytorch-ft-entrypoint.sh) script into the directory containing your finetuning scripts and configuration files and run the following command to make it executable:

```bash
chmod +x $PWD/pytorch-ft-entrypoint.sh
```

On your primary node, deploy the Finetuning multi-node stack by downloading the [**docker-compose.yml**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/docker-compose.yml) file into the same directory as in the previous step and running the following command:
```bash
docker stack deploy -c $PWD/docker-compose.yml finetuning-multinode
```
> [!NOTE]
> Ensure you download both files into the same directory from which you are running the command.

You can verify the status of your worker nodes using the following
```bash
docker stack ps finetuning-multinode
```

If everything is healthy, you should see a similar output to the following:
```
nvidia@spark-1b3b:~$ docker stack ps finetuning-multinode
ID             NAME                                IMAGE                              NODE         DESIRED STATE   CURRENT STATE            ERROR     PORTS
vlun7z9cacf9   finetuning-multinode_finetunine.1   nvcr.io/nvidia/pytorch:25.10-py3   spark-1d84   Running         Starting 2 seconds ago             
tjl49zicvxoi   finetuning-multinode_finetunine.2   nvcr.io/nvidia/pytorch:25.10-py3   spark-1b3b   Running         Starting 2 seconds ago             

```

> [!NOTE]
> If your "Current state" is not "Running", see troubleshooting section for more information.

### Step 7. Find your Docker container ID

You can use `docker ps` to find your Docker container ID. You can save the container ID in a variable as shown below. Run this command on both nodes.
```bash
export FINETUNING_CONTAINER=$(docker ps -q -f name=finetuning-multinode)
```

### Step 9. Adapt the configuration files

For multi-node runs, we provide 2 configuration files:
- [**config_finetuning.yaml**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/configs/config_finetuning.yaml) used for full finetuning of Llama3 3B.
- [**config_fsdp_lora.yaml**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets/configs/config_fsdp_lora.yaml) used for finetuning with LoRa and FSDP of Llama3 8B and Llama3 70B.

These configuration files need to be adapted:
- Set `machine_rank` on each of your nodes according to its rank. Your master node should have a rank `0`. The second node has a rank `1`.
- Set `main_process_ip` using the IP address of your master node. Ensure that both configuration files have the same value. Use `ifconfig` on your main node to find the correct value for the CX-7 IP address on this node.
- Set a port number that can be used on your main node.

The fields that need to be filled in your YAML files:

```bash
machine_rank: 0
main_process_ip: < TODO: specify IP >
main_process_port: < TODO: specify port >
```

All the scripts and configuration files are available in this [**repository**](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/pytorch-fine-tune/assets).

### Step 10. Run finetuning scripts

Once you successfully run the previous steps, you can use one of the `run-multi-llama_*` scripts for finetuning. Here is an example for Llama3 70B using LoRa for finetuning and FSDP2.

```bash
## Need to specify huggingface token for model download.
export HF_TOKEN=<your-huggingface-token>

docker exec \
  -e HF_TOKEN=$HF_TOKEN \
  -it $FINETUNING_CONTAINER bash -c '
  bash /workspace/install-requirements;
  accelerate launch --config_file=/workspace/configs/config_fsdp_lora.yaml /workspace/Llama3_70B_LoRA_finetuning.py'
```

During the run, the progress bar of the finetuning will appear on your main node's stdout only. This is an expected behavior as `accelerate` uses a wrapper around `tqdm` to display the progress on the main process only as explained [**here**](https://github.com/huggingface/accelerate/blob/main/src/accelerate/utils/tqdm.py#L25). Using `nvidia-smi` on the worker node should show that the GPU is used.

### Step 14. Cleanup and rollback

Stop and remove containers by using the following command on the leader node:

```bash
docker stack rm finetuning-multinode
```

Remove downloaded models to free disk space:

```bash
rm -rf $HOME/.cache/huggingface/hub/models--meta-llama* $HOME/.cache/huggingface/hub/datasets*
```

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```