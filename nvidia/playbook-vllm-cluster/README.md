# Serve LLMs with vLLM Across DGX Sparks

> Guided and configurable model serving across DGX Spark clusters

## Table of Contents

- [Overview](#overview)
- [Beginner](#beginner)
- [Intermediate](#intermediate)
- [Advanced](#advanced)
  - [Step 1. Prepare the cluster and repository](#step-1-prepare-the-cluster-and-repository)
  - [Step 2. Prepare the image and both models](#step-2-prepare-the-image-and-both-models)
  - [Step 3. Launch with explicit vLLM settings](#step-3-launch-with-explicit-vllm-settings)
  - [Step 4. Check and stop the cluster](#step-4-check-and-stop-the-cluster)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

If you are trying to run a workload or a model that won't fit on a single DGX Spark, you can connect two or more Sparks into a high-speed cluster and then distribute the workload or model across the devices.

However, first setting up a cluster and then distributing the workload requires significant technical experience and a lot of trial and error. 
It's not for the average user.

This playbook shows you three "friendlier" paths to create a cluster and then set up containerized vLLM to run a model distributed across the nodes ([see here for a single node](https://build.nvidia.com/playbooks/vllm)). 


## What you'll accomplish

Pick a path below to set up the cluster, distribute the model and make it available through an endpoint at `localhost` on your laptop.

- **Beginner:** Use NVIDIA Sync to set up the cluster and serve a fixed model (two or four Sparks).
- **Intermediate:** Use NVIDIA Sync to set up the cluster, then choose a model recipe and use NVIDIA provided scripts to set `spark-vllm-docker` to serve it (two Sparks).
- **Advanced:** Follow the manual instructions and use NVIDIA provided scripts (two Sparks).  

## What to know before starting

- Using [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html) to connect to devices and launch an application. **Beginner · Intermediate** (optional for Advanced)
- Choosing a model recipe and understanding model size, precision, and hardware fit. **Intermediate · Advanced**
- Reading Docker and Hugging Face CLI commands. **Intermediate · Advanced**
- Reading shell scripts and vLLM launch options, including tensor parallelism. **Advanced**

## Supported hardware platforms

The guided path supports two to four DGX Sparks. The command-line examples currently cover two.

| Hardware platform | OS | Memory per device | Beginner | Intermediate | Advanced |
| --- | --- | --- | :---: | :---: | :---: |
| **Two DGX Sparks** | DGX OS (Linux) | 128 GB Unified Memory | ✅ | ✅ | ✅ |
| **Three or four DGX Sparks** | DGX OS (Linux) | 128 GB Unified Memory | ✅ | — | — |

The dashes mean this draft has no worked command-line example for those cluster sizes.

## Prerequisites

**Hardware requirements**

- Two to four DGX Sparks with the necessary QSFP cables (switch required for four Sparks). **All paths**
- Two DGX Sparks for the current command-line examples. **Intermediate · Advanced**

**Software and access requirements**

- [NVIDIA Sync installed](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding) on your laptop and each Spark added before using Cluster Assistant. **Beginner · Intermediate** (optional for Advanced)
- Docker installed and available to your user on each Spark. **All paths**
- A Hugging Face token if the model launcher asks for one. **Beginner, if prompted**
- An installed, authenticated Hugging Face CLI on each Spark. **Intermediate · Advanced**
- The `uv` package manager installed on the Spark you use for the head node. **Intermediate · Advanced**

## Find model recipes

The two-Spark worked example uses [`qwen3.8-27b-nvfp4-dflash2.yaml`](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-27b-nvfp4-dflash2.yaml) from `spark-vllm-docker` to serve `nvidia/Qwen3.8-27B-NVFP4`. The Beginner path uses the fixed model offered by NVIDIA Sync's model launcher.

## Time & risk

- **Estimated time:** Plan for 30 minutes or longer when downloading a model for the first time. Cluster setup and download time vary with the number of devices and chosen model.
- **Risk level:** Medium; the procedure changes the cluster network configuration and launches processes on multiple Sparks.
- **Stop or rollback:** For Beginner, stop the model in NVIDIA Sync. For Intermediate and Advanced, stop the deployment on both Sparks from the head node before changing or deleting the cluster. Remove the cluster in NVIDIA Sync if you want to undo its network setup.
- **Last Updated:** 09/28/2026
  - Split the cluster workflow from the single-device vLLM playbook.

## Beginner

## Step 1. Connect two to four DGX Sparks

Follow [Connect Multiple DGX Sparks](https://build.nvidia.com/playbooks/connect-multiple-sparks). Cable the devices and add each Spark to NVIDIA Sync through a direct SSH connection.

## Step 2. Configure the cluster with NVIDIA Sync

Use NVIDIA Sync Cluster Assistant to configure and check the ConnectX-7 network and interdevice SSH for the connected Sparks.

**Success:** Cluster Assistant reports that the devices and their connections pass its checks.

## Step 3. Launch a model with NVIDIA Sync

Use NVIDIA Sync's model launcher to start one of its fixed models on the cluster. The model launcher starts the workload after Cluster Assistant has configured the devices.

When finished, stop the model in NVIDIA Sync. You can leave the cluster configured for another session.

## Next steps

Use **Intermediate** if you want to choose a compatible recipe, or **Advanced** if you want to control the vLLM launch settings. Their current worked examples use two Sparks.

## Intermediate

## Step 1. Configure the cluster (one time)

**Follow the** [Connect Multiple Sparks playbook](https://build.nvidia.com/playbooks/connect-multiple-sparks).

It shows you how to connect the two Sparks with a cable or switch and uses the NVIDIA Sync Clustering Assistant ([see demo video](https://www.youtube.com/watch?v=MehBUQtb9qM)).

## Step 2. Check Docker access and the Hugging Face CLI (one time)

**Make sure the** `docker` **group is configured and the Hugging Face CLI is installed and authenticated on each device.**

Follow Step 2 in the [single-device vLLM instructions](https://build.nvidia.com/playbooks/vllm/intermediate) on both Sparks.

## Step 3. Select a cluster head node, install `uv`, and clone the script repository from GitHub (one time)

**First, choose one Spark from the cluster as the head node and install the Python package manager `uv`.**

1. Connect to the device via NVIDIA Sync
2. Open a terminal on the device 
3. Then install the `uv` package manager with `curl -LsSf https://astral.sh/uv/install.sh | sh`.

**Next, clone the community script repository** [spark-vllm-docker](https://github.com/eugr/spark-vllm-docker) **to the head node.**

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
```

## Step 4. Serve the model

Change into the repository and run the predefined recipe for **nvidia/Qwen3.8-27B-NVFP4** across both Sparks. The recipe's `--setup` option prepares its image and model assets before launch:

```bash
cd spark-vllm-docker
./run-recipe.sh recipes/qwen3.8-27b-nvfp4-dflash2.yaml --setup
```

**Success:** The setup and launch commands finish their preparation steps and vLLM starts on both Sparks. Keep the launch terminal open while you use the service.

> [!NOTE]
> See `./run-recipe.sh --help` for full usage. To customize the image, model, or serving command, see [Advanced](advanced.md).

## Step 5. Test inference

Run these checks on the head node. Laptop access needs a separate connection to the head Spark; see [Advanced](advanced.md) for the existing SSH tunnel guidance.

First wait for the API to become ready:

```bash
curl --fail --show-error http://localhost:8000/health
curl --fail --show-error http://localhost:8000/v1/models
```

```bash
curl -s http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3.8-27B-NVFP4",
    "messages": [
      {"role": "user", "content": "Write a haiku about a GPU"}
    ],
    "max_tokens": 64
  }'
```

## Next steps

Consider for production:

- Health checks and automatic restarts
- Log rotation for long-running services
- Persistent model caching across restarts

## Advanced

## Two DGX Sparks: customize the cluster launch

### Step 1. Prepare the cluster and repository

First, cable and configure the two devices using either path in [Connect Multiple Sparks](https://build.nvidia.com/playbooks/connect-multiple-sparks). Confirm that interdevice SSH works. Choose one Spark as the head node and use a terminal on it for the remaining commands.

Check Docker access and Hugging Face authentication on both devices as described in [single-device vLLM instructions](https://build.nvidia.com/playbooks/vllm/intermediate#step-2-open-a-remote-terminal-to-verify-the-docker-group-configuration-and-the-hugging-face-cli-one-time).

Clone the community launcher repository on the head node and enter it:

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Install `uv` if it is not already available:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Before preparing artifacts, run `./run-recipe.sh --discover` and confirm that it finds the intended two Sparks and network interfaces. It saves the cluster configuration for the following commands.

### Step 2. Prepare the image and both models

The commands below prepare the `vllm-node` image and copy the same image and both model repositories to the other Spark. They match the [Qwen3.8-27B NVFP4 with DFlash2 recipe](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-27b-nvfp4-dflash2.yaml).

```bash
./build-and-copy.sh -c --copy-parallel
./hf-download.sh nvidia/Qwen3.8-27B-NVFP4 -c --copy-parallel
./hf-download.sh z-lab/Qwen3.8-27B-DFlash2 -c --copy-parallel
```

### Step 3. Launch with explicit vLLM settings

Run this command on the head node. The launcher uses the saved cluster configuration to start vLLM across both Sparks. These settings follow the selected recipe; if you change the model or image, review the full command for compatibility.

```bash
./launch-cluster.sh -t vllm-node exec \
  vllm serve nvidia/Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":8,"draft_tensor_parallel_size":2}' \
    --load-format instanttensor \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --tensor-parallel-size 2
```

### Step 4. Check and stop the cluster

On the head node, wait for the API to become ready and confirm the served model:

```bash
curl --fail --show-error http://127.0.0.1:8000/health
curl --fail --show-error http://127.0.0.1:8000/v1/models
```

Then send a chat request:

```bash
curl --fail --show-error http://127.0.0.1:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":64}'
```

For laptop access, open an SSH tunnel from your computer to the head Spark's management address, as shown in the [single-device Advanced tab](https://build.nvidia.com/playbooks/vllm/advanced). To stop the deployment on both Sparks, run `./launch-cluster.sh stop` from the repository directory on the head node.

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| Cluster Assistant cannot finish its checks | One or both Sparks or their ConnectX-7 link needs attention | Return to [Connect Multiple DGX Sparks](https://build.nvidia.com/playbooks/connect-multiple-sparks) and complete its checks before running the recipe. |
| `docker` reports permission denied on either Spark | The device user cannot access Docker | Follow the Docker access check in the [single-device instructions](https://build.nvidia.com/playbooks/vllm/intermediate). |
| Model download reports an authentication error | Hugging Face CLI access is missing or invalid on a Spark | Check `hf auth whoami` on both devices and authenticate as described in the [Hugging Face guide](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication). |
| The API does not respond on the head Spark | The recipe may still be loading or launch may have failed | Check the output in the launch terminal before retrying the `/health` request. |

The cluster recipe and manual launcher have separate setup steps. Use the commands from one path consistently; see **Intermediate** for the recipe or **Advanced** for explicit launch settings.
