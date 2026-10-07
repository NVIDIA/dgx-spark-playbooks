# Serve LLMs with vLLM

> High-throughput serving for 30+ models, with continuous batching and an OpenAI-compatible API

## Table of Contents

- [Overview](#overview)
- [Choose a Recipe](#choose-a-recipe)
- [Single device](#single-device)
  - [Step 1. Install NVIDIA Sync locally and add the DGX Spark or DGX Station device (one time)](#step-1-install-nvidia-sync-locally-and-add-the-dgx-spark-or-dgx-station-device-one-time)
- [Multi-node DGX Spark](#multi-node-dgx-spark)
  - [Step 1. Install NVIDIA Sync on your laptop and add the two DGX Spark devices](#step-1-install-nvidia-sync-on-your-laptop-and-add-the-two-dgx-spark-devices)
  - [Step 2. Physically connect the two Sparks and configure the ConnectX-7 network using NVIDIA Sync](#step-2-physically-connect-the-two-sparks-and-configure-the-connectx-7-network-using-nvidia-sync)
  - [Step 2. Check the Docker group on each device and configure if needed](#step-2-check-the-docker-group-on-each-device-and-configure-if-needed)
  - [Step 2. Prepare the environment](#step-2-prepare-the-environment)
  - [Step 3. Serve the model](#step-3-serve-the-model)
  - [Step 4. Test inference](#step-4-test-inference)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

vLLM is a highly performant inference engine for serving large language models through an OpenAI-compatible API.
It uses efficient memory management and continuous batching to increase serving throughput.
You can learn more about vLLM [in their blog](https://vllm.ai/blog).

This playbook walks you through two situations for configuring a vLLM container to serve a model.

- **Single remote device**: How to configure and access a model in a vLLM container on a DGX Spark or DGX Station
- **DGX Spark cluster**: How to serve a model across vLLM containers running on two clustered DGX Sparks

The playbook focuses on the recommended paths using NVIDIA Sync.
Advanced users can take the manual path to get under the hood for details.

## What you'll accomplish

Choose a model and vLLM recipe for your hardware, launch the model, and send a test request to its OpenAI-compatible API.

**Recommended path:** Use [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html)

- One DGX Spark or Station: Use NVIDIA Sync to start/stop the remote container and handle port forwarding for the API
  - One time: Download the recommended container and model
  - One time: Save the launch script as an NVIDIA Sync custom application
  - Repeat use: Start/stop the vLLM container from NVIDIA Sync

- Two DGX Sparks: Use NVIDIA Sync to set up the cluster, then run a recipe from the head Spark to serve the model across both devices
  - One time: Physically connect the Sparks and use the Cluster Assistant to configure and test the high-speed network and interdevice SSH
  - One time: Prepare the recommended container and model on both Sparks
  - Repeat use: Run the multi-node recipe from the head Spark to start vLLM

**Advanced path:** Run the setup and recipe commands directly on the devices when you need to customize the deployment.

## What to know before starting

**Required**:

- Familiarity with [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html) and adding devices ([see steps here](https://docs.nvidia.com/sync/latest/direct-connections.html))
- How to run simple terminal commands in Linux
- Basic familiarity with [Hugging Face](https://huggingface.co/) and the [CLI](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended)
- If using a DGX Spark cluster:
  - How to physically connect the devices ([see here](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#the-qsfp-ports-and-cables))
  - How to use NVIDIA Sync Cluster Assistant (see [here](https://build.nvidia.com/playbooks/connect-multiple-sparks) and the [demo video](https://www.youtube.com/watch?v=MehBUQtb9qM))
  - How to run shell scripts

**Suggested**:

- How to use the NVIDIA Sync Custom App feature ([see here](https://docs.nvidia.com/sync/latest/applications.html#adding-and-editing-a-custom-script-to-a-remote-device))
- How to download and run containers
- How to edit and run shell scripts


## Supported hardware platforms

Check the table below to confirm which path this playbook supports for your hardware.

| Hardware platform | OS | Memory | One device | Clustered Sparks |
| :---- | :---- | :---- | :----: | :----: |
| **DGX Spark** | DGX OS (Linux) | 128 GB Unified Memory | ✅ | ✅ Two devices |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | ✅ | — |

> [!NOTE]
> Cluster Assistant can configure two, three, or four DGX Sparks, but model sharding depends on the model and cluster configuration. For simplicity, this playbook limits cluster serving to two Sparks.

## Prerequisites

**Hardware requirements**

- Single device: a DGX Spark or DGX Station device — see Supported hardware platforms matrix above
- DGX Spark cluster: Two DGX Sparks and a single QSFP cable ([see here for cabling two devices](https://docs.nvidia.com/dgx/dgx-spark/spark-clustering.html#the-qsfp-ports-and-cables))

**Software requirements**

- NVIDIA Sync installed on your laptop ([see installation instructions here](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding))
- Each remote device added to NVIDIA Sync ([see how to do this here](https://docs.nvidia.com/sync/latest/direct-connections.html#adding-a-device-for-a-direct-connection))
- Each remote device has Docker installed and the user in the Docker group ([see here for how](https://docs.docker.com/engine/install/linux-postinstall/#add-your-user-to-the-docker-group))
- The Hugging Face CLI installed and authenticated on each device
  - [See here](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended) on Hugging Face for CLI installation
  - [See here](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication) on Hugging Face for token authentication

## Time & risk

- **Estimated time:** 30 minutes for one device; longer for a cluster or the first model download
- **Risk level:** Low for one device; medium when using a cluster
- **Rollback:** For one device, stop the vLLM custom application in NVIDIA Sync or stop the container you launched manually. For a cluster, stop vLLM on both devices before deleting or changing the cluster.
- **Last Updated:** 09/14/2026
  - Reorganized the playbook around the recommended NVIDIA Sync paths for one device and DGX Spark clusters.

## Choose a Recipe

## Step 1. Identify your configuration

Choose the configuration that matches the hardware you will use:

| Configuration | How vLLM will run |
| --- | --- |
| **One DGX Spark** | On the GB10 GPU in one device |
| **One DGX Station** | On the GB300 GPU in one device |
| **Two DGX Sparks** | Across one GB10 GPU in each device |

If you are unsure which NVIDIA GPU is available, connect to the device with NVIDIA Sync, open **Terminal**, and run:

```bash
nvidia-smi --query-gpu=name --format=csv,noheader
```

## Step 2. Use the recommended recipe

Select the row for your configuration. These recipes were chosen to provide reasoning and tool calling while making effective use of the available hardware.

| Configuration | Recommended model | Why this recipe | Continue with |
| --- | --- | --- | --- |
| **One DGX Spark** | [Qwen3.8-27B NVFP4](https://recipes.vllm.ai/Qwen/Qwen3.8-27B?hardware=dgx_spark_gb10) | The quantized model fits one Spark and has a hardware-specific vLLM configuration. | **Single device** |
| **One DGX Station** | [Qwen3.8-Flash-Next NVFP4](https://recipes.vllm.ai/Qwen/Qwen3.8-Flash-Next?hardware=dgx_station_gb300) | The larger mixture-of-experts model takes advantage of the Station's greater memory and uses a dedicated container. | **Single device** |
| **Two DGX Sparks** | [Qwen3.8-27B NVFP4](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-27b-nvfp4-dflash2.yaml) | The predefined cluster recipe uses tensor parallelism across both Sparks and adds DFlash2 speculative decoding. | **Multi-node DGX Spark** |

The launch tabs provide the complete model, container, and serving configuration for these recommended recipes. You do not need to translate the recipe commands yourself.

## Step 3. Decide whether to use another recipe

For the simplest path, keep the recommended recipe and continue to the tab shown in the table.

If you are comfortable selecting a different model, container, and launch configuration, continue to Step 4.

> [!IMPORTANT]
> The copy-and-paste launch configurations in this playbook are tested for the recommended recipes. Another recipe may require different model-download, container, environment, memory, parser, or parallelism settings.

## Step 4. Generate another recipe

To use another recipe:

1. Open the filtered recipe catalog for your hardware:
   - [DGX Spark recipes](https://recipes.vllm.ai/browse?panel=open&hw=dgx_spark_gb10)
   - [DGX Station recipes](https://recipes.vllm.ai/browse?panel=open&hw=dgx_station_gb300)
2. Select a model that supports your hardware configuration.
3. Select the model variant and precision.
4. Enable the capabilities you need, such as tool calling or reasoning.
5. Copy the model ID, container image, environment variables, and complete `vllm serve` command from the generated recipe.
6. Keep all values from the same generated recipe. Do not combine settings from different model variants or hardware configurations.
7. For two-Spark serving, confirm that a matching recipe exists in [`spark-vllm-docker`](https://github.com/eugr/spark-vllm-docker/tree/main/recipes). Do not assume a single-device recipe can run across two devices.

> [!NOTE]
> Model size is not the only compatibility requirement. The container architecture, vLLM version, quantization format, parsers, and parallel configuration must also match the selected model and hardware.

## Step 5. Continue to the launch instructions

- For a recommended one-Spark or one-Station recipe, continue with **Single device**.
- For the recommended two-Spark recipe, continue with **Multi-node DGX Spark**.
- For another recipe, use its generated installation and serving commands as the manual path. Substitute its settings only where the following launch instructions explicitly tell you to do so.

## Single device

## Install with an agent

```text
Use https://build.nvidia.com/playbooks/vllm.md and complete this playbook on this machine: Serve LLMs with vLLM. Fetch that .md URL directly; the non-.md page is a JavaScript shell. When finished, state clearly whether the playbook succeeded, and give the evidence: what you installed, where, and the output of a command proving it works (a version string, a container listing, or an HTTP status).
```

> [!NOTE]
> This has been tested using OpenCode and Qwen 3.8 35B A3B NVFP4. Performance and outcome may vary based on agent and model selection.

### Step 1. Install NVIDIA Sync locally and add the DGX Spark or DGX Station device (one time)

Follow the [Connect to Your Spark](https://build.nvidia.com/spark/connect-to-your-spark) playbook.

## Step 2. Open a remote terminal to check Docker and the Hugging Face CLI (one time)

**First, use NVIDIA Sync to open a terminal on the remote device.**

1. On your computer, open NVIDIA Sync and select the device.
3. Then select **Connect**.
4. After the device connects, open **Terminal**.

**Next, check that your user is in the Docker group.**

```bash
docker ps > /dev/null
```

If it returns a blank line, then the group is already configured.
If it reports a permission error, add your user to the `docker` group as follows:

1. Add your user: `sudo usermod -aG docker "$USER"`
2. Activate the group: `newgrp docker`
3. Check status again: `docker ps > /dev/null`

**Success**: It returns a blank line.

**Finally, check that the Hugging Face CLI is installed and authenticated.**

```bash
hf auth whoami
```

If this returns `command not found` or `Not logged in`, then install ([see here](https://huggingface.co/docs/huggingface_hub/main/en/guides/cli#standalone-installer-recommended)) and/or authenticate the CLI ([see here](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication)).

> [!NOTE]
> If you install the CLI, you need to refresh the terminal with the command `source "$HOME/.bashrc"`.

If `hf` is still not found after refreshing the terminal, add the local user binary directory to your current `PATH`:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

**Success**: The command `hf auth whoami` returns your username and org.

## Step 3. Download the vLLM container and model for the recipe (one time)

**First, pull the appropriate container and follow progress in the terminal.**

| Device | Docker Command to Run |
| --- | --- |
| DGX Spark | `docker pull vllm/vllm-openai:qwen38` |
| DGX Station | `docker pull vllm/vllm-openai:qwen38-flash-next` |

**Success**: The Docker CLI reports that the download has succeeded.

**Then, download the appropriate model and follow progress in the terminal**.

| Device | HF CLI Command to Run |
| --- | --- |
| Single DGX Spark | `hf download nvidia/Qwen3.8-27B-NVFP4 --cache-dir "$HOME/.cache/huggingface/hub"` |
| Single DGX Station | `hf download Inferact/Qwen3.8-Flash-Next-NVFP4 --cache-dir "$HOME/.cache/huggingface/hub"` |

**Success**: The command output will stop and print a path under `$HOME/.cache/huggingface/hub`.

## Step 4. Add the vLLM launch script as a custom app (one time)

**First, open the launch script for your device.**

| Device | Launch Script to Copy |
| --- | --- |
| Single DGX Spark | [open the DGX Spark Qwen3.8 27B script](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-single-spark.sh) |
| Single DGX Station | [open the DGX Station Qwen3.8 Flash script](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm/assets/sync-vllm-single-station.sh) |

**Next, create the custom app in NVIDIA Sync.**

1. Select **Custom > Add New** to open the form.
2. Name the app, e.g "vLLM-Qwen3.827B" for DGX Spark or "vllm-Qwen3.8Flash" for DGX Station.
3. Enter **8000** for the port. If it is already in use, follow [Use another API port](troubleshooting.md#use-another-api-port) before copying the script.
4. Leave **Auto open in browser** turned off because vLLM serves an API, not a web app.
5. Copy the appropriate script into the **Launch Script** field.
6. Select **Add**.

**Success**: The application name shows up in the **Custom** section.

## Step 5. Launch vLLM and watch the model load (repeat use)

1. In NVIDIA Sync, select the custom app you added in Step 4.
2. Then open **Resource Monitor** to watch GPU activity and memory use as the model loads.
3. Follow the container logs in an NVIDIA Sync terminal with the command for your device:

| Device | Docker Command to See Logs |
| --- | --- |
| DGX Spark | `docker logs --tail 50 --follow vllm-qwen38` |
| DGX Station | `docker logs --tail 50 --follow vllm-qwen38-flash` |

**Success**: The logs show `OpenAI server is ready to accept requests` or `Application startup complete`. You can close the log terminal without stopping vLLM.

> [!IMPORTANT]
> GPU activity in the Resource Monitor shows that vLLM is working but does not mean the API is ready. Wait for a readiness message before you test the endpoint.

## Step 6. Once vLLM is ready, test the API from your laptop (one time)

> [!IMPORTANT]
> Run these commands on the laptop where NVIDIA Sync is running, not in the remote terminal. On Windows, use PowerShell, not a WSL terminal.
> Use the custom app port in every URL. The examples use 8000; if you assigned 8001, change every `localhost:8000` to `localhost:8001`.

**First, check the endpoint health and list the models.**

| Laptop terminal | Health check command | List models command |
| --- | --- | --- |
| Windows (PowerShell) | `curl.exe -i http://localhost:8000/health` | `curl.exe -sS http://localhost:8000/v1/models` |
| macOS or Linux (terminal) | `curl -i http://localhost:8000/health` | `curl -sS http://localhost:8000/v1/models` |

**Success**: The health check returns HTTP `200`, and the model list shows the model for your device.

**Next, send a chat request using the command for your laptop terminal and DGX device.**

| Laptop terminal | DGX Spark | DGX Station |
| --- | --- | --- |
| Windows (PowerShell) | `Invoke-RestMethod -Uri http://localhost:8000/v1/chat/completions -Method Post -ContentType application/json -Body '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}' \| ConvertTo-Json -Depth 8` | `Invoke-RestMethod -Uri http://localhost:8000/v1/chat/completions -Method Post -ContentType application/json -Body '{"model":"Inferact/Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}' \| ConvertTo-Json -Depth 8` |
| macOS or Linux (terminal) | `curl -sS http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"nvidia/Qwen3.8-27B-NVFP4","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}'` | `curl -sS http://localhost:8000/v1/chat/completions -H "Content-Type: application/json" -d '{"model":"Inferact/Qwen3.8-Flash-Next-NVFP4","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}'` |

**Success**: The response should contain a `choices` array and the model's answer.

## Next steps

- To serve the model across two DGX Sparks, go to [Multi-node DGX Spark](multi-node-spark.md).
- To choose another model, return to [Choose a Recipe](choose-recipe.md).
- If vLLM does not start or respond, see [Troubleshooting](troubleshooting.md).

## Multi-node DGX Spark

### Step 1. Install NVIDIA Sync on your laptop and add the two DGX Spark devices

Follow the [Connect to Your Spark](https://build.nvidia.com/spark/connect-to-your-spark) playbook.
Estimate time is 5 minutes. 

### Step 2. Physically connect the two Sparks and configure the ConnectX-7 network using NVIDIA Sync

Follow the [Connect Multiple Sparks](https://build.nvidia.com/playbooks/connect-multiple-sparks) playbook.

It shows you how to connect the devices with a cable or switch and then walks you the NVIDIA Sync Clustering Assistant ([see demo video](https://www.youtube.com/watch?v=MehBUQtb9qM)). 

### Step 2. Check the Docker group on each device and configure if needed

We will be using a vLLM container on both devices.
This simplifies things in a variety of ways, a major way being the elimination of installing and configuring NCCL on the two devices.

Docker commands 


If you have not used Cluster Assistant, follow [Configure Manually](https://build.nvidia.com/playbooks/connect-multiple-sparks/manual) in the Connect Multiple Sparks playbook to set up the two-node cluster: physical cabling, network configuration, passwordless SSH, and connectivity verification.

> **Manual setup only:** the connectivity script writes its SSH key to `~/.ssh/` and fails if the directory does not exist. Run `mkdir -p ~/.ssh && chmod 700 ~/.ssh` on both nodes first if you have never used SSH on them.

### Step 2. Prepare the environment

On the **`head node`** (first node in your cluster), clone the DGX Spark community container repo [**spark-vllm-docker**](https://github.com/eugr/spark-vllm-docker)

```bash
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

Install `uv` if not present

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Download the model you want to serve on all nodes in the cluster eg. **nvidia/Qwen3.8-27B-NVFP4**

```bash
./hf-download.sh nvidia/Qwen3.8-27B-NVFP4 -c --copy-parallel
```

### Step 3. Serve the model

To serve the model across the cluster, you can use two options

**Option 1**: Serve using a pre-defined recipe

If a pre-defined recipe exists in the repo then you can run it directly like below. This runs the **nvidia/Qwen3.8-27B-NVFP4** model across a two node cluster.

```bash
./run-recipe.sh recipes/qwen3.8-27b-nvfp4-dflash2.yaml --setup
```

> !NOTE
> See ./run-recipe.sh --help for full usage

**Option 2**: Serve with manual command

If a pre-defined recipe does not exist or if you want to run with your own custom arguments, you can run it like below. 

```bash
./launch-cluster.sh -t vllm/vllm-openai:v0.28.0 \
  --earlyoom \
  -e VLLM_USE_V2_MODEL_RUNNER="1" \
  -e VLLM_FLOAT32_MATMUL_PRECISION=high \
  -e PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  exec \
  vllm serve nvidia/Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --speculative-config '{"method":"dflash","model": "z-lab/Qwen3.8-27B-DFlash2", "num_speculative_tokens":8, "draft_tensor_parallel_size": 2}' \
    --load-format safetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice \
    --tensor-parallel-size 2
```

> [!NOTE]
> You can specify a different VLLM container image using the -t flag. Eg. eugr/spark-vllm-b12x:latest, vllm/vllm-openai:latest etc.
> --earlyoom helps detect OOM early and kills the VLLM process to avoid system hang due to OOM
> See ./launch-cluster.sh --help for full usage.

### Step 4. Test inference

Run on **`head node`**. If you want to run from an external client, replace `localhost` with head node's reachable IP.

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

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| "permission denied" when running docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Container fails to start with GPU error | All hardware platforms | NVIDIA Container Toolkit not configured | Run `nvidia-ctk runtime configure --runtime=docker` and restart Docker |
| HuggingFace authentication failure, gated model access denied, or model download hangs/fails | All hardware platforms | Missing/invalid token, restricted model access, or network issue | Export `HF_TOKEN` before running docker; regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens) and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated) if needed; check internet connection and verify the token is valid |
| CUDA out of memory | All hardware platforms | Context length too large / model too big | Reduce `--max-model-len` and `--max-num-seqs`, or lower `--gpu-memory-utilization` |
| Port 8000 is in use | All hardware platforms | Another app is using the port | Follow **Use another API port** below |
| Laptop cannot reach the API, but `/health` returns HTTP `200` on the DGX device | All hardware platforms | NVIDIA Sync is not forwarding the expected port, or the test is running in WSL on Windows | Keep the Sync custom app running and match its port to Docker's left-hand port. Use that port in the laptop URL; on Windows, test in PowerShell, not WSL |
| PowerShell prompts for `Uri` after `curl -i` | All hardware platforms | `curl` is a PowerShell alias for `Invoke-WebRequest` | Use `curl.exe` for the Step 6 health and model checks |
| Chat request returns `The model 'unknown' does not exist` | All hardware platforms | The request omits `model` | Use the Step 6 command for your DGX device; its model ID must match `/v1/models` |
| NGC authentication fails | All hardware platforms | Invalid or missing credentials | Run `docker login nvcr.io` with your NGC API key |
| `rm: cannot remove '.../.cache/huggingface/hub/models--...': Permission denied` | All hardware platforms | The container downloads weights as root into the mounted hub cache, so cached model files are root-owned | Remove with `sudo rm -rf $HOME/.cache/huggingface/hub/"<downloaded model name>"` |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |
| Container startup fails / missing ARM64 image | DGX Spark | Image not built for ARM64 | Use the default NGC image for your hardware platform from the Instructions tab |
| Model runs on wrong GPU | DGX Station | Default GPU selection with two GPUs | Use `--gpus '"device=N"'` to pin the GB300 (`N` from `nvidia-smi`) |
| EngineCore failed / FlashInfer "Buffer overflow when allocating memory for batch_prefill_tmp_v" | DGX Station | CUDA graph capture failure during batch prefill | Use the recommended container image: `nvcr.io/nvidia/vllm:26.01-py3` |
| Chat completion returns `content: null` with `finish_reason: length` | All hardware platforms | `max_tokens` was exhausted during reasoning | Raise `max_tokens` in the request (Step 6 uses `4096`) so the model can finish with a visible answer |

## Use another API port

If port 8000 is in use, you can use 8001 without changing the port inside the container:

1. Set the NVIDIA Sync custom app port to **8001**.
2. In its launch script, change the Docker mapping to `-p 8001:8000`. Leave `vllm serve --port 8000` unchanged.
3. Restart the custom app. In every **Step 6** URL, use `http://localhost:8001` instead of `http://localhost:8000`.

Docker's left-hand port is on the remote device; the right-hand port is inside the container. NVIDIA Sync forwards your laptop's port 8001 to port 8001 on the remote device.

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!NOTE]
> **Monitoring GPU memory with UMA.** Because of unified memory, `nvidia-smi --query-gpu` memory fields report `N/A`. Use plain `nvidia-smi` instead.
