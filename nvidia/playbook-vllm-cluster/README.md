# Serve LLMs with vLLM Across DGX Sparks

> Guided and configurable model serving across DGX Spark clusters

## Table of Contents

- [Overview](#overview)
- [Intermediate](#intermediate)
  - [Step 1. Connect and set up the two Sparks](#step-1-connect-and-set-up-the-two-sparks)
  - [Step 2. Set up the launcher on the head Spark](#step-2-set-up-the-launcher-on-the-head-spark)
  - [Step 3. Choose a model and save its recipe](#step-3-choose-a-model-and-save-its-recipe)
  - [Step 4. Download the image and launch the model](#step-4-download-the-image-and-launch-the-model)
  - [Step 5. Check the API on the head Spark](#step-5-check-the-api-on-the-head-spark)
  - [Step 6. Reach the model from your laptop](#step-6-reach-the-model-from-your-laptop)
  - [Step 7. Stop and restart the model](#step-7-stop-and-restart-the-model)
- [Advanced](#advanced)
  - [Step 1. Choose a model and prepare the cluster](#step-1-choose-a-model-and-prepare-the-cluster)
  - [Step 2. Read the distributed and memory settings](#step-2-read-the-distributed-and-memory-settings)
  - [Step 3. Start a container on each Spark](#step-3-start-a-container-on-each-spark)
  - [Step 4. Run the model on the head and worker](#step-4-run-the-model-on-the-head-and-worker)
  - [Step 5. Test the API and tool calling](#step-5-test-the-api-and-tool-calling)
  - [Step 6. Connect a client and tune the workload](#step-6-connect-a-client-and-tune-the-workload)
  - [Step 7. Stop or save the service](#step-7-stop-or-save-the-service)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

If you want to run a workload or a model that won't fit on a single DGX Spark, you can connect two or more Sparks into a cluster and distribute the workload or model across the devices.

However, this takes a bit of set up and experience. 
Even if you have done it before, it can still involve a lot of trial and error. 

This playbook shows you two paths to set up containerized vLLM to run a model distributed across the nodes ([see here for a single node](https://build.nvidia.com/playbooks/vllm)). 


## What you'll accomplish

Pick a path below to set up the cluster, distribute the model and make it available through an endpoint at `localhost` on your laptop.

- **Intermediate:** Use NVIDIA Sync to set up the cluster, then choose a model recipe and use NVIDIA provided scripts to set `spark-vllm-docker` to serve it (two Sparks).
- **Advanced:** Follow the manual instructions and use NVIDIA provided scripts (two Sparks).  

## What to know before starting

- Using [NVIDIA Sync](https://docs.nvidia.com/sync/latest/index.html) to connect to devices and launch an application. **Intermediate** 
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

- [NVIDIA Sync installed](https://docs.nvidia.com/sync/latest/getting-started.html#installation-and-onboarding) on your laptop and each Spark added before using Cluster Assistant. **Intermediate**
- Docker installed and available to your user on each Spark. **All paths**
- An installed, authenticated Hugging Face CLI on each Spark. **Intermediate · Advanced**
- The `uv` package manager installed on the Spark you use for the head node. **Intermediate · Advanced**

## Find model recipes

The two-Spark worked example uses [`qwen3.8-27b-nvfp4-dflash2.yaml`](https://github.com/eugr/spark-vllm-docker/blob/main/recipes/qwen3.8-27b-nvfp4-dflash2.yaml) from `spark-vllm-docker` to serve `nvidia/Qwen3.8-27B-NVFP4`. 

## Time & risk

- **Estimated time:** Plan for 30 minutes or longer when downloading a model for the first time. Cluster setup and download time vary with the number of devices and chosen model.
- **Risk level:** Medium; the procedure changes the cluster network configuration and launches processes on multiple Sparks.
- **Stop or rollback:** For Beginner, stop the model in NVIDIA Sync. For Intermediate and Advanced, stop the deployment on both Sparks from the head node before changing or deleting the cluster. Remove the cluster in NVIDIA Sync if you want to undo its network setup.
- **Last Updated:** 09/28/2026
  - Split the cluster workflow from the single-device vLLM playbook.

## Intermediate

## Run a vLLM model across two DGX Sparks

You will use NVIDIA Sync to set up a two-Spark cluster, then use a supplied recipe to run one model across both devices.

The recipe launcher starts a container on each Spark and runs vLLM across the pair.
You will send requests to the head Spark and use an SSH tunnel to reach its API from your laptop.

### Step 1. Connect and set up the two Sparks

**First, connect two Sparks with the same memory size.**

Use two 64GB Sparks or two 128GB Sparks.
Each recipe below is set for one of those pairs.

Follow the [Connect Multiple Sparks playbook](https://build.nvidia.com/playbooks/connect-multiple-sparks) to cable the devices and set up the cluster with NVIDIA Sync.
Add both Sparks to Sync and allow it to set up the Docker group if prompted.

**Then, save the cluster's network details.**

Keep the head and worker addresses, SSH aliases, and ConnectX interface names from the cluster setup.
The launcher will use the ConnectX link to pass model data between the two Sparks.

**Success:** Both Sparks are in the cluster, and you can open a terminal on each through Sync.

### Step 2. Set up the launcher on the head Spark

**First, choose one Spark as the head and open its terminal through Sync.**

The head runs the API and starts the model on both devices.
Use this Spark for the rest of the commands unless a step says otherwise.

**Then, install `uv` and get the launcher.**

Run:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
git clone https://github.com/eugr/spark-vllm-docker.git
cd spark-vllm-docker
```

These are [community scripts](https://github.com/eugr/spark-vllm-docker) maintained outside NVIDIA.
The commands below use `uv` to run the recipe tool with its Python dependency.

**Next, find the cluster and check the saved settings.**

```bash
uv run --with pyyaml python run-recipe.py --discover
uv run --with pyyaml python run-recipe.py --show-env
```

Check that the saved `.env` file names exactly two Sparks, with this Spark first as the head.
Check the addresses and ConnectX interfaces against the details from Step 1.
If the launcher cannot reach the worker through SSH, fix that connection before you go on.

**Success:** The launcher has saved the intended head, worker, and network settings in `.env`.

### Step 3. Choose a model and save its recipe

**First, choose the row for your pair of Sparks.**

Memory in the table is **per Spark**.
These models use both devices to serve one endpoint.

| Cluster | Model | Recipe file |
| --- | --- | --- |
| 2 × DGX Spark, 64GB each | [Qwen3.8 Flash-Next](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark-2gpu/index.html?model=qwen%2Fqwen3.8-flash-next&os=linux&backend=vllm&memory=64&precision=fp4-local&memoryConfig=64) | [qwen38-flash-next-2x64gb.yaml](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm-cluster/assets/qwen38-flash-next-2x64gb.yaml) |
| 2 × DGX Spark, 128GB each | [GLM 5.3 Flash](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark-2gpu/index.html?model=z-ai%2Fglm-5.3-flash&os=linux&backend=vllm&memory=128&precision=fp4-local&memoryConfig=128) | [glm53-flash-2x128gb.yaml](https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/playbook-vllm-cluster/assets/glm53-flash-2x128gb.yaml) |

**Then, save the recipe on the head Spark.**

1. Open the recipe file from your row
2. Copy its full contents into a file in the launcher's `recipes` folder on the head Spark
3. Use the file name shown in the table

For example, open this file in a text editor on the head Spark for the 64GB pair:

```bash
nano recipes/qwen38-flash-next-2x64gb.yaml
```

For the 128GB pair, use `recipes/glm53-flash-2x128gb.yaml` instead.
The supplied files keep the image, model, and serving settings from the matching NVIDIA recipe together.

**Success:** Your chosen recipe is saved in the head Spark's `recipes` folder.

### Step 4. Download the image and launch the model

**First, download the container image on the head Spark.**

Both recipes use this image:

```bash
docker pull eugr/spark-vllm-b12x:nightly-20261015
```

The recipe uses this tag so the model runs with the version of the backend it calls for.

**Then, run the command for your pair.**

For two 64GB Sparks:

```bash
uv run --with pyyaml python run-recipe.py recipes/qwen38-flash-next-2x64gb.yaml --setup
```

For two 128GB Sparks:

```bash
uv run --with pyyaml python run-recipe.py recipes/glm53-flash-2x128gb.yaml --setup
```

The setup step copies the prepared image and model files to the worker, then starts vLLM on both devices.
Allow time and free disk space on both Sparks for the image and model files.
If a model download asks for access, sign in on the head with `uvx hf auth login`, then run the command again.

**Next, keep the launch terminal open while the model runs.**

The first start can take a while as the model loads and the backend builds runtime files.
Wait for the head's logs to show that the API server has started.

**Success:** vLLM is running on both Sparks, and the head has started its API on port `8000`.

### Step 5. Check the API on the head Spark

**First, open a second terminal on the head through Sync.**

Leave the launch terminal open and run:

```bash
curl --fail-with-body -i http://127.0.0.1:8000/health
curl --fail-with-body -sS http://127.0.0.1:8000/v1/models
```

The health check should return HTTP `200`.
Copy the model ID from the model-list reply.
If the server is still loading, wait and try again.

**Then, send a chat request.**

Replace `<model-id>` with the ID from the model list:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}'
```

These models may use some output tokens for reasoning before the final answer.
The larger output limit gives that work room to finish.

**Success:** The API returns a chat response with a final answer.

### Step 6. Reach the model from your laptop

**First, open a terminal on your laptop and start an SSH tunnel.**

Use the head Spark's management address or an SSH alias that works from your laptop.
Replace `<user>` and `<head-host>` below.
This command works in Windows PowerShell, macOS, and Linux terminals with OpenSSH installed:

```text
ssh -N -L 8000:127.0.0.1:8000 <user>@<head-host>
```

Keep this terminal open.
The tunnel sends requests from port `8000` on your laptop to port `8000` on the head Spark.
Sync's cluster setup does not create this API tunnel.

If port `8000` is in use on your laptop, change the first `8000` to `8001` and use `8001` in your laptop's API URLs.

**Then, check the API from a second laptop terminal.**

| Laptop terminal | Health check | List served models |
| --- | --- | --- |
| Windows PowerShell | `curl.exe --fail-with-body -i http://localhost:8000/health` | `curl.exe --fail-with-body -sS http://localhost:8000/v1/models` |
| macOS or Linux | `curl --fail-with-body -i http://localhost:8000/health` | `curl --fail-with-body -sS http://localhost:8000/v1/models` |

For an OpenAI-compatible client on your laptop, use `http://localhost:8000/v1` as the base URL and the model ID from the list.
Both recipes support automatic tool calling; see the Advanced Tab for a test request.

**Success:** Your laptop can reach the model through the head Spark.

### Step 7. Stop and restart the model

**First, stop the model when you are done.**

Press `Ctrl+C` in the launch terminal on the head Spark.
The foreground launcher stops the containers on both Sparks when it exits.

You can also stop them from a second terminal on the head:

```bash
cd ~/spark-vllm-docker
./launch-cluster.sh stop
./launch-cluster.sh status
```

Close the laptop tunnel with `Ctrl+C` when you no longer need it.
The cluster's network settings and downloaded files stay in place.

**Then, restart with the same recipe when you need it again.**

From the repository on the head, run your Step 4 recipe command without `--setup`.
Keep that terminal open and start the laptop tunnel again if you need it.
Use `--setup` again if you need to restore missing model files or copy the image to the worker.
If the image is missing on the head, first run the `docker pull` command from Step 4 again.

**Success:** You can start and stop the model on both Sparks from the head.

## Advanced

## Run and tune a vLLM recipe across two DGX Sparks

You will run one vLLM service across two Sparks, using the Docker and server commands from an NVIDIA Local AI Inference Recipe.

This path uses SSH and a container on each node.
It covers the settings that split the model across the pair, move data between nodes, and set the capacity of the service.

### Step 1. Choose a model and prepare the cluster

**First, choose the recipe for your pair of Sparks.**

Memory in this table is per device.
Use two Sparks of the same memory size for these configurations.

| Cluster | Model and recipe | Checkpoint |
| --- | --- | --- |
| 2 × DGX Spark, 64GB each | [Qwen3.8 Flash-Next](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark-2gpu/index.html?model=qwen%2Fqwen3.8-flash-next&os=linux&backend=vllm&memory=64&precision=fp4-local&memoryConfig=64) | `nvidia/Qwen3.8-Flash-Next-FP4-Local` |
| 2 × DGX Spark, 128GB each | [GLM 5.3 Flash](https://nvidia.github.io/local-ai-inference-recipes/hardware/dgx-spark-2gpu/index.html?model=z-ai%2Fglm-5.3-flash&os=linux&backend=vllm&memory=128&precision=fp4-local&memoryConfig=128) | `nvidia/GLM-5.3-Flash-FP4-Local` |

Both recipes use `eugr/spark-vllm-b12x:nightly-20261015`, a community image with the B12X backend.
Keep the image, checkpoint, environment settings, and server flags from the same recipe.
The 64GB Qwen configuration has its own memory settings; the settings for a pair of 128GB Sparks are different.

**Then, cable and configure the two nodes.**

Use **Configure Manually** in the [Connect Multiple Sparks playbook](https://build.nvidia.com/playbooks/connect-multiple-sparks) if the cluster is not already set up.
Confirm interdevice SSH and Docker access on both nodes.
Choose a head Spark and a worker Spark, then open an SSH session to each.

Keep these values from the network setup:

| Value | Use |
| --- | --- |
| Head's ConnectX IP address | Rendezvous address shared by both vLLM processes |
| Each node's ConnectX Ethernet interface | Socket and control traffic on that node |
| Each node's RoCE HCA names | GPU data transfers over the cluster link |
| Head's management address or SSH alias | Laptop access to the API |

**Success:** You have a complete recipe and can reach both nodes over the cluster link.

### Step 2. Read the distributed and memory settings

**First, check how the recipe splits work across the pair.**

Each Spark runs one tensor-parallel rank.
The ranks hold parts of the model and exchange data during inference.
The head exposes the API; the worker joins the same service.
This gives you one model endpoint across two GPUs.

| Setting | Why it is in the command |
| --- | --- |
| `--tensor-parallel-size 2` | Splits supported model tensors across the two GPUs. Communication uses the link between the Sparks. |
| `--nnodes 2` | Sets the number of nodes in the distributed executor. |
| `--node-rank 0` or `1` | Gives each node a unique rank. The head is rank `0`. |
| `--master-addr` and `--master-port` | Set the shared rendezvous address and port. Use the head's ConnectX address on both nodes. |
| `--headless` | Runs the worker without an API server. |
| `--pipeline-parallel-size 1` | GLM keeps one pipeline stage; its model split uses tensor parallelism. |

These commands use native multi-node execution.
They do not need a separate Ray cluster.
API port `8000` and rendezvous port `29501` serve different roles.
See the [vLLM multi-node multiprocessing guide](https://docs.vllm.ai/en/latest/serving/parallelism_scaling/#running-vllm-with-multiprocessing) for the rank and rendezvous settings.

**Next, check the memory and scheduling limits.**

Each node must fit its share of the weights, caches, and runtime work.
The pair does not act as one flat pool of memory.

| Setting | Why it is in the recipe |
| --- | --- |
| `--quantization modelopt_mixed` and `--load-format b12x` | Select the weight format and loader supported by the checkpoint and image. |
| `--kv-cache-dtype fp8` | Sets KV-cache precision apart from weight precision. |
| `--gpu-memory-utilization` | Sets the fraction of GPU memory vLLM budgets for its executor on each node. |
| `--kv-cache-memory-bytes 3G` | Gives Qwen an explicit KV-cache budget per GPU. This takes precedence over the utilization setting for KV-cache sizing. |
| `--max-model-len` | Caps the total input and output tokens in one request. |
| `--max-num-seqs` | Caps the number of sequences processed in one iteration. |
| `--max-num-batched-tokens` | Sets the token budget for one scheduler iteration. |
| `--enable-chunked-prefill` | Lets long prompt processing share batches with decode work. |
| `--enable-prefix-caching` | Reuses cached prompt work when requests share a prefix. |

The context limit is a per-request ceiling.
It does not mean every active request can use that full length at once.
Actual capacity also depends on the model's cache layout and the other requests in flight.

**Then, keep the model-specific backend settings together for the first run.**

The B12X kernels, compilation settings, and environment variables are part of these recipes.
For example, the 64GB Qwen recipe uses `VLLM_PLE_TABLE_MEMORY=disk`, limits compile workers, and sets a `3G` KV-cache budget.
GLM uses a different attention backend, block size, and memory budget.

Both recipes use native multi-token prediction (MTP) for speculative decoding.
They do not call for a separate draft checkpoint.
Keep the reasoning and tool-call parsers if the service will be used by an agent.

See the [vLLM serve options](https://docs.vllm.ai/en/latest/cli/serve/), [memory and scheduling guide](https://docs.vllm.ai/en/latest/configuration/optimization/), and [speculative decoding guide](https://docs.vllm.ai/en/latest/features/speculative_decoding/) for the core settings.
The community image supplies the B12X-specific code paths used here.

**Success:** You know which settings define the cluster, which set its capacity, and which belong to the chosen model and backend.

### Step 3. Start a container on each Spark

**First, set the network values in each node's host shell.**

Use the same head address and rendezvous port on both nodes.
Set `ETH_IF` and `IB_IF` to the local node's interfaces; their names may differ between nodes.
Keep both RoCE HCA names called for by the recipe.

```bash
export HEAD_IP="<head-connectx-ip>"
export ETH_IF="<local-connectx-interface>"
export IB_IF="<local-roce-hca-1>,<local-roce-hca-2>"
export MASTER_PORT=29501
export IMAGE="eugr/spark-vllm-b12x:nightly-20261015"
export HF_CACHE_DIR="$HOME/.cache/huggingface"
```

**Then, run this Docker command on each node.**

If checkpoint access needs a Hugging Face token, set `HF_TOKEN` in the host shell first.
The container downloads the image if needed, and vLLM downloads the model when it starts.
Both nodes need the model files and enough free disk space for their local caches.

```bash
mkdir -p "$HF_CACHE_DIR" "$HOME/.cache/vllm" \
  "$HOME/.cache/flashinfer" "$HOME/.triton" "$HOME/.cache/b12x"

docker run --rm -it \
  --name vllm_cluster \
  --privileged --gpus all --ipc host --network host \
  --shm-size=16g \
  --ulimit nofile=1048576:1048576 \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -e HF_TOKEN="${HF_TOKEN:-}" \
  -e MN_IF_NAME="$ETH_IF" -e UCX_NET_DEVICES="$ETH_IF" \
  -e NCCL_SOCKET_IFNAME="$ETH_IF" -e NCCL_IB_HCA="$IB_IF" \
  -e NCCL_IB_DISABLE=0 -e NCCL_IGNORE_CPU_AFFINITY=1 \
  -e OMPI_MCA_btl_tcp_if_include="$ETH_IF" \
  -e GLOO_SOCKET_IFNAME="$ETH_IF" -e TP_SOCKET_IFNAME="$ETH_IF" \
  -e HEAD_IP="$HEAD_IP" -e MASTER_PORT="$MASTER_PORT" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.triton:/root/.triton" \
  -v "$HOME/.cache/b12x:/root/.cache/b12x" \
  --entrypoint /bin/bash "$IMAGE"
```

Host networking and the interface variables let the processes use the ConnectX link.
The mounts keep model and runtime files across container restarts.
The API will bind to the head's host interfaces, so use the service on your trusted network.

**Success:** You have an open Bash shell inside the container on each node.

### Step 4. Run the model on the head and worker

**First, set the rank arguments inside each container.**

On the head, run:

```bash
NODE_ARGS=(--node-rank 0)
```

On the worker, run:

```bash
NODE_ARGS=(--node-rank 1 --headless)
```

**Then, run the full command for your model in both containers.**

Use the same model, environment settings, and serving flags on both nodes.
Only the rank arguments differ.
Start the head command, then start the worker command in its session; the distributed startup waits for both ranks.

#### Qwen3.8 Flash-Next on two 64GB Sparks

```bash
export VLLM_PLE_TABLE_MEMORY="disk"
export CUTE_DSL_ARCH="sm_121a"
export SAFETENSORS_FAST_GPU="1"
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SSM_CONV_STATE_LAYOUT="DS"
export VLLM_USE_AOT_COMPILE="1"
export VLLM_USE_MEGA_AOT_ARTIFACT="1"
export VLLM_USE_V2_MODEL_RUNNER="1"
export B12X_POLICY_MODE="auto"
export VLLM_ENABLE_ROCE_ALLREDUCE="1"
export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
export NCCL_BUFFSIZE="1048576"
export NCCL_MAX_CTAS="8"
export B12X_WEIGHTS_COMPILE_WORKERS="4"
export B12X_STATE_COMPILE_WORKERS="4"
export B12X_BIND_COMPILE_WORKERS="4"
export VLLM_B12X_MOE_FP4_FORCE_A16="1"
export B12X_W4A16_A4_PREFILL="1"

vllm serve nvidia/Qwen3.8-Flash-Next-FP4-Local \
  --host 0.0.0.0 \
  --port 8000 \
  --trust-remote-code \
  --tensor-parallel-size 2 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --block-size 16 \
  --load-format b12x \
  --max-model-len 262144 \
  --max-num-seqs 8 \
  --max-num-batched-tokens 2048 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":4}' \
  --gdn-decode-kernel b12x \
  --linear-backend b12x \
  --moe-backend b12x \
  --no-enable-flashinfer-autotune \
  --mm-encoder-tp-mode data \
  --reasoning-parser qwen3 \
  --tool-call-parser qwen3_xml \
  --enable-auto-tool-choice \
  --compilation-config '{"pass_config":{"fuse_act_quant":true},"max_cudagraph_capture_size":40,"cudagraph_capture_sizes":[1,2,4,5,8,10,16,20,24,32,40]}' \
  --mm-processor-cache-gb 0 \
  --limit-mm-per-prompt '{"image":1}' \
  --gpu-memory-utilization 0.8 \
  --kv-cache-memory-bytes 3G \
  --nnodes 2 \
  --master-addr "$HEAD_IP" \
  --master-port "$MASTER_PORT" \
  "${NODE_ARGS[@]}"
```

#### GLM 5.3 Flash on two 128GB Sparks

```bash
export CUTE_DSL_ARCH="sm_121a"
export SAFETENSORS_FAST_GPU="1"
export VLLM_ENABLE_ROCE_ALLREDUCE="1"
export VLLM_ROCE_ALLREDUCE_MAX_SIZE="2MB"
export NCCL_BUFFSIZE=1048576
export NCCL_MAX_CTAS=8
export VLLM_WORKER_MULTIPROC_METHOD="spawn"
export VLLM_SSM_CONV_STATE_LAYOUT="DS"
export VLLM_USE_AOT_COMPILE="1"
export VLLM_USE_MEGA_AOT_ARTIFACT="1"
export VLLM_USE_V2_MODEL_RUNNER="1"
export VLLM_ENABLE_PCIE_ALLREDUCE="0"
export B12X_POLICY_MODE="auto"
export VLLM_B12X_MOE_FP4_FORCE_A16=1
export B12X_W4A16_A4_PREFILL=1

vllm serve nvidia/GLM-5.3-Flash-FP4-Local \
  --host 0.0.0.0 \
  --port 8000 \
  --tensor-parallel-size 2 \
  --pipeline-parallel-size 1 \
  --decode-context-parallel-size 1 \
  --mamba-cache-mode align \
  --enable-prefix-caching \
  --enable-chunked-prefill \
  --dtype bfloat16 \
  --kv-cache-dtype fp8 \
  --quantization modelopt_mixed \
  --attention-backend B12X \
  --block-size 256 \
  --moe-backend b12x \
  --linear-backend b12x \
  --no-enable-flashinfer-autotune \
  --load-format b12x \
  --max-model-len 500000 \
  --max-num-seqs 4 \
  --max-num-batched-tokens 4096 \
  --speculative-config '{"method":"mtp","num_speculative_tokens":5,"moe_backend":"humming","attention_backend":"B12X"}' \
  --reasoning-parser glm45 \
  --tool-call-parser glm47 \
  --enable-auto-tool-choice \
  --gpu-memory-utilization 0.87 \
  --nnodes 2 \
  --master-addr "$HEAD_IP" \
  --master-port "$MASTER_PORT" \
  "${NODE_ARGS[@]}"
```

**Success:** Both ranks load the model and join the distributed executor, and the head starts the API server.
Keep both container sessions open.

### Step 5. Test the API and tool calling

**First, check readiness from a second host shell on the head.**

```bash
curl --fail-with-body -i http://127.0.0.1:8000/health
curl --fail-with-body -sS http://127.0.0.1:8000/v1/models
```

Wait for HTTP `200`, then copy the model ID from the model list.
If startup fails, check both container sessions: a worker failure can block the head.

**Next, send a chat request with that model ID.**

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Write a haiku about a GPU."}],"max_tokens":4096}'
```

Allow enough output tokens for reasoning and a final answer.
Check `finish_reason` if the response stops before the answer is complete.

**Then, test automatic tool calling.**

Replace `<model-id>` and run:

```bash
curl --fail-with-body -sS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"<model-id>","messages":[{"role":"user","content":"Use get_weather to look up the weather in Boston."}],"tools":[{"type":"function","function":{"name":"get_weather","description":"Get current weather for a location.","parameters":{"type":"object","properties":{"location":{"type":"string"}},"required":["location"]}}}],"tool_choice":"auto","max_tokens":4096}'
```

Look for a `tool_calls` entry for `get_weather` with Boston in its arguments, and `finish_reason` set to `tool_calls`.
The request declares the tool; your client would execute it.
If the reply is plain text, check the model's tool support, chat template, and parser settings.
See the [vLLM tool-calling guide](https://docs.vllm.ai/en/latest/features/tool_calling/).

**Success:** Chat and automatic tool calling work through the head's API.

### Step 6. Connect a client and tune the workload

**First, forward the head's API port to your laptop if needed.**

Run this in a local laptop terminal, using the head's management address or a working SSH alias:

```text
ssh -N -L 8000:127.0.0.1:8000 <user>@<head-host>
```

Keep the tunnel open and use `http://localhost:8000/v1` as the client's base URL.
Set the client's model to the ID from `/v1/models`.
The worker does not need a separate client connection.

**Then, measure the service with prompts and concurrency that match your workload.**

Record prompt length, output length, active requests, time to first token, and output tokens per second.
Watch memory and link use on both Sparks.
Two nodes add capacity and communication work; measure the result for the model and request mix you plan to serve.

Change one setting at a time and apply the same change on both ranks.

| Goal | Settings to review |
| --- | --- |
| Fit the workload within memory | Active sequences, context length, and the recipe's KV-cache budget |
| Change the balance between prompt work and active replies | `--max-num-batched-tokens` with chunked prefill |
| Reuse repeated prompt text | Prefix caching with matching system prompts or document prefixes |
| Check the benefit of MTP | Compare the recipe's speculative settings with a supported baseline under the same request mix |

Keep a copy of the working commands, image tag or digest, checkpoint revision, network values, and measurements before you tune.

**Success:** You have a working client and a measured baseline for further changes.

### Step 7. Stop or save the service

**First, stop both containers when you are done.**

Run this from a host shell on **each Spark**:

```bash
docker stop vllm_cluster
```

The containers are removed when they stop.
The mounted model and runtime caches remain for the next start.
Close the laptop tunnel with `Ctrl+C`.

**Then, save the host and server commands if you want a repeatable launch.**

Keep the per-node network values and rank arguments with the full model command.
If you use a wrapper, have it start and stop both ranks and report a failed worker.
The Intermediate Tab includes supplied YAML recipes for the community launcher; those files keep the model environment and serve flags together while the launcher supplies the node ranks and starts both containers.

**Success:** Both nodes have stopped, and you have the commands needed to start the same service again.

## Troubleshooting

## Common issues

| Symptom | Cause | Fix |
| --- | --- | --- |
| Cluster Assistant cannot finish its checks | One or both Sparks or their ConnectX-7 link needs attention | Return to [Connect Multiple DGX Sparks](https://build.nvidia.com/playbooks/connect-multiple-sparks) and complete its checks before running the recipe. |
| `docker` reports permission denied on either Spark | The device user cannot access Docker | Follow the Docker access check in the [single-device instructions](https://build.nvidia.com/playbooks/vllm/intermediate). |
| Model download reports an authentication error | Hugging Face CLI access is missing or invalid on a Spark | Check `hf auth whoami` on both devices and authenticate as described in the [Hugging Face guide](https://huggingface.co/docs/huggingface_hub/main/en/quick-start#authentication). |
| The API does not respond on the head Spark | The recipe may still be loading or launch may have failed | Check the output in the launch terminal before retrying the `/health` request. |

The cluster recipe and manual launcher have separate setup steps. Use the commands from one path consistently; see **Intermediate** for the recipe or **Advanced** for explicit launch settings.
