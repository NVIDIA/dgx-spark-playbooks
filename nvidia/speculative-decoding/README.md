# Speculative Decoding

> Learn how to set up speculative decoding for fast inference on Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Option 1: EAGLE-3](#option-1-eagle-3)
  - [Option 2: Draft Target](#option-2-draft-target)
- [Run on Two Sparks](#run-on-two-sparks)
  - [Step 1. Configure Docker Permissions](#step-1-configure-docker-permissions)
  - [Step 2. Network Setup](#step-2-network-setup)
  - [Step 3. Set Container Name Variable](#step-3-set-container-name-variable)
  - [Step 4. Start the TRT-LLM Multi-Node Container](#step-4-start-the-trt-llm-multi-node-container)
  - [Step 5. Configure OpenMPI Hostfile](#step-5-configure-openmpi-hostfile)
  - [Step 6. Launch Eagle3 Speculative Decoding](#step-6-launch-eagle3-speculative-decoding)
  - [Step 7. Validate the API](#step-7-validate-the-api)
  - [Step 8. Cleanup](#step-8-cleanup)
  - [Step 9. Next Steps](#step-9-next-steps)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Speculative decoding speeds up text generation by using a **small, fast model** to draft several tokens ahead, then having the **larger model** quickly verify or adjust them.
This way, the big model doesn't need to predict every token step-by-step, reducing latency while keeping output quality.

## What you'll accomplish

You'll explore speculative decoding using TensorRT-LLM on NVIDIA Spark using two approaches: EAGLE-3 and Draft-Target.
These examples demonstrate how to accelerate large language model inference while maintaining output quality.

## Why two Sparks?

A single DGX Spark has 128 GB of unified memory shared between the CPU and GPU. This is sufficient to run models like GPT-OSS-120B with EAGLE-3 or Llama-3.3-70B with Draft-Target, as shown in the **Instructions** tab.

Larger models like **Qwen3-235B-A22B** exceed what a single Spark can hold in memory — even with FP4 quantization, the model weights, KV cache, and Eagle3 draft head together require more than 128 GB. By connecting two Sparks, you double the available memory to 256 GB, making it possible to serve these larger models.

The **Run on Two Sparks** tab walks through this setup. The two Sparks are connected via QSFP cable and use **tensor parallelism (TP=2)** to split the model — each Spark holds half of every layer's weight matrices and computes its portion of each forward pass. The nodes communicate intermediate results over the high-bandwidth link using NCCL and OpenMPI, so the model operates as a single logical instance across both devices.

In short: two Sparks let you run models that are too large for one, while speculative decoding (Eagle3) on top further accelerates inference by drafting and verifying multiple tokens in parallel.

## What to know before starting

- Experience with Docker and containerized applications
- Understanding of speculative decoding concepts
- Familiarity with TensorRT-LLM serving and API endpoints
- Knowledge of GPU memory management for large language models

## Prerequisites

- NVIDIA Spark device with sufficient GPU memory available
- Docker with GPU support enabled

  ```bash
  docker run --gpus all nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 nvidia-smi
  ```
- Active HuggingFace Token for model access
- Network connectivity for model downloads


## Time & risk

* **Duration:** 10-20 minutes for setup, additional time for model downloads (varies by network speed)
* **Risks:** GPU memory exhaustion with large models, container registry access issues, network timeouts during downloads
* **Rollback:** Stop Docker containers and optionally clean up downloaded model cache.
* **Last Updated:** 01/02/2026
  * Upgrade to latest container v1.2.0rc6
  * Add EAGLE-3 Speculative Decoding example with GPT-OSS-120B

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

## Step 2. Set Environment Variables

Set up the environment variables for downstream services:

 ```bash
export HF_TOKEN=<your_huggingface_token>
 ```

## Step 3. Run Speculative Decoding Methods

### Option 1: EAGLE-3

Run EAGLE-3 Speculative Decoding by executing the following command:

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
  bash -c '
    hf download openai/gpt-oss-120b && \
    hf download nvidia/gpt-oss-120b-Eagle3-long-context \
        --local-dir /opt/gpt-oss-120b-Eagle3/ && \
    cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 5
    speculative_model_dir: /opt/gpt-oss-120b-Eagle3/

kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
    export TIKTOKEN_ENCODINGS_BASE="/tmp/harmony-reqs" && \
    mkdir -p $TIKTOKEN_ENCODINGS_BASE && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/o200k_base.tiktoken && \
    wget -P $TIKTOKEN_ENCODINGS_BASE https://openaipublic.blob.core.windows.net/encodings/cl100k_base.tiktoken
    trtllm-serve openai/gpt-oss-120b \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --extra_llm_api_options /tmp/extra-llm-api-config.yml'
```

Once the server is running, test it by making an API call from another terminal:

```bash
## Test completion endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-120b",
    "prompt": "Solve the following problem step by step. If a train travels 180 km in 3 hours, and then slows down by 20% for the next 2 hours, what is the total distance traveled? Show all intermediate calculations and provide a final numeric answer.",
    "max_tokens": 300,
    "temperature": 0.7
  }'
```

**Key Features of EAGLE-3 Speculative Decoding**

- **Simpler deployment** — Instead of managing a separate draft model, EAGLE-3 uses a built-in drafting head that generates speculative tokens internally.

- **Better accuracy** — By fusing features from multiple layers of the model, draft tokens are more likely to be accepted, reducing wasted computation.

- **Faster generation** — Multiple tokens are verified in parallel per forward pass, cutting down the latency of autoregressive inference.

### Option 2: Draft Target

Execute the following command to set up and run draft target speculative decoding:

```bash
docker run \
  -e HF_TOKEN=$HF_TOKEN \
  -v $HOME/.cache/huggingface/:/root/.cache/huggingface/ \
  --rm -it --ulimit memlock=-1 --ulimit stack=67108864 \
  --gpus=all --ipc=host --network host nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
  bash -c "
#    # Download models
    hf download nvidia/Llama-3.3-70B-Instruct-FP4 && \
    hf download nvidia/Llama-3.1-8B-Instruct-FP4 \
    --local-dir /opt/Llama-3.1-8B-Instruct-FP4/ && \

#    # Create configuration file
    cat <<EOF > extra-llm-api-config.yml
print_iter_log: false
disable_overlap_scheduler: true
speculative_config:
  decoding_type: DraftTarget
  max_draft_len: 4
  speculative_model_dir: /opt/Llama-3.1-8B-Instruct-FP4/
kv_cache_config:
  enable_block_reuse: false
EOF

#    # Start TensorRT-LLM server
    trtllm-serve nvidia/Llama-3.3-70B-Instruct-FP4 \
      --backend pytorch --tp_size 1 \
      --max_batch_size 1 \
      --kv_cache_free_gpu_memory_fraction 0.9 \
      --extra_llm_api_options ./extra-llm-api-config.yml
  "
```

Once the server is running, test it by making an API call from another terminal:

```bash
## Test completion endpoint
curl -X POST http://localhost:8000/v1/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Llama-3.3-70B-Instruct-FP4",
    "prompt": "Explain the benefits of speculative decoding:",
    "max_tokens": 150,
    "temperature": 0.7
  }'
```

**Key features of draft-target:**

- **Efficient resource usage**: 8B draft model accelerates 70B target model
- **Flexible configuration**: Adjustable draft token length for optimization
- **Memory efficient**: Uses FP4 quantized models for reduced memory footprint
- **Compatible models**: Uses Llama family models with consistent tokenization

## Step 4.  Cleanup

Stop the Docker container when finished:

```bash
## Find and stop the container
docker ps
docker stop <container_id>

## Optional: Clean up downloaded models from cache
## rm -rf $HOME/.cache/huggingface/hub/models--*gpt-oss*
```

## Step 5. Next Steps

- Experiment with different `max_draft_len` values (1, 2, 3, 4, 8)
- Monitor token acceptance rates and throughput improvements
- Test with different prompt lengths and generation parameters
- Read more on Speculative Decoding [here](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html).

## Run on Two Sparks

### Step 1. Configure Docker Permissions

**Run on both Spark A and Spark B:**

```bash
sudo usermod -aG docker $USER
newgrp docker
```

### Step 2. Network Setup

Follow the network setup instructions from the **[Connect Two Sparks](https://build.nvidia.com/spark/connect-two-sparks/stacked-sparks)** playbook.

> [!NOTE]
> Complete Steps 1-3 from the Connect Two Sparks playbook before proceeding:
>
> - **Step 1**: Ensure same username on both systems
> - **Step 2**: Physical hardware connection (QSFP cable)
> - **Step 3**: Network interface configuration
>   - Use **Option 2: Manual IP Assignment with the netplan configure file**
>   - Each Spark has two pairs of network ports. When you physically connect a cable between two Sparks, the connected ports will show as **Up**. You can use whichever pair is Up — either **`enp1s0f0np0`** and **`enP2p1s0f0np0`**, or **`enp1s0f1np1`** and **`enP2p1s0f1np1`**
>   - This playbook assumes you are using **`enp1s0f1np1`** and **`enP2p1s0f1np1`**. If your Up interfaces are different, substitute your interface names in the commands below

**For this playbook, we will use the following IP addresses:**

**Spark A (Node 1):**
- `enp1s0f1np1`: 192.168.200.12/24
- `enP2p1s0f1np1`: 192.168.200.14/24

**Spark B (Node 2):**
- `enp1s0f1np1`: 192.168.200.13/24
- `enP2p1s0f1np1`: 192.168.200.15/24

After completing the Connect Two Sparks setup, return here to continue with the TRT-LLM container setup.

### Step 3. Set Container Name Variable

**Run on both Spark A and Spark B:**

```bash
export TRTLLM_MN_CONTAINER=trtllm-multinode
```

### Step 4. Start the TRT-LLM Multi-Node Container

**Run on both Spark A and Spark B:**

```bash
docker run -d --rm \
  --name $TRTLLM_MN_CONTAINER \
  --gpus '"device=all"' \
  --network host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --device /dev/infiniband:/dev/infiniband \
  -e UCX_NET_DEVICES="enp1s0f1np1,enP2p1s0f1np1" \
  -e NCCL_SOCKET_IFNAME="enp1s0f1np1,enP2p1s0f1np1" \
  -e OMPI_MCA_btl_tcp_if_include="enp1s0f1np1,enP2p1s0f1np1" \
  -e OMPI_MCA_orte_default_hostfile="/etc/openmpi-hostfile" \
  -e OMPI_MCA_rmaps_ppr_n_pernode="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT="1" \
  -e OMPI_ALLOW_RUN_AS_ROOT_CONFIRM="1" \
  -e CPATH="/usr/local/cuda/include" \
  -e TRITON_PTXAS_PATH="/usr/local/cuda/bin/ptxas" \
  -v ~/.cache/huggingface/:/root/.cache/huggingface/ \
  -v ~/.ssh:/tmp/.ssh:ro \
  nvcr.io/nvidia/tensorrt-llm/release:1.2.0rc6 \
  bash -c "curl https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/trt-llm/assets/trtllm-mn-entrypoint.sh | bash"
```

Verify:

```bash
docker logs -f $TRTLLM_MN_CONTAINER
```

Expected output at the end:

```
total 56K
drwx------ 2 root root 4.0K Jan 13 05:13 .
drwx------ 1 root root 4.0K Jan 13 05:12 ..
-rw------- 1 root root  100 Jan 13 05:13 authorized_keys
-rw------- 1 root root   45 Jan 13 05:13 config
-rw------- 1 root root  411 Jan 13 05:13 id_ed25519
-rw-r--r-- 1 root root  102 Jan 13 05:13 id_ed25519.pub
-rw------- 1 root root  411 Jan 13 05:13 id_ed25519_shared
-rw-r--r-- 1 root root  100 Jan 13 05:13 id_ed25519_shared.pub
-rw------- 1 root root 3.4K Jan 13 05:13 id_rsa
-rw-r--r-- 1 root root  743 Jan 13 05:13 id_rsa.pub
-rw------- 1 root root 5.0K Jan 13 05:13 known_hosts
-rw------- 1 root root 3.2K Jan 13 05:13 known_hosts.old
Starting SSH
```

### Step 5. Configure OpenMPI Hostfile

The hostfile tells MPI which nodes participate in distributed execution. Use the IPs from the `enp1s0f1np1` interface configured in Step 2.

**On both Spark A and Spark B**, create the hostfile:

```bash
cat > ~/openmpi-hostfile <<EOF
192.168.200.12
192.168.200.13
EOF
```

**Run on both Spark A and Spark B** to copy the hostfile into each container:

```bash
docker cp ~/openmpi-hostfile $TRTLLM_MN_CONTAINER:/etc/openmpi-hostfile
```

Verify connectivity:

```bash
docker exec -it $TRTLLM_MN_CONTAINER bash -c "mpirun -np 2 hostname"
```

Expected output:

```
nvidia@spark-afe0:~$ docker exec -it $TRTLLM_MN_CONTAINER bash -c "mpirun -np 2 hostname"
Warning: Permanently added '[192.168.200.13]:2233' (ED25519) to the list of known hosts.
spark-afe0
spark-ae11
nvidia@spark-afe0:~$
```

### Step 6. Launch Eagle3 Speculative Decoding

Eagle3 speculative decoding accelerates inference by predicting multiple tokens ahead, then validating them in parallel. This can provide significant speedup compared to standard autoregressive generation.

#### Set your Hugging Face token

```bash
export HF_TOKEN=your_huggingface_token_here
```

#### Download the Eagle3 speculative model on both nodes

```bash
docker exec \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c "
    mpirun -x HF_TOKEN -np 2 bash -c 'hf download nvidia/Qwen3-235B-A22B-Eagle3 --local-dir /opt/Qwen3-235B-A22B-Eagle3/'
"
```

#### Create the Eagle3 speculative decoding configuration

This configuration enables Eagle speculative decoding with 3 draft tokens and conservative memory settings.

```bash
docker exec -it $TRTLLM_MN_CONTAINER bash -c "cat > /tmp/extra-llm-api-config.yml <<EOF
enable_attention_dp: false
disable_overlap_scheduler: false
enable_autotuner: false
enable_chunked_prefill: false
cuda_graph_config:
    max_batch_size: 1
speculative_config:
    decoding_type: Eagle
    max_draft_len: 3
    speculative_model_dir: /opt/Qwen3-235B-A22B-Eagle3/
kv_cache_config:
    free_gpu_memory_fraction: 0.9
    enable_block_reuse: false
EOF
"
```

#### Launch the server with Eagle3 speculative decoding

**Run on Spark A only.** This starts the TensorRT-LLM API server using the FP4 base model with Eagle3 speculative decoding enabled. The `mpirun` command coordinates execution across both nodes, so it only needs to be launched from Spark A. The maximum token length is set to 1024 (adjust as needed).

```bash
docker exec \
  -e MODEL="nvidia/Qwen3-235B-A22B-FP4" \
  -e HF_TOKEN=$HF_TOKEN \
  -it $TRTLLM_MN_CONTAINER bash -c '
    mpirun -x CPATH=/usr/local/cuda/include \
           -x TRITON_PTXAS_PATH=/usr/local/cuda/bin/ptxas \
           -x HF_TOKEN \
           trtllm-llmapi-launch \
           trtllm-serve \
           $MODEL \
           --backend pytorch \
           --tp_size 2 \
           --max_num_tokens 1024 \
           --extra_llm_api_options /tmp/extra-llm-api-config.yml \
           --port 8355 --host 0.0.0.0
'
```

Expected output when the endpoint is ready:

```
[01/13/2026-06:16:56] [TRT-LLM] [I] get signal from executor worker
INFO:     Started server process [2011]
INFO:     Waiting for application startup.
INFO:     Application startup complete.
```

### Step 7. Validate the API

**Run on Spark A only.** The server is listening on Spark A, so test the endpoint from there:

```bash
curl -s http://localhost:8355/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/Qwen3-235B-A22B-FP4",
    "messages": [{"role": "user", "content": "Paris is great because"}],
    "max_tokens": 64
  }'
```

Expected: A JSON response with generated text. This confirms the multi-node TensorRT-LLM server with Eagle3 speculative decoding is working correctly.

### Step 8. Cleanup

#### Stop the containers

**Run on both Spark A & B:**

```bash
docker stop $TRTLLM_MN_CONTAINER
```

The containers will be automatically removed due to the `--rm` flag.

#### (Optional) Remove downloaded models

If you need to free up disk space:

**Run on both Spark A & B:**

```bash
rm -rf $HOME/.cache/huggingface/hub/models--nvidia--Qwen3*
```

This removes the model files (~hundreds of GB). Skip this if you plan to run the setup again.

### Step 9. Next Steps

Now that you have Eagle3 speculative decoding running, consider these optimizations and experiments:

- **Adjust draft length:** Modify `max_draft_len` in the configuration (try values between 2-5) to balance speculation speed vs. accuracy
- **Try different models:** Experiment with other model pairs that support Eagle speculative decoding
- **Optimize batch size:** Adjust `max_batch_size` in `cuda_graph_config` for throughput-latency tradeoffs
- **Learn more:** Review the [TensorRT-LLM Speculative Decoding documentation](https://nvidia.github.io/TensorRT-LLM/advanced/speculative-decoding.html) for advanced tuning options
- **Benchmark performance:** Compare inference speeds with and without speculative decoding to measure speedup gains

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| "CUDA out of memory" error | Insufficient GPU memory | Reduce `kv_cache_free_gpu_memory_fraction` to 0.9 or use a device with more VRAM |
| Container fails to start | Docker GPU support issues | Verify `nvidia-docker` is installed and `--gpus=all` flag is supported |
| Model download fails | Network or authentication issues | Check HuggingFace authentication and network connectivity |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| Server doesn't respond | Port conflicts or firewall | Check if port 8000 is available and not blocked |
| `mpirun` fails with SSH connection refused | SSH not configured between containers or nodes | Complete SSH setup from Connect Two Sparks playbook; verify `ssh <node_ip>` works without password from both nodes |
| `mpirun` hangs or times out connecting to remote node | Hostfile IPs don't match actual node IPs | Verify IPs in `/etc/openmpi-hostfile` match the IPs assigned to network interfaces with `ip addr show` |
| NCCL error: "Socket operation on non-socket" | Wrong network interface specified | Check `ibdev2netdev` output and ensure `NCCL_SOCKET_IFNAME` and `UCX_NET_DEVICES` match the active interfaces `enp1s0f1np1,enP2p1s0f1np1` |
| `Permission denied (publickey)` during mpirun | SSH keys not exchanged between containers | Re-run SSH setup from Connect Two Sparks playbook or manually verify `/root/.ssh/authorized_keys` contains public keys from both nodes |
| Model download fails silently in multi-node setup | HF_TOKEN not propagated to mpirun | Add `-e HF_TOKEN=$HF_TOKEN` to `docker exec` command and `-x HF_TOKEN` to `mpirun` command |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU.
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
