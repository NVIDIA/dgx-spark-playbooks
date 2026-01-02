# Speculative Decoding

> Learn how to set up speculative decoding for fast inference on Spark

## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
  - [Option 1: EAGLE-3](#option-1-eagle-3)
  - [Option 2: Draft Target](#option-2-draft-target)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

Speculative decoding speeds up text generation by using a **small, fast model** to draft several tokens ahead, then having the **larger model** quickly verify or adjust them.
This way, the big model doesn't need to predict every token step-by-step, reducing latency while keeping output quality.

## What you'll accomplish

You'll explore speculative decoding using TensorRT-LLM on NVIDIA Spark using two approaches: EAGLE-3 and Draft-Target.
These examples demonstrate how to accelerate large language model inference while maintaining output quality.

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

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| "CUDA out of memory" error | Insufficient GPU memory | Reduce `kv_cache_free_gpu_memory_fraction` to 0.9 or use a device with more VRAM |
| Container fails to start | Docker GPU support issues | Verify `nvidia-docker` is installed and `--gpus=all` flag is supported |
| Model download fails | Network or authentication issues | Check HuggingFace authentication and network connectivity |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| Server doesn't respond | Port conflicts or firewall | Check if port 8000 is available and not blocked |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
