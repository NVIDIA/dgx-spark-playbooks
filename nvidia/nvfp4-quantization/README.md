# NVFP4 Quantization

> Quantize Qwen3.6 MoE to NVFP4 with TensorRT Model Optimizer recipes and serve it on Spark with vLLM


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVFP4 is a 4-bit floating-point format introduced with NVIDIA Blackwell GPUs to maintain model accuracy while reducing memory bandwidth and storage requirements for inference workloads. 
Unlike uniform INT4 quantization, NVFP4 retains floating-point semantics with a shared exponent and a compact mantissa, allowing higher dynamic range and more stable convergence.
NVIDIA Blackwell Tensor Cores natively support mixed-precision execution across FP16, FP8, and FP4, enabling models to use FP4 for weights and activations while accumulating in higher precision (typically FP16). 
This design minimizes quantization error during matrix multiplications and supports efficient conversion pipelines in TensorRT-LLM for fine-tuned layer-wise quantization.

Immediate benefits are:
  - Cut memory use ~3.5x vs FP16 and ~1.8x vs FP8
  - Maintain accuracy close to FP8 (usually <1% loss)
  - Improve speed and energy efficiency for inference


## What you'll accomplish

You'll quantize the Qwen3.6-35B-A3B Mixture-of-Experts model using NVIDIA's TensorRT Model Optimizer
inside a TensorRT-LLM container, producing an NVFP4 quantized Hugging Face checkpoint, and then serve
it on NVIDIA DGX Spark with vLLM.

Quantization is driven by Model Optimizer **recipes** — declarative YAML configurations selected with
the `--recipe` flag. The playbook offers two: a Qwen3.5/3.6-MoE dedicated W4A16 recipe (NVFP4 weights
on the MoE MLPs, FP8 attention) recommended for interactive use on DGX Spark, and a general-purpose
NVFP4 experts-only recipe (W4A4) for compute-bound, higher-concurrency serving.

Depending on the recipe, quantization reduces model size by roughly 3x compared to the BF16 model.
This quantization approach aims to preserve accuracy while providing significant throughput improvements. However, it's important to note that quantization can potentially impact model accuracy - we recommend running evaluations to verify if the quantized model maintains acceptable performance for your use case.

## What to know before starting

- Working with Docker containers and GPU-accelerated workloads
- Understanding of model quantization concepts and their impact on inference performance
- Experience with NVIDIA TensorRT and CUDA toolkit environments
- Familiarity with Hugging Face model repositories and authentication

## Prerequisites

- NVIDIA Spark device with Blackwell architecture GPU
- Docker installed with GPU support
- NVIDIA Container Toolkit configured
- Available storage for model files and outputs (about 100 GB free: the BF16 model download is roughly 70 GB and the quantized output roughly 25 GB)
- Hugging Face account with access to the target model

Verify your setup:
```bash
## Check Docker GPU access
docker run --rm --gpus all nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev nvidia-smi

## Verify sufficient disk space
df -h .
```

## Time & risk

* **Estimated duration**: 1.5-3 hours, usually dominated by the model download
* **Risks**:
  * Model download may fail due to network issues or Hugging Face authentication problems
  * Quantization process is memory-intensive and may fail on systems with insufficient GPU memory
  * Output files are large (several GB) and require adequate storage space
* **Rollback**: Remove the output directory and any pulled Docker images to restore original state.
* **Last Updated**: 07/21/2026
  * Update TensorRT Model Optimizer to the 0.45.0 release
  * Switch the example model from DeepSeek-R1-Distill-Llama-8B to Qwen3.6-35B-A3B (MoE)
  * Use recipe-driven PTQ (`hf_ptq.py --recipe`) instead of the removed `--export_fmt` flow, offering the Qwen3.5/3.6-MoE dedicated W4A16 recipe and the general-purpose NVFP4 experts-only recipe
  * Serve the quantized checkpoint with the NGC vLLM container

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

## Step 2. Prepare the environment

Create a local output directory where the quantized model files will be stored. This directory will be mounted into the container to persist results after the container exits.

```bash
mkdir -p ./output_models
chmod 755 ./output_models
```

## Step 3. Authenticate with Hugging Face

Set your Hugging Face authentication token so the container can download the Qwen3.6-35B-A3B model (roughly 70 GB).

```bash
## Export your Hugging Face token as an environment variable
## Get your token from: https://huggingface.co/settings/tokens
export HF_TOKEN="your_token_here"
```

The token will be automatically used by the container for model downloads.

## Step 4. Choose a quantization recipe

TensorRT Model Optimizer drives post-training quantization through **recipes** — declarative YAML files that bundle the full quantization configuration (per-layer formats, calibration algorithm, and KV-cache setting). A recipe is selected with the `--recipe` flag of `hf_ptq.py` and replaces the older `--qformat`/`--quant` flags.

This playbook quantizes the Qwen3.6-35B-A3B Mixture-of-Experts model and offers two recipes:

| Recipe | What it does | When to use it |
|--------|--------------|----------------|
| `huggingface/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast` | Dedicated to the Qwen3.5/Qwen3.6 MoE architecture: NVFP4 weight-only (W4A16) on the MoE expert and shared-expert MLP weights and `lm_head`, FP8 on self-attention and the linear-attention projections, FP8 KV cache. Same layout as the published [nvidia/Qwen3.6-35B-A3B-NVFP4](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) checkpoint. | **Recommended on DGX Spark.** Weight-only quantization keeps the memory-bound decode path fast, so it gives the best interactive (low-concurrency) performance, and it usually does not regress accuracy versus the BF16 model. |
| `general/ptq/nvfp4_experts_only-kv_fp8_cast` | General-purpose MoE recipe: NVFP4 weights *and* activations (W4A4) on the routed experts, FP8 KV cache; attention and dense layers stay in high precision. | Maximum compute throughput once serving becomes compute-bound (higher concurrency or larger batches), and a good template to reuse with other MoE models. |

Set the recipe you want to use:

```bash
## Recommended for DGX Spark: Qwen3.5/3.6-MoE dedicated W4A16 recipe
export RECIPE="huggingface/qwen3_5_moe/ptq/w4a16_nvfp4-fp8_attn-kv_fp8_cast"
export EXPORT_NAME="Qwen3.6-35B-A3B-W4A16-NVFP4"

## Alternative: general-purpose NVFP4 MoE recipe (W4A4 experts)
## export RECIPE="general/ptq/nvfp4_experts_only-kv_fp8_cast"
## export EXPORT_NAME="Qwen3.6-35B-A3B-NVFP4-EXPERTS"
```

## Step 5. Run the quantization

Launch the TensorRT-LLM container with GPU access, IPC settings optimized for multi-GPU workloads, and volume mounts for model caching and output persistence, then run recipe-driven PTQ with `hf_ptq.py`.

```bash
docker run --rm -it --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "./output_models:/workspace/output_models" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev \
  bash -c "
    git clone -b 0.45.0 --single-branch https://github.com/NVIDIA/Model-Optimizer.git /app/Model-Optimizer && \
    cd /app/Model-Optimizer && pip install -e '.[hf]' && \
    cd examples/llm_ptq && \
    python hf_ptq.py \
      --pyt_ckpt_path 'Qwen/Qwen3.6-35B-A3B' \
      --recipe $RECIPE \
      --export_path /workspace/output_models/$EXPORT_NAME
  "
```

Note: You may see a message like `Failed to get GPU memory info: ... Stopping GPU memory monitor.` on DGX Spark. This is expected in some environments and does not affect quantization results.
Note: Please be aware that if your model is too large, you may encounter an out of memory error. You can try quantizing a smaller model instead.

This command:
- Runs the container with full GPU access and optimized shared memory settings
- Mounts your output directory to persist quantized model files
- Mounts your Hugging Face cache to avoid re-downloading the model
- Clones and installs the TensorRT Model Optimizer 0.45.0 release from source
- Runs recipe-driven PTQ and exports a unified Hugging Face checkpoint to `./output_models/$EXPORT_NAME`

The process will display progress information including the model download from Hugging Face, quantization calibration steps (512 calibration samples by default), and the model export phase. End-to-end time is usually dominated by the model download.

## Step 6. Validate the quantized model

After the container completes, verify that the quantized model files were created successfully.

```bash
## Check output directory contents
ls -la ./output_models/$EXPORT_NAME/

## Verify model files are present
find ./output_models/$EXPORT_NAME/ -name "*.safetensors" -o -name "config.json" -o -name "hf_quant_config.json"
```

You should see model weight files, configuration files (including `hf_quant_config.json` with the quantization metadata), and tokenizer files in the output directory.

## Step 7. Serve the model with vLLM

Quantized Qwen3.6 MoE checkpoints are served with vLLM — this is also how NVIDIA validates the published Qwen3.6 NVFP4 checkpoint. Find the latest container build from https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm:

```bash
export LATEST_VLLM_VERSION=<latest_container_version>
## example
## export LATEST_VLLM_VERSION=26.05.post1-py3

## Path to the quantized model produced in Step 5
export MODEL_PATH="./output_models/$EXPORT_NAME/"

docker run --rm -it --gpus all --ipc=host --network host \
  --ulimit memlock=-1 --ulimit stack=67108864 \
  -v "$MODEL_PATH:/workspace/model" \
  -e HF_TOKEN=$HF_TOKEN \
  nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION} \
  vllm serve /workspace/model \
    --served-model-name qwen3.6-35b-a3b \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.4 \
    --max-model-len 262144 \
    --max-num-seqs 4 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --enable-prefix-caching \
    --reasoning-parser qwen3
```

> [!NOTE]
> For the W4A16 recipe, the [nvidia/Qwen3.6-35B-A3B-NVFP4 model card](https://huggingface.co/nvidia/Qwen3.6-35B-A3B-NVFP4) lists additional DGX Spark tuning flags (FlashInfer attention backend, Marlin MoE backend, MTP speculative decoding, and fastsafetensors loading) that can further improve performance.

In a second terminal, wait for the server to become ready (model loading may take several minutes):

```bash
timeout 900 bash -c 'until curl -sf http://localhost:8000/health > /dev/null 2>&1; do sleep 10; done' || echo "Server failed to start within 900s"
```

## Step 8. Test the server

Run the following to test the server with a client CURL request:

```bash
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.6-35b-a3b",
    "messages": [{"role": "user", "content": "What is artificial intelligence?"}],
    "max_tokens": 100,
    "temperature": 0.7,
    "stream": false
  }'
```

## Step 9. Cleanup and rollback

To clean up the environment and remove generated files:

> [!WARNING]
> This will permanently delete all quantized model files and cached data.

```bash
## Remove output directory and all quantized models
rm -rf ./output_models

## Remove Hugging Face cache (optional)
rm -rf ~/.cache/huggingface

## Remove Docker images (optional)
docker rmi nvcr.io/nvidia/tensorrt-llm/release:spark-single-gpu-dev
docker rmi nvcr.io/nvidia/vllm:${LATEST_VLLM_VERSION}
```

## Step 10. Next steps

The quantized model is now ready for deployment. Common next steps include:
- Quantizing with the other recipe from Step 4 and comparing accuracy and throughput for your workload.
- Benchmarking inference performance compared to the original model.
- Integrating the quantized model into your inference pipeline.
- Running additional validation tests on your specific use cases.

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| "Permission denied" when accessing Hugging Face | Missing or invalid HF token | Run `huggingface-cli login` with valid token |
| Container exits with CUDA out of memory | Insufficient GPU memory | Reduce batch size or use a machine with more GPU memory |
| Model files not found in output directory | Volume mount failed or wrong path | Verify `$(pwd)/output_models` resolves correctly |
| Git clone fails inside container | Network connectivity issues | Check internet connection and retry |
| Quantization process hangs | Container resource limits | Increase Docker memory limits or use `--ulimit` flags |
| Cannot access gated repo for URL | Certain HuggingFace models have restricted access | Regenerate your [HuggingFace token](https://huggingface.co/docs/hub/en/security-tokens); and request access to the [gated model](https://huggingface.co/docs/hub/en/models-gated#customize-requested-information) on your web browser |
| `KeyError: 'qwen3_5_moe'` or "unrecognized model type" during quantization | The container's `transformers` version predates Qwen3.6 support | Run `pip install -U transformers` inside the container before running `hf_ptq.py` |
| vLLM fails to load the quantized checkpoint | Older vLLM containers may not support this model architecture or the ModelOpt quantization layout | Use the latest [NGC vLLM container](https://catalog.ngc.nvidia.com/orgs/nvidia/containers/vllm) |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
