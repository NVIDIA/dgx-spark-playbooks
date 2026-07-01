# Nemotron Model Family on DGX Spark

> Deploy Nemotron 3 model family (Nemotron-3-Nano or Nemotron-3-Super) on DGX Spark


## Table of Contents

- [Overview](#overview)
- [Run Nemotron Nano](#run-nemotron-nano)
- [Run Nemotron Super](#run-nemotron-super)
- [Troubleshooting](#troubleshooting)
  - [Nemotron Super — Docker and GPU (both paths)](#nemotron-super-docker-and-gpu-both-paths)
  - [Nemotron Super — vLLM (Steps 3–6)](#nemotron-super-vllm-steps-36)
  - [Nemotron Super — TensorRT-LLM (Steps 7–11)](#nemotron-super-tensorrt-llm-steps-711)

---

## Overview

## Basic idea

**Nemotron** is NVIDIA’s open language model family built for real workloads: strong reasoning, tool use, and long-context chat in a stack you can run on your own hardware. Models in the line share a similar product shape—capable assistants you can serve behind an OpenAI-style API and integrate with agents, IDEs, or custom apps—while checkpoints differ in size, architecture (including efficient MoE designs), and the serving stack that fits them best.

The family lives in the open [Nemotron](https://github.com/NVIDIA-NeMo/Nemotron) project and on [Hugging Face](https://huggingface.co/nvidia) alongside documentation and cookbooks from NVIDIA and the community. This playbook is your on-ramp for **running Nemotron on a single DGX Spark**: you get a working inference endpoint, not a tour of every training or export option.

## What you'll accomplish

You will deploy a **production-style Nemotron inference server** on DGX Spark with an **OpenAI-compatible HTTP API**, so you can send chat completions from `curl`, scripts, or client libraries. You will use checkpoints and runtimes that are known to work on Spark’s GB10 GPU and unified memory, with steps for verifying the server and (where the model supports it) reasoning-style behavior.

The instruction tabs are **two supported ways to get there**—different models and engines—picked to match what actually fits on one machine. You do not need both; choose the tab that matches the model you want.

## What to know before starting

- Comfortable using a Linux terminal, environment variables, and **running Docker** with GPU access.
- Able to use **Hugging Face Hub** for downloads and, for gated assets, an account and token.
- Basic familiarity with **REST APIs** or `curl` for a quick smoke test.

## Prerequisites

**Hardware**

- NVIDIA **DGX Spark** with GB10
- Enough disk for model weights, build artifacts or container layers, and caches (plan for **tens of gigabytes** minimum; large checkpoints need more)

**Software (baseline)**

- Network access to GitHub, Hugging Face, and (for containerized serving) container registries
- Current NVIDIA GPU driver stack appropriate for your Spark image

**Choosing a tab (only split here)**

- **Run Nemotron Nano** — You serve **Nemotron-3-Nano** on Docker with **[vLLM](https://github.com/vllm-project/vllm)** (`vllm/vllm-openai:v0.20.0`), pointing the container at local **Nemotron-3-Nano Omni** weights and running `vllm serve` on port **8000**. You will want the **NVIDIA Container Toolkit** and Docker with GPU access.
- **Run Nemotron Super** — You serve the **NVFP4** Nemotron-3-Super checkpoint on Docker with **either** **[vLLM](https://github.com/vllm-project/vllm)** (`vllm/vllm-openai:cu130-nightly`, reasoning-parser plugin, tuned env vars and flags) **or** **[TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM)** (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9`, `hf download` into a local folder, `extra-llm-api-config.yml`, and `trtllm-serve`). You will want the **NVIDIA Container Toolkit**, Docker with GPU access, and a **Hugging Face token** where the model card requires it.

## Time & risk

- **Time:** First-time setup usually **on the order of tens of minutes to longer**, dominated by downloads (weights, and for Super, the inference image). Exact times depend on bandwidth and cache state.
- **Risk level:** **Low** — work is in user space (Docker, weight directories, cache dirs); you are not asked to reimage the system.
- **Rollback:** Remove the model/weight directories you created, prune Docker images/containers, and clear Hugging Face cache only if you intend to reclaim disk.
- **Last updated:** 07/01/2026
  — Switch Nemotron Nano tab to vLLM

## Run Nemotron Nano

## Step 1. Overview and prerequisites

This tab serves **NVIDIA Nemotron-3-Nano** on a single DGX Spark with **vLLM**, using the upstream multi-arch image **`vllm/vllm-openai:v0.20.0`**. The server exposes an OpenAI-compatible HTTP API on port **8000**. The example uses the multimodal **Nemotron-3-Nano Omni** weights (text, image, audio, and video), so the launch command installs the optional audio packages and enables the multimodal limits.

DGX Spark ships a single Grace–Blackwell GB10 GPU with **128 GB of unified memory**—the batch, context, and cache choices below assume that footprint.

This tab covers the **Spark-specific setup**. For everything not covered here—**API examples, reasoning mode, and video tuning**—follow the general Nemotron instructions and the model card.

**Requirements**

- DGX Spark with GB10 and sufficient disk for weights, caches, and the container image
- NVIDIA Container Toolkit and Docker with GPU access
- Local **Nemotron-3-Nano Omni** weights on disk (point `WEIGHTS` at that directory in Step 3)

---

## Step 2. Pull the vLLM container image

Pull the upstream multi-arch vLLM **v0.20.0** image. Docker automatically pulls the **arm64** variant on Spark.

```bash
docker pull vllm/vllm-openai:v0.20.0
```

---

## Step 3. Launch the vLLM server on Spark

Point `WEIGHTS` at your local Nemotron-3-Nano Omni weights directory, then start the server. The base image does **not** include audio packages, so the command installs them with `pip install vllm[audio]` before running `vllm serve`.

```bash
WEIGHTS=/path/to/nemotron-3-nano-omni-weights

docker run --rm -it \
  --gpus all \
  --ipc=host -p 8000:8000 \
  --shm-size=16g \
  --name vllm-nemotron-omni \
  -v "${WEIGHTS}:/model:ro" \
  --entrypoint /bin/bash \
  vllm/vllm-openai:v0.20.0 -c  \
  "pip install vllm[audio] && vllm serve /model \
  --served-model-name=nemotron_3_nano_omni \
  --max-num-seqs 8 \
  --max-model-len 131072 \
  --port 8000 \
  --trust-remote-code \
  --gpu-memory-utilization 0.8 \
  --limit-mm-per-prompt '{\"video\": 1, \"image\": 1, \"audio\": 1}' \
  --media-io-kwargs '{\"video\": {\"fps\": 2,  \"num_frames\": 256}}' \
  --allowed-local-media-path=/ \
  --enable-prefix-caching \
  --max-num-batched-tokens 32768 \
  --reasoning-parser nemotron_v3 \
  --enable-auto-tool-choice \
  --tool-call-parser qwen3_coder"
```

**Flag rationale (summary)**

| Flag | Role |
| ---- | ---- |
| `--served-model-name=nemotron_3_nano_omni` | The `model` id clients pass to the API |
| `--max-num-seqs 8` | Conservative concurrency for memory headroom |
| `--max-model-len 131072` | Context window; reduce first if you hit OOM (see Step 5) |
| `--gpu-memory-utilization 0.8` | Fraction of unified memory vLLM may use |
| `--limit-mm-per-prompt` | Caps multimodal inputs per request (1 each of video, image, audio) |
| `--media-io-kwargs` | Video decode settings (2 fps, up to 256 frames) |
| `--allowed-local-media-path=/` | Lets the server read local media files by path |
| `--enable-prefix-caching` | Reuses shared prompt prefixes across requests |
| `--reasoning-parser nemotron_v3` | Parses Nemotron reasoning output |
| `--enable-auto-tool-choice` / `--tool-call-parser qwen3_coder` | Tool calling support |

**Key Spark-specific flags**

| Flag | Purpose | Spark guidance |
| ---- | ------- | -------------- |
| `--gpus all` | Select GPU | Spark has one GB10 GPU; `all` is equivalent to `device=0` |
| `--max-model-len` | Max context window | Start at `131072`; reduce if you hit OOM (see Step 5) |

---

## Step 4. Verify the server and test the API

In another terminal, confirm the server is ready and reports the served model:

```bash
curl -sS http://localhost:8000/v1/models | python3 -m json.tool
```

Once the model is listed, send a chat completion. Use the served model name from Step 3:

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron_3_nano_omni",
    "messages": [{"role": "user", "content": "New York is a great city because..."}],
    "max_tokens": 100
  }'
```

For multimodal requests, reasoning-mode prompting, and video tuning, follow the general Nemotron instructions and the model card.

---

## Step 5. Memory tuning on Spark

Spark uses unified LPDDR5X memory (~128 GB shared between CPU and GPU), not separate system + VRAM pools. If you hit OOM, use these two levers, in order of impact:

1. **Lower `--gpu-memory-utilization`** from `0.8` toward `0.70` to free memory back to the OS and re-enable weight prefetch. Cost: a smaller KV cache budget.
2. **Lower `--max-model-len`** to reduce KV cache allocation (for example, halving the context window halves the KV cache at `--max-num-seqs=1`).

Combined override for a tight-memory run:

```bash
  --gpu-memory-utilization=0.70 \
  --max-model-len=32768 \
```

---

## Step 6. Cleanup

Stop the running container with `Ctrl+C`, or from another terminal:

```bash
docker stop vllm-nemotron-omni
```

Remove the image only if you want to reclaim disk:

```bash
docker rmi vllm/vllm-openai:v0.20.0
```

Delete the local weights directory only if you no longer need it.

For the larger **Nemotron Super** deployment on the same hardware, use the **Run Nemotron Super** tab in this playbook.

## Run Nemotron Super

## Step 1. Overview and prerequisites

This tab serves **NVIDIA Nemotron-3-Super** (NVFP4 checkpoint) on a single DGX Spark. Deploy with **vLLM** (`vllm/vllm-openai:cu130-nightly`) or **TensorRT-LLM** (`nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9`) using the commands and tuning from the [Nemotron Spark deployment guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide).

DGX Spark ships a single Grace–Blackwell GPU with **128 GB of unified memory**—the guide’s KV, batch, and cache choices assume that footprint.

**Architecture refresher of Nemotron 3 Super**

- **LatentMoE** — Expert computation runs in a compressed latent dimension (`d=4096 → ℓ=1024`); all-to-all routing is reduced ~4× versus a standard MoE. On a single GPU, expert parallelism does not apply: use `--tensor-parallel-size 1` (vLLM) and `--tp_size 1 --ep_size 1` (TensorRT-LLM).
- **MTP (multi-token prediction)** — One MTP layer is baked into the checkpoint for speculative decoding, with minimal extra KV versus an external draft model.
- **Mamba-2 hybrid** — SSM state cache (`mamba_ssm_cache`) is separate from the KV cache. vLLM uses `float32` for that cache in the recipe below; TensorRT-LLM uses FP16 SSM cache plus stochastic rounding in `extra-llm-api-config.yml`, as in the guide.

**Requirements**

- DGX Spark with GB10 and sufficient disk for weights, caches, and images
- NVIDIA Container Toolkit and Docker with GPU access
- Hugging Face token (`HF_TOKEN`) when the model card requires it

**How the steps are grouped**

- **Step 2:** Download the reasoning parser file—the guide asks for this before starting **either** server.
- **Steps 3–6:** vLLM (pull image, environment, run, test on port **8000**). Skip if you only use TensorRT-LLM.
- **Steps 7–11:** TensorRT-LLM (pull image, checkpoint, YAML, run, test on port **8123**). Skip Steps 3–6 if you only use TensorRT-LLM.
- **Step 12:** Cleanup.

The configurations in the upstream guide were contributed by Izzy Putterman, Nave Assaf, Joyjit Daw, and other NVIDIA engineers.

---

## Step 2. Download the Nemotron Super reasoning parser

Download the reasoning parser **before** starting the server (for both vLLM and TensorRT-LLM):

```bash
wget https://huggingface.co/nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4/raw/main/super_v3_reasoning_parser.py
```

- **vLLM:** Mount this file in Step 5 (`docker run` uses `-v …/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py`).
- **TensorRT-LLM:** The documented `trtllm-serve` line uses `--reasoning_parser nano-v3`; still keep this file in your project directory so your setup matches the guide.

---

## Step 3. Pull the vLLM container image

**vLLM path only.**

```bash
docker pull vllm/vllm-openai:cu130-nightly
```

MTP + NVFP4 on DGX Spark requires this **cu130 nightly** image. Pinned releases such as **0.17.1** do not support this combination on single-GPU Spark for this model.

---

## Step 4. Configure vLLM environment variables

**vLLM path only.**

On the host, export the vLLM variables (they match the upstream Spark guide) and your Hugging Face token before `docker run`. Copy and paste:

```bash
export VLLM_NVFP4_GEMM_BACKEND=marlin
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
export VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm
export VLLM_USE_FLASHINFER_MOE_FP4=0
export HF_TOKEN=<your_huggingface_token>
```

Why these environment variables:
- `VLLM_NVFP4_GEMM_BACKEND` — Marlin GEMM for NVFP4 on Spark.
- `VLLM_ALLOW_LONG_MAX_MODEL_LEN` — Required for `--max-model-len 1000000` on one GPU.
- `VLLM_FLASHINFER_ALLREDUCE_BACKEND` — Allreduce fix for single-GPU topology.
- `VLLM_USE_FLASHINFER_MOE_FP4` — Disable FlashInfer FP4 MoE (multi-GPU Blackwell); Marlin handles FP4 on Spark.

The Step 5 `docker run` passes the same values with `-e`; keeping these exports on the host is optional but makes the flags easy to reuse or inspect.

---

## Step 5. Run Nemotron Super with vLLM

**vLLM path only.**

From the directory that contains `super_v3_reasoning_parser.py`, in the **same shell** where you ran Step 4 (so `HF_TOKEN` is set):

```bash
docker run --rm -it --gpus all \
  -e VLLM_NVFP4_GEMM_BACKEND=marlin \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm \
  -e VLLM_USE_FLASHINFER_MOE_FP4=0 \
  -e HF_TOKEN=$HF_TOKEN \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -v $(pwd)/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py \
  -p 8000:8000 \
  vllm/vllm-openai:cu130-nightly \
    --model nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
    --served-model-name nemotron-3-super \
    --host 0.0.0.0 \
    --port 8000 \
    --async-scheduling \
    --dtype auto \
    --kv-cache-dtype fp8 \
    --tensor-parallel-size 1 \
    --pipeline-parallel-size 1 \
    --data-parallel-size 1 \
    --trust-remote-code \
    --gpu-memory-utilization 0.90 \
    --enable-chunked-prefill \
    --max-num-seqs 4 \
    --max-model-len 1000000 \
    --moe-backend marlin \
    --mamba_ssm_cache_dtype float32 \
    --quantization fp4 \
    --speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}' \
    --reasoning-parser-plugin /app/super_v3_reasoning_parser.py \
    --reasoning-parser super_v3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder
```

**Flag rationale (summary)**

| Flag | Role |
| ---- | ---- |
| `--tensor-parallel-size 1` | Single GPU; no expert parallelism on Spark |
| `--kv-cache-dtype fp8` | Smaller KV footprint for long context in unified memory |
| `--max-num-seqs 4` | Conservative concurrency for memory headroom |
| `--moe-backend marlin` / `--quantization fp4` | NVFP4 + Marlin MoE on one GPU |
| `--speculative_config` … MTP | Baked-in MTP head (3 draft tokens); Triton MoE on speculative path |
| `--mamba_ssm_cache_dtype float32` | SSM cache separate from KV; TensorRT-LLM can use FP16 + stochastic rounding instead |
| `--async-scheduling` | Better throughput on single GPU |

---

## Step 6. Test the vLLM API

**vLLM path only.**

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nemotron-3-super",
    "messages": [{"role": "user", "content": "Summarize what Latent MoE changes for routing traffic."}],
    "max_tokens": 256
  }'
```

If you enabled API key auth in vLLM, add the matching `Authorization` header.

---

## Step 7. Pull the TensorRT-LLM container image

**TensorRT-LLM path only** — skip Steps 3–6 if you use this stack.

```bash
docker pull nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9
```

---

## Step 8. Download the NVFP4 checkpoint for TensorRT-LLM

**TensorRT-LLM path only.**

`trtllm-serve` expects a **local NVFP4 checkpoint directory**. Download the Hugging Face model into a folder you will mount into the container (here `./nemotron-super-nvfp4`):

```bash
export HF_TOKEN=<your_huggingface_token>
hf download nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4 \
  --local-dir ./nemotron-super-nvfp4
```

If your TensorRT-LLM build expects engines or another layout, follow [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) documentation for `release:1.3.0rc9`.

---

## Step 9. Create extra-llm-api-config.yml for TensorRT-LLM

**TensorRT-LLM path only.**

In the same parent directory as `nemotron-super-nvfp4`, create `extra-llm-api-config.yml`:

```yaml
kv_cache_config:
  dtype: fp8
  enable_block_reuse: false
  free_gpu_memory_fraction: 0.9
  mamba_ssm_cache_dtype: float16
  mamba_ssm_stochastic_rounding: true
  mamba_ssm_philox_rounds: 5
moe_config:
  backend: CUTLASS
cuda_graph_config:
  enable_padding: true
  max_batch_size: 8
enable_attention_dp: false
enable_chunked_prefill: true
stream_interval: 1
print_iter_log: true
speculative_config:
  decoding_type: MTP
  num_nextn_predict_layers: 3
  allow_advanced_sampling: true
```

**Config rationale**

| Setting | Rationale |
| ------- | --------- |
| `kv_cache_config.dtype: fp8` | FP8 KV cache to maximize context in 128 GB unified memory |
| `mamba_ssm_cache_dtype: float16` | FP16 SSM cache saves memory vs float32 on one GPU |
| `mamba_ssm_stochastic_rounding: true` | Mitigates FP16 SSM precision loss (e.g. 5 Philox rounds) |
| `enable_block_reuse: false` | Mamba recurrent state is not prefix-cacheable |
| `free_gpu_memory_fraction: 0.9` | Aggressive allocator use on single GPU |
| `moe_config.backend: CUTLASS` | CUTLASS MoE for single-GPU NVFP4 |
| `max_batch_size: 8` | Conservative for memory headroom |
| `num_nextn_predict_layers: 3` | MTP with three speculative steps |
| `max_seq_len: 1048576` | Full 1M token context window (set on `trtllm-serve` in Step 10). |

---

## Step 10. Run Nemotron Super with TensorRT-LLM

**TensorRT-LLM path only.**

From the directory that contains both `nemotron-super-nvfp4` and `extra-llm-api-config.yml`:

```bash
export HF_TOKEN=<your_huggingface_token>

docker run --rm -it --gpus all \
  -e HF_TOKEN=$HF_TOKEN \
  -e TLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -v "$(pwd)":/workspace \
  -w /workspace \
  -p 8123:8123 \
  nvcr.io/nvidia/tensorrt-llm/release:1.3.0rc9 \
  trtllm-serve nemotron-super-nvfp4 \
    --host 0.0.0.0 \
    --port 8123 \
    --max_batch_size 8 \
    --tp_size 1 --ep_size 1 \
    --max_num_tokens 8192 \
    --trust_remote_code \
    --reasoning_parser nano-v3 \
    --tool_parser qwen3_coder \
    --extra_llm_api_options extra-llm-api-config.yml \
    --max_seq_len 1048576
```

Change `/workspace/nemotron-super-nvfp4` if your checkpoint folder name or path differs.

---

## Step 11. Test the TensorRT-LLM API

**TensorRT-LLM path only.**

Use the same shape as Step 6 on port **8123**. Set `"model"` to the served name printed in `trtllm-serve` logs (the example below uses a placeholder you should replace).

```bash
curl http://localhost:8123/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "YOUR_SERVED_MODEL_NAME",
    "messages": [{"role": "user", "content": "Summarize what Latent MoE changes for routing traffic."}],
    "max_tokens": 256
  }'
```

---

## Step 12. Cleanup

Stop the running container with `Ctrl+C`.

Remove the vLLM reasoning parser if you no longer need it:

```bash
rm -f super_v3_reasoning_parser.py
```

Remove local checkpoint directories or Hugging Face cache only if you want to reclaim disk.

For **Nemotron Nano** model, use the **Run Nemotron Nano** tab.

## Troubleshooting

## Troubleshooting for Nemotron Nano

The **Run Nemotron Nano** tab serves the model on Docker with **vLLM** (`vllm/vllm-openai:v0.20.0`) on port **8000**. Full flags and paths are in that tab.

| Symptom | Cause | Fix |
|---------|-------|-----|
| `docker: could not select device driver` or no GPU in container | NVIDIA Container Toolkit / driver | Install or restart the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); keep `docker run --gpus all` as in Step 3 |
| Container exits immediately or model fails to load | Wrong or incomplete weights path | Point `WEIGHTS` at the local Nemotron-3-Nano Omni weights directory and keep `-v "${WEIGHTS}:/model:ro"` so the container sees `/model` (Step 3) |
| `ModuleNotFoundError` / audio backend errors | Audio packages not installed | The base image omits audio; keep `pip install vllm[audio] && vllm serve /model …` in the launch command (Step 3) |
| "CUDA out of memory" when starting server | Context or concurrency too large for free memory | Lower `--gpu-memory-utilization` (e.g. `0.70`) first, then `--max-model-len` (e.g. `32768`), per Step 5 |
| Reasoning or tool output malformed | Parser flags mismatch | Keep `--reasoning-parser nemotron_v3`, `--enable-auto-tool-choice`, and `--tool-call-parser qwen3_coder` as in Step 3 |
| `curl` returns model / 404 errors | Name does not match served id | Use `"model": "nemotron_3_nano_omni"` to match `--served-model-name` in Step 3 |
| "Connection refused" on port 8000 | Port or container | Map `-p 8000:8000`, `--port 8000`, and confirm the container is still running (`docker ps`) |

## Troubleshooting for Nemotron Super

The **Run Nemotron Super** tab uses **Step 2** (reasoning parser download) for both stacks, then **vLLM** (Steps **3–6**) or **TensorRT-LLM** (Steps **7–11**). Match the table to the stack you run. Full flags and paths are in that tab; the upstream [Spark deployment guide](https://github.com/NVIDIA-NeMo/Nemotron/tree/main/usage-cookbook/Nemotron-3-Super/SparkDeploymentGuide) is the source of truth for tuned settings.

### Nemotron Super — Docker and GPU (both paths)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `docker: Error response from daemon: could not select device driver` or no GPU in container | NVIDIA Container Toolkit / driver | Install or restart the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html); use `docker run --gpus all` as in the Super tab |
| Image pull failures | Auth or network | For NGC (`nvcr.io/...`), `docker login nvcr.io` with your API key if required; check proxy and registry access |

### Nemotron Super — vLLM (Steps 3–6)

| Symptom | Cause | Fix |
|---------|-------|-----|
| Container exits immediately or model fails to load | Missing HF access or token | Set `HF_TOKEN` (Step 4) in the **same shell** as Step 5 so `-e HF_TOKEN=$HF_TOKEN` is not empty. Confirm your account can use `nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-NVFP4` |
| Stuck or repeated downloads inside the container | Cache not persisted | Keep `-v ~/.cache/huggingface:/root/.cache/huggingface` so weights reuse across runs |
| Error loading reasoning parser / plugin | Missing file or bad mount | Run Step 2 `wget`, start Step 5 `docker run` from the directory that contains `super_v3_reasoning_parser.py`, and keep `-v $(pwd)/super_v3_reasoning_parser.py:/app/super_v3_reasoning_parser.py` |
| Errors about max model length | Long context blocked | Export or pass `VLLM_ALLOW_LONG_MAX_MODEL_LEN=1` with large `--max-model-len` (e.g. 1000000), per Step 4–5 |
| FP4 / MoE kernel errors on Spark | Wrong image or backend | Use `vllm/vllm-openai:cu130-nightly` (not an older pin such as `0.17.1` for this recipe). Keep `VLLM_NVFP4_GEMM_BACKEND=marlin`, `VLLM_USE_FLASHINFER_MOE_FP4=0`, and `--moe-backend marlin` / `--quantization fp4` |
| Allreduce / distributed warnings on one GPU | Single-GPU topology | Set `VLLM_FLASHINFER_ALLREDUCE_BACKEND=trtllm` (Steps 4–5) as in the Spark guide; see [vLLM PR #35793](https://github.com/vllm-project/vllm/pull/35793) |
| MTP / speculative decoding errors | Bad JSON or flags | Copy `--speculative_config '{"method":"mtp","num_speculative_tokens":3,"moe_backend":"triton"}'` exactly; fix shell quoting if the JSON was split or escaped wrong |
| OOM or server killed under load | Context × concurrency too high | Lower `--max-num-seqs`, `--max-model-len`, or `--gpu-memory-utilization`; free GPU memory from other jobs. KV uses `--kv-cache-dtype fp8` to save space |
| Reasoning or tool output malformed | Parser flags mismatch | Keep `--reasoning-parser-plugin /app/super_v3_reasoning_parser.py`, `--reasoning-parser super_v3`, `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder` as in Step 5 |
| `curl` returns model / 404 errors | Name does not match served id | Use `"model": "nemotron-3-super"` to match `--served-model-name` in Step 5 (Step 6 `curl`) |
| "Connection refused" on port 8000 | Port or container | Map `-p 8000:8000`, `--host 0.0.0.0`, `--port 8000`, and confirm the container is still running |

### Nemotron Super — TensorRT-LLM (Steps 7–11)

| Symptom | Cause | Fix |
|---------|-------|-----|
| `trtllm-serve` exits or cannot open model | Wrong checkpoint path or incomplete download | Finish Step 8 `hf download` into `./nemotron-super-nvfp4`. Mount the parent with `-v "$(pwd)":/workspace` and pass the same folder name as in Step 10 (e.g. `/workspace/nemotron-super-nvfp4`) |
| Long context / max seq errors | Long-seq guard | Keep `TLLM_ALLOW_LONG_MAX_MODEL_LEN=1` on the container when using `--max_seq_len 1048576` (Step 10) |
| Config or YAML parse errors | Missing or invalid `extra-llm-api-config.yml` | Place the file next to the checkpoint directory, mount `/workspace`, and use `--extra_llm_api_options /workspace/extra-llm-api-config.yml`. Match YAML indentation to Step 9 |
| OOM or process killed | Batch or sequence too large | Reduce `--max_batch_size`, `--max_num_tokens`, or `--max_seq_len`; in the YAML, try lowering `cuda_graph_config.max_batch_size` or `kv_cache_config.free_gpu_memory_fraction` slightly |
| MoE / NVFP4 backend errors | Backend mismatch for single GPU | Keep `moe_config.backend: CUTLASS` in `extra-llm-api-config.yml` for this Spark recipe |
| Reasoning or tools look wrong | Parser confusion with vLLM | TensorRT-LLM uses `--reasoning_parser nano-v3` and `--tool_parser qwen3_coder` (Step 10), not the vLLM `super_v3` plugin file |
| "Connection refused" on port 8123 | Port mapping | Use `-p 8123:8123`, `--host 0.0.0.0`, `--port 8123` as in Step 10 |
| `curl` fails or wrong model in JSON | Served name differs | Read `trtllm-serve` startup logs for the actual model id and substitute `YOUR_SERVED_MODEL_NAME` in Step 11 |
| Checkpoint layout errors | TRT-LLM expects engines or another format | Your image may need a converted engine or different layout; follow [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM) docs for `release:1.3.0rc9` and this checkpoint |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic sharing between GPU and CPU. Some workloads can still hit memory pressure while reporting headroom. If you see unexplained OOM or stalls, try flushing the page cache (administrative host only):

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For platform known issues, see the [DGX Spark known issues](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html) page.
