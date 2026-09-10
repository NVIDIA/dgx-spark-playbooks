# Run NVFP4 Pretraining with Megatron Bridge

> Higher Llama 3.1 8B throughput with mixed-precision training

## Table of Contents

- [Overview](#overview)
- [Pretrain with NVFP4](#pretrain-with-nvfp4)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

NVFP4 is a 4-bit floating-point format natively supported by NVIDIA Blackwell Tensor Cores.
When applied during **pretraining**, NVFP4 reduces memory bandwidth and compute cost for matrix multiplications while preserving model quality through mixed-precision accumulation in higher precision (BF16/FP32).

Megatron-Bridge is NVIDIA's library for large-scale distributed training built on top of Megatron-Core.
It provides composable recipe configs for models, optimizers, and mixed-precision strategies — including the first-class `bf16_with_nvfp4_mixed` recipe used in this playbook.

Combining the two lets you pretrain LLMs at lower memory cost and higher throughput compared to BF16-only training, with minimal accuracy trade-off.

Key benefits:

- **~2× higher training throughput vs BF16** — Higher TFLOPs at minimal loss in model quality
- **Native Blackwell NVFP4 GEMMs** — FP4 matmuls run as a single Tensor Core instruction, no software emulation overhead
- **Recipe-based configuration** — swap between `bf16_mixed`, `bf16_with_fp8_current_scaling_mixed`, and `bf16_with_nvfp4_mixed` with a single line
- **Stability controls** — pin the first/last N transformer layers in BF16 (this playbook keeps the last 4 layers in BF16 via `first_last_layers_bf16`)
- **~2× memory reduction** — For inference weight storage vs FP8, ~3.5× vs FP16

## What you'll accomplish

You'll pretrain a **Llama 3.1 8B** model using Megatron-Bridge with NVFP4 mixed precision on your **hardware platform**.
You'll run a short training loop with mock data to verify the full pipeline end-to-end, compare against a plain BF16 baseline via the `--disable-fp4` flag, and learn how to point the recipe at real data if required.

## Measured results

Run settings:

- Model: Llama 3.1 8B (`llama3_8b_pretrain_config()`)
- 50 iterations, 2 warmup
- Global batch size 64, micro batch size 4, sequence length 4096
- Dummy data (Megatron-Core's built-in `MockGPTDataset` — synthetic random token IDs, no real corpus)
- Single GB300 GPU, `nvcr.io/nvidia/nemo:26.04` container
- Latency: average of iterations 20–50 (iter 10 includes one-time CUDA-graph/compile overhead)
- VRAM: peak of `nvidia-smi --query-compute-apps=used_memory` sampled every 2 s during the run

| Precision | Recipe | Avg step time | Throughput (Model TFLOP/s/GPU) | Peak VRAM |
|---|---|---|---|---|
| BF16 baseline | `bf16_mixed()` | 9.05 s | ~1399 | 221.6 GB |
| NVFP4 (last-4 BF16) | `bf16_with_nvfp4_mixed()` + `first_last_layers_bf16=True`, `num_layers_at_end_in_bf16=4` | **5.39 s** | **~2347** | **207.8 GB** |

NVFP4 is **1.68× faster** than BF16 (≈68% higher throughput) with ≈13.8 GB (≈6%) less peak VRAM — the regime NVFP4 was designed for, where matmul FLOPs dominate each step and quantization overhead is amortized over wide linear projections.

## What to know before starting

**Required:**

- Basic Python and PyTorch usage
- Familiarity with distributed training concepts (`torchrun`)
- Understanding of mixed precision training (FP16/BF16/FP8)
- Experience with Docker containers and GPU-accelerated workloads

**Optional:**

- Experience preparing Megatron-format datasets for real-data training runs

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Station** | DGX OS (Linux) | Large HBM + Grace DRAM | `nvcr.io/nvidia/nemo:26.04` on GB300 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Blackwell-architecture GPU with NVFP4 Tensor Core support (GB300)
- Enough free disk space for the NeMo container image and experiment checkpoints

**Software requirements**

- Docker installed with GPU support: `docker --version`
- NVIDIA Container Toolkit configured
- Hugging Face account with an access token and access to `meta-llama/Meta-Llama-3-8B`
- Network access to NGC / container registry and Hugging Face

Verify your setup:

```bash
## Check GPU availability and architecture
nvidia-smi

## Verify Python and torch on the host (optional; training runs inside the container)
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Ancillary files

All required assets can be found [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-nvfp4-pretraining/).

- `assets/pretrain_llama.py` — Llama 3.1 8B Megatron-Bridge pretraining script with NVFP4 mixed precision and a BF16 baseline flag

## Time & risk

- **Estimated time:** 30 MIN (quick test loop with default `--train-iters 50`; longer for real data)
- **Risk level:** Medium
  - NVFP4 requires Blackwell GPUs — will fail on Hopper or older architectures
  - Mock data is used by default (`eval_iters=0`); real data requires a preprocessed Megatron-format dataset
  - Large container pulls and checkpoint directories need adequate storage and network bandwidth
- **Rollback:** Stop the `torchrun` process and remove any checkpoint directories (see Cleanup in Instructions). The container is launched with `--rm`, so exiting removes it.
- **Last Updated:** 08/03/2026
  - Megatron-Bridge NVFP4 mixed-precision pretraining workflow for Llama 3.1 8B on supported hardware platforms

## Pretrain with NVFP4

> [!NOTE]
> These instructions target **Linux** on a supported hardware platform. Use the default container from the Supported hardware platforms matrix in Overview.

## Step 1. Set up the environment

The recommended way to run Megatron-Bridge on your hardware platform is through the [NeMo Framework container](https://github.com/NVIDIA-NeMo/Megatron-Bridge#-nemo-framework-container), which includes Megatron-Bridge, Megatron-Core, Transformer Engine, and all CUDA dependencies pre-installed. Running outside the container is not supported in this playbook — the NVFP4 kernels rely on the exact Transformer Engine / CUDA versions shipped inside the image.

The training recipe fetches the Llama 3 8B architecture config from Hugging Face, so export a [Hugging Face token](https://huggingface.co/settings/tokens) with access to `meta-llama/Meta-Llama-3-8B` before launching the container:

```bash
export HF_TOKEN=<YOUR_HF_TOKEN>
```

Clone the playbook assets, then launch the container. If your system has more than one GPU, pin the GB300 device:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-nvfp4-pretraining/assets

## Use the latest nemo tag
export TAG=26.04

## Pin the GB300 when multiple GPUs are present
GB300_DEVICE=$(nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/GB300/ {print $1; exit}')

docker run --rm -it \
  --gpus device=${GB300_DEVICE} \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -e HF_TOKEN="$HF_TOKEN" \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(pwd):/workdir" \
  -w /workdir \
  --entrypoint bash \
  nvcr.io/nvidia/nemo:${TAG}
```

On a single-GPU system (GB300 only), `GB300_DEVICE` resolves to `0`.

All subsequent `torchrun` / `python` commands in this playbook are meant to be executed **from the shell inside this container**.

## Step 2. Review the pretraining script

The pretraining script is `pretrain_llama.py`. The key piece is the NVFP4 precision config, built on top of Megatron-Bridge's prebuilt `bf16_with_nvfp4_mixed` recipe:

```python
from megatron.bridge.training.mixed_precision import bf16_with_nvfp4_mixed

def nvfp4_mixed_precision():
    cfg = bf16_with_nvfp4_mixed()
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 0
    cfg.num_layers_at_end_in_bf16 = 4
    return cfg
```

`bf16_with_nvfp4_mixed()` already sets `fp8="e4m3"` and `fp8_recipe="nvfp4"` under the hood; we just toggle the layer-pinning knobs on top:

- **Last 4 layers in BF16** (`num_layers_at_end_in_bf16=4`) for training stability (adjustable per model)
- **No start-layer pinning** (`num_layers_at_start_in_bf16=0`) — last-layer stability is usually enough

> [!NOTE]
> The script uses `llama3_8b_pretrain_config()` which defaults to `context_parallel_size=2`. The script overrides this to `context_parallel_size=1` for single-GPU runs. If you swap in a larger recipe (e.g. `nemotron_3_nano_pretrain_config`, which defaults to TP=4), you **must** either launch `torchrun --nproc_per_node=4` on a 4-GPU node or override `config.model.tensor_model_parallel_size = 1` before calling `pretrain(...)`, or you will hit:
> `AssertionError: world size (1) is not divisible by total_model_size (...tensor_model_parallel_size=4 * ...)`.

## Step 3. Launch NVFP4 pretraining

Launch a short training run with mock data and tee the output to a log file so you can inspect VRAM and per-iteration latency afterwards:

```bash
torchrun --nproc_per_node=1 pretrain_llama.py > nvfp4.log 2>&1
```

Expected output (see `nvfp4.log`):

- Model initialization logs and a `Theoretical memory footprints: weight and optimizer=...` line
- Iteration progress printed every step (`log_interval=1`), e.g. `iteration 10/50 | ... elapsed time per iteration (ms): ... | lm loss: ...`
- A `[Rank 0] ... memory (GB) | mem-max-reserved-gigabytes: ...` line — this is your peak VRAM
- A checkpoint saved to `/workdir/nemo_experiments/default/checkpoints`

If the run finishes with `EXIT=0` (or no traceback), your NVFP4 pretraining setup is working.

## Step 4. Compare with BF16 baseline

Run the same script with `--disable-fp4` to establish a BF16 baseline, again logging to a file:

```bash
## Remove the prior checkpoint directory so the two runs don't interfere
rm -rf nemo_experiments

torchrun --nproc_per_node=1 pretrain_llama.py --disable-fp4 > bf16.log 2>&1
```

To compare the two runs on **latency** and **throughput**, grep the per-iteration lines out of each log:

```bash
grep -E "elapsed time per iteration|MODEL_TFLOP" nvfp4.log
grep -E "elapsed time per iteration|MODEL_TFLOP" bf16.log
```

Each step prints two lines:

- `Step Time : 5.39s GPU utilization: 2347.0MODEL_TFLOP/s/GPU` — step latency and throughput
- `iteration 10/50 | ... elapsed time per iteration (ms): 5390 | ... lm loss: ...` — same latency in ms plus loss

Iteration 10 includes one-time CUDA-graph/compile overhead, so average iterations 20–50 for a fair per-step latency number.

#### Measuring peak VRAM (from `nvidia-smi`)

Megatron's in-log memory numbers (`mem-max-reserved-gigabytes`) reflect PyTorch's caching-allocator reservation, which can drift from what the device actually holds. For an accurate read, watch `nvidia-smi` live from a second shell while training runs:

```bash
watch -n 1 nvidia-smi
```

See the measured numbers in Overview for expected VRAM and latency on 1× GB300 with Llama 3.1 8B.

## Step 5. Script arguments

`pretrain_llama.py` accepts the following arguments:

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--disable-fp4` | flag | off | Disable NVFP4; use plain BF16 mixed precision as a baseline |
| `--train-iters` | int | 50 | Number of training iterations |
| `--warmup-iters` | int | 2 | Number of warmup iterations |
| `--global-batch-size` | int | 64 | Global batch size |
| `--micro-batch-size` | int | 4 | Micro batch size (drives peak VRAM; increase to use more memory) |
| `--seq-length` | int | 4096 | Sequence length |

Example combining several flags:

```bash
torchrun --nproc_per_node=1 pretrain_llama.py \
    --train-iters 50 --warmup-iters 2 \
    --global-batch-size 64 --micro-batch-size 4 --seq-length 4096
```

## Step 6. Point to real data

To train on your own dataset, modify the config in the script:

```python
config = llama3_8b_pretrain_config()
config.dataset.data_path = ["/path/to/your/preprocessed/dataset"]
config.train.train_iters = 5000
config.train.global_batch_size = 256
config.train.micro_batch_size = 2
```

Megatron-Bridge expects preprocessed data in Megatron format. See the [Megatron-Bridge data preparation guide](https://docs.nvidia.com/nemo/megatron-bridge/latest/) for details.

## Step 7. Cleanup

Remove checkpoints and log files generated by the runs:

```bash
rm -rf nemo_experiments/ nvfp4.log bf16.log
```

Then exit the container shell (`exit`) — the `--rm` flag in Step 1 deletes it automatically.

## Troubleshooting

## Common issues

The **Hardware platform** column shows where an issue is most relevant. "All hardware platforms" applies to every supported platform.

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `RuntimeError: NVFP4 is not supported on this GPU` or similar FP4 error | All hardware platforms | GPU is not Blackwell architecture | NVFP4 requires Blackwell GPUs (for example GB300). Confirm the device with `nvidia-smi` |
| `ModuleNotFoundError: No module named 'megatron.bridge'` | All hardware platforms | Megatron-Bridge not available in the environment | Use the NeMo Framework NGC container from Instructions; do not run outside the container |
| `CUDA out of memory` during model init | All hardware platforms | Insufficient GPU memory for Llama 3.1 8B + optimizer states | Reduce `--micro-batch-size` or use `--nproc_per_node` for model parallelism |
| `torchrun` hangs or times out | All hardware platforms | NCCL communication failure between GPUs | Run with `NCCL_DEBUG=INFO torchrun ...`; verify the intended GPUs are visible |
| Training loss is NaN | All hardware platforms | Precision instability | Increase `num_layers_at_end_in_bf16` (e.g., from 4 to 8) or reduce learning rate |
| `--disable-fp4` works but NVFP4 crashes | All hardware platforms | Transformer Engine / container mismatch | Use the playbook container tag (`nvcr.io/nvidia/nemo:26.04`); do not mix host TE packages |
| Slow training throughput | All hardware platforms | Tensor Cores underutilized | Ensure batch dimensions are multiples of 8; confirm high GPU utilization in `nvidia-smi` |
| Permission denied on Docker | All hardware platforms | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
| Training runs on the wrong GPU | All hardware platforms | Multi-GPU default selection | Set `GB300_DEVICE` from `nvidia-smi` and pass `--gpus device=${GB300_DEVICE}` |
| `AssertionError: world size (1) is not divisible by total_model_size` | All hardware platforms | Recipe parallel sizes exceed launch GPU count | Override `tensor_model_parallel_size` / `context_parallel_size` for single-GPU, or launch with matching `--nproc_per_node` |

For latest known issues, see the documentation linked under **Resources**.
