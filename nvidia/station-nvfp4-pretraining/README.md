# NVFP4 Pretraining with Megatron Bridge

> Pretrain Llama 3.1 8B with NVFP4 mixed precision on DGX Station using Megatron Bridge


## Table of Contents

- [Overview](#overview)
- [Pretrain with NVFP4](#pretrain-with-nvfp4)
- [Troubleshooting](#troubleshooting)

---

## Overview

## NVFP4 training

NVFP4 is a 4-bit floating-point format natively supported by NVIDIA Blackwell Tensor Cores.
When applied during **pretraining**, NVFP4 reduces memory bandwidth and compute cost for matrix multiplications while preserving model quality through mixed-precision accumulation in higher precision (BF16/FP32).

Megatron-Bridge is NVIDIA's library for large-scale distributed training built on top of Megatron-Core.
It provides composable recipe configs for models, optimizers, and mixed-precision strategies — including the first-class `bf16_with_nvfp4_mixed` recipe used in this playbook.

Combining the two lets you pretrain LLMs at lower memory cost and higher throughput compared to BF16-only training, with minimal accuracy trade-off.

Key benefits:

- **~2× higher training throughput vs BF16** - Higher TFLOPs at minimal loss in model quality
- **Native Blackwell NVFP4 GEMMs** — FP4 matmuls run as a single Tensor Core instruction, no software emulation overhead
- **Recipe-based configuration** — swap between `bf16_mixed`, `bf16_with_fp8_current_scaling_mixed`, and `bf16_with_nvfp4_mixed` with a single line
- **Stability controls** — pin the first/last N transformer layers in BF16 (this playbook keeps the last 4 layers in BF16 via `first_last_layers_bf16`)
- **~2× memory reduction** - For inference weight storage vs FP8, ~3.5× vs FP16

## What you'll accomplish

Pretrain a **Llama 3.1 8B** model using Megatron-Bridge with NVFP4 mixed precision on NVIDIA DGX Station.
You'll run a short training loop with mock data to verify the full pipeline end-to-end, compare against a plain BF16 baseline via the `--disable-fp4` flag and then learn how to point it at real data if required.

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

- Basic Python and PyTorch usage
- Familiarity with distributed training concepts (`torchrun`)
- Understanding of mixed precision training (FP16/BF16/FP8)

## Prerequisites

- NVIDIA DGX Station with Blackwell architecture GPU (GB300 chip)
- Docker installed with GPU support
- NVIDIA Container Toolkit configured
- Megatron-Bridge installed (via the NeMo Framework NGC container)

Verify your setup:

```bash
## Check GPU availability and architecture
nvidia-smi

## Verify Python and torch
python3 -c "import torch; print(torch.cuda.get_device_name(0))"
```

## Time & risk

* **Estimated duration**: 20-30 minutes (quick test loop with default `--train-iters 50`); longer for real data
* **Risks**:
  * NVFP4 requires Blackwell GPUs — will fail on Hopper or older
  * Mock data is used by default (`eval_iters=0`); real data requires a preprocessed Megatron-format dataset
* **Rollback**: Stop the `torchrun` process and remove any checkpoint directories
* **Last Updated:** 05/26/2026
  * First Publication

## Pretrain with NVFP4

## Step 1. Set up the environment

The recommended way to run Megatron-Bridge on DGX Station is through the [NeMo Framework container](https://github.com/NVIDIA-NeMo/Megatron-Bridge#-nemo-framework-container), which includes Megatron-Bridge, Megatron-Core, Transformer Engine, and all CUDA dependencies pre-installed. Running outside the container is not supported in this playbook — the NVFP4 kernels rely on the exact Transformer Engine / CUDA versions shipped inside the image.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/station-nvfp4-pretraining/assets

## Use the latest nemo tag
export TAG=26.04

docker run --rm -it \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v "$(pwd):/workdir" \
  -w /workdir \
  --entrypoint bash \
  nvcr.io/nvidia/nemo:${TAG}
```

All subsequent `torchrun` / `python` commands in this playbook are meant to be executed **from the shell inside this container**.

## Step 2. Review the pretraining script

The pretraining script can be found at `pretrain_llama.py`. The key piece is the NVFP4 precision config, built on top of Megatron-Bridge's prebuilt `bf16_with_nvfp4_mixed` recipe:

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

## Step 3. Launch NVFP4 pre-training

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

See the measured numbers in `overview.md` for expected VRAM and latency on 1× GB300 with Llama 3.1 8B.

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
config.data.data_path = "/path/to/your/preprocessed/dataset"
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

## References

- Quickstart: https://github.com/NVIDIA-NeMo/Megatron-Bridge/blob/main/tutorials/recipes/llama/00_quickstart_pretrain.py
- Mixed precision: https://docs.nvidia.com/nemo/megatron-bridge/latest/training/mixed-precision.html
- API: https://docs.nvidia.com/nemo/megatron-bridge/latest/apidocs/bridge/bridge.training.mixed_precision.html

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `RuntimeError: NVFP4 is not supported on this GPU` or similar FP4 error | GPU is not Blackwell architecture | NVFP4 requires Blackwell GPUs (GB200, GB300). Check with `nvidia-smi` |
| `ModuleNotFoundError: No module named 'megatron.bridge'` | Megatron Bridge not installed | Run `pip install megatron-bridge` or use the NGC container |
| `CUDA out of memory` during model init | Insufficient GPU memory for Llama 3.1 8B + optimizer states | Reduce `micro_batch_size` or use `--nproc_per_node` for model parallelism |
| `torchrun` hangs or times out | NCCL communication failure between GPUs | Check `NCCL_DEBUG=INFO torchrun ...` for details; verify all GPUs are visible |
| Training loss is NaN | Precision instability | Increase `num_layers_at_end_in_bf16` (e.g., from 4 to 8) or reduce learning rate |
| `--disable-fp4` works but NVFP4 crashes | Transformer Engine version mismatch | Ensure Transformer Engine supports NVFP4; update with `pip install --upgrade transformer-engine` |
| Slow training throughput | Not using Tensor Cores efficiently | Ensure batch dimensions are multiples of 8; check that `nvidia-smi` shows high GPU utilization |
| Permission denied on Docker | User not in docker group | Run `sudo usermod -aG docker $USER && newgrp docker` |
