# Run cuTile Kernels on DGX Spark and B300

> Python GPU kernels you can benchmark, optimize, and reuse

## Table of Contents

- [Overview](#overview)
- [Kernel Benchmarks](#kernel-benchmarks)
- [End-to-End Inference](#end-to-end-inference)
- [FMHA Implementation](#fmha-implementation)
  - [Attention basics](#attention-basics)
  - [Flash Attention algorithm](#flash-attention-algorithm)
  - [cuTile pseudocode → actual mapping](#cutile-pseudocode-actual-mapping)
  - [Kernel pseudocode](#kernel-pseudocode)
  - [cuTile implementation](#cutile-implementation)
  - [Launching the kernel](#launching-the-kernel)
  - [Optimizations](#optimizations)
  - [Platform configuration](#platform-configuration)
  - [Performance results](#performance-results)
  - [Common issues](#common-issues)
  - [Companion scripts](#companion-scripts)
  - [References](#references)
- [Platform Comparison](#platform-comparison)
  - [End-to-end throughput](#end-to-end-throughput)
  - [CUDA kernel time](#cuda-kernel-time)
  - [cuTile kernel breakdown](#cutile-kernel-breakdown)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

[TileGym](https://github.com/NVIDIA/TileGym) is NVIDIA's benchmark suite and integration framework for cuTile kernels — high-performance GPU kernels written with the cuTile Python DSL. cuTile compiles to Tile IR, so you can write efficient kernels without low-level CUDA programming.

This playbook covers three workflows:

1. **Kernel Benchmarks** — Run standalone cuTile kernel benchmarks (FMHA, MatMul, RMSNorm, and more)
2. **End-to-End Inference** — Run LLM inference with cuTile-optimized kernels via monkey-patching
3. **FMHA Implementation** — Build a Flash Multi-Head Attention kernel from pseudocode to optimized cuTile, with companion scripts to run and benchmark

The same cuTile code runs across supported hardware platforms — cuTile JIT compiles to the appropriate GPU architecture automatically.

## What you'll accomplish

You'll set up a cuTile / TileGym environment on your **hardware platform** and:

- Run the TileGym benchmark suite for attention, matrix, and activation kernels
- Run Qwen2-7B or DeepSeek-V2-Lite inference with cuTile-optimized kernels
- Observe performance scaling across supported hardware platforms
- Build an FMHA kernel step-by-step from pseudocode to an optimized cuTile implementation

## What to know before starting

**Required:**

- Basic familiarity with Docker and command-line tools
- Understanding of GPU compute concepts (TFLOPS, memory bandwidth)

**Optional:**

- HuggingFace account with an access token (for LLM inference)
- No CUDA programming experience required

## Supported hardware platforms

Use the matrix below to confirm your hardware platform, recommended default local settings, and whether multi-node applies.

| Hardware platform | OS | Memory | Recommended default local settings | Multi-node capable hardware |
| :---- | :---- | :---- | :---- | :---- |
| **DGX Spark** | Linux (Ubuntu 24.04 / DGX OS) | 128 GB Unified Memory | CUDA NGC `nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04`; TileGym v1.3.0 | — |
| **B300** | Linux (cloud instance) | 192 GB HBM3e | Same CUDA NGC image and TileGym pin; cuTile JIT targets sm_103 | — |

## Prerequisites

**Hardware requirements**

- Supported hardware platform — see Supported hardware platforms matrix above
- Minimum 16 GB GPU memory for LLM inference
- At least 50 GB available storage for model downloads

**Software requirements**

- Docker installed and configured: `docker ps`
- CUDA Toolkit 13.x with Tile IR support (provided by the CUDA NGC devel container)
- HuggingFace token for model access (LLM inference only)
- Network access for pulling containers and downloading models

Verify Docker is available:

```bash
docker ps
```

If you get a permission error:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

## Kernel support matrix

| Kernel | Category | Data Types | Description |
|--------|----------|------------|-------------|
| **FMHA** | Attention | float16, float8 | Flash Multi-Head Attention |
| **MLA** | Attention | bfloat16, float8 | Multi-head Latent Attention |
| **MLA Decoding** | Attention | float16, float8 | MLA for decode phase |
| **MatMul** | Matrix Ops | float16, float8 | Matrix multiplication |
| **BMM** | Matrix Ops | float16 | Batched matrix multiplication |
| **Group GEMM** | Matrix Ops | float16, float8 | Grouped GEMM for MoE |
| **RMSNorm** | Normalization | float16, bfloat16 | Root mean square normalization |
| **RoPE** | Positional | float16 | Rotary position embedding |
| **SiLU** | Activation | float16, float32 | SiLU activation with multiply |
| **SwiGLU** | Activation | float16, float32 | SwiGLU fused operation |
| **Softmax** | Activation | float16 | Softmax normalization |
| **Dropout** | Regularization | float16, float32 | Dropout forward |

## Model support for LLM inference

| Model | Supported Kernels | Batch Size | Output Tokens | Notes |
|-------|-------------------|------------|---------------|-------|
| **Qwen2-7B** | RoPE, RMSNorm, SwiGLU, FMHA | 16 | 50 | Standard transformer |
| **DeepSeek-V2-Lite** | RoPE, RMSNorm, SiLU, MLA, MoE | 1 | 100 | MLA attention, MoE layers |

## Ancillary files

TileGym sources and scripts live in the [TileGym repository](https://github.com/NVIDIA/TileGym). Companion FMHA tutorial scripts ship with this playbook [in the playbook repository](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/playbook-cutile-kernels/).

- `tests/benchmark/run_all.sh` — Run all kernel benchmarks (TileGym)
- `modeling/transformers/bench_qwen.sh` — Qwen2-7B benchmark script (TileGym)
- `modeling/transformers/bench_deepseek.sh` — DeepSeek-V2-Lite benchmark script (TileGym)
- `modeling/transformers/infer.py` — Main inference script with TileGym integration (TileGym)
- `assets/fmha_optimization_tutorial.py` — FMHA step-by-step optimization tutorial
- `assets/fmha_scaling_analysis.py` — FMHA scaling analysis across sequence lengths

## Time & risk

- **Estimated time:** 60 MIN (including model download for LLM inference; full benchmark suite may take longer)
- **Risk level:** Low
  - Large downloads may fail due to network issues
  - First run includes JIT compilation overhead
- **Rollback:** Exit and remove the Docker container to undo environment changes
- **Last Updated:** 08/03/2026
  - Set up cuTile / TileGym workflows for kernel benchmarks, end-to-end LLM inference, and FMHA implementation on supported hardware platforms

## Kernel Benchmarks

## Step 1. Pull CUDA NGC container with CTK 13.x

```bash
docker pull nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04
```

Launch an interactive session with GPU access:

```bash
docker run --gpus all -it --rm \
  -v ~/TileGym:/workspace/TileGym \
  nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04 \
  /bin/bash
```

> [!NOTE]
> The `-v` flag mounts a local directory to persist the TileGym repository. The `--rm` flag automatically removes the container when you exit; omit it if you want to keep the container for later use.

Prepare the container for installing TileGym:

```bash
apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python-is-python3 \
    git wget curl build-essential nsight-systems-2025.1.3
update-alternatives --install /usr/bin/nsys nsys /opt/nvidia/nsight-systems/2025.1.3/bin/nsys 100 && hash -r
python -m pip install --upgrade pip setuptools wheel

pip install --no-cache-dir --pre "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu130

pip install --no-cache-dir --no-deps accelerate==1.13.0 && \
    pip install --no-cache-dir sentencepiece protobuf
```

## Step 2. Clone TileGym repository

```bash
git clone https://github.com/NVIDIA/TileGym
cd TileGym
git checkout v1.3.0
pip install .
```

## Step 3. Run individual benchmarks

To run specific kernel benchmarks:

```bash
cd tests/benchmark/

## Flash Multi-Head Attention
python bench_fused_attention.py

## Matrix Multiplication
python bench_matrix_multiplication.py

## RMSNorm
python bench_rmsnorm.py

## RoPE
python bench_rope.py

## SwiGLU
python bench_swiglu.py
```

## Step 4. View results

Results show cuTile performance for each kernel and sequence length.

Expected output should look like:

```text
==========================================
Running bench_fused_attention.py...
==========================================
fused-attention-batch4-head32-d128-fwd-causal=True-float16-TFLOPS:
     N_CTX     CuTile
0   1024.0  58.188262
1   2048.0  80.906892
2   4096.0  86.189532
3   8192.0  88.891086
4  16384.0  89.491869
✓ PASSED: bench_fused_attention.py
```

## Step 5. Run benchmark suite

```bash
cd tests/benchmark/
bash run_all.sh
```

> [!NOTE]
> The benchmark runs sequentially to ensure accurate timing results. Completing all kernels may take 40–60 minutes.

## Step 6. Clean up

Exit the container:

```bash
exit
```

Remove this workflow's containers (if you ran without `--rm`):

```bash
## Preferred: remove only containers from this workflow's image
docker ps -a --filter ancestor=nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04 -q | xargs -r docker rm

## Alternative: prune all stopped containers (will prompt for confirmation)
## docker container prune
```

Remove the image (optional):

```bash
docker rmi nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04
```

## Step 7. Compare across hardware platforms

Repeat Steps 1–6 on another supported hardware platform to observe scaling. See the **Platform Comparison** tab for reference ratios.

## End-to-End Inference

## Step 1. Set up environment

If you haven't already, pull the CUDA container and clone TileGym (see the **Kernel Benchmarks** tab for details).

First, clone TileGym on the host:

```bash
mkdir -p ~/TileGym
git clone https://github.com/NVIDIA/TileGym ~/TileGym
cd ~/TileGym
git checkout v1.3.0
```

Then launch the container with the repository mounted:

```bash
docker run --gpus all -it --rm \
  -v ~/TileGym:/workspace/TileGym \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  nvcr.io/nvidia/cuda:13.2.0-devel-ubuntu22.04 \
  /bin/bash
```

> [!NOTE]
> The `-v ~/.cache/huggingface:/root/.cache/huggingface` mount reuses your HuggingFace cache so models are not re-downloaded.

Prepare the container for installing TileGym:

```bash
apt-get update && apt-get install -y --no-install-recommends \
    python3-pip python3-dev python-is-python3 \
    git wget curl build-essential nsight-systems-2025.1.3
update-alternatives --install /usr/bin/nsys nsys /opt/nvidia/nsight-systems/2025.1.3/bin/nsys 100 && hash -r
python -m pip install --upgrade pip setuptools wheel

pip install --no-cache-dir --pre "torch==2.9.1" --index-url https://download.pytorch.org/whl/cu130

pip install --no-cache-dir --no-deps accelerate==1.13.0 && \
    pip install --no-cache-dir sentencepiece protobuf
```

Install TileGym inside the container:

```bash
cd /workspace/TileGym
pip install .
```

Set your HuggingFace token for accessing gated models:

```bash
export HF_TOKEN=<your_huggingface_token>
```

> [!WARNING]
> You need a HuggingFace account and access token. Get one at https://huggingface.co/settings/tokens

## Step 2. Run inference benchmark

Navigate to the transformers benchmark directory:

```bash
cd /workspace/TileGym/modeling/transformers
```

**Option A: Run Qwen2-7B benchmark**

```bash
./bench_qwen.sh
```

Configuration: Model `Qwen/Qwen2-7B`, batch size 16, output length 50 tokens.

**Option B: Run DeepSeek-V2-Lite benchmark**

```bash
./bench_deepseek.sh
```

Configuration: Model `deepseek-ai/DeepSeek-V2-Lite-Chat`, batch size 1, output length 100 tokens.

Both scripts run two configurations:

1. **PyTorch baseline** — Standard HuggingFace inference
2. **TileGym cuTile** — With cuTile kernel replacements

## Step 3. View results

Sample results on a supported hardware platform (Qwen2-7B):

```text
========================================
  Benchmark Results
========================================
Qwen2-7B_naive_bfloat16    |  15.66 tokens/s |  51.10s |  51151.0ms CUDA
Qwen2-7B_cutile_attn       |  18.52 tokens/s |  43.20s |  43079.7ms CUDA
========================================
```

**cuTile kernel breakdown (same run):**

| Kernel | CUDA Time (ms) | Calls |
|--------|----------------|-------|
| `fmha_kernel` | 4185.9 | 28 |
| `swiglu_forward_kernel` | 2459.8 | 1400 |
| `attention_decode_kernel_grouped` | 2271.8 | 1372 |
| `rms_norm_kernel_static_persistent` | 634.7 | 57 |
| `rope_kernel` | 355.6 | 1400 |

Throughput and kernel timings vary by hardware platform. See the **Platform Comparison** tab for cross-platform ratios.

## Step 4. How TileGym monkey-patching works

TileGym replaces PyTorch model operations with cuTile kernels. The snippet below is taken from TileGym's [`src/tilegym/transformers/monkey_patch.py`](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/transformers/monkey_patch.py) and invoked from [`modeling/transformers/infer.py`](https://github.com/NVIDIA/TileGym/blob/main/modeling/transformers/infer.py):

```python
from tilegym.transformers import apply_tilegym_kernel_to_qwen2

apply_tilegym_kernel_to_qwen2(
    rope=True,      # Replace RoPE with cuTile kernel
    rms_norm=True,  # Replace RMSNorm with cuTile kernel
    swiglu=True,    # Replace SwiGLU with cuTile kernel
    attn=True,      # Replace attention with cuTile FMHA
    use_cutile=True # Use cuTile backend (vs Triton)
)
```

**Patched kernels for Qwen2:**

| Kernel | PyTorch Operation | cuTile Replacement |
|--------|-------------------|-------------------|
| `rms_norm_kernel_static_persistent` | `nn.RMSNorm` | Persistent RMSNorm |
| `rope_kernel` | Rotary position embedding | Fused RoPE |
| `fmha_kernel` | `F.scaled_dot_product_attention` | Flash Attention |
| `swiglu_forward_kernel` | SiLU + Mul | Fused SwiGLU |
| `attention_decode_kernel_grouped` | Decode attention | Grouped decode |

**Patched kernels for DeepSeek-V2** (see [`src/tilegym/transformers/monkey_patch.py`](https://github.com/NVIDIA/TileGym/blob/main/src/tilegym/transformers/monkey_patch.py)):

```python
from tilegym.transformers import apply_tilegym_kernel_to_deepseek_v2

apply_tilegym_kernel_to_deepseek_v2(
    rope=True,      # Replace RoPE with cuTile kernel
    rms_norm=True,  # Replace RMSNorm with cuTile kernel
    swiglu=True,    # Replace SiLU+Mul with cuTile kernel
    attn=True,      # Replace MLA attention with cuTile
    moe=True,       # Replace MoE routing with cuTile
    use_cutile=True
)
```

| Kernel | PyTorch Operation | cuTile Replacement |
|--------|-------------------|-------------------|
| `prefill_mla` | MLA prefill attention | Multi-head Latent Attention |
| `_mla_decoding_split_kv` | MLA decode attention | Split-KV decoding |
| `fused_moe_kernel` | MoE expert routing | Fused MoE |
| `group_gemm_kernel` | Expert FFN | Grouped GEMM |

## Step 5. Architecture-specific tuning (advanced)

cuTile exposes complementary performance-tuning mechanisms:

- **[`ct.ByTarget`](https://docs.nvidia.com/cuda/cutile-python/performance.html)** — Select different kernel launch parameters per GPU architecture (`sm_<major><minor>`). The compiler picks the value matching the current target at JIT time; if no entry matches, the `default` value is used. See the [Performance Tuning](https://docs.nvidia.com/cuda/cutile-python/performance.html) and [Execution Model](https://docs.nvidia.com/cuda/cutile-python/execution.html) pages.
- **`num_ctas`** — Number of Cooperative Thread Arrays (thread blocks) launched per kernel invocation. Tune to the number of SMs on the target GPU.
- **`occupancy`** — Hint for the number of concurrent CTAs the compiler should target per SM. Higher occupancy hides memory latency but increases register/shared-memory pressure.
- **[`ct.autotune`](https://docs.nvidia.com/cuda/cutile-python/performance.html)** — Search a list of candidate values at runtime and pick the fastest configuration. Results are reported via [`cuda.tile.tune.TuningResult`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.TuningResult.html) / [`Measurement`](https://docs.nvidia.com/cuda/cutile-python/generated/cuda.tile.tune.Measurement.html).

```python
import cuda.tile as ct

@ct.kernel(
#    # num_ctas: how many thread blocks to launch.
#    # Use ByTarget to pick an arch-specific value at JIT time.
    num_ctas=ct.ByTarget({
        "sm_103": 8,   # More SMs — launch more CTAs
        "sm_121": 4,   # Fewer SMs — use fewer CTAs
        "default": 1,  # Fallback for any other GPU architecture
    }),
#    # occupancy: hint for concurrent CTAs per SM (latency hiding vs. register pressure).
    occupancy=ct.ByTarget({
        "sm_103": 16,  # High occupancy when registers/SMEM allow
        "sm_121": 12,  # Moderate occupancy
        "default": 8,  # Conservative fallback
    }),
    opt_level=3       # Maximum compiler optimization level
)
def optimized_kernel(A, B, C):
#    # Same kernel code works across architectures;
#    # ByTarget swaps in the arch-specific launch params automatically.
    ...
```

For automatic tuning, use [`ct.autotune`](https://docs.nvidia.com/cuda/cutile-python/performance.html) to search over candidate values and pick the fastest configuration at runtime:

```python
@ct.kernel(
#    # autotune: benchmark each value and pick the fastest.
    num_ctas=ct.autotune([1, 2, 4, 8, 16]),
    occupancy=ct.autotune([8, 12, 16, 24]),
    opt_level=3
)
def autotuned_kernel(A, B, C):
    ...
```

## Step 6. Compare across hardware platforms

Repeat Steps 1–3 on another supported hardware platform. The **same code runs without modification** — cuTile JIT compiles for the local GPU architecture automatically.

See the **Platform Comparison** tab for detailed scaling results.

## FMHA Implementation

## FMHA Implementation Guide

> [!NOTE]
> This is a guide to understanding FMHA implementation in cuTile, not a complete reference. For comprehensive documentation, see the [cuTile Python Documentation](https://docs.nvidia.com/cuda/cutile-python/).

### Attention basics

Attention allows a neural network to focus on relevant parts of the input. In transformers (GPT, LLaMA, Qwen), each position computes how much to attend to every other position using three vectors:

- **Query (Q)**: "What am I looking for?"
- **Key (K)**: "What do I contain?"
- **Value (V)**: "Here is my content"

```text
Attention(Q, K, V) = softmax(Q × K^T / √d) × V

Shapes:
  Q, K, V = [batch, heads, seq_len, head_dim]
  Q × K^T = [batch, heads, seq_len, seq_len]  # Attention scores
  Output  = [batch, heads, seq_len, head_dim]
```

For autoregressive models, **causal masking** ensures each token only attends to previous tokens by setting future scores to -infinity before softmax.

### Flash Attention algorithm

Standard attention materializes a [seq_len × seq_len] matrix (e.g., 2 GB for seq_len=32768). Flash Attention avoids this by processing in tiles with **online softmax**:

```text
m = -infinity    # Running maximum
l = 0            # Running sum of exp(x - m)
acc = 0          # Running weighted sum of values

FOR each K,V tile:
    scores = Q_tile @ K_tile.T * scale
    m_new = max(m, max(scores))
    correction = exp(m - m_new)
    l = l * correction + sum(exp(scores - m_new))
    acc = acc * correction + exp(scores - m_new) @ V_tile
    m = m_new

output = acc / l
```

### cuTile pseudocode → actual mapping

| Concept | Pseudocode | cuTile |
|---|---|---|
| Define kernel | `KERNEL fmha(...)` | `@ct.kernel()` |
| Get block ID | `block_x = BLOCK_ID_X` | `bid_x = ct.bid(0)` |
| Create indices | `range(0, N)` | `ct.arange(N, dtype=ct.int32)` |
| Create constant tile | `tile = zeros(M, N)` | `ct.full((M, N), 0.0, dtype)` |
| Load from memory | `tile = LOAD(ptr, shape)` | `ct.load(tensor, index, shape)` |
| Store to memory | `STORE(ptr, tile)` | `ct.store(tensor, index, tile)` |
| Matrix multiply | `C = A @ B + C` | `ct.mma(A, B, C)` |
| Reduction | `max_val = MAX(tile, axis)` | `ct.max(tile, axis, keepdims)` |

### Kernel pseudocode

```text
KERNEL fmha(Q, K, V, Out, scale, TILE_M, TILE_N):
    tile_row = BLOCK_ID_X
    batch_head = BLOCK_ID_Y
    batch = batch_head // num_heads
    head = batch_head % num_heads

    m_i = full(TILE_M, -infinity)
    l_i = full(TILE_M, 0)
    acc = zeros(TILE_M, head_dim)

    q = LOAD(Q[batch, head, tile_row*TILE_M : (tile_row+1)*TILE_M, :])

    FOR j = 0 to num_k_tiles:
        k = LOAD(K[batch, head, j*TILE_N : (j+1)*TILE_N, :])
        v = LOAD(V[batch, head, j*TILE_N : (j+1)*TILE_N, :])
        scores = MMA(q, transpose(k)) * scale
        IF causal AND in_mask_region:
            scores = WHERE(valid_mask, scores, -infinity)
        m_new = max(m_i, row_max(scores))
        correction = exp(m_i - m_new)
        p = exp(scores - m_new)
        l_i = l_i * correction + row_sum(p)
        acc = acc * correction + MMA(p, v)
        m_i = m_new

    out = acc / l_i
    STORE(Out[batch, head, tile_row*TILE_M :, :], out)
```

### cuTile implementation

```python
import cuda.tile as ct
import math
ConstInt = ct.Constant[int]
ConstBool = ct.Constant[bool]

@ct.kernel()
def fmha_kernel(Q, K, V, Out, qk_scale: float, TILE_D: ConstInt, H: ConstInt,
                TILE_M: ConstInt, TILE_N: ConstInt, CAUSAL: ConstBool):
    bid_x, bid_y = ct.bid(0), ct.bid(1)
    batch_idx, head_idx = bid_y // H, bid_y % H

    offs_m = (bid_x * TILE_M + ct.arange(TILE_M, dtype=ct.int32))[:, None]
    offs_n_tile = ct.arange(TILE_N, dtype=ct.int32)[None, :]

    m_i = ct.full((TILE_M, 1), -math.inf, dtype=ct.float32)
    l_i = ct.full((TILE_M, 1), 0.0, dtype=ct.float32)
    acc = ct.full((TILE_M, TILE_D), 0.0, dtype=ct.float32)

    q = ct.load(Q, index=(batch_idx, head_idx, bid_x, 0),
                shape=(1, 1, TILE_M, TILE_D)).reshape((TILE_M, TILE_D))

    k_seqlen = K.shape[2]
    if CAUSAL:
        Tc = ct.cdiv(min((bid_x + 1) * TILE_M, k_seqlen), TILE_N)
        mask_start = (bid_x * TILE_M) // TILE_N
    else:
        Tc = ct.cdiv(k_seqlen, TILE_N)
        mask_start = k_seqlen // TILE_N

    for j in range(0, Tc):
        k_tile = ct.load(K, index=(batch_idx, head_idx, j, 0),
                        shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        k_t = ct.permute(k_tile, (1, 0))

        qk = ct.mma(q, k_t, ct.full((TILE_M, TILE_N), 0.0, dtype=ct.float32))
        qk = qk * qk_scale

        if CAUSAL and j >= mask_start:
            offs_n = j * TILE_N + offs_n_tile
            qk = ct.where(offs_m >= offs_n, qk,
                         ct.full((TILE_M, TILE_N), -math.inf, dtype=ct.float32))

        m_ij = ct.maximum(m_i, ct.max(qk, axis=-1, keepdims=True))
        qk = qk - m_ij
        p = ct.exp(qk)
        alpha = ct.exp(m_i - m_ij)
        l_i = l_i * alpha + ct.sum(p, axis=-1, keepdims=True)
        acc = acc * alpha

        v_tile = ct.load(V, index=(batch_idx, head_idx, j, 0),
                        shape=(1, 1, TILE_N, TILE_D)).reshape((TILE_N, TILE_D))
        acc = ct.mma(p.astype(Q.dtype), v_tile, acc)
        m_i = m_ij

    acc = (acc / l_i).reshape((1, 1, TILE_M, TILE_D)).astype(Out.dtype)
    ct.store(Out, index=(batch_idx, head_idx, bid_x, 0), tile=acc)
```

### Launching the kernel

```python
def run_fmha(q, k, v, sm_scale, is_causal=True):
    import torch
    TILE_M, TILE_N = 64, 64  # Tune per hardware platform (see below)
    batch, num_heads, seq_len, head_dim = q.shape
    out = torch.empty_like(q)
    grid = (math.ceil(seq_len / TILE_M), batch * num_heads, 1)
    ct.launch(
        torch.cuda.current_stream(), grid, fmha_kernel,
        (q, k, v, out, sm_scale, head_dim, num_heads, TILE_M, TILE_N, is_causal)
    )
    return out
```

### Optimizations

#### exp2 + flush_to_zero

`exp2(x) = 2^x` is faster than `exp(x)` on GPU. Requires scale adjustment by `1/log(2)`.

```python
## Convert natural-exp scale to base-2 so we can use the faster ct.exp2 intrinsic.
## exp(x) == exp2(x / log(2)) == exp2(x * INV_LOG_2).
INV_LOG_2 = 1.0 / math.log(2)  # ≈ 1.4427
qk_scale_log2 = qk_scale * INV_LOG_2  # Pre-multiply the softmax scale once

## ... in loop:
## Fuse the running-max update with the scale multiplication.
m_ij = ct.max(qk, axis=-1, keepdims=True) * qk_scale_log2
## Subtract the running max for numerical stability (online softmax).
qk = qk * qk_scale_log2 - m_ij
## flush_to_zero=True: flush denormals to 0 -> avoids slow denormal handling on GPU.
p = ct.exp2(qk, flush_to_zero=True)
alpha = ct.exp2(m_i - m_ij, flush_to_zero=True)  # Correction factor for previous acc/l_i
```

#### Load order transpose

Load K already transposed using `order` parameter, avoiding explicit permute.

```python
## order=(0,1,3,2) swaps the last two axes during the load,
## producing K^T directly in registers -- no extra ct.permute() needed.
## shape is expressed in the transposed layout: (1, 1, TILE_D, TILE_N).
k_t = ct.load(K, index=(..., 0, j), shape=(1,1,TILE_D,TILE_N),
              order=(0,1,3,2)).reshape((TILE_D, TILE_N))
```

#### Latency hints

Prefetch data to overlap memory loads with computation. See the [Performance Tuning docs](https://docs.nvidia.com/cuda/cutile-python/performance.html) for the full list of load/store hints (e.g. `allow_tma`, `latency`).

```python
## latency=N tells the compiler to issue this load N loop iterations in
## advance of its use, so the memory transfer overlaps with the MMA work
## from earlier iterations. Larger latency = deeper software pipeline but
## more register pressure.
k_t = ct.load(K, ..., latency=2)    # Prefetch K 2 iterations ahead
v_tile = ct.load(V, ..., latency=4) # Prefetch V 4 iterations ahead (used later in the loop)
```

#### Occupancy

Allow multiple thread blocks per SM to hide memory latency. See the [Execution Model docs](https://docs.nvidia.com/cuda/cutile-python/execution.html) for details on how `occupancy` interacts with registers and shared memory.

```python
## occupancy=N is a hint to the compiler to target N concurrent CTAs per SM.
## Higher occupancy -> more warps available to hide memory latency,
## but constrains the per-CTA register/SMEM budget.
@ct.kernel(occupancy=2)  # 2 thread blocks (CTAs) co-resident per SM
def fmha_optimized(...):
```

#### Approximate division

Use fast approximate division for final normalization.

```python
from cuda.tile import RoundingMode as RMd
## RMd.APPROX -> hardware approximate reciprocal/divide (MUFU), much faster
## than IEEE-compliant division. Safe here because it's the final softmax
## normalization step where a small ULP error is acceptable.
## flush_to_zero=True flushes denormals to 0 to avoid the slow path.
acc = ct.truediv(acc, l_i, flush_to_zero=True, rounding_mode=RMd.APPROX)
```

### Platform configuration

The same kernel code works across hardware platforms; only configuration parameters change. Use [`ct.ByTarget`](https://docs.nvidia.com/cuda/cutile-python/performance.html) to select values per architecture, or [`ct.autotune`](https://docs.nvidia.com/cuda/cutile-python/performance.html) to search candidate values automatically.

| Hardware platform | Arch | TILE_M | TILE_N | Occupancy | Rationale |
|---|---|---|---|---|---|
| **DGX Spark** | sm_121 | 64 | 64 | 2 | Smaller tiles, higher occupancy for 48 SMs |
| **B300** | sm_103 | 256 | 128 | 1 | Large tiles maximize HBM3e throughput |
| **B300** (alternate) | sm_103 | 128 | 128 | 2 | Higher occupancy, balanced parallelism |

```python
import cuda.tile as ct

@ct.kernel(
#    # TILE_M / TILE_N: rows/cols of the Q and K/V tiles processed per CTA.
#    # Larger tiles -> more arithmetic intensity; smaller tiles -> higher occupancy.
#    # occupancy: target concurrent CTAs per SM (latency hiding vs. register pressure).
    occupancy=ct.ByTarget({
        "sm_121": 2,   # Fewer SMs: 2 CTAs/SM for latency hiding
        "sm_100": 1,   # Larger tiles already saturate the SM
        "default": 1,  # Conservative fallback for other architectures
    }),
    opt_level=3        # Maximum compiler optimization level
)
def fmha_kernel(...):
    ...
```

### Performance results

> **Note:** PyTorch SDPA is used for correctness verification only, not performance comparison.

#### Optimization ladder (sm_121, seq 2048)

| Step | Optimization | Latency (ms) | TFLOPS |
|---|---|---|---|
| 1 | Basic cuTile | 2.19 | 62.8 |
| 2 | + exp2 | 2.07 | 66.5 |
| 3 | + Load Order | 2.07 | 66.3 |
| 4 | + Latency Hints | 2.07 | 66.5 |
| 5 | + Occupancy=2 | 1.73 | 79.5 |
| 6 | + Approx Div (Final) | 1.69 | 81.1 |

#### Scaling across sequence lengths (sm_103)

| Seq Len | Latency (ms) | TFLOPS | vs sm_121 |
|---|---|---|---|
| 1024 | 0.074 | 465 | 5.7x |
| 2048 | 0.178 | 770 | 9.5x |
| 4096 | 0.550 | 999 | 15.1x |
| 8192 | 1.897 | 1159 | 14.6x |
| 16384 | 7.014 | 1254 | 14.2x |

### Common issues

| Issue | Solution |
|---|---|
| Shape mismatch in ct.mma | Ensure A is (M,K), B is (K,N), C is (M,N) |
| dtype errors | Use `.astype()` before mma; accumulator should be float32 |
| Incorrect results with causal | Check mask_start calculation and `offs_m >= offs_n` logic |
| Low performance | Try different TILE_M/N, check occupancy, verify latency hints |

### Companion scripts

Clone the playbook assets, then run the companion scripts on your hardware platform:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd client-hardware-playbooks/nvidia/playbook-cutile-kernels/assets
```

- **`fmha_optimization_tutorial.py`** — Step-by-step optimization tutorial. Builds the FMHA kernel from basic to fully optimized, matching the progression in this guide.
- **`fmha_scaling_analysis.py`** — Scaling analysis across sequence lengths. Benchmarks each optimization level and generates performance data.

```bash
## Run the optimization tutorial
python fmha_optimization_tutorial.py --correctness-check

## Run the scaling analysis
python fmha_scaling_analysis.py --iterations 100
```

### References

- [cuTile Python Documentation](https://docs.nvidia.com/cuda/cutile-python/)
- [Tile IR Specification](https://docs.nvidia.com/cuda/tile-ir/)
- [TileGym (pre-optimized kernels)](https://github.com/NVIDIA/TileGym)
- [NVIDIA Blog: Tuning Flash Attention for Peak Performance in CUDA Tile](https://developer.nvidia.com/blog/tuning-flash-attention-for-peak-performance-in-nvidia-cuda-tile/)
- [Flash Attention Paper](https://arxiv.org/abs/2205.14135)

## Platform Comparison

## Hardware platform performance comparison

This page summarizes performance scaling between the hardware platforms listed in the Supported hardware platforms table — **DGX Spark** (GB10, sm_121) and **B300** (sm_103) — for kernel benchmarks and end-to-end LLM inference.

## Kernel benchmark scaling

Use the ratios below as a reference for how kernel performance scales from DGX Spark (GB10) to B300.

| Kernel | Metric | B300 / GB10 |
|--------|--------|-------------|
| FMHA (causal, 8192) | TFLOPS | 13.7x |
| FMHA (non-causal, 8192) | TFLOPS | 15.1x |
| MatMul (8192) | TFLOPS | 18.9x |
| BMM (batch8, 4096) | TFLOPS | 19.4x |
| Group GEMM (4096) | TFLOPS | 23.9x |
| RMSNorm (4096) | GB/s | 33.1x |
| RoPE (16384) | GB/s | 22.8x |

**Key observations:**

- Compute-heavy kernels typically scale 14–24x from GB10 to B300
- Memory-bound kernels can scale 20–33x due to HBM bandwidth advantage

## Qwen2-7B performance

### End-to-end throughput

| Configuration | DGX Spark | B300 | Platform speedup |
|---------------|-----------|------|------------------|
| **cuTile** | 18.52 tok/s | 257.33 tok/s | **13.9x** |

### CUDA kernel time

| Configuration | DGX Spark | B300 | Platform speedup |
|---------------|-----------|------|------------------|
| **cuTile** | 43,080 ms | 2,954 ms | **14.6x** |

### cuTile kernel breakdown

**DGX Spark (GB10):**

| Kernel | CUDA Time (ms) | Calls |
|--------|----------------|-------|
| `fmha_kernel` | 4,185.9 | 28 |
| `swiglu_forward_kernel` | 2,459.8 | 1,400 |
| `attention_decode_kernel_grouped` | 2,271.8 | 1,372 |
| `rms_norm_kernel_static_persistent` | 634.7 | 57 |
| `rope_kernel` | 355.6 | 1,400 |

**B300:**

| Kernel | CUDA Time (ms) | Speedup vs Spark |
|--------|----------------|------------------|
| `fmha_kernel` | 337.9 | 12.4x |
| `swiglu_forward_kernel` | 226.3 | 10.9x |
| `attention_decode_kernel_grouped` | 111.0 | 20.5x |
| `rms_norm_kernel_static_persistent` | 29.7 | 21.4x |
| `rope_kernel` | 16.7 | 21.3x |

**Same code, different architectures** — cuTile JIT compiles for sm_121 (DGX Spark) and sm_103 (B300).

## Platform specifications

| Specification | DGX Spark (GB10) | B300 |
|---------------|------------------|------|
| Compute capability | sm_121 (12.1) | sm_103 (10.3) |
| SMs | 48 | 132 |
| Memory | 128 GB LPDDR5x | 192 GB HBM3e |
| Memory bandwidth | 273 GB/s | 8 TB/s |

## Troubleshooting

| Symptom | Hardware platform | Cause | Fix |
|---------|-------------------|-------|-----|
| `docker: permission denied` | All hardware platforms | User not in docker group | `sudo usermod -aG docker $USER && newgrp docker` |
| `401 Client Error: Unauthorized` | All hardware platforms | Missing HuggingFace token | `export HF_TOKEN=<your_token>` |
| `ModuleNotFoundError: tilegym` | All hardware platforms | TileGym not installed | `cd TileGym && pip install .` |
| `RuntimeError: CUDA out of memory` | All hardware platforms | Model too large | Reduce batch size or use a smaller model |
| `Killed` during model load | All hardware platforms | Out of system memory | Clear cache: `sync; echo 3 > /proc/sys/vm/drop_caches` |
| Slow first run | All hardware platforms | JIT compilation | Normal — cuTile compiles kernels on first run |
| `FileNotFoundError: input_prompt_small.txt` | All hardware platforms | Missing input file | Run from `modeling/transformers` directory |
| `torch.cuda.OutOfMemoryError` | All hardware platforms | Insufficient GPU memory | Reduce `--batch_size` parameter |
| `ImportError: cuda.tile` | All hardware platforms | Missing Tile IR | Install: `apt-get install cuda-tile-ir-13-2` |
| Benchmark hangs | All hardware platforms | GPU busy or locked | Check `nvidia-smi` for other processes |
| Memory pressure within capacity | DGX Spark | UMA buffer cache not released | See UMA note below |

> [!NOTE]
> **Unified memory (UMA).** On hardware platforms with unified memory, GPU and CPU share memory dynamically. Some applications have not yet been updated for UMA, so you may hit memory issues even within capacity. If that happens, manually flush the buffer cache:
> ```bash
> sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
> ```

> [!TIP]
> First run of cuTile kernels includes JIT compilation overhead. Subsequent runs are faster as compiled kernels are cached.

For latest known issues, see the documentation linked under **Resources** for your hardware platform.
