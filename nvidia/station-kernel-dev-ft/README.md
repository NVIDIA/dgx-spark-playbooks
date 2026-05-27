# Profiler-Driven Kernel Optimization for Fine-Tuning

> Use torch.profiler to find training bottlenecks, then write custom Triton kernels to optimize LLaMA 8B fine-tuning


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

DGX Station puts a full Blackwell GPU on your desk, which makes it an ideal environment for profiling and optimizing GPU kernels used during model training. This playbook walks through a real optimization workflow: **profiling a LLaMA 3.1 8B fine-tuning run to identify bottlenecks, then writing custom Triton kernels that eliminate those bottlenecks** — specifically a fused RMSNorm and a fused cross-entropy loss using online softmax.

For inference workloads, tools like `torch.compile` and serving frameworks (vLLM, TensorRT-LLM) already ship highly optimized fused kernels. But training workloads are different. Backward passes double the kernel count, large vocabularies create massive intermediate tensors during loss computation, and `torch.compile` does not restructure algorithms to avoid these allocations. Projects like [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) and [Unsloth](https://github.com/unslothai/unsloth) demonstrate that custom training kernels deliver real results: 20-60% memory reduction and 10-30% throughput improvement.

This playbook uses **Triton** instead of raw CUDA C++. Triton is a Python-native GPU programming language that JIT-compiles to optimized GPU code — no `nvcc` compiler, no C++ build systems, no manual thread indexing. It is the standard for custom training kernels: Liger-Kernel, Unsloth, and FlashAttention are all written in Triton.

**No prior Triton, CUDA, or GPU programming experience is required.** The instructions explain each concept as it comes up.

## What you'll accomplish

You will profile a LLaMA 3.1 8B fine-tuning workload, identify the key performance bottlenecks, and write custom Triton kernels that address them.

- **Profile** a baseline fine-tuning step using `torch.profiler` and interpret the results to identify two targets: RMSNorm (memory-bandwidth-bound) and cross-entropy loss (memory-capacity-bound).
- **Write a fused RMSNorm kernel** in Triton that processes normalization in a single GPU pass instead of multiple separate operations, improving memory bandwidth utilization from ~11% to ~80-90% of peak.
- **Write a fused cross-entropy kernel** using the online softmax algorithm (Milakov-Gimelshein) that computes loss without materializing intermediate softmax tensors, achieving ~6x memory reduction and up to 4x latency improvement at realistic batch sizes.
- **Verify correctness** of both kernels (forward and backward passes) against PyTorch reference implementations.
- **Benchmark** the kernels to measure latency, throughput, and memory savings.
- **Integrate** both kernels into an end-to-end LLaMA 3.1 8B fine-tuning loop and measure real training throughput and memory improvements.

## What to know before starting

- Comfortable with Linux command line and shell scripting.
- Basic familiarity with Python and PyTorch (tensors, autograd, training loops).
- Understanding of what fine-tuning is (training a pre-trained model on new data).
- No Triton, CUDA, or GPU programming experience required — all code is explained.

## Prerequisites

**Hardware:**
- NVIDIA DGX Station with GB300 Ultra Superchip.
- At least 150 GB available storage for the container image, model weights (~16 GB for LLaMA 3.1 8B in BF16), profiler traces, and optimizer states.

**Software:**
- Docker with NVIDIA Container Toolkit: `docker run --rm --gpus all nvcr.io/nvidia/cuda:12.8.0-devel-ubuntu24.04 nvidia-smi`
- On a DGX Station, immediately confirm which device index belongs to the GB300 so later steps can target it explicitly. Run `nvidia-smi --query-gpu=index,name --format=csv,noheader` and note the index for the row showing `NVIDIA GB300`. Subsequent steps recommend `--gpus '"device=N"'` (with `N` = that index) instead of `--gpus all` so profiling and benchmark numbers stay on a single, known GPU.
- Network access to pull container images from NGC and download model weights from Hugging Face.
- A Hugging Face account with access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B) and a [Hugging Face access token](https://huggingface.co/settings/tokens).

## Ancillary files

All required assets are in the playbook directory `nvidia/station-kernel-dev-ft/assets` (see the [dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks) repository).

- `assets/Dockerfile` — Development container based on NVIDIA's PyTorch NGC image with Triton, transformers, and profiling dependencies.
- `assets/requirements.txt` — Python dependencies installed inside the container.
- `assets/profile_baseline.py` — Profiling script that captures a `torch.profiler` trace of a LLaMA 3.1 8B training step and prints a breakdown of GPU time by operation. Supports flags to enable custom kernels for re-profiling.
- `assets/rmsnorm_kernel.py` — Fused RMSNorm Triton kernel with forward and backward passes, wrapped as a drop-in `torch.nn.Module` replacement. Heavily commented with explanations of each Triton concept.
- `assets/rmsnorm_test.py` — Correctness tests comparing the custom RMSNorm against PyTorch's reference implementation (forward and backward, FP32 and BF16).
- `assets/cross_entropy_kernel.py` — Fused cross-entropy Triton kernel using online softmax, with forward and backward passes. Processes the vocabulary in chunks to avoid materializing the full logit tensor.
- `assets/cross_entropy_test.py` — Correctness tests and memory usage comparison against `torch.nn.CrossEntropyLoss`.
- `assets/benchmark_kernels.py` — Benchmarking script that measures latency, throughput, bandwidth utilization, and peak memory for both custom kernels.
- `assets/finetune_baseline.py` — Minimal LLaMA 3.1 8B fine-tuning script using vanilla PyTorch, reporting tokens/sec and peak memory.
- `assets/finetune_optimized.py` — Identical fine-tuning script with both custom kernels monkey-patched in for direct comparison.

## Time & risk

* **Estimated time:** About 2 hours. Steps 1-4 (setup through baseline profiling) take about 30 minutes. Steps 5-7 (RMSNorm kernel) take about 30 minutes. Steps 8-10 (cross-entropy kernel) take about 40 minutes. Step 11 (end-to-end integration) takes about 20 minutes. Steps 12-13 (cleanup and next steps) are a few minutes.
* **Risk level:** Low
  * All work runs inside a Docker container — no host system modifications.
  * LLaMA 3.1 8B model weights (~16 GB in BF16) are downloaded from Hugging Face on first run and cached locally.
  * Requires a Hugging Face token with access to the LLaMA 3.1 model.
* **Rollback:** Exit the container. Your source files are preserved in the mounted `assets/` directory; everything else is discarded.
* **Last Updated:** 05/26/2026
  * First publication

## Instructions

## Step 1. Set up the development environment

Before profiling or writing any GPU kernels, you need a development environment with PyTorch, Triton (the GPU programming language we'll use), and the tools to load LLaMA 3.1 8B. We use a Docker container so everything is pre-configured and isolated from your host system.

Clone the playbook repository and navigate to the assets directory:

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/station-kernel-dev-ft/assets
```

Build the development container. This creates a Docker image based on NVIDIA's PyTorch NGC container with additional libraries for model loading and benchmarking:

```bash
docker build -t kernel-dev-ft .
```

Identify the GB300's device index so the container can target it explicitly. On multi-GPU DGX Station systems, pinning to a single, known GPU keeps profiling and benchmark numbers consistent across runs:

```bash
nvidia-smi --query-gpu=index,name --format=csv,noheader
```

Look for the row showing `NVIDIA GB300` and note its index (commonly `0` or `1`). Use that value as `N` in the next command.

Start the container with GPU access. Pass your Hugging Face token so the container can download LLaMA 3.1 8B:

```bash
## Replace N with the GB300 index from the command above.
## On a single-GPU Station you may substitute --gpus all.
docker run -it --rm \
  --name kernel-dev-ft \
  --gpus '"device=N"' \
  --ipc host \
  -e HF_TOKEN=$HF_TOKEN \
  -v "$(pwd):/workspace" \
  -v ~/.cache/huggingface:/root/.cache/huggingface \
  -w /workspace \
  kernel-dev-ft
```

> [!NOTE]
> The `-v "$(pwd):/workspace"` flag mounts the current directory into the container. Any files you create or modify inside `/workspace` persist on your host machine after the container exits. The `-v ~/.cache/huggingface:/root/.cache/huggingface` mount persists downloaded model weights across container restarts so you don't need to re-download the 16 GB model each time. Everything outside these mounted paths is discarded when the container stops.

> [!IMPORTANT]
> Targeting the GB300 explicitly with `--gpus '"device=N"'` (rather than `--gpus all`) ensures `torch.cuda` and `nvidia-smi` inside the container both see the **GB300** as device `0`. Profiling and benchmark numbers later in this playbook assume a single Blackwell GPU; mixing a workstation GPU in via `--gpus all` can change scheduling and skew tokens/sec and bandwidth utilization figures.

> [!NOTE]
> If you haven't set `HF_TOKEN` in your shell, export it first: `export HF_TOKEN=hf_your_token_here`. You need a Hugging Face token with access to [meta-llama/Llama-3.1-8B](https://huggingface.co/meta-llama/Llama-3.1-8B). You must first accept the LLaMA 3.1 Community License Agreement on the [model page](https://huggingface.co/meta-llama/Llama-3.1-8B) before your token can download the weights.

Verify the toolchain inside the container:

```bash
python -c "import triton; print(f'Triton {triton.__version__}')"
python -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')"
nvidia-smi --query-gpu=name,compute_cap --format=csv,noheader
```

Expected output should show:
- Triton version 3.0 or later
- PyTorch with CUDA support enabled
- A Blackwell GPU with a `10.x` compute capability. The exact minor version depends on the SKU — for example, `nvidia-smi` reports **`10.0`** on GB200 / B200 (standard Blackwell) and **`10.3`** on GB300 / B300 (Blackwell Ultra; same GPU silicon, different host packaging). Any `10.x` value is fine for this playbook; the kernels target the Blackwell family, not a specific minor.

> [!NOTE]
> Unlike the CUDA C++ workflow (which requires the `nvcc` compiler and a separate compilation step), Triton is a Python library that JIT-compiles GPU code at runtime. There is no build step — you write Python, and Triton compiles it to optimized GPU machine code when you first call the kernel.

## Step 2. Understand the fine-tuning workload

Before profiling, let's build a mental model of where GPU time goes during LLaMA 3.1 8B fine-tuning and why certain operations are candidates for custom kernels.

**LLaMA 3.1 8B architecture at a glance:**

| Property | Value |
|----------|-------|
| Parameters | 8.03 billion |
| Layers | 32 transformer blocks |
| Hidden size | 4096 |
| Attention heads | 32 |
| Key/value heads | 8 (grouped-query attention) |
| Vocabulary | 128,256 tokens |
| Normalization | RMSNorm (not LayerNorm) |
| Activation | SwiGLU (SiLU-gated MLP) |

**Memory budget for full fine-tuning in BF16:**
- Model weights: 8B params x 2 bytes = ~16 GB
- AdamW optimizer states: 8B params x 8 bytes (FP32 copy + first moment + second moment) = ~64 GB
- Gradients: 8B params x 2 bytes = ~16 GB
- Activations: varies with batch size and sequence length
- **Total: ~96 GB minimum**, fitting comfortably in DGX Station's 252 GB HBM3e

**Why training is different from inference for kernel optimization:**

For inference, `torch.compile` and serving frameworks like vLLM already fuse most pointwise operations automatically. Writing a custom SiLU or SwiGLU kernel for inference is reinventing what's already solved.

Training is different for three reasons:
1. **Backward passes double the kernel count.** Every forward operation has a corresponding backward operation for gradient computation. `torch.compile` handles some of these but cannot restructure algorithms (like how loss is computed).
2. **Large vocabularies create massive intermediate tensors.** LLaMA's 128K vocabulary means the logit tensor for a single batch is enormous. Standard cross-entropy materializes this entire tensor in memory.
3. **Memory is the binding constraint.** Unlike inference (where latency matters most), training is often limited by how much data fits in GPU memory. Kernels that reduce memory enable larger batch sizes, which improve GPU utilization across *all* operations.

**Memory-bound vs compute-bound (where to spend effort):**

- **Memory-bound** regions are limited by how fast you can move bytes through HBM (read/write bandwidth). Symptoms: small kernels, low achieved GB/s vs peak, profiler shows many narrow ops or fusion gaps. **Optimize** by fusing passes, reducing tensor materialization, using narrower dtypes where safe, and improving coalescing so each byte does more useful math.
- **Compute-bound** regions are limited by arithmetic throughput (Tensor Cores, FP32/FP16 units). Symptoms: large `aten::mm` / matmul and attention dominating self CUDA time with high utilization. **Optimize** with better tiling, larger batch sizes (more work per launch), kernels that keep math in registers, and libraries (cuBLAS, FlashAttention) before hand-writing alternatives.

A single training step usually mixes both: matmuls tend toward **compute-bound** on large batches, while pointwise norm/loss paths are often **memory-bound**. Profiling tells you which bucket your hotspot falls into.

## Step 3. Profile a baseline training step

Now let's see where GPU time actually goes. We'll use `torch.profiler` to capture a detailed trace of a single forward + backward + optimizer step.

Run the profiling script:

```bash
python profile_baseline.py
```

> [!NOTE]
> The first run downloads LLaMA 3.1 8B weights (~16 GB in BF16) from Hugging Face. This takes several minutes depending on network speed. Subsequent runs use the cached weights and start immediately.

> [!NOTE]
> **Repeat runs:** `profile_baseline.py` removes any prior trace directory and Chrome JSON for the same flags before recording, so you can re-run baseline profiling without a "trace is already saved" error.

> [!NOTE]
> **Ranking variance:** The exact ordering and percentages in the "Top 20 CUDA operations" table can change between runs, PyTorch / CUDA versions, and GPU generation. You should still see the same *categories* of work (matmuls, FlashAttention, RMSNorm decompositions, cross-entropy). **"Command Buffer Full"** (or similar) sometimes appears at the top of self-time tables: that reflects the GPU driver's **submission queue / scheduling**, not a user kernel to optimize. The script filters that row from the printed table; the raw trace in Perfetto still contains the underlying kernels.

> [!TIP]
> **Optional Nsight Systems timeline:** For a visual timeline with CUDA API and GPU work (outside or alongside `torch.profiler`), install [NVIDIA Nsight Systems](https://developer.nvidia.com/nsight-systems) and run something like: `nsys profile -o llama_ft_repro --trace=cuda,nvtx python profile_baseline.py` from an environment where `nsys` is on `PATH` (often the host, or a devel image with CUDA toolkit). Open the `.nsys-rep` file in the Nsight Systems GUI.

The script loads LLaMA 3.1 8B, runs one training step under `torch.profiler`, and prints a table like this:

```
======================================================================
  Top 20 CUDA Operations by Total GPU Time
======================================================================
Name                              Self CUDA   Self CUDA %   # Calls
------------------------------------  ----------  -----------  --------
aten::mm                             152.3ms       42.1%        258
aten::_flash_attention_forward        48.7ms       13.5%         32
aten::_flash_attention_backward       41.2ms       11.4%         32
aten::_scaled_mm                      28.1ms        7.8%          2
aten::pow                             12.4ms        3.4%         65
aten::mean                            11.8ms        3.3%         65
aten::rsqrt                            8.2ms        2.3%         65
aten::mul                             15.6ms        4.3%        198
aten::_log_softmax                     8.9ms        2.5%          1
aten::nll_loss_forward                 3.2ms        0.9%          1
...
```

**How to read these results:**

- **`aten::mm`** (matrix multiplications): The largest single category (~42% of GPU time). These are already highly optimized by cuBLAS. Not a target for custom kernels.
- **`aten::_flash_attention_forward/backward`** (~25% combined): Already optimized by FlashAttention. Not a target.
- **`aten::pow`, `aten::mean`, `aten::rsqrt`, and some `aten::mul` calls**: These are the **RMSNorm** operations, broken into separate kernels. Individually small, but there are many of them (32 layers x 2 norms per layer + 1 model norm = 65 in the forward pass, plus corresponding backward operations). In aggregate, they consume significant time and make many redundant memory round-trips.
- **`aten::_log_softmax` + `aten::nll_loss_forward`**: This is the **cross-entropy loss** computation. Only called once, but it operates over the full `[batch_size * seq_len, 128256]` logit tensor.

The profiler also saves a Chrome trace file. You can inspect it visually:

> [!TIP]
> Open the Chrome trace JSON in [Perfetto UI](https://ui.perfetto.dev/) for an interactive timeline view. Look for sequences of narrow bars (small kernels with gaps between them) — these represent unfused operations where the GPU reads and writes the same data multiple times.

## Step 4. Understand why these operations are slow

Before writing kernels, let's understand *why* our two targets are slow. This understanding will guide the kernel design.

**RMSNorm is memory-bandwidth-bound.**

The formula for RMSNorm is:

```
RMSNorm(x) = (x / sqrt(mean(x^2) + eps)) * weight
```

PyTorch's default implementation breaks this into separate GPU operations:
1. `x.pow(2)` — square each element → writes result to memory
2. `.mean(-1)` — reduce across hidden dimension → reads result, writes mean
3. `+ eps` then `.rsqrt()` — reads mean, writes inverse RMS
4. `x * rnorm` — reads x again, reads rnorm, writes normalized output
5. `* weight` — reads output, reads weight, writes final result

Each of these reads from and writes to GPU memory (HBM). For `hidden_size=4096` in BF16, a single row is 8 KB. The unfused version reads and writes this data **5+ times**. A fused kernel reads it **once** and writes **once**.

The DGX Station GB300's HBM3e has ~8 TB/s of bandwidth. PyTorch's unfused RMSNorm typically achieves only ~11% of this peak. A fused kernel can reach ~80-90% — a dramatic improvement for an operation that runs 66 times per training step.

**Cross-entropy is memory-capacity-bound.**

Standard cross-entropy computes `softmax(logits)` over the full vocabulary for every token position. For LLaMA 3.1 8B:

```
logit tensor shape: [batch_size * seq_len, 128256]
For batch_size=1, seq_len=512: [512, 128256]
Memory: 512 * 128256 * 4 bytes (float32) ≈ 250 MB
```

PyTorch also saves the softmax output for the backward pass, roughly doubling this to ~500 MB. As batch size or sequence length grows, this scales linearly.

The **online softmax** trick (Milakov & Gimelshein, 2018) avoids materializing the full logit tensor. Instead of computing softmax all at once, it processes the vocabulary in chunks while maintaining two running values:
- **`m`**: the running maximum logit (for numerical stability)
- **`d`**: the running sum of `exp(logit - m)` (the softmax denominator)

Here's the algorithm with a small example. Suppose we have 8 logits `[2, 5, 1, 3, 4, 7, 2, 6]` and process them in chunks of 4:

**Chunk 1: `[2, 5, 1, 3]`**
- `m = 5` (max of chunk)
- `d = exp(2-5) + exp(5-5) + exp(1-5) + exp(3-5) = 0.050 + 1.0 + 0.018 + 0.135 = 1.203`

**Chunk 2: `[4, 7, 2, 6]`**
- `chunk_max = 7`, `new_m = max(5, 7) = 7`
- Rescale previous `d`: `d = 1.203 * exp(5 - 7) + exp(4-7) + exp(7-7) + exp(2-7) + exp(6-7)`
- `d = 1.203 * 0.135 + 0.050 + 1.0 + 0.007 + 0.368 = 1.587`

After all chunks: `loss = log(d) + m - logit[target]`. No `[8]`-sized softmax tensor was ever allocated — just two scalars (`m`, `d`) maintained across chunks.

For `V=128256`, this reduces the *algorithmic* memory from `O(B*T*V)` to `O(B*T)` per row. In practice, the input logit tensor is still allocated (PyTorch needs it for the backward pass), so the measured end-to-end reduction is ~6x — still a significant saving that frees hundreds of megabytes at realistic batch sizes.

## Step 5. Write the fused RMSNorm Triton kernel

Let's start with the simpler kernel. Open `rmsnorm_kernel.py` to review the implementation:

```bash
cat rmsnorm_kernel.py
```

This file contains four components:
1. **`_rmsnorm_fwd_kernel`** — The forward pass Triton kernel
2. **`_rmsnorm_bwd_kernel`** — The backward pass Triton kernel
3. **`TritonRMSNormFunction`** — A `torch.autograd.Function` that connects the kernels to PyTorch's autograd
4. **`TritonRMSNorm`** — A drop-in `nn.Module` replacement for `LlamaRMSNorm`

**Key Triton concepts in the forward kernel:**

```python
@triton.jit
def _rmsnorm_fwd_kernel(X_ptr, W_ptr, Y_ptr, Rnorm_ptr, stride_x, hidden_size, eps, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    ...
```

- **`@triton.jit`** marks a function for GPU compilation. This is Triton's equivalent of CUDA's `__global__` keyword, but instead of writing C++, you write Python-like code. Triton's compiler handles thread management, memory coalescing, and vectorization automatically.

- **`tl.program_id(0)`** returns a unique index for each "program" (similar to a CUDA thread block). Each program handles one row of the input tensor. For a batch of 512 tokens with hidden_size=4096, we launch 512 programs.

- **`tl.load(X_ptr + row_start + offsets, mask=mask, other=0.0)`** loads a vector of values from GPU memory into registers. The `mask` ensures we don't read beyond the row boundary. The `other=0.0` provides a default value for masked-out elements.

- **`BLOCK_SIZE: tl.constexpr`** is a compile-time constant. Triton generates specialized GPU code for each value of `BLOCK_SIZE`. For `hidden_size=4096`, we use `BLOCK_SIZE=4096` (the next power of 2), meaning each program loads the entire row in one batch.

**The key optimization:**

```python
## One pass: read x, compute variance, normalize, multiply by weight, write y
x_fp32 = x.to(tl.float32)
variance = tl.sum(x_fp32 * x_fp32, axis=0) / hidden_size
rnorm = 1.0 / tl.sqrt(variance + eps)
y = (x_fp32 * rnorm).to(x.dtype) * w
tl.store(Y_ptr + row_start + offsets, y, mask=mask)
```

The entire RMSNorm computation — square, mean, rsqrt, normalize, scale by weight — happens in registers without intermediate writes to GPU memory. Compare this to PyTorch's 5 separate kernel launches, each with a full memory round-trip.

**The backward kernel** follows the same pattern: load everything needed for one row, compute both `grad_x` and `grad_w` in registers, write once. The mathematical derivation is documented in the kernel source comments.

**The autograd wrapper** (`TritonRMSNormFunction`) connects the kernels to PyTorch's automatic differentiation:
- `forward()` calls the forward kernel and saves `x`, `weight`, and `rnorm` for later.
- `backward()` receives the upstream gradient, calls the backward kernel, and returns gradients for `x` and `weight`.

> [!NOTE]
> **Triton vs. CUDA C++:** In the [Custom CUDA Kernel Development](https://build.nvidia.com/nvidia/station-kernel-dev) playbook, we wrote CUDA C++ with explicit thread indexing (`blockIdx.x`, `threadIdx.x`), manual `float4` vectorization, `nvcc` compilation, and `ctypes` bindings. Triton abstracts all of that — you write Python-like code, and the compiler handles vectorization, memory coalescing, and PTX generation automatically. The tradeoff is less fine-grained hardware control, but for operations like RMSNorm, Triton matches hand-tuned CUDA performance.

## Step 6. Test RMSNorm for correctness

Before measuring performance, verify the kernel produces the same results as PyTorch's implementation. Even small numerical errors can cascade through a 32-layer transformer and produce garbage gradients.

Run the correctness tests:

```bash
python rmsnorm_test.py
```

Expected output:

```
RMSNorm Correctness Tests
============================================================

Test 1: Float32
  FP32 Forward  — max diff: 9.54e-07  PASSED
  FP32 Backward (dx)  — max diff: 1.43e-06  PASSED
  FP32 Backward (dw)  — max diff: 2.29e-05  PASSED

Test 2: BFloat16 (relaxed tolerance)
  BF16 Forward  — max diff: 1.56e-02  PASSED
  BF16 Backward (dx)  — max diff: 1.56e-02  PASSED
  BF16 Backward (dw)  — max diff: 5.00e-01  PASSED

============================================================
All RMSNorm correctness tests PASSED
```

The tests compare the custom kernel against PyTorch's reference `LlamaRMSNorm` at shapes matching LLaMA 3.1 8B (`batch=4, seq_len=512, hidden_size=4096`), testing both the forward output and the backward gradients for `x` and `weight`.

> [!WARNING]
> BF16 has only 7 bits of mantissa (vs. 23 for FP32). Per-element differences of ~0.01-0.02 are normal for forward and `grad_x`. The **weight gradient** (`dw`) shows larger absolute differences (up to ~0.5) because it sums per-element contributions across all 2,048 token positions — different summation order between our FP32-accumulated kernel and PyTorch's autograd produces BF16 rounding differences that accumulate. The test uses relaxed tolerance for `dw` to account for this.

The FP32 test uses tolerance `atol=1e-4` and the BF16 test uses `atol=1e-2` for per-element values, with a more relaxed threshold for the accumulated weight gradient. Both forward and backward must pass — many kernel bugs only manifest in the backward pass.

## Step 7. Benchmark and re-profile RMSNorm

Now let's measure the performance improvement. Run the RMSNorm benchmark:

```bash
python benchmark_kernels.py --kernel rmsnorm
```

Example output:

```
======================================================================
  RMSNorm Benchmark — Custom Triton vs. PyTorch Reference
======================================================================
  GPU: NVIDIA GB300

Tokens    Custom (us)    PyTorch (us)    Custom (GB/s)    PyTorch (GB/s)    Speedup
--------  -------------  --------------  ---------------  ----------------  ---------
256             313.5           479.6               40                 26  1.53x
1,024           313.6           495.8              161                102  1.58x
4,096           319.4           576.5              630                349  1.80x
16,384          298.9         2,041.7            2,694                394  6.83x
```

**How to read these results (what "better" means):**

| Column / metric | Better when… |
|-----------------|---------------|
| **Custom (us)** | **Lower** is faster (fewer microseconds per forward+backward pass). |
| **PyTorch (us)** | Reference only; same rule (lower is faster). |
| **Custom (GB/s)** | **Higher** means you move closer to HBM peak (more useful bytes per second for this fused region). |
| **Speedup** | **Higher** means the custom kernel beats PyTorch by a larger factor on that row. |

- **Custom (GB/s)** shows effective memory bandwidth. On large inputs (16K tokens), the fused kernel typically reaches much higher GB/s than the unfused PyTorch path.
- **Speedup** often ranges from roughly **1.5x** on small inputs to **6x+** on large inputs in internal runs.
- These numbers measure **forward + backward combined**, which is what matters for training.

> [!NOTE]
> **Treat the table as illustrative, not a target.** Absolute microsecond and GB/s values **can differ by an order of magnitude** between GB300 stacks (different driver versions, NGC PyTorch builds, clock states, autograd overhead between iterations). On the validation run for this playbook the same shapes measured ~4,000–5,000 µs per fwd+bwd instead of ~300 µs, while still showing **custom faster than PyTorch and the gap widening with `num_tokens`**. The **direction of the speedup** (and the GB/s ratio between custom and PyTorch in the same run) is the stable signal — match those, and your kernel is healthy.

Now re-profile the full training step with the custom RMSNorm to confirm the bottleneck is resolved:

```bash
python profile_baseline.py --use-custom-rmsnorm
```

Compare the profiler output to Step 3. The `aten::pow`, `aten::mean`, and `aten::rsqrt` calls from RMSNorm should be gone, replaced by fewer, faster Triton kernel calls. The remaining top operations should be matrix multiplications and FlashAttention — operations already handled by highly optimized libraries.

## Step 8. Write the fused cross-entropy Triton kernel

Now for the more complex kernel. Open `cross_entropy_kernel.py`:

```bash
cat cross_entropy_kernel.py
```

This implements the online softmax algorithm from Step 4 as a Triton kernel. The structure mirrors the RMSNorm kernel (forward kernel, backward kernel, autograd function, nn.Module), but the forward kernel is more complex because it loops over the vocabulary in chunks.

**The forward kernel, annotated:**

```python
@triton.jit
def _cross_entropy_fwd_kernel(Logits_ptr, Targets_ptr, Losses_ptr, Max_ptr, Denom_ptr,
                               vocab_size, stride_logits, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    ...
    m = float("-inf")   # Running maximum logit
    d = 0.0             # Running sum of exp(logit_i - m)
    target_logit = 0.0  # Logit at the target index

    for start in range(0, vocab_size, BLOCK_SIZE):
        ...
```

Key differences from the RMSNorm kernel:

- **Loop over vocabulary chunks.** The RMSNorm kernel loads the entire row at once (4096 elements fits in registers). The cross-entropy kernel can't do that — 128,256 vocabulary entries is too large. Instead, it processes `BLOCK_SIZE` elements at a time (e.g., 4096 per iteration, 32 iterations total). Triton unrolls this loop for efficiency.

- **Running state across iterations.** The kernel maintains `m` (running max) and `d` (running sum-of-exp) across loop iterations. The update rule handles the rescaling when a new maximum is found:

  ```python
  new_m = tl.maximum(m, chunk_max)
  d = d * tl.exp(m - new_m) + tl.sum(tl.exp(logits_chunk - new_m), axis=0)
  m = new_m
  ```

  The `d * tl.exp(m - new_m)` term rescales the previous sum to account for a potentially larger maximum. This is the core of the online softmax algorithm.

- **No intermediate tensor allocation.** The standard approach would allocate a `[num_tokens, 128256]` tensor for the softmax output. This kernel only stores three scalars per row (`loss`, `m`, `d`) plus the target logit.

**The backward kernel** also loops over the vocabulary in chunks. For each chunk, it computes `softmax(logit) = exp(logit - m) / d` using the saved `m` and `d` values, subtracts 1 at the target position, and writes the gradient. Like the forward kernel, it never materializes the full softmax vector.

> [!TIP]
> This kernel is inspired by the [Liger-Kernel](https://github.com/linkedin/Liger-Kernel) project from LinkedIn. Liger-Kernel also implements a more advanced variant called **Fused Linear Cross-Entropy** that fuses the final linear projection (`hidden_states @ lm_head_weight`) with the cross-entropy loss, computing logits chunk-by-chunk and never materializing them at all. This is even more memory-efficient but significantly more complex (it requires tiled matrix multiplication within the kernel). See the Next Steps section for pointers.

## Step 9. Test cross-entropy for correctness

Run the correctness and memory tests:

```bash
python cross_entropy_test.py
```

Expected output:

```
Cross-Entropy Correctness Tests
============================================================

Test 1: Float32
  FP32 Loss      — ref: 12.331120  custom: 12.331120  diff: 0.00e+00  PASSED
  FP32 Gradient  — max diff: 9.09e-13  PASSED

Test 2: BFloat16 (relaxed tolerance)
  BF16 Loss      — ref: 12.250000  custom: 12.247243  diff: 2.76e-03  PASSED
  BF16 Gradient  — max diff (fp32 compare): 1.23e-01  PASSED

Memory Comparison
------------------------------------------------------------
  Standard PyTorch CE — peak memory: 504.0 MB
  Fused Triton CE     — peak memory: 252.0 MB
  Memory reduction: 2.0x

============================================================
All cross-entropy tests PASSED
```

The **memory comparison** shows that standard PyTorch cross-entropy allocates ~500 MB (for the softmax output and other intermediates), while the fused kernel uses ~250 MB. The 2x reduction measured here understates the real benefit: in the benchmark (Step 10), where memory is measured more precisely per-operation, the reduction is **~6x**. The larger benefit appears because the benchmark isolates just the cross-entropy overhead, while this test includes the base logit tensor allocation in both measurements.

> [!NOTE]
> Cross-entropy involves `log(sum(exp(...)))`, which is numerically sensitive. The online softmax algorithm maintains stability through the running-max trick — subtracting the maximum logit before exponentiating prevents overflow. FP32 checks use tight tolerances. **BF16** compares loss with relaxed `atol/rtol` and compares **gradients in float32** with wider tolerances (`atol=2e-1`, `rtol=2e-1`) so chunked reductions over 128K vocabulary do not false-fail against PyTorch's different accumulation order.

> [!WARNING]
> BF16 tolerances are intentionally looser than FP32: they assert the custom kernel matches the reference **within training-usable error**, not bitwise. Tighten tolerances only if you change the algorithm or dtype strategy.

## Step 10. Benchmark and re-profile cross-entropy

Run the cross-entropy benchmark:

```bash
python benchmark_kernels.py --kernel cross_entropy
```

Example output:

```
======================================================================
  Cross-Entropy Benchmark — Custom Triton (online softmax) vs. PyTorch
======================================================================
  GPU: NVIDIA GB300
  Vocabulary size: 128,256 (LLaMA 3.1)

Tokens    Custom (us)    PyTorch (us)    Speedup    Custom Mem (MB)    PyTorch Mem (MB)    Mem Reduction
--------  -------------  --------------  ---------  -----------------  ------------------  ---------------
128               311             220     0.71x                 32                   188  5.9x
256               300             338     1.12x                 63                   378  6.0x
512               306             676     2.21x                126                   752  6.0x
1,024             315           1,277     4.06x                251                 1,506  6.0x
```

**How to read these results:** For latency columns, **lower microseconds is better**. For **Speedup**, **higher is better** (custom faster than PyTorch). For **Mem Reduction**, **higher is better** (more peak memory saved).

- **Speedup** grows from slower at 128 tokens (kernel launch overhead dominates) to several times faster at 1,024 tokens in typical runs.
- **Memory reduction** (~6x in the table): PyTorch allocates separate tensors for the logits, softmax output, and loss gradients. The fused kernel avoids the softmax intermediary. For 1,024 tokens, this saves over 1 GB of GPU memory, room for larger batches or longer sequences.
- At very small token counts (128), the fused kernel can be **slower**. That is expected: the online softmax loop has fixed per-row overhead. The crossover is often near 256–1,024 tokens depending on stack.

> [!NOTE]
> Same caveat as in Step 7: **absolute microseconds in the example table are illustrative**. On the validation run for this playbook the per-iteration latencies were several thousand µs rather than the ~300 µs printed above, while the **memory reduction** (~6x) and the **speedup direction** (fused becomes faster as `num_tokens` grows) remained stable. Trust the **shape** of the table and the **memory column**, not the absolute latency numbers.

Now re-profile with both custom kernels active:

```bash
python profile_baseline.py --use-custom-rmsnorm --use-custom-ce
```

The profiler output should now show matrix multiplications and FlashAttention as the dominant operations. The RMSNorm and cross-entropy bottlenecks from Step 3 have been eliminated. The remaining operations are already handled by cuBLAS and FlashAttention — the most highly optimized GPU libraries available.

## Step 11. Run end-to-end fine-tuning with custom kernels

Let's put it all together: run a real fine-tuning loop and measure the end-to-end impact.

First, run the baseline (vanilla PyTorch):

```bash
python finetune_baseline.py
```

Then run the optimized version with both custom kernels:

```bash
python finetune_optimized.py
```

Example comparison on **GB300** (default `--batch-size 1`, `--seq-len 512`; throughput is `batch * seq_len / step_time`; numbers below are **illustrative**, not a target):

```
======================================================================
  Baseline Results
======================================================================
  Average time per step:  0.201 s
  Average throughput:     2540 tokens/sec (illustrative)
  Peak GPU memory:        112.4 GB

======================================================================
  Optimized Results (illustrative)
======================================================================
  Average time per step:  0.194 s
  Average throughput:     2640 tokens/sec (illustrative)
  Peak GPU memory:        78.6 GB
```

> [!NOTE]
> **Treat the throughput numbers above as illustrative, not a target** — same caveat as the RMSNorm (Step 7) and cross-entropy (Step 10) benchmark notes. Absolute tok/s and step time **vary** with GPU generation, clocks, PyTorch / CUDA builds, and whether the warm-up pass included JIT; older runs near **~280 tok/s** were observed on different stacks. The **stable signals** are (1) the **relative gap** — optimized > baseline — and (2) the **peak GPU memory delta** (the cross-entropy memory reduction is what frees room for larger batch sizes). Match those, not the absolute tok/s.

**How the custom kernels are integrated:**

The `finetune_optimized.py` script uses the "surgical replacement" pattern to swap in custom kernels without modifying the model source code:

```python
## Walk the model tree and collect every LlamaRMSNorm for replacement.
## We collect first, then apply — modifying the tree during iteration is unsafe.
replacements = []
for name, module in model.named_modules():
    if type(module).__name__ == "LlamaRMSNorm":
        parts = name.split(".")
        parent = model.get_submodule(".".join(parts[:-1])) if len(parts) > 1 else model
        replacements.append((parent, parts[-1], module))

for parent, attr_name, old_module in replacements:
    setattr(parent, attr_name, TritonRMSNorm.from_llama_rmsnorm(old_module))

## Use custom cross-entropy instead of the model's built-in loss
outputs = model(input_ids=input_ids)  # Forward without computing loss
logits = outputs.logits[:, :-1, :].contiguous()
loss = custom_ce(logits, labels[:, 1:].contiguous())
```

This pattern — find modules by type, create optimized replacements, swap them in — is widely used in production inference and training optimization.

> [!NOTE]
> **Amdahl's law in action.** An 8x faster RMSNorm does not make training 8x faster. If RMSNorm was 10% of total step time, making it 8x faster saves about 8.75% of total time. The cross-entropy memory reduction has an outsized impact because it frees GPU memory that enables larger batch sizes, which improves GPU utilization across *all* operations — including the matrix multiplications and attention that dominate the compute profile.

## Step 12. Cleanup

When you're finished, exit the container:

```bash
exit
```

Since we used `--rm`, the container is automatically removed. Your source code and profiler traces are preserved in the `assets/` directory on your host machine. Model weights are cached in `~/.cache/huggingface/` on the host (via the volume mount).

To remove the container image:

> [!WARNING]
> This deletes the built Docker image. You'll need to rebuild it if you want to use it again.

```bash
docker rmi kernel-dev-ft
```

To remove downloaded model weights cached by Hugging Face:

```bash
rm -rf ~/.cache/huggingface/hub/models--meta-llama--Llama-3.1-8B
```

## Step 13. Next steps

You profiled a real training workload, identified bottlenecks, shipped custom Triton kernels, and measured end-to-end impact. Continue from here with:

- **Fused Linear Cross-Entropy:** The kernel in this playbook takes pre-computed logits. A more advanced variant fuses the `lm_head` linear projection with the cross-entropy so logits are produced chunk-by-chunk and the full `[B*T, V]` tensor is never stored. See [Liger-Kernel's FusedLinearCrossEntropy](https://github.com/linkedin/Liger-Kernel).
- **Fused SwiGLU with backward:** The [Custom CUDA Kernel Development](https://build.nvidia.com/nvidia/station-kernel-dev) playbook covered inference-only SwiGLU. Training needs the backward pass; use the same `torch.autograd.Function` pattern as here.
- **Liger-Kernel integration:** `pip install liger-kernel` and `apply_liger_kernel_to_llama()`, then compare throughput to your hand-written kernels.
- **Larger batch sizes:** Fused cross-entropy frees memory. Re-profile with `--batch-size 2` or `--batch-size 4` to see utilization when more matmul work sits behind each step.
- **LoRA fine-tuning:** Re-run the profiling methodology on LoRA or QLoRA. Bottlenecks shift (fewer optimizer states, different activation pressure).
- **Multi-GPU training:** These kernels compose with FSDP and DDP unchanged (each rank runs its own Triton programs).

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `ModuleNotFoundError: No module named 'triton'` | Container missing Triton | Use the `kernel-dev-ft` container built from the playbook's Dockerfile. Triton ships with PyTorch NGC containers. Verify: `python -c "import triton; print(triton.__version__)"`. |
| `triton.compiler.errors.CompilationError` referencing `sm_100` | Triton version too old for Blackwell | Use PyTorch NGC container 26.01+ which includes Triton with Blackwell support. Check: `python -c "import triton; print(triton.__version__)"`. |
| Cross-entropy BF16 test fails on loss or gradient | BF16 + 128K vocab accumulate drift vs PyTorch's CE path | `cross_entropy_test.py` uses relaxed loss tolerances and compares **gradients in float32** with wider `atol/rtol`. If it still fails, check PyTorch / CUDA versions; file an issue with `torch.__version__`. |
| `RuntimeError: Trace is already saved` from profiler | Stale `traces/` directory from a previous run | Use the latest `profile_baseline.py` (it deletes the prior trace dir and Chrome JSON). Or run `rm -rf traces/trace traces/trace_*` before profiling. |
| `torch.cuda.OutOfMemoryError` during baseline profiling | Batch size or sequence length too large | Reduce `--batch-size` or `--seq-len` in `profile_baseline.py`. LLaMA 3.1 8B in BF16 needs ~16 GB for weights alone, plus ~32 GB for AdamW optimizer states. |
| `torch.cuda.OutOfMemoryError` during PyTorch cross-entropy but NOT during custom kernel | Standard cross-entropy materializes full `[B*T, V]` logit tensor | This demonstrates exactly why the custom kernel is needed. Reduce batch size or sequence length for the baseline comparison, or run only the custom kernel path. |
| Profiler trace JSON is very large (>1 GB) | Too many training steps profiled | Reduce `wait`, `warmup`, `active` in the profiler schedule. The default script profiles only 1 active step. |
| `401 Client Error` when downloading LLaMA 3.1 8B | Missing or invalid Hugging Face token, or no LLaMA access | Set `HF_TOKEN` environment variable. Accept the LLaMA 3.1 license at `https://huggingface.co/meta-llama/Llama-3.1-8B`. Verify token: `huggingface-cli whoami`. |
| Custom RMSNorm backward produces NaN gradients | Epsilon value too small or input contains extreme values | Ensure epsilon is `1e-6` (LLaMA default). Check input tensor for NaN/Inf with `torch.isfinite(x).all()`. |
| Benchmark shows no speedup for RMSNorm on small hidden dimensions | Kernel launch overhead dominates for small tensors | RMSNorm speedup is most visible at `hidden_size >= 2048`. LLaMA 3.1 8B uses 4096, which is well above the threshold. |
| `docker: Error response from daemon: could not select device driver` | NVIDIA Container Toolkit not installed or Docker not restarted | Install: `sudo apt install nvidia-container-toolkit && sudo systemctl restart docker`. Verify: `docker run --rm --gpus all nvidia/cuda:12.8.0-base-ubuntu24.04 nvidia-smi`. |
| Fused cross-entropy loss differs from PyTorch by more than 0.1 | Bug in the chunked online softmax implementation | Verify the running-max update: `m_new = max(m_old, chunk_max)` must happen BEFORE updating the running sum-of-exp `d`. Check that the target index masking uses the correct chunk offset. |
| Fine-tuning throughput is not improved despite faster kernels | GPU is compute-bound on matmuls, not bandwidth-bound on norms/loss | This is expected if batch size is large enough that matmuls dominate. The primary benefit is memory reduction (enabling larger batches or longer sequences) rather than pure latency. |
| `ImportError: cannot import name 'LlamaForCausalLM'` | `transformers` library version too old | Update: `pip install --upgrade transformers>=4.45.0`. The container's Dockerfile pins a compatible version. |
| Chrome trace file won't open in browser | Trace file too large for `chrome://tracing` | Use [Perfetto UI](https://ui.perfetto.dev/) instead, which handles larger traces. Or reduce the number of profiled steps. |
