"""
Benchmark script for custom Triton kernels.

Measures latency, throughput, bandwidth utilization, and peak memory for the
fused RMSNorm and fused cross-entropy kernels, comparing against PyTorch
reference implementations.

Usage:
    python benchmark_kernels.py --kernel rmsnorm
    python benchmark_kernels.py --kernel cross_entropy
    python benchmark_kernels.py --kernel all
"""

import argparse
import torch
from tabulate import tabulate


def benchmark_fn(fn, warmup=10, iters=100):
    """Time a GPU function using CUDA events for accurate measurement."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()

    return start.elapsed_time(end) / iters  # ms per iteration


def benchmark_rmsnorm():
    """Benchmark fused RMSNorm vs PyTorch reference."""
    from rmsnorm_kernel import TritonRMSNorm

    print("=" * 70)
    print("  RMSNorm Benchmark — Custom Triton vs. PyTorch Reference")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}\n")

    hidden_size = 4096  # LLaMA 3.1 8B
    eps = 1e-6

    # Reference: PyTorch implementation (matches HuggingFace LlamaRMSNorm)
    class RefRMSNorm(torch.nn.Module):
        def __init__(self, hs, eps):
            super().__init__()
            self.weight = torch.nn.Parameter(torch.ones(hs, device="cuda", dtype=torch.bfloat16))
            self.eps = eps

        def forward(self, x):
            x32 = x.to(torch.float32)
            v = x32.pow(2).mean(-1, keepdim=True)
            return self.weight * (x32 * torch.rsqrt(v + self.eps)).to(x.dtype)

    ref_norm = RefRMSNorm(hidden_size, eps)
    custom_norm = TritonRMSNorm(hidden_size, eps).to(device="cuda", dtype=torch.bfloat16)
    custom_norm.weight.data.copy_(ref_norm.weight.data)

    rows = []
    for num_tokens in [256, 1024, 4096, 16384]:
        x = torch.randn(num_tokens, hidden_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        grad = torch.randn_like(x)

        # Benchmark forward + backward together (training workload)
        def run_ref():
            x_r = x.detach().requires_grad_(True)
            y = ref_norm(x_r)
            y.backward(grad)

        def run_custom():
            x_c = x.detach().requires_grad_(True)
            y = custom_norm(x_c)
            y.backward(grad)

        ref_ms = benchmark_fn(run_ref)
        custom_ms = benchmark_fn(run_custom)

        # Bandwidth calculation:
        # Forward: read x + read weight + write y + write rnorm = 2*N*H*2 + H*2 + N*4 bytes (BF16)
        # Backward: read dy + read x + read w + read rnorm + write dx + write dw_rows
        # Approximate total: ~6 * N * H * 2 bytes for fwd+bwd
        total_bytes = 6 * num_tokens * hidden_size * 2
        ref_gbps = total_bytes / (ref_ms * 1e-3) / 1e9
        custom_gbps = total_bytes / (custom_ms * 1e-3) / 1e9

        speedup = ref_ms / custom_ms

        rows.append([
            f"{num_tokens:,}",
            f"{custom_ms * 1000:.1f}",
            f"{ref_ms * 1000:.1f}",
            f"{custom_gbps:.0f}",
            f"{ref_gbps:.0f}",
            f"{speedup:.2f}x",
        ])

    headers = ["Tokens", "Custom (us)", "PyTorch (us)", "Custom (GB/s)", "PyTorch (GB/s)", "Speedup"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


def benchmark_cross_entropy():
    """Benchmark fused cross-entropy vs PyTorch reference."""
    from cross_entropy_kernel import TritonCrossEntropyLoss

    print("=" * 70)
    print("  Cross-Entropy Benchmark — Custom Triton (online softmax) vs. PyTorch")
    print("=" * 70)

    gpu_name = torch.cuda.get_device_name(0)
    print(f"  GPU: {gpu_name}")
    print(f"  Vocabulary size: 128,256 (LLaMA 3.1)\n")

    vocab_size = 128256
    ref_ce = torch.nn.CrossEntropyLoss()
    custom_ce = TritonCrossEntropyLoss()

    rows = []
    for num_tokens in [128, 256, 512, 1024]:
        # --- Latency comparison ---
        logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        targets = torch.randint(0, vocab_size, (num_tokens,), device="cuda")
        grad = torch.tensor(1.0, device="cuda")

        def run_ref():
            l = logits.detach().requires_grad_(True)
            loss = ref_ce(l.float(), targets)  # PyTorch CE needs float32 internally
            loss.backward(grad)

        def run_custom():
            l = logits.detach().requires_grad_(True)
            loss = custom_ce(l, targets)
            loss.backward(grad)

        ref_ms = benchmark_fn(run_ref, warmup=5, iters=20)
        custom_ms = benchmark_fn(run_custom, warmup=5, iters=20)

        # --- Memory comparison ---
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.empty_cache()
        l_std = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        base_mem = torch.cuda.memory_allocated()
        loss_std = ref_ce(l_std.float(), targets)
        loss_std.backward()
        std_peak = torch.cuda.max_memory_allocated() - base_mem
        del l_std, loss_std
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        l_fused = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
        base_mem = torch.cuda.memory_allocated()
        loss_fused = custom_ce(l_fused, targets)
        loss_fused.backward()
        fused_peak = torch.cuda.max_memory_allocated() - base_mem
        del l_fused, loss_fused
        torch.cuda.empty_cache()

        speedup = ref_ms / custom_ms
        mem_reduction = std_peak / fused_peak if fused_peak > 0 else float("inf")

        rows.append([
            f"{num_tokens:,}",
            f"{custom_ms * 1000:.0f}",
            f"{ref_ms * 1000:.0f}",
            f"{speedup:.2f}x",
            f"{fused_peak / 1024 / 1024:.0f}",
            f"{std_peak / 1024 / 1024:.0f}",
            f"{mem_reduction:.1f}x",
        ])

    headers = ["Tokens", "Custom (us)", "PyTorch (us)", "Speedup", "Custom Mem (MB)", "PyTorch Mem (MB)", "Mem Reduction"]
    print(tabulate(rows, headers=headers, tablefmt="simple"))
    print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Benchmark custom Triton kernels")
    parser.add_argument("--kernel", choices=["rmsnorm", "cross_entropy", "all"], default="all")
    args = parser.parse_args()

    if args.kernel in ("rmsnorm", "all"):
        benchmark_rmsnorm()
    if args.kernel in ("cross_entropy", "all"):
        benchmark_cross_entropy()
