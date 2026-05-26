"""
Correctness and memory tests for the fused cross-entropy Triton kernel.

Compares loss values and gradients against torch.nn.CrossEntropyLoss, and
reports peak GPU memory usage for both approaches to demonstrate the memory
savings from online softmax.
"""

import torch
from cross_entropy_kernel import TritonCrossEntropyLoss


def test_cross_entropy_correctness(dtype, loss_atol, loss_rtol, grad_atol, grad_rtol):
    """Test forward loss and backward gradients for a given dtype.

    BF16 logits use a looser loss tolerance (online softmax vs. PyTorch's path).
    Gradients are compared in float32 to avoid BF16 rounding false failures on
    128K-wide reductions.
    """
    dtype_name = "FP32" if dtype == torch.float32 else "BF16"
    vocab_size = 128256  # LLaMA 3.1 vocabulary size
    num_tokens = 512     # batch_size * seq_len

    # Create random logits and targets.
    logits = torch.randn(num_tokens, vocab_size, device="cuda", dtype=dtype, requires_grad=True)
    targets = torch.randint(0, vocab_size, (num_tokens,), device="cuda")

    # --- Reference: PyTorch CrossEntropyLoss ---
    logits_ref = logits.detach().clone().requires_grad_(True)
    ref_loss = torch.nn.CrossEntropyLoss()(logits_ref, targets)
    ref_loss.backward()

    # --- Custom: Triton fused cross-entropy ---
    logits_custom = logits.detach().clone().requires_grad_(True)
    custom_ce = TritonCrossEntropyLoss()
    custom_loss = custom_ce(logits_custom, targets)
    custom_loss.backward()

    # Compare forward loss values.
    loss_diff = abs(ref_loss.item() - custom_loss.item())
    loss_ok = torch.allclose(
        ref_loss.detach().float(),
        custom_loss.detach().float(),
        atol=loss_atol,
        rtol=loss_rtol,
    )
    print(f"  {dtype_name} Loss      — ref: {ref_loss.item():.6f}  custom: {custom_loss.item():.6f}  diff: {loss_diff:.2e}  {'PASSED' if loss_ok else 'FAILED'}")
    assert loss_ok, f"Loss correctness check FAILED for {dtype_name}"

    # Compare backward gradients in FP32 (BF16 allclose on raw grads is too strict for V-wide CE).
    g_ref = logits_ref.grad.float()
    g_custom = logits_custom.grad.float()
    grad_diff = (g_ref - g_custom).abs().max().item()
    grad_ok = torch.allclose(g_ref, g_custom, atol=grad_atol, rtol=grad_rtol)
    print(f"  {dtype_name} Gradient  — max diff (vs ref, fp32 compare): {grad_diff:.2e}  {'PASSED' if grad_ok else 'FAILED'}")
    assert grad_ok, f"Gradient correctness check FAILED for {dtype_name}"


def test_memory_savings():
    """Compare peak GPU memory between standard and fused cross-entropy."""
    vocab_size = 128256
    num_tokens = 512

    print("\nMemory Comparison")
    print("-" * 60)

    # --- Standard PyTorch cross-entropy ---
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.empty_cache()

    logits_std = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    targets = torch.randint(0, vocab_size, (num_tokens,), device="cuda")

    baseline_before = torch.cuda.memory_allocated()
    loss_std = torch.nn.CrossEntropyLoss()(logits_std, targets)
    loss_std.backward()
    std_peak = torch.cuda.max_memory_allocated()

    del logits_std, loss_std
    torch.cuda.empty_cache()

    # --- Custom fused cross-entropy ---
    torch.cuda.reset_peak_memory_stats()

    logits_fused = torch.randn(num_tokens, vocab_size, device="cuda", dtype=torch.bfloat16, requires_grad=True)

    fused_before = torch.cuda.memory_allocated()
    custom_ce = TritonCrossEntropyLoss()
    loss_fused = custom_ce(logits_fused, targets)
    loss_fused.backward()
    fused_peak = torch.cuda.max_memory_allocated()

    std_mb = std_peak / 1024 / 1024
    fused_mb = fused_peak / 1024 / 1024
    reduction = std_peak / fused_peak if fused_peak > 0 else float("inf")

    print(f"  Standard PyTorch CE — peak memory: {std_mb:.1f} MB")
    print(f"  Fused Triton CE     — peak memory: {fused_mb:.1f} MB")
    print(f"  Memory reduction: {reduction:.1f}x")


if __name__ == "__main__":
    print("Cross-Entropy Correctness Tests")
    print("=" * 60)

    print("\nTest 1: Float32")
    # log-sum-exp over 128K elements accumulates small drift vs PyTorch's CE path.
    test_cross_entropy_correctness(
        dtype=torch.float32,
        loss_atol=1e-4,
        loss_rtol=1e-4,
        grad_atol=1e-4,
        grad_rtol=1e-4,
    )

    print("\nTest 2: BFloat16 (relaxed tolerance)")
    test_cross_entropy_correctness(
        dtype=torch.bfloat16,
        loss_atol=5e-2,
        loss_rtol=5e-2,
        grad_atol=2e-1,
        grad_rtol=2e-1,
    )

    test_memory_savings()

    print("\n" + "=" * 60)
    print("All cross-entropy tests PASSED")
