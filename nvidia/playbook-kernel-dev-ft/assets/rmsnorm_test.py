"""
Correctness tests for the fused RMSNorm Triton kernel.

Compares forward and backward outputs against PyTorch's reference LlamaRMSNorm
at shapes matching LLaMA 3.1 8B (hidden_size=4096). Tests both FP32 and BF16
to verify numerical correctness under realistic training precision.
"""

import torch
from rmsnorm_kernel import TritonRMSNorm


class ReferenceLlamaRMSNorm(torch.nn.Module):
    """PyTorch reference implementation of RMSNorm (from HuggingFace transformers)."""

    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)


def test_rmsnorm(dtype, atol, rtol, dw_atol=None, dw_rtol=None):
    """Test forward and backward for a given dtype.

    dw_atol/dw_rtol: separate tolerance for weight gradient, which accumulates
    across all rows and is more sensitive to FP ordering differences.
    """
    dtype_name = "FP32" if dtype == torch.float32 else "BF16"
    hidden_size = 4096  # LLaMA 3.1 8B hidden dimension
    batch_size = 4
    seq_len = 512

    if dw_atol is None:
        dw_atol = atol
    if dw_rtol is None:
        dw_rtol = rtol

    # Create input tensor with gradients enabled.
    x = torch.randn(batch_size, seq_len, hidden_size, device="cuda", dtype=dtype, requires_grad=True)

    # Reference implementation
    ref_norm = ReferenceLlamaRMSNorm(hidden_size).to(device="cuda", dtype=dtype)

    # Custom Triton implementation (copy weights from reference)
    custom_norm = TritonRMSNorm(hidden_size).to(device="cuda", dtype=dtype)
    custom_norm.weight.data.copy_(ref_norm.weight.data)

    # --- Forward pass ---
    x_ref = x.detach().clone().requires_grad_(True)
    x_custom = x.detach().clone().requires_grad_(True)

    y_ref = ref_norm(x_ref)
    y_custom = custom_norm(x_custom)

    fwd_max_diff = (y_ref - y_custom).abs().max().item()
    fwd_pass = torch.allclose(y_ref, y_custom, atol=atol, rtol=rtol)

    print(f"  {dtype_name} Forward  — max diff: {fwd_max_diff:.2e}  {'PASSED' if fwd_pass else 'FAILED'}")
    assert fwd_pass, f"Forward correctness check FAILED for {dtype_name}"

    # --- Backward pass ---
    # Use the same upstream gradient for both.
    grad_output = torch.randn_like(y_ref)

    y_ref.backward(grad_output)
    y_custom.backward(grad_output)

    grad_x_diff = (x_ref.grad - x_custom.grad).abs().max().item()
    grad_x_pass = torch.allclose(x_ref.grad, x_custom.grad, atol=atol, rtol=rtol)

    # Weight gradient uses separate (potentially relaxed) tolerance because it
    # sums per-row contributions across batch*seq_len rows. Different summation
    # order between our FP32-accumulated kernel and PyTorch's autograd produces
    # larger absolute differences, especially in BF16 where the final cast
    # has 0.25-0.5 step size for typical gradient magnitudes.
    grad_w_diff = (ref_norm.weight.grad - custom_norm.weight.grad).abs().max().item()
    grad_w_pass = torch.allclose(ref_norm.weight.grad, custom_norm.weight.grad, atol=dw_atol, rtol=dw_rtol)

    print(f"  {dtype_name} Backward (dx)  — max diff: {grad_x_diff:.2e}  {'PASSED' if grad_x_pass else 'FAILED'}")
    print(f"  {dtype_name} Backward (dw)  — max diff: {grad_w_diff:.2e}  {'PASSED' if grad_w_pass else 'FAILED'}")

    assert grad_x_pass, f"Backward grad_x check FAILED for {dtype_name}"
    assert grad_w_pass, f"Backward grad_w check FAILED for {dtype_name}"


if __name__ == "__main__":
    print("RMSNorm Correctness Tests")
    print("=" * 60)

    print("\nTest 1: Float32")
    # Slightly relaxed tolerance: the weight gradient is accumulated across all
    # rows (batch_size * seq_len = 2048), and different summation order between
    # our per-row accumulation and PyTorch's autograd causes FP32 rounding diffs.
    test_rmsnorm(dtype=torch.float32, atol=1e-4, rtol=1e-4)

    print("\nTest 2: BFloat16 (relaxed tolerance)")
    # BF16 has only 7 bits of mantissa, so larger differences are expected.
    # The weight gradient tolerance is more relaxed because BF16 accumulation
    # across 2048 rows amplifies rounding differences. The key check is that
    # the relative error is small (< 2%), not the absolute difference.
    test_rmsnorm(dtype=torch.bfloat16, atol=1e-2, rtol=1e-2,
                 dw_atol=2.0, dw_rtol=2e-2)

    print("\n" + "=" * 60)
    print("All RMSNorm correctness tests PASSED")
