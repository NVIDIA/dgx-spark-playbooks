"""
Optimized LLaMA 3.1 8B fine-tuning script — with custom Triton kernels.

Identical to finetune_baseline.py but with both custom kernels monkey-patched
in: fused RMSNorm replaces all LlamaRMSNorm modules, and fused cross-entropy
replaces the standard loss computation.

Usage:
    python finetune_optimized.py
    python finetune_optimized.py --steps 50 --batch-size 2 --seq-len 256
"""

import argparse
import time
import torch


def replace_rmsnorm_modules(model):
    """Walk the model tree and replace every LlamaRMSNorm with TritonRMSNorm.

    This is the 'surgical replacement' pattern commonly used in production
    inference and training optimization. We find modules by type, create a
    Triton-backed replacement that copies the learned weights, and swap it
    into the model's module tree.
    """
    from rmsnorm_kernel import TritonRMSNorm

    count = 0
    # We need to collect replacements first, then apply them, because
    # modifying the module tree during iteration is not safe.
    replacements = []

    for name, module in model.named_modules():
        if type(module).__name__ == "LlamaRMSNorm":
            # Split the dotted name to find the parent module and attribute name.
            parts = name.split(".")
            parent_name = ".".join(parts[:-1])
            attr_name = parts[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attr_name, module))

    for parent, attr_name, old_module in replacements:
        new_module = TritonRMSNorm.from_llama_rmsnorm(old_module)
        setattr(parent, attr_name, new_module)
        count += 1

    return count


def main():
    parser = argparse.ArgumentParser(description="Optimized LLaMA 3.1 8B fine-tuning")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps (default: 20)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    args = parser.parse_args()

    print("=" * 70)
    print("  LLaMA 3.1 8B Fine-Tuning — Optimized (Custom Triton Kernels)")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Steps: {args.steps}, Batch size: {args.batch_size}, Seq len: {args.seq_len}")
    print()

    # Load model
    print("Loading meta-llama/Llama-3.1-8B...")
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.train()

    param_count = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {param_count:,}")

    # --- Apply custom kernels ---
    # 1. Replace all RMSNorm modules with fused Triton implementation.
    n_replaced = replace_rmsnorm_modules(model)
    print(f"  Replaced {n_replaced} RMSNorm module(s) with custom Triton kernel")

    # 2. Use fused cross-entropy loss.
    from cross_entropy_kernel import TritonCrossEntropyLoss
    custom_ce = TritonCrossEntropyLoss()
    print("  Using custom Triton cross-entropy loss (online softmax)")

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Synthetic data
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    labels = input_ids.clone()

    # Warm-up step (triggers Triton JIT compilation)
    print("\nRunning warm-up step (includes Triton JIT compilation)...")
    outputs = model(input_ids=input_ids)
    # Use custom cross-entropy: extract logits, shift for next-token prediction.
    logits = outputs.logits[:, :-1, :].contiguous()
    target = labels[:, 1:].contiguous()
    loss = custom_ce(logits, target)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    # Reset memory tracking after warm-up
    torch.cuda.reset_peak_memory_stats()

    # Training loop
    print(f"Running {args.steps} training steps...\n")
    tokens_per_step = args.batch_size * args.seq_len
    step_times = []

    for step in range(args.steps):
        torch.cuda.synchronize()
        t0 = time.perf_counter()

        outputs = model(input_ids=input_ids)
        logits = outputs.logits[:, :-1, :].contiguous()
        target = labels[:, 1:].contiguous()
        loss = custom_ce(logits, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        torch.cuda.synchronize()
        t1 = time.perf_counter()
        step_times.append(t1 - t0)

        if (step + 1) % 5 == 0:
            tokens_sec = tokens_per_step / step_times[-1]
            print(f"  Step {step + 1:3d}/{args.steps}  loss: {loss.item():.4f}  time: {step_times[-1]:.3f}s  tokens/sec: {tokens_sec:.0f}")

    # Summary
    avg_time = sum(step_times) / len(step_times)
    avg_tokens_sec = tokens_per_step / avg_time
    peak_mem_gb = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024

    print()
    print("=" * 70)
    print("  Optimized Results")
    print("=" * 70)
    print(f"  Average time per step:  {avg_time:.3f} s")
    print(f"  Average throughput:     {avg_tokens_sec:.0f} tokens/sec")
    print(f"  Peak GPU memory:        {peak_mem_gb:.1f} GB")
    print()


if __name__ == "__main__":
    main()
