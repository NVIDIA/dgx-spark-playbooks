"""
Profile a LLaMA 3.1 8B fine-tuning step to identify GPU bottlenecks.

Runs a single forward + backward + optimizer step under torch.profiler, then
exports a Chrome trace and prints a summary table of the most time-consuming
GPU operations. Supports optional flags to enable custom Triton kernels for
re-profiling after optimization.

Usage:
    python profile_baseline.py                                    # Baseline profile
    python profile_baseline.py --use-custom-rmsnorm               # With custom RMSNorm
    python profile_baseline.py --use-custom-rmsnorm --use-custom-ce  # With both custom kernels
    python profile_baseline.py --batch-size 2 --seq-len 256       # Custom dimensions
"""

import argparse
import os
import torch
from torch.profiler import profile, ProfilerActivity, schedule


def _filter_profiler_table(table: str) -> str:
    """Drop noisy driver / submission rows that confuse beginners (e.g. Command Buffer Full)."""
    lines = table.splitlines()
    out = []
    for line in lines:
        low = line.lower()
        if "command buffer full" in low:
            continue
        out.append(line)
    return "\n".join(out)


def replace_rmsnorm(model):
    """Replace all LlamaRMSNorm modules with the custom Triton implementation."""
    from rmsnorm_kernel import TritonRMSNorm

    # Collect replacements first, then apply — modifying the module tree during
    # named_modules() iteration can cause skipped modules.
    replacements = []
    for name, module in model.named_modules():
        if type(module).__name__ == "LlamaRMSNorm":
            parent_name = ".".join(name.split(".")[:-1])
            attr_name = name.split(".")[-1]
            parent = model.get_submodule(parent_name) if parent_name else model
            replacements.append((parent, attr_name, module))

    for parent, attr_name, old_module in replacements:
        setattr(parent, attr_name, TritonRMSNorm.from_llama_rmsnorm(old_module))

    return len(replacements)


def main():
    parser = argparse.ArgumentParser(description="Profile LLaMA 3.1 8B fine-tuning step")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--use-custom-rmsnorm", action="store_true", help="Use custom Triton RMSNorm")
    parser.add_argument("--use-custom-ce", action="store_true", help="Use custom Triton cross-entropy")
    args = parser.parse_args()

    print("=" * 70)
    print("  LLaMA 3.1 8B Fine-Tuning Profiler")
    print("=" * 70)
    print(f"  GPU: {torch.cuda.get_device_name(0)}")
    print(f"  Batch size: {args.batch_size}, Sequence length: {args.seq_len}")
    print(f"  Custom RMSNorm: {'ON' if args.use_custom_rmsnorm else 'OFF'}")
    print(f"  Custom Cross-Entropy: {'ON' if args.use_custom_ce else 'OFF'}")
    print()

    # Load model
    print("Loading meta-llama/Llama-3.1-8B...")
    from transformers import AutoModelForCausalLM, AutoConfig

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B",
        torch_dtype=torch.bfloat16,
        device_map="cuda",
    )
    model.train()
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Apply custom kernels if requested
    if args.use_custom_rmsnorm:
        n = replace_rmsnorm(model)
        print(f"  Replaced {n} RMSNorm module(s) with custom Triton kernel")

    if args.use_custom_ce:
        from cross_entropy_kernel import TritonCrossEntropyLoss
        custom_ce = TritonCrossEntropyLoss()
        print("  Using custom Triton cross-entropy loss")
    else:
        custom_ce = None

    # Create synthetic training data
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    labels = input_ids.clone()

    # Set up optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Warm-up step (not profiled) — this triggers Triton JIT compilation
    # and CUDA lazy initialization so they don't appear in the profile.
    print("\nRunning warm-up step...")
    if custom_ce:
        outputs = model(input_ids=input_ids)
        loss = custom_ce(outputs.logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous())
    else:
        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    torch.cuda.synchronize()

    # Profiled step
    print("Running profiled training step...")
    os.makedirs("traces", exist_ok=True)
    trace_name = "trace"
    if args.use_custom_rmsnorm:
        trace_name += "_custom_rmsnorm"
    if args.use_custom_ce:
        trace_name += "_custom_ce"

    chrome_trace_path = os.path.join("traces", f"{trace_name}_chrome.json")
    # Remove a prior Chrome trace for these flags so re-runs start clean.
    if os.path.isfile(chrome_trace_path):
        os.remove(chrome_trace_path)

    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        with_stack=True,
        schedule=schedule(wait=0, warmup=0, active=1, repeat=1),
    ) as prof:
        if custom_ce:
            outputs = model(input_ids=input_ids)
            loss = custom_ce(outputs.logits[:, :-1, :].contiguous(), labels[:, 1:].contiguous())
        else:
            outputs = model(input_ids=input_ids, labels=labels)
            loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        prof.step()

    # Export a single Chrome trace JSON for manual inspection (open in Perfetto UI).
    prof.export_chrome_trace(chrome_trace_path)

    # Print summary table
    print(f"\nChrome trace saved to: {chrome_trace_path}")
    print("\n" + "=" * 70)
    print("  Top 20 CUDA Operations by Total GPU Time")
    print("=" * 70)
    raw_table = prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=20)
    print(_filter_profiler_table(raw_table))

    # Print peak memory
    peak_mem = torch.cuda.max_memory_allocated() / 1024 / 1024 / 1024
    print(f"\nPeak GPU memory allocated: {peak_mem:.1f} GB")


if __name__ == "__main__":
    main()
