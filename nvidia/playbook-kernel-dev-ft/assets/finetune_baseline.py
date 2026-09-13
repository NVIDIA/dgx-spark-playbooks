"""
Baseline LLaMA 3.1 8B fine-tuning script — vanilla PyTorch.

Runs a minimal fine-tuning loop with synthetic data and reports training
throughput (tokens/sec), peak GPU memory, and time per step. This establishes
the baseline that finetune_optimized.py improves upon with custom kernels.

Usage:
    python finetune_baseline.py
    python finetune_baseline.py --steps 50 --batch-size 2 --seq-len 256
"""

import argparse
import time
import torch


def main():
    parser = argparse.ArgumentParser(description="Baseline LLaMA 3.1 8B fine-tuning")
    parser.add_argument("--steps", type=int, default=20, help="Number of training steps (default: 20)")
    parser.add_argument("--batch-size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    args = parser.parse_args()

    print("=" * 70)
    print("  LLaMA 3.1 8B Fine-Tuning — Baseline (vanilla PyTorch)")
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

    # Optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5)

    # Synthetic data (the playbook is about kernel optimization, not data pipelines).
    # To use a real dataset, replace this with a HuggingFace dataset loader:
    #   from datasets import load_dataset
    #   dataset = load_dataset("tatsu-lab/alpaca", split="train")
    input_ids = torch.randint(0, model.config.vocab_size, (args.batch_size, args.seq_len), device="cuda")
    labels = input_ids.clone()

    # Warm-up step (not timed)
    print("\nRunning warm-up step...")
    outputs = model(input_ids=input_ids, labels=labels)
    outputs.loss.backward()
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

        outputs = model(input_ids=input_ids, labels=labels)
        loss = outputs.loss
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
    print("  Baseline Results")
    print("=" * 70)
    print(f"  Average time per step:  {avg_time:.3f} s")
    print(f"  Average throughput:     {avg_tokens_sec:.0f} tokens/sec")
    print(f"  Peak GPU memory:        {peak_mem_gb:.1f} GB")
    print()


if __name__ == "__main__":
    main()
