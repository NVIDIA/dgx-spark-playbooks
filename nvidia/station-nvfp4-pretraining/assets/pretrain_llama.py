import argparse

import torch

from megatron.bridge.recipes.llama import llama3_8b_pretrain_config
from megatron.bridge.training.gpt_step import forward_step
from megatron.bridge.training.pretrain import pretrain
from megatron.bridge.training.mixed_precision import MixedPrecisionConfig, bf16_mixed, bf16_with_nvfp4_mixed


def nvfp4_mixed_precision() -> MixedPrecisionConfig:
    """NVFP4 mixed precision config with last 4 layers in BF16."""
    cfg = bf16_with_nvfp4_mixed()
    cfg.first_last_layers_bf16 = True
    cfg.num_layers_at_start_in_bf16 = 0
    cfg.num_layers_at_end_in_bf16 = 4
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Llama 3.1 8B NVFP4 pretraining")
    parser.add_argument(
        "--disable-fp4",
        action="store_true",
        help="Disable NVFP4; use plain BF16 mixed precision as a baseline",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=50,
        help="Number of training iterations",
    )
    parser.add_argument(
        "--warmup-iters",
        type=int,
        default=2,
        help="Number of warmup iterations",
    )
    parser.add_argument(
        "--global-batch-size",
        type=int,
        default=64,
        help="Global batch size",
    )
    parser.add_argument(
        "--micro-batch-size",
        type=int,
        default=4,
        help="Micro batch size (drives peak VRAM; increase to use more memory)",
    )
    parser.add_argument(
        "--seq-length",
        type=int,
        default=4096,
        help="Sequence length (recipe default is 8192; halved here to fit single GPU)",
    )
    args = parser.parse_args()

    config = llama3_8b_pretrain_config()

    # Single-GPU override: recipe defaults to context_parallel_size=2
    config.model.context_parallel_size = 1

    config.model.seq_length = args.seq_length
    config.dataset.sequence_length = args.seq_length

    config.train.train_iters = args.train_iters
    config.scheduler.lr_warmup_iters = args.warmup_iters
    config.train.global_batch_size = args.global_batch_size
    config.train.micro_batch_size = args.micro_batch_size

    config.logger.log_interval = 1
    config.dataset.num_workers = 2
    config.train.eval_iters = 0

    if args.disable_fp4:
        config.mixed_precision = bf16_mixed()
    else:
        config.mixed_precision = nvfp4_mixed_precision()

    pretrain(config=config, forward_step_func=forward_step)

    if torch.distributed.is_initialized() and torch.distributed.get_rank() == 0:
        print(
            f"FINAL mem-reserved-gigabytes: {torch.cuda.memory_reserved() / 1e9:.3f} | "
            f"mem-max-reserved-gigabytes: {torch.cuda.max_memory_reserved() / 1e9:.3f} | "
            f"mem-max-allocated-gigabytes: {torch.cuda.max_memory_allocated() / 1e9:.3f}",
            flush=True,
        )


if __name__ == "__main__":
    main()
