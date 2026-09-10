#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import torch
import argparse
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer



# Define prompt templates
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction: {}

### Input: {}

### Response: {}"""

def get_alpaca_dataset(eos_token, dataset_size=512):
    # Preprocess the dataset
    def preprocess(x):
        texts = [
            ALPACA_PROMPT_TEMPLATE.format(instruction, input, output) + eos_token
            for instruction, input, output in zip(x["instruction"], x["input"], x["output"])
        ]
        return {"text": texts}

    dataset = load_dataset("tatsu-lab/alpaca", split="train").select(range(dataset_size)).shuffle(seed=42)
    return dataset.map(preprocess, remove_columns=dataset.column_names, batched=True)


def main(args):
    # Load the model and tokenizer
    print(f"Loading model: {args.model_name}")
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=args.dtype,
        device_map="auto",
        trust_remote_code=True
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    # Print model information
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} (100% - Full Fine-tuning)")

    # Load and preprocess the dataset
    print(f"Loading dataset with {args.dataset_size} samples...")
    dataset = get_alpaca_dataset(tokenizer.eos_token, args.dataset_size)

    # Configure the SFT config
    config = {
        "per_device_train_batch_size": args.batch_size,
        "num_train_epochs": 0.05,  # Warmup epoch
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "save_strategy": 'no',
        "remove_unused_columns": False,
        "seed": 42,
        "dataset_text_field": "text",
        "packing": False,
        "max_length": args.seq_length,
        "report_to": "none",
        "logging_dir": args.log_dir,
        "logging_steps": args.logging_steps,
        "gradient_checkpointing": args.gradient_checkpointing,  # Save memory
    }

    # Compile model for faster training
    print("Compiling model with torch.compile()...")
    model = torch.compile(model)

    # Warmup for torch compile
    print("Running warmup for torch.compile()...")
    SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**config),
    ).train()

    # Train the model
    print(f"\nStarting full fine-tuning for {args.num_epochs} epoch(s)...")
    config["num_train_epochs"] = args.num_epochs
    config["report_to"] = "tensorboard"

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**config),
    )

    trainer_stats = trainer.train()

    # Print training statistics
    print(f"\n{'='*60}")
    print("TRAINING COMPLETED")
    print(f"{'='*60}")
    print(f"Training runtime: {trainer_stats.metrics['train_runtime']:.2f} seconds")
    print(f"Samples per second: {trainer_stats.metrics['train_samples_per_second']:.2f}")
    print(f"Steps per second: {trainer_stats.metrics['train_steps_per_second']:.2f}")
    print(f"Train loss: {trainer_stats.metrics['train_loss']:.4f}")
    print(f"{'='*60}\n")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama 3.2 3B Full Fine-tuning (SFT)")

    # Model configuration
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.2-3B-Instruct",
                        help="Model name or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")

    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Per device training batch size")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")

    # Dataset configuration
    parser.add_argument("--dataset_size", type=int, default=512,
                        help="Number of samples to use from dataset")

    # Logging configuration
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every N steps")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for logs")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"\n{'='*60}")
    print("LLAMA 3.2 3B FULL FINE-TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Training mode: Full SFT ")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"{'='*60}\n")

    main(args)
