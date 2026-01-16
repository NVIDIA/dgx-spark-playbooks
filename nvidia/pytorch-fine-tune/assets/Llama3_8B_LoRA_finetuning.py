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
from peft import get_peft_model, LoraConfig, TaskType


# Define prompt templates
ALPACA_PROMPT_TEMPLATE = """Below is an instruction that describes a task, paired with an input that provides further context. Write a response that appropriately completes the request.
### Instruction: {}

### Input: {}

### Response: {}"""

def get_alpaca_dataset(eos_token, dataset_size=500):
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
        device_map="auto"
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Configure LoRA config
    peft_config = LoraConfig(
        r=args.lora_rank,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_alpha=16,
        lora_dropout=0,
        task_type=TaskType.CAUSAL_LM)
    model = get_peft_model(model, peft_config)

    print(f"Trainable parameters = {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")

    # Load and preprocess the dataset
    print(f"Loading dataset with {args.dataset_size} samples...")
    dataset = get_alpaca_dataset(tokenizer.eos_token, args.dataset_size)

    # Configure the SFT config
    config = {
        "per_device_train_batch_size": args.batch_size,
        "num_train_epochs": 0.05,
        "gradient_accumulation_steps": args.gradient_accumulation_steps,
        "learning_rate": args.learning_rate,
        "optim": "adamw_torch",
        "save_strategy": 'no',
        "remove_unused_columns": False,
        "seed": 42,
        "dataset_text_field": "text",
        "packing": False,
        "max_length": args.seq_length,
        "torch_compile": False,
        "report_to": "none",
        "logging_dir": args.log_dir,
        "logging_steps": args.logging_steps
    }

    # Compile model if requested
    if args.use_torch_compile:
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
    print(f"\nStarting LoRA fine-tuning for {args.num_epochs} epoch(s)...")
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
    parser = argparse.ArgumentParser(description="Llama 3.1 8B Fine-tuning with LoRA")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="meta-llama/Llama-3.1-8B-Instruct",
                        help="Model name or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        choices=["float32", "float16", "bfloat16"],
                        help="Model dtype")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Per device training batch size")
    parser.add_argument("--seq_length", type=int, default=2048,
                        help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    
    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    
    # Dataset configuration
    parser.add_argument("--dataset_size", type=int, default=500,
                        help="Number of samples to use from dataset")
    
    # Logging configuration
    parser.add_argument("--logging_steps", type=int, default=1,
                        help="Log every N steps")
    parser.add_argument("--log_dir", type=str, default="logs",
                        help="Directory for logs")
    # Compilation
    parser.add_argument("--use_torch_compile", action="store_true",
                        help="Use torch.compile() for faster training")
    
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    print(f"\n{'='*60}")
    print("LLAMA 3.1 8B LoRA FINE-TUNING CONFIGURATION")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Batch size: {args.batch_size}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Torch compile: {args.use_torch_compile}")
    print(f"{'='*60}\n")
    
    main(args)