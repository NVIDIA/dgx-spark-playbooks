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
import os
from datasets import load_dataset
from trl import SFTConfig, SFTTrainer
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training


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
    print(f"Training mode: QLoRA (4-bit quantization)")
    
    # Use balanced device map for QLoRA to avoid device placement issues
    # "balanced" distributes model across available GPUs more reliably than "auto"
    device_map_config = "balanced" if torch.cuda.device_count() > 1 else {"": 0}
    
    # Configure 4-bit quantization for QLoRA
    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type='nf4',
        bnb_4bit_compute_dtype=args.dtype,
        bnb_4bit_quant_storage=args.dtype
    )
    
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        quantization_config=quantization_config,
        dtype=args.dtype,
        device_map="cuda",
    )
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    tokenizer.pad_token = tokenizer.eos_token

    # Prepare model for QLoRA training
    print(f"Preparing model for QLoRA (4-bit) with rank {args.lora_rank}...")
    model = prepare_model_for_kbit_training(model)
    
    peft_config = LoraConfig(
        r=args.lora_rank,
        target_modules="all-linear",
        lora_alpha=16,
        lora_dropout=0,
        task_type=TaskType.CAUSAL_LM
    )
    
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Trainable parameters: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")

    # Load and preprocess the dataset
    print(f"Loading dataset with {args.dataset_size} samples...")
    dataset = get_alpaca_dataset(tokenizer.eos_token, args.dataset_size)

    # Configure the SFT config
    config = {
        "per_device_train_batch_size": args.batch_size,
        "num_train_epochs": args.num_epochs,
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
        "gradient_checkpointing": args.gradient_checkpointing
    }

    # Train the model
    print(f"\nStarting QLoRA fine-tuning for {args.num_epochs} epoch(s)...")
    config["report_to"] = "tensorboard"
    
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        args=SFTConfig(**config),
        peft_config=peft_config,
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
    
    # Save model if requested
    if args.output_dir:
        print(f"Saving model to {args.output_dir}...")
        trainer.save_model(args.output_dir)
        tokenizer.save_pretrained(args.output_dir)
        print("Model saved successfully!")


def parse_arguments():
    parser = argparse.ArgumentParser(description="Llama 3.1 70B Fine-tuning with QLoRA")
    
    # Model configuration
    parser.add_argument("--model_name", type=str, default="unsloth/Meta-Llama-3.1-70B-bnb-4bit",
                        help="Model name or path")
    parser.add_argument("--dtype", type=str, default="bfloat16",
                        help="Model dtype (e.g., float32, float16, bfloat16)")
    
    # Training configuration
    parser.add_argument("--batch_size", type=int, default=8,
                        choices=[1, 2, 4, 8, 16, 32],
                        help="Per device training batch size")
    parser.add_argument("--seq_length", type=int, default=2048,
                        choices=[256, 512, 1024, 2048, 4096, 8192],
                        help="Maximum sequence length")
    parser.add_argument("--num_epochs", type=int, default=1,
                        help="Number of training epochs")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1,
                        help="Gradient accumulation steps")
    parser.add_argument("--learning_rate", type=float, default=1e-4,
                        help="Learning rate")
    parser.add_argument("--gradient_checkpointing", action="store_true",
                        help="Enable gradient checkpointing to save memory")
    
    # LoRA configuration
    parser.add_argument("--lora_rank", type=int, default=8,
                        help="LoRA rank")
    
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
    print("LLAMA 3.1 70B QLoRA FINE-TUNING")
    print(f"{'='*60}")
    print(f"Model: {args.model_name}")
    print(f"Training mode: QLoRA (4-bit quantization)")
    print(f"Batch size: {args.batch_size}")
    print(f"Gradient accumulation: {args.gradient_accumulation_steps}")
    print(f"Effective batch size: {args.batch_size * args.gradient_accumulation_steps}")
    print(f"Sequence length: {args.seq_length}")
    print(f"Number of epochs: {args.num_epochs}")
    print(f"Learning rate: {args.learning_rate}")
    print(f"LoRA rank: {args.lora_rank}")
    print(f"Dataset size: {args.dataset_size}")
    print(f"Gradient checkpointing: {args.gradient_checkpointing}")
    print(f"{'='*60}\n")
    
    main(args)
