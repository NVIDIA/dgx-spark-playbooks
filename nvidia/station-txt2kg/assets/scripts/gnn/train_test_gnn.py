#!/usr/bin/env python3
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

import argparse
import os
import torch
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

# Import the necessary modules from PyTorch Geometric
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn import (
    GAT, LLM, GRetriever, LLMJudge
)

# Define constants for better readability
NV_NIM_MODEL_DEFAULT = "nvidia/llama-3.1-nemotron-70b-instruct"
LLM_GENERATOR_NAME_DEFAULT = "meta-llama/Meta-Llama-3.1-8B-Instruct"
GNN_HID_CHANNELS_DEFAULT = 1024
GNN_LAYERS_DEFAULT = 4
LR_DEFAULT = 1e-5
EPOCHS_DEFAULT = 2
BATCH_SIZE_DEFAULT = 1
EVAL_BATCH_SIZE_DEFAULT = 2
LLM_GEN_MODE_DEFAULT = "full"
DEFAULT_ENDPOINT_URL = "https://integrate.api.nvidia.com/v1"

# File paths and directories
DATASET_FILE = "tech_qa.pt"
MODEL_SAVE_PATH = "tech-qa-model.pt"
OUTPUT_DIR = "output"

# Prompt template for questions
prompt_template = """Answer this question based on retrieved contexts. Just give the answer without explanation.
[QUESTION] {question} [END_QUESTION]
[RETRIEVED_CONTEXTS] {context} [END_RETRIEVED_CONTEXTS]
Answer: """

def parse_args():
    parser = argparse.ArgumentParser()
    # Model and training related arguments
    parser.add_argument('--NV_NIM_MODEL', type=str, default=NV_NIM_MODEL_DEFAULT, help="The NIM LLM to use for evaluation with LLMJudge")
    parser.add_argument('--NV_NIM_KEY', type=str, default="", help="NVIDIA API key")
    parser.add_argument(
        '--ENDPOINT_URL', type=str, default=DEFAULT_ENDPOINT_URL, help=
        "The URL hosting your model, in case you are not using the public NIM."
    )
    
    parser.add_argument('--gnn_hidden_channels', type=int, default=GNN_HID_CHANNELS_DEFAULT, help="Hidden channels for GNN")
    parser.add_argument('--num_gnn_layers', type=int, default=GNN_LAYERS_DEFAULT, help="Number of GNN layers")
    parser.add_argument('--lr', type=float, default=LR_DEFAULT, help="Learning rate")
    parser.add_argument('--epochs', type=int, default=EPOCHS_DEFAULT, help="Number of epochs")
    parser.add_argument('--batch_size', type=int, default=BATCH_SIZE_DEFAULT, help="Batch size")
    parser.add_argument('--eval_batch_size', type=int, default=EVAL_BATCH_SIZE_DEFAULT, help="Evaluation batch size")
    parser.add_argument('--llm_generator_name', type=str, default=LLM_GENERATOR_NAME_DEFAULT, help="The LLM to use for Generation")
    parser.add_argument(
        '--llm_generator_mode', type=str, default=LLM_GEN_MODE_DEFAULT, choices=["frozen", "lora", "full"],
        help="Whether to freeze the Generator LLM, use LORA, or fully finetune"
    )
    parser.add_argument('--dont_save_model', action="store_true", help="Whether to skip model saving.")
    parser.add_argument('--eval_only', action="store_true", help="Skip training and only run evaluation")
    
    # File path arguments
    parser.add_argument('--dataset_file', type=str, default=DATASET_FILE, help="Path to load dataset")
    parser.add_argument('--model_save_path', type=str, default=MODEL_SAVE_PATH, help="Path to save/load model")
    parser.add_argument('--output_dir', type=str, default=OUTPUT_DIR, help="Directory for output files")
    
    return parser.parse_args()

def load_params_dict(model, load_path):
    """
    Load model parameters from a saved checkpoint
    """
    print(f"Loading model parameters from {load_path}")
    state_dict = torch.load(load_path, weights_only=True)
    model.load_state_dict(state_dict)
    return model

def save_params_dict(model, save_path):
    """
    Save model parameters to a checkpoint
    """
    print(f"Saving model parameters to {save_path}")
    torch.save(model.state_dict(), save_path)

def adjust_learning_rate(param_group, base_lr, progress, num_training_steps):
    """
    Implement learning rate schedule with warmup and decay
    """
    if progress < 0.1:
        # Linear warmup for first 10% of training
        lr = base_lr * progress / 0.1
    else:
        # Cosine decay for remaining 90%
        progress = (progress - 0.1) / 0.9
        lr = base_lr * 0.5 * (1.0 + math.cos(math.pi * progress))
    
    param_group["lr"] = lr
    return lr

def get_loss(model, batch):
    """
    Calculate loss for a batch
    """
    return model(
        input_question=batch.question,
        input_graph=batch,
        output_labels=batch.label
    )

def inference_step(model, batch):
    """
    Run inference on a batch and return predictions
    """
    with torch.no_grad():
        preds = model.generate(
            input_question=batch.question,
            input_graph=batch
        )
    return preds

def train(args, data_lists):
    """
    Train the GNN model
    
    Args:
        args: Command line arguments
        data_lists: Dictionary containing train, validation, and test datasets
        
    Returns:
        Trained model and test dataloader
    """
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    hidden_channels = args.gnn_hidden_channels
    num_gnn_layers = args.num_gnn_layers
    
    train_loader = DataLoader(data_lists["train"], batch_size=batch_size,
                             drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(data_lists["validation"], batch_size=eval_batch_size,
                           drop_last=False, pin_memory=True, shuffle=False)
    test_loader = DataLoader(data_lists["test"], batch_size=eval_batch_size,
                            drop_last=False, pin_memory=True, shuffle=False)
    
    gnn = GAT(in_channels=768, hidden_channels=hidden_channels,
             out_channels=1024, num_layers=num_gnn_layers, heads=4)
    
    if args.llm_generator_mode == "full":
        llm = LLM(model_name=args.llm_generator_name)
        model = GRetriever(llm=llm, gnn=gnn)
    elif args.llm_generator_mode == "lora":
        llm = LLM(model_name=args.llm_generator_name, dtype=torch.float32)
        model = GRetriever(llm=llm, gnn=gnn, use_lora=True)
    else:  # frozen
        llm = LLM(model_name=args.llm_generator_name, dtype=torch.float32).eval()
        for _, p in llm.named_parameters():
            p.requires_grad = False
        model = GRetriever(llm=llm, gnn=gnn)
    
    # Use the path from arguments
    model_path = os.path.join(args.output_dir, args.model_save_path)
    if os.path.exists(model_path):
        print(f"Re-using saved G-retriever model from {model_path}...")
        model = load_params_dict(model, model_path)
        
        if args.eval_only:
            print("Skipping training as --eval_only flag is set")
            return model, test_loader
    
    if not args.eval_only:
        params = [p for _, p in model.named_parameters() if p.requires_grad]
        lr = args.lr
        optimizer = torch.optim.AdamW([{
            'params': params, 'lr': lr, 'weight_decay': 0.05
        }], betas=(0.9, 0.95))
        
        for epoch in range(args.epochs):
            model.train()
            epoch_loss = 0
            epoch_str = f'Epoch: {epoch + 1}|{args.epochs}'
            loader = tqdm(train_loader, desc=epoch_str)
            
            for step, batch in enumerate(loader):
                new_qs = []
                for i, q in enumerate(batch["question"]):
                    # insert VectorRAG context
                    new_qs.append(
                        prompt_template.format(question=q, context=batch.text_context[i]))
                batch.question = new_qs
                
                optimizer.zero_grad()
                loss = get_loss(model, batch)
                loss.backward()
                clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
                
                if (step + 1) % 2 == 0:
                    adjust_learning_rate(optimizer.param_groups[0], lr,
                                        step / len(train_loader) + epoch, args.epochs)
                
                optimizer.step()
                epoch_loss += float(loss)
                
                if (step + 1) % 2 == 0:
                    lr = optimizer.param_groups[0]['lr']
            
            train_loss = epoch_loss / len(train_loader)
            print(epoch_str + f', Train Loss: {train_loss:4f}')
            
            # Eval Step
            val_loss = 0
            model.eval()
            with torch.no_grad():
                for step, batch in enumerate(val_loader):
                    new_qs = []
                    for i, q in enumerate(batch["question"]):
                        # insert VectorRAG context
                        new_qs.append(
                            prompt_template.format(question=q, context=batch.text_context[i]))
                    batch.question = new_qs
                    
                    loss = get_loss(model, batch)
                    val_loss += loss.item()
            
            val_loss = val_loss / len(val_loader)
            print(epoch_str + f", Val Loss: {val_loss:4f}")
            torch.cuda.empty_cache()
            torch.cuda.reset_max_memory_allocated()
        
        model.eval()
        if not args.dont_save_model:
            # Create output directory if it doesn't exist
            os.makedirs(args.output_dir, exist_ok=True)
            save_params_dict(model, save_path=model_path)
    
    return model, test_loader

def test(model, test_loader, args):
    """
    Test the GNN model and calculate evaluation metrics
    
    Args:
        model: Trained GNN model
        test_loader: DataLoader for test dataset
        args: Command line arguments
    """
    llm_judge = LLMJudge(args.NV_NIM_MODEL, args.NV_NIM_KEY, args.ENDPOINT_URL)
    
    def eval(question: str, pred: str, correct_answer: str):
        # calculate the score based on pred and correct answer
        return llm_judge.score(question, pred, correct_answer)
    
    scores = []
    eval_tuples = []
    
    for test_batch in tqdm(test_loader, desc="Testing"):
        new_qs = []
        for i, q in enumerate(test_batch["question"]):
            # insert VectorRAG context
            new_qs.append(
                prompt_template.format(question=q, context=test_batch.text_context[i]))
        test_batch.question = new_qs
        
        preds = inference_step(model, test_batch)
        for question, pred, label in zip(test_batch.question, preds, test_batch.label):
            eval_tuples.append((question, pred, label))
    
    for question, pred, label in tqdm(eval_tuples, desc="Evaluating"):
        scores.append(eval(question, pred, label))
    
    avg_scores = sum(scores) / len(scores)
    print("Avg marlin accuracy =", avg_scores)
    
    # Save results to file
    results_path = os.path.join(args.output_dir, "test_results.txt")
    with open(results_path, "w") as f:
        f.write(f"Average marlin accuracy: {avg_scores}\n\n")
        f.write("Example predictions:\n")
        for i, (question, pred, label) in enumerate(eval_tuples[:5]):  # Show first 5 examples
            f.write(f"Example {i+1}:\n")
            f.write(f"Question: {question}\n")
            f.write(f"Prediction: {pred}\n")
            f.write(f"Ground Truth: {label}\n")
            f.write(f"Score: {scores[i]}\n\n")
    
    print(f"Test results saved to {results_path}")

def load_dataset(args):
    """
    Load preprocessed dataset from file
    """
    dataset_path = os.path.join(args.output_dir, args.dataset_file)
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please run preprocess_data.py first.")
    
    print(f"Loading dataset from {dataset_path}...")
    data_lists = torch.load(dataset_path, weights_only=False)
    print("Dataset loaded successfully!")
    print(f"Train set size: {len(data_lists['train'])}")
    print(f"Validation set size: {len(data_lists['validation'])}")
    print(f"Test set size: {len(data_lists['test'])}")
    
    return data_lists

if __name__ == '__main__':
    import math
    
    # for reproducibility
    seed_everything(50)
    args = parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load preprocessed dataset
    data_lists = load_dataset(args)
    
    # Train model
    model, test_loader = train(args, data_lists)
    
    # Test model
    test(model, test_loader, args) 