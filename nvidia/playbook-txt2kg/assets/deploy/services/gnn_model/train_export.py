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

import os
import argparse
import torch
from torch_geometric import seed_everything
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GAT, LLM, GRetriever

def parse_args():
    parser = argparse.ArgumentParser(description='Train and export GNN model for service')
    parser.add_argument('--dataset_file', type=str, default='tech_qa.pt', help='Path to load dataset')
    parser.add_argument('--output_dir', type=str, default='models', help='Directory to save model')
    parser.add_argument('--model_save_path', type=str, default='tech-qa-model.pt', help='Model file name')
    parser.add_argument('--gnn_hidden_channels', type=int, default=1024, help='Hidden channels for GNN')
    parser.add_argument('--num_gnn_layers', type=int, default=4, help='Number of GNN layers')
    parser.add_argument('--llm_generator_name', type=str, default='meta-llama/Meta-Llama-3.1-8B-Instruct', 
                        help='LLM to use for generation')
    parser.add_argument('--epochs', type=int, default=2, help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=1, help='Training batch size')
    parser.add_argument('--eval_batch_size', type=int, default=2, help='Evaluation batch size')
    parser.add_argument('--lr', type=float, default=1e-5, help='Learning rate')
    
    return parser.parse_args()

def load_dataset(dataset_path):
    """
    Load preprocessed dataset from file
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(f"Dataset file not found at {dataset_path}. Please run preprocess_data.py first.")
    
    print(f"Loading dataset from {dataset_path}...")
    data_lists = torch.load(dataset_path, weights_only=False)
    print("Dataset loaded successfully!")
    print(f"Train set size: {len(data_lists['train'])}")
    print(f"Validation set size: {len(data_lists['validation'])}")
    print(f"Test set size: {len(data_lists['test'])}")
    
    return data_lists

def train_model(args, data_lists):
    """
    Train the GNN model
    """
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    hidden_channels = args.gnn_hidden_channels
    num_gnn_layers = args.num_gnn_layers
    
    train_loader = DataLoader(data_lists["train"], batch_size=batch_size,
                             drop_last=True, pin_memory=True, shuffle=True)
    val_loader = DataLoader(data_lists["validation"], batch_size=eval_batch_size,
                           drop_last=False, pin_memory=True, shuffle=False)
    
    # Create GNN model
    gnn = GAT(in_channels=768, hidden_channels=hidden_channels,
             out_channels=1024, num_layers=num_gnn_layers, heads=4)
    
    # Create LLM model
    llm = LLM(model_name=args.llm_generator_name)
    
    # Create the combined GRetriever model
    model = GRetriever(llm=llm, gnn=gnn)
    
    # Training setup
    params = [p for _, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW([{
        'params': params, 'lr': args.lr, 'weight_decay': 0.05
    }], betas=(0.9, 0.95))
    
    # Prompt template for questions
    prompt_template = """Answer this question based on retrieved contexts. Just give the answer without explanation.
[QUESTION] {question} [END_QUESTION]
[RETRIEVED_CONTEXTS] {context} [END_RETRIEVED_CONTEXTS]
Answer: """
    
    # Training loop
    for epoch in range(args.epochs):
        model.train()
        epoch_loss = 0
        print(f'Epoch: {epoch + 1}/{args.epochs}')
        
        for batch in train_loader:
            new_qs = []
            for i, q in enumerate(batch["question"]):
                # insert context
                new_qs.append(
                    prompt_template.format(question=q, context=batch.text_context[i]))
            batch.question = new_qs
            
            optimizer.zero_grad()
            loss = model(
                input_question=batch.question,
                input_graph=batch,
                output_labels=batch.label
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(optimizer.param_groups[0]['params'], 0.1)
            optimizer.step()
            
            epoch_loss += float(loss)
        
        avg_train_loss = epoch_loss / len(train_loader)
        print(f'Train Loss: {avg_train_loss:.4f}')
        
        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                new_qs = []
                for i, q in enumerate(batch["question"]):
                    # insert context
                    new_qs.append(
                        prompt_template.format(question=q, context=batch.text_context[i]))
                batch.question = new_qs
                
                loss = model(
                    input_question=batch.question,
                    input_graph=batch,
                    output_labels=batch.label
                )
                val_loss += float(loss)
        
        avg_val_loss = val_loss / len(val_loader)
        print(f'Validation Loss: {avg_val_loss:.4f}')
    
    return model

def save_model(model, save_path):
    """
    Save the trained model
    """
    directory = os.path.dirname(save_path)
    if not os.path.exists(directory):
        os.makedirs(directory)
    
    print(f"Saving model to {save_path}")
    torch.save(model.state_dict(), save_path)
    print("Model saved successfully!")

if __name__ == '__main__':
    import math
    
    # Set seed for reproducibility
    seed_everything(50)
    
    # Parse arguments
    args = parse_args()
    
    # Load dataset
    dataset_path = os.path.join(args.output_dir, args.dataset_file)
    data_lists = load_dataset(dataset_path)
    
    # Train model
    model = train_model(args, data_lists)
    
    # Save model
    model_path = os.path.join(args.output_dir, args.model_save_path)
    save_model(model, model_path)
    
    print(f"Model has been trained and saved to {model_path}")
    print("This model can now be used by the GNN service.") 