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

import requests
import json
import argparse

def parse_args():
    parser = argparse.ArgumentParser(description='Client for GNN Model Service')
    parser.add_argument('--url', type=str, default='http://localhost:5000', 
                        help='URL of the GNN model service')
    parser.add_argument('--question', type=str, required=True,
                        help='Question to ask')
    parser.add_argument('--context', type=str, required=True,
                        help='Context information to provide')
    
    return parser.parse_args()

def query_gnn_model(url, question, context):
    """
    Query the GNN model service with a question and context
    """
    endpoint = f"{url}/predict"
    
    payload = {
        "question": question,
        "context": context
    }
    
    headers = {
        "Content-Type": "application/json"
    }
    
    try:
        response = requests.post(endpoint, json=payload, headers=headers)
        if response.status_code == 200:
            return response.json()
        else:
            print(f"Error: {response.status_code}")
            print(response.text)
            return None
    except Exception as e:
        print(f"Error connecting to GNN service: {e}")
        return None

def query_rag_model(question, context):
    """
    Simple Pure RAG approach for comparison
    This is a placeholder - in a real implementation, you would have a separate RAG service
    or use a local LLM with context insertion
    """
    # This would typically call an API or use a local LLM
    print("Note: This is a placeholder for a Pure RAG implementation")
    return {
        "question": question,
        "answer": "Placeholder RAG answer. Implement real RAG for comparison."
    }

def compare_approaches(gnn_result, rag_result):
    """
    Compare the results from GNN and Pure RAG approaches
    """
    print("\n----- COMPARISON -----")
    print(f"Question: {gnn_result['question']}")
    print(f"GNN Answer: {gnn_result['answer']}")
    print(f"RAG Answer: {rag_result['answer']}")
    print("----------------------\n")

if __name__ == "__main__":
    args = parse_args()
    
    print(f"Querying GNN model at {args.url}...")
    gnn_result = query_gnn_model(args.url, args.question, args.context)
    
    if gnn_result:
        print("GNN Query successful!")
        
        # Get RAG result for comparison
        rag_result = query_rag_model(args.question, args.context)
        
        # Compare the approaches
        compare_approaches(gnn_result, rag_result)
    else:
        print("Failed to get response from GNN model service.") 