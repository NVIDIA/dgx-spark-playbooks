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
import torch
from flask import Flask, request, jsonify
import torch_geometric
from torch_geometric.nn import GAT, LLM, GRetriever

app = Flask(__name__)

# Constants
MODEL_PATH = os.environ.get('MODEL_PATH', '/app/models/tech-qa-model.pt')
LLM_GENERATOR_NAME = os.environ.get('LLM_GENERATOR_NAME', 'meta-llama/Meta-Llama-3.1-8B-Instruct')
GNN_HID_CHANNELS = int(os.environ.get('GNN_HID_CHANNELS', '1024'))
GNN_LAYERS = int(os.environ.get('GNN_LAYERS', '4'))

# Prompt template for questions
prompt_template = """Answer this question based on retrieved contexts. Just give the answer without explanation.
[QUESTION] {question} [END_QUESTION]
[RETRIEVED_CONTEXTS] {context} [END_RETRIEVED_CONTEXTS]
Answer: """

# Load the model
def load_model():
    print(f"Loading model from {MODEL_PATH}")
    
    # Create the GNN component
    gnn = GAT(in_channels=768, hidden_channels=GNN_HID_CHANNELS,
             out_channels=1024, num_layers=GNN_LAYERS, heads=4)
    
    # Create the LLM component
    llm = LLM(model_name=LLM_GENERATOR_NAME)
    
    # Create the GRetriever model
    model = GRetriever(llm=llm, gnn=gnn)
    
    # Load trained weights
    if os.path.exists(MODEL_PATH):
        state_dict = torch.load(MODEL_PATH, weights_only=True)
        model.load_state_dict(state_dict)
        model.eval()
        print("Model loaded successfully")
    else:
        print(f"WARNING: Model file not found at {MODEL_PATH}. Using untrained model.")
    
    return model

# Initialize model
model = None

@app.before_first_request
def initialize():
    global model
    model = load_model()

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"})

@app.route('/predict', methods=['POST'])
def predict():
    if not request.is_json:
        return jsonify({"error": "Request must be JSON"}), 400
    
    data = request.get_json()
    
    if 'question' not in data:
        return jsonify({"error": "Question is required"}), 400
    
    if 'context' not in data:
        return jsonify({"error": "Context is required"}), 400
    
    question = data['question']
    context = data['context']
    
    # Format the question with context using the prompt template
    formatted_question = prompt_template.format(question=question, context=context)
    
    # Prepare input for the model
    # Note: In a real implementation, you'd need to convert text to graph structure
    # Here we're assuming a simplified interface for demonstration
    try:
        # Create a PyTorch Geometric Data object
        # This is simplified and would need to be adapted to your actual graph structure
        graph_data = create_graph_from_text(context)
        
        # Generate prediction
        with torch.no_grad():
            prediction = model.generate(
                input_question=[formatted_question],
                input_graph=graph_data
            )[0]  # Get first prediction since we're processing one sample
        
        return jsonify({
            "question": question,
            "answer": prediction
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

def create_graph_from_text(text):
    """
    Convert text to a graph structure for the GNN.
    This is a placeholder - you'll need to implement the actual conversion
    based on your specific graph construction approach.
    """
    # This would need to be implemented based on how your graphs are constructed
    # For now, return a dummy graph
    raise NotImplementedError("Graph creation from text needs to be implemented")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 