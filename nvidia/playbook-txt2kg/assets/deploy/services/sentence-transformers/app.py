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
from flask import Flask, request, jsonify
from sentence_transformers import SentenceTransformer
import os
import time
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Get model name from environment variable
model_name = os.environ.get("MODEL_NAME", "all-MiniLM-L6-v2")
logger.info(f"Loading model: {model_name}")

# Load model during startup
start_time = time.time()
try:
    model = SentenceTransformer(model_name)
    logger.info(f"Model loaded in {time.time() - start_time:.2f} seconds")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    raise

@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy", "model": model_name})

@app.route("/embed", methods=["POST"])
def embed():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        texts = data.get("texts", [])
        if not texts:
            return jsonify({"error": "No texts provided"}), 400
            
        # Process in batches if needed
        batch_size = data.get("batch_size", 32)
        
        start_time = time.time()
        embeddings = model.encode(texts, batch_size=batch_size).tolist()
        processing_time = time.time() - start_time
        
        logger.info(f"Processed {len(texts)} texts in {processing_time:.2f} seconds")
        
        return jsonify({
            "embeddings": embeddings,
            "model": model_name,
            "processing_time": processing_time
        })
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500

# Add compatibility with the /embeddings endpoint for the EmbeddingsService class
@app.route("/embeddings", methods=["POST"])
def embeddings():
    try:
        data = request.json
        if not data:
            return jsonify({"error": "No JSON data provided"}), 400
            
        texts = data.get("input", [])
        if not texts:
            return jsonify({"error": "No input texts provided"}), 400
            
        batch_size = data.get("batch_size", 32)
        
        start_time = time.time()
        embeddings = model.encode(texts, batch_size=batch_size).tolist()
        processing_time = time.time() - start_time
        
        # Format response for compatibility with the EmbeddingsService
        response_data = {
            "data": [{"embedding": embedding} for embedding in embeddings],
            "model": model_name,
            "processing_time": processing_time
        }
        
        logger.info(f"Processed {len(texts)} texts in {processing_time:.2f} seconds for /embeddings endpoint")
        
        return jsonify(response_data)
    except Exception as e:
        logger.error(f"Error generating embeddings: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", 80))) 