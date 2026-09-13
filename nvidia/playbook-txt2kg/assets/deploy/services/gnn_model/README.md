# GNN Model Service (Experimental)

**Status**: This is an experimental service for serving Graph Neural Network models trained for enhanced RAG retrieval.

**Note**: This service is **not included** in the default docker-compose configurations and must be deployed separately.

## Overview

This service provides a REST API for serving predictions from a Graph Neural Network (GNN) model that enhances knowledge graph retrieval:

- Load pre-trained GNN models (GAT architecture)
- Process queries with graph-structured knowledge
- Combine GNN embeddings with LLM generation
- Compare GNN-based retrieval vs traditional RAG

## Getting Started

### Prerequisites

- Python 3.8+
- PyTorch and PyTorch Geometric
- A trained model file (created using `train_export.py` in `scripts/gnn/`)
- Docker (optional)

### Training the Model

Before using the service, you must train a GNN model using the training pipeline:

```bash
# See scripts/gnn/README.md for full instructions

# 1. Preprocess data from ArangoDB
python scripts/gnn/preprocess_data.py --use_arango --output_dir ./output

# 2. Train the model
python scripts/gnn/train_test_gnn.py --output_dir ./output

# 3. Export model for serving
python deploy/services/gnn_model/train_export.py --output_dir models
```

This creates the `tech-qa-model.pt` file needed by the service.

### Running the Service

#### Option A: Direct Python

```bash
cd deploy/services/gnn_model
pip install -r requirements.txt
python app.py
```

Service runs on: http://localhost:5000

#### Option B: Docker

```bash
cd deploy/services/gnn_model
docker build -t gnn-model-service .
docker run -p 5000:5000 -v $(pwd)/models:/app/models gnn-model-service
```


## API Endpoints

### Health Check

```
GET /health
```

Returns the health status of the service.

### Prediction

```
POST /predict
```

Request body:
```json
{
  "question": "Your question here",
  "context": "Retrieved context information"
}
```

Response:
```json
{
  "question": "Your question here",
  "answer": "The generated answer"
}
```

## Using the Client Example

A simple client script is provided to test the service:

```bash
python deploy/services/gnn_model/client_example.py --question "What is the capital of France?" --context "France is a country in Western Europe. Its capital is Paris, which is known for the Eiffel Tower."
```

This script also includes a placeholder for comparing the GNN-based approach with a traditional RAG approach.

## Architecture

The GNN model service uses:
- A Graph Attention Network (GAT) to process graph structured data
- A Language Model (LLM) to generate answers
- A combined architecture (GRetriever) that leverages both components

## Integration with txt2kg

To integrate this service with the main txt2kg application:

1. Train a model using the GNN training pipeline
2. Deploy the GNN service on a separate port
3. Update the frontend to call the GNN service endpoints
4. Compare GNN-enhanced retrieval vs standard RAG

## Current Status

This is an experimental feature. The service code exists but requires:
- A trained GNN model
- Integration with the frontend query pipeline
- Graph construction from txt2kg knowledge graphs
- Performance benchmarking vs traditional RAG

## Future Enhancements

- Docker Compose integration for easier deployment
- Automatic model training from txt2kg graphs
- Real-time model updates as graphs grow
- Comparison UI in the frontend 