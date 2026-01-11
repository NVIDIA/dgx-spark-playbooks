# Text to Knowledge Graph

> Transform unstructured text into interactive knowledge graphs with LLM inference and graph visualization


## Table of Contents

- [Overview](#overview)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook demonstrates how to build and deploy a comprehensive knowledge graph generation and visualization solution that serves as a reference for knowledge graph extraction.
The unified memory architecture enables running larger, more accurate models that produce higher-quality knowledge graphs and deliver superior downstream GraphRAG performance.

This txt2kg playbook transforms unstructured text documents into structured knowledge graphs using:
- **Knowledge Triple Extraction**: Using Ollama with GPU acceleration for local LLM inference to extract subject-predicate-object relationships
- **Graph Database Storage**: ArangoDB for storing and querying knowledge triples with relationship traversal
- **GPU-Accelerated Visualization**: Three.js WebGPU rendering for interactive 2D/3D graph exploration

> **Future Enhancements**: Vector embeddings and GraphRAG capabilities are planned enhancements.

## What you'll accomplish

You will have a fully functional system capable of processing documents, generating and editing knowledge graphs, and providing querying, accessible through an interactive web interface.
The setup includes:
- **Local LLM Inference**: Ollama for GPU-accelerated LLM inference with no API keys required
- **Graph Database**: ArangoDB for storing and querying triples with relationship traversal
- **Interactive Visualization**: GPU-accelerated graph rendering with Three.js WebGPU
- **Modern Web Interface**: Next.js frontend with document management and query interface
- **Fully Containerized**: Reproducible deployment with Docker Compose and GPU support

## Prerequisites

-  DGX Spark with latest NVIDIA drivers
-  Docker installed and configured with NVIDIA Container Toolkit
-  Docker Compose


## Time & risk

- **Duration**:
  - 2-3 minutes for initial setup and container deployment
  - 5-10 minutes for Ollama model download (depending on model size)
  - Immediate document processing and knowledge graph generation

- **Risks**:
  - GPU memory requirements depend on chosen Ollama model size
  - Document processing time scales with document size and complexity

- **Rollback**: Stop and remove Docker containers, delete downloaded models if needed
- **Last Updated**: 12/02/2025
  - Knowledge graph search with multi-hop graph traversal
  - Improved UI/UX

## Instructions

## Step 1. Clone the repository

In a terminal, clone the txt2kg repository and navigate to the project directory.

```bash
git clone https://github.com/NVIDIA/dgx-spark-playbooks
cd dgx-spark-playbooks/nvidia/txt2kg/assets
```

## Step 2. Start the txt2kg services

Use the provided start script to launch all required services. This will set up Ollama, ArangoDB, and the Next.js frontend:

```bash
./start.sh
```

The script will automatically:
- Check for GPU availability
- Start Docker Compose services
- Set up ArangoDB database
- Launch the web interface

## Step 3. Pull an Ollama model (optional)

Download a language model for knowledge extraction. The default model loaded is Llama 3.1 8B:

```bash
docker exec ollama-compose ollama pull <model-name>
```

Browse available models at [https://ollama.com/search](https://ollama.com/search)

> [!NOTE]
> The unified memory architecture enables running larger models like 70B parameters, which produce significantly more accurate knowledge triples.

## Step 4. Access the web interface

Open your browser and navigate to:

```
http://localhost:3001
```

You can also access individual services:
- **ArangoDB Web Interface**: http://localhost:8529 
- **Ollama API**: http://localhost:11434

## Step 5. Upload documents and build knowledge graphs

#### 5.1. Document Upload
- Use the web interface to upload text documents (markdown, text, CSV supported)
- Documents are automatically chunked and processed for triple extraction

#### 5.2. Knowledge Graph Generation
- The system extracts subject-predicate-object triples using Ollama
- Triples are stored in ArangoDB for relationship querying

#### 5.3. Interactive Visualization
- View your knowledge graph in 2D or 3D with GPU-accelerated rendering
- Explore nodes and relationships interactively

#### 5.4. Graph-based Queries
- Ask questions about your documents using the query interface
- Graph traversal enhances context with entity relationships from ArangoDB
- LLM generates responses using the enriched graph context

> **Future Enhancement**: GraphRAG capabilities with vector-based KNN search for entity retrieval are planned.

## Step 6. Cleanup and rollback

Stop all services and optionally remove containers:

```bash
## Stop services
docker compose down

## Remove containers and volumes (optional)
docker compose down -v

## Remove downloaded models (optional)
docker exec ollama-compose ollama rm llama3.1:8b
```

## Step 7. Next steps

- Experiment with different Ollama models for varied extraction quality
- Customize triple extraction prompts for domain-specific knowledge
- Explore advanced graph querying and visualization features

## Troubleshooting

| Symptom | Cause | Fix |
|---------|--------|-----|
| Ollama performance issues | Suboptimal settings for DGX Spark | Set environment variables:<br>`OLLAMA_FLASH_ATTENTION=1` (enables flash attention for better performance)<br>`OLLAMA_KEEP_ALIVE=30m` (keeps model loaded for 30 minutes)<br>`OLLAMA_MAX_LOADED_MODELS=1` (avoids VRAM contention)<br>`OLLAMA_KV_CACHE_TYPE=q8_0` (reduces KV cache VRAM with minimal performance impact) |
| VRAM exhausted or memory pressure (e.g. when switching between Ollama models) | Linux buffer cache consuming GPU memory | Flush buffer cache: `sudo sync; sudo sh -c 'echo 3 > /proc/sys/vm/drop_caches'` |
| Slow triple extraction | Large model or large context window | Reduce document chunk size or use faster models |
| ArangoDB connection refused | Service not fully started | Wait 30s after start.sh, verify with `docker ps` |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. 
> With many applications still updating to take advantage of UMA, you may encounter memory issues even when within 
> the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:
```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```
