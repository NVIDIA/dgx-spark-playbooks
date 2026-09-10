# Deployment Configuration

This directory contains all deployment-related configuration for the txt2kg project.

## Structure

- **compose/**: Docker Compose configuration
  - `docker-compose.yml`: ArangoDB + Ollama (default)
  - `docker-compose.vllm.yml`: Neo4j + vLLM (GPU-accelerated)

- **app/**: Frontend application Docker configuration
  - Dockerfile for Next.js application

- **services/**: Containerized services
  - **ollama/**: Ollama LLM inference service (default)
  - **vllm/**: vLLM inference service with GPU support (via `--vllm` flag)
  - **sentence-transformers/**: Sentence transformer service for embeddings (via `--vector-search` flag)
  - **gpu-viz/**: GPU-accelerated graph visualization services (run separately)
  - **gnn_model/**: Graph Neural Network model service (experimental)

## Usage

**Recommended: Use the start script**

```bash
# Default: ArangoDB + Ollama
./start.sh

# Use Neo4j + vLLM (GPU-accelerated, for DGX Spark/GB300)
./start.sh --vllm

# Enable vector search (Qdrant + Sentence Transformers)
./start.sh --vector-search

# Combine options
./start.sh --vllm --vector-search

# Development mode (run frontend without Docker)
./start.sh --dev-frontend
```

**Manual Docker Compose commands:**

```bash
# Default: ArangoDB + Ollama
docker compose -f deploy/compose/docker-compose.yml up -d

# Neo4j + vLLM
docker compose -f deploy/compose/docker-compose.vllm.yml up -d

# With vector search services (add --profile vector-search)
docker compose -f deploy/compose/docker-compose.yml --profile vector-search up -d
docker compose -f deploy/compose/docker-compose.vllm.yml --profile vector-search up -d
```

## Services Included

### Default Stack (ArangoDB + Ollama)
- **Next.js App**: Web UI on port 3001
- **ArangoDB**: Graph database on port 8529
- **Ollama**: Local LLM inference on port 11434

### vLLM Stack (`--vllm` flag) - Neo4j + vLLM
- **Next.js App**: Web UI on port 3001
- **Neo4j**: Graph database on ports 7474 (HTTP) and 7687 (Bolt)
- **vLLM**: GPU-accelerated LLM inference on port 8001

### Vector Search (`--vector-search` profile)
- **Qdrant**: Vector database on port 6333
- **Sentence Transformers**: Embedding generation on port 8000

### Optional Services (run separately)
- **GPU-Viz Services**: See `services/gpu-viz/README.md` for GPU-accelerated visualization
- **GNN Model Service**: See `services/gnn_model/README.md` for experimental GNN-based RAG

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│  Default Stack (./start.sh)          │  vLLM Stack (--vllm)     │
├──────────────────────────────────────┼──────────────────────────┤
│                                      │                          │
│  ┌─────────────┐                     │  ┌─────────────┐         │
│  │   Next.js   │ port 3001           │  │   Next.js   │ 3001    │
│  └──────┬──────┘                     │  └──────┬──────┘         │
│         │                            │         │                │
│  ┌──────┴──────┐  ┌─────────────┐    │  ┌──────┴──────┐  ┌─────┐│
│  │  ArangoDB   │  │   Ollama    │    │  │   Neo4j     │  │vLLM ││
│  │  port 8529  │  │ port 11434  │    │  │  port 7474  │  │8001 ││
│  └─────────────┘  └─────────────┘    │  └─────────────┘  └─────┘│
│                                      │                          │
└──────────────────────────────────────┴──────────────────────────┘

Optional (--vector-search): Qdrant (6333) + Sentence Transformers (8000)
```
