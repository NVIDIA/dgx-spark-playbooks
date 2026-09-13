# Frontend Application

This directory contains the Next.js frontend application for the txt2kg project.

## Structure

- **app/**: Next.js 15 app directory with pages and API routes
  - API routes for LLM providers (Ollama, vLLM, NVIDIA API)
  - Triple extraction and graph query endpoints
  - Settings and health check endpoints
- **components/**: React 19 components
  - Graph visualization (Three.js WebGPU)
  - PyGraphistry integration for GPU-accelerated rendering
  - RAG query interface
  - Document upload and processing
- **contexts/**: React context providers for state management
- **hooks/**: Custom React hooks
- **lib/**: Utility functions and shared logic
  - LLM service (Ollama, vLLM, NVIDIA API integration)
  - Graph database services (ArangoDB, Neo4j)
  - Qdrant vector database integration
  - RAG service for knowledge graph querying
- **public/**: Static assets
- **types/**: TypeScript type definitions for graph data structures

## Technology Stack

- **Next.js 15**: React framework with App Router
- **React 19**: Latest React with improved concurrent features
- **TypeScript**: Type-safe development
- **Tailwind CSS**: Utility-first styling
- **Three.js**: WebGL/WebGPU 3D graph visualization
- **D3.js**: Data-driven visualizations
- **LangChain**: LLM orchestration and chaining

## Development

To start the development server:

```bash
cd frontend
npm install
npm run dev
```

Or use the start script from project root:

```bash
./start.sh --dev-frontend
```

The development server will run on http://localhost:3000

## Building for Production

```bash
cd frontend
npm run build
npm start
```

Or use Docker (recommended):

```bash
# From project root
./start.sh
```

The production app will run on http://localhost:3001

## Environment Variables

Required environment variables are configured in docker-compose files:

- `ARANGODB_URL`: ArangoDB connection URL
- `OLLAMA_BASE_URL`: Ollama API endpoint
- `VLLM_BASE_URL`: vLLM API endpoint (optional)
- `NVIDIA_API_KEY`: NVIDIA API key (optional)
- `QDRANT_URL`: Qdrant vector database URL (optional)
- `SENTENCE_TRANSFORMER_URL`: Embeddings service URL (optional)

## Features

- **Knowledge Graph Extraction**: Extract triples from text using LLMs
- **Graph Visualization**: Interactive 3D visualization with Three.js WebGPU
- **RAG Queries**: Query knowledge graphs with retrieval-augmented generation
- **Multiple LLM Providers**: Support for Ollama, vLLM, and NVIDIA API
- **GPU-Accelerated Rendering**: Optional PyGraphistry integration for large graphs
- **Vector Search**: Qdrant integration for semantic search 