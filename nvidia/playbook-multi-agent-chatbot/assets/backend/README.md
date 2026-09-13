# Backend

FastAPI Python application serving as the API backend for the chatbot demo.

## Overview

The backend handles:
- Multi-model LLM integration (local models)
- Document ingestion and vector storage for RAG
- WebSocket connections for real-time chat streaming
- Image processing and analysis
- Chat history management
- Model Control Protocol (MCP) integration

## Key Features

- **Multi-model support**: Integrates various LLM providers and local models
- **RAG pipeline**: Document processing, embedding generation, and retrieval
- **Streaming responses**: Real-time token streaming via WebSocket
- **Image analysis**: Multi-modal capabilities for image understanding
- **Vector database**: Efficient similarity search for document retrieval
- **Session management**: Chat history and context persistence

## Architecture

FastAPI application with async support, integrated with vector databases for RAG functionality and WebSocket endpoints for real-time communication.

## Docker Troubleshooting

### Container Issues
- **Port conflicts**: Ensure port 8000 is not in use
- **Memory issues**: Backend requires significant RAM for model loading
- **Startup failures**: Check if required environment variables are set

### Model Loading Problems
```bash
# Check model download status
docker logs backend | grep -i "model"

# Verify model files exist
docker exec -it cbackend ls -la /app/models/

# Check available disk space
docker exec -it backend df -h
```

### Common Commands
```bash
# View backend logs
docker logs -f backend

# Restart backend container
docker restart backend

# Rebuild backend
docker-compose up --build -d backend

# Access container shell
docker exec -it backend /bin/bash

# Check API health
curl http://localhost:8000/health
```

### Performance Issues
- **Slow responses**: Check GPU availability and model size
- **Memory errors**: Increase Docker memory limit or use smaller models
- **Connection timeouts**: Verify WebSocket connections and firewall settings
