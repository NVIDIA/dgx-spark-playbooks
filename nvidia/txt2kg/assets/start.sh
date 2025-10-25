#!/bin/bash

# Setup script for txt2kg project

# Parse command line arguments
DEV_FRONTEND=false
USE_COMPLETE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev-frontend)
      DEV_FRONTEND=true
      shift
      ;;
    --complete)
      USE_COMPLETE=true
      shift
      ;;
    --help|-h)
      echo "Usage: ./start.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --dev-frontend    Run frontend in development mode (without Docker)"
      echo "  --complete        Use complete stack (vLLM, Pinecone, Sentence Transformers)"
      echo "  --help, -h        Show this help message"
      echo ""
      echo "Default: Starts minimal stack with Ollama, ArangoDB, and Next.js frontend"
      echo ""
      echo "Examples:"
      echo "  ./start.sh                # Start minimal demo (recommended)"
      echo "  ./start.sh --complete     # Start with all optional services"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './start.sh --help' for usage information"
      exit 1
      ;;
  esac
done

if [ "$DEV_FRONTEND" = true ]; then
  echo "Starting frontend in development mode..."
  cd frontend
  if ! command -v pnpm &> /dev/null; then
    echo "Error: pnpm is not installed. Install it with: npm install -g pnpm"
    exit 1
  fi
  pnpm run dev
  exit 0
fi

# Check for GPU support
echo "Checking for GPU support..."
if command -v nvidia-smi &> /dev/null; then
  if nvidia-smi &> /dev/null; then
    echo "✓ NVIDIA GPU detected"
    GPU_INFO=$(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -n1)
    echo "  GPU: $GPU_INFO"
  else
    echo "⚠ NVIDIA GPU not accessible. Services will run in CPU mode (slower)."
  fi
else
  echo "⚠ nvidia-smi not found. Services will run in CPU mode (slower)."
fi

# Check which Docker Compose version is available
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
  echo "Using Docker Compose V2"
elif command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
  echo "Using Docker Compose V1 (deprecated - consider upgrading)"
else
  echo "Error: Neither 'docker compose' nor 'docker-compose' is available"
  echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
  exit 1
fi

# Check Docker daemon permissions
echo "Checking Docker permissions..."
if ! docker info &> /dev/null; then
  echo ""
  echo "=========================================="
  echo "ERROR: Docker Permission Denied"
  echo "=========================================="
  echo ""
  echo "You don't have permission to connect to the Docker daemon."
  echo ""
  echo "To fix this, run one of the following:"
  echo ""
  echo "Option 1 (Recommended): Add your user to the docker group"
  echo "  sudo usermod -aG docker \$USER"
  echo "  newgrp docker"
  echo ""
  echo "Option 2: Run this script with sudo (not recommended)"
  echo "  sudo ./start.sh"
  echo ""
  echo "After adding yourself to the docker group, you may need to log out"
  echo "and log back in for the changes to take effect."
  echo ""
  exit 1
fi
echo "✓ Docker permissions OK"

# Build the docker-compose command
if [ "$USE_COMPLETE" = true ]; then
  CMD="$DOCKER_COMPOSE_CMD -f $(pwd)/deploy/compose/docker-compose.complete.yml"
  echo "Using complete stack (Ollama, vLLM, Pinecone, Sentence Transformers)..."
else
  CMD="$DOCKER_COMPOSE_CMD -f $(pwd)/deploy/compose/docker-compose.yml"
  echo "Using minimal configuration (Ollama + ArangoDB only)..."
fi

# Execute the command
echo ""
echo "Starting services..."
echo "Running: $CMD up -d"
cd $(dirname "$0")
eval "$CMD up -d"

echo ""
echo "=========================================="
echo "txt2kg is now running!"
echo "=========================================="
echo ""
echo "Core Services:"
echo "  • Web UI: http://localhost:3001"
echo "  • ArangoDB: http://localhost:8529"
echo "  • Ollama API: http://localhost:11434"
echo ""

if [ "$USE_COMPLETE" = true ]; then
  echo "Additional Services (Complete Stack):"
  echo "  • Local Pinecone: http://localhost:5081"
  echo "  • Sentence Transformers: http://localhost:8000"
  echo "  • vLLM API: http://localhost:8001"
  echo ""
fi

echo "Next steps:"
echo "  1. Pull an Ollama model (if not already done):"
echo "     docker exec ollama-compose ollama pull llama3.1:8b"
echo ""
echo "  2. Open http://localhost:3001 in your browser"
echo "  3. Upload documents and start building your knowledge graph!"
echo ""
echo "Other options:"
echo "  • Stop services: ./stop.sh"
echo "  • Run frontend in dev mode: ./start.sh --dev-frontend"
echo "  • Use complete stack: ./start.sh --complete"
echo "  • View logs: docker compose logs -f"
echo "" 
