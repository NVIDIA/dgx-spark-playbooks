#!/bin/bash
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

# Setup script for txt2kg project

# Parse command line arguments
# Default: vLLM (supported on DGX Station). Use --ollama for ArangoDB + Ollama.
DEV_FRONTEND=false
USE_VLLM=true
USE_VECTOR_SEARCH=false
WAIT_FOR_VLLM=true

while [[ $# -gt 0 ]]; do
  case $1 in
    --dev-frontend)
      DEV_FRONTEND=true
      shift
      ;;
    --ollama)
      USE_VLLM=false
      shift
      ;;
    --vllm)
      USE_VLLM=true
      shift
      ;;
    --no-wait)
      WAIT_FOR_VLLM=false
      shift
      ;;
    --vector-search)
      USE_VECTOR_SEARCH=true
      shift
      ;;
    --help|-h)
      echo "Usage: ./start.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --dev-frontend    Run frontend in development mode (without Docker)"
      echo "  --ollama          Use ArangoDB + Ollama (alternative to default vLLM)"
      echo "  --vllm            Use Neo4j + vLLM (default on DGX Station)"
      echo "  --no-wait         Do not wait for vLLM to be ready (default: wait)"
      echo "  --vector-search   Enable vector search services (Qdrant + Sentence Transformers)"
      echo "  --help, -h        Show this help message"
      echo ""
      echo "Default: Neo4j + vLLM (GPU-accelerated for DGX Station)"
      echo ""
      echo "Examples:"
      echo "  ./start.sh                       # Default: Neo4j + vLLM (DGX Station)"
      echo "  ./start.sh --ollama              # ArangoDB + Ollama instead"
      echo "  ./start.sh --vector-search       # Add Qdrant + Sentence Transformers"
      echo "  ./start.sh --no-wait             # Start vLLM but do not wait for readiness"
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

# Select compose file and build command
COMPOSE_DIR="$(pwd)/deploy/compose"
PROFILES=""

if [ "$USE_VLLM" = true ]; then
  COMPOSE_FILE="$COMPOSE_DIR/docker-compose.vllm.yml"
  echo "Using Neo4j + vLLM (GPU-accelerated)..."
  echo "  ⚡ Optimized for DGX Station GB300 Ultra with high GPU memory"
else
  COMPOSE_FILE="$COMPOSE_DIR/docker-compose.yml"
  echo "Using ArangoDB + Ollama configuration..."
fi

CMD="$DOCKER_COMPOSE_CMD -f $COMPOSE_FILE"

if [ "$USE_VECTOR_SEARCH" = true ]; then
  PROFILES="--profile vector-search"
  echo "Enabling vector search (Qdrant + Sentence Transformers)..."
fi

# Execute the command
echo ""
echo "Starting services..."
echo "Running: $CMD $PROFILES up -d"
cd $(dirname "$0")
eval "$CMD $PROFILES up -d"

# When using vLLM, optionally wait for backend to be ready and show progress
if [ "$USE_VLLM" = true ] && [ "$WAIT_FOR_VLLM" = true ]; then
  echo ""
  echo "Waiting for vLLM backend to be ready (model load can take 30+ minutes)..."
  WAIT_START=$(date +%s)
  while true; do
    if curl -sf http://localhost:8001/v1/models >/dev/null 2>&1; then
      echo ""
      echo "  vLLM backend is ready."
      break
    fi
    ELAPSED=$(($(date +%s) - WAIT_START))
    printf "\r  Elapsed: %d min (check logs: docker logs vllm-service -f)   " $((ELAPSED / 60))
    sleep 30
  done
  echo ""
fi

echo ""
echo "=========================================="
echo "txt2kg is now running!"
echo "=========================================="
echo ""
echo "Core Services:"
echo "  • Web UI: http://localhost:3001"
if [ "$USE_VLLM" = true ]; then
  echo "  • Neo4j Browser: http://localhost:7474"
  echo "  • vLLM API: http://localhost:8001 (GPU-accelerated)"
else
  echo "  • ArangoDB: http://localhost:8529"
  echo "  • Ollama API: http://localhost:11434"
fi
echo ""

if [ "$USE_VECTOR_SEARCH" = true ]; then
  echo "Vector Search Services:"
  echo "  • Qdrant: http://localhost:6333"
  echo "  • Sentence Transformers: http://localhost:8000"
  echo ""
fi

echo "Next steps:"
if [ "$USE_VLLM" = true ]; then
  echo "  1. Open http://localhost:3001 in your browser"
  echo "  2. Upload documents and start building your knowledge graph!"
else
  echo "  1. Pull the Llama 3.1 405B model (if not already done):"
  echo "     docker exec ollama-compose ollama pull llama3.1:405b"
  echo ""
  echo "  2. Open http://localhost:3001 in your browser"
  echo "  3. Upload documents and start building your knowledge graph!"
fi
echo ""
echo "Other options:"
echo "  • Stop services: ./stop.sh"
echo "  • Run frontend in dev mode: ./start.sh --dev-frontend"
if [ "$USE_VLLM" = true ]; then
  echo "  • Use Ollama instead: ./start.sh --ollama"
else
  echo "  • Use vLLM (default): ./start.sh or ./start.sh --vllm"
fi
echo "  • Add vector search: ./start.sh --vector-search"
echo "  • View logs: docker compose logs -f"
echo "  • Skip vLLM wait next time: ./start.sh --no-wait"
echo ""
