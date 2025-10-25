#!/bin/bash

# Stop script for txt2kg project

# Parse command line arguments
USE_COMPLETE=false

while [[ $# -gt 0 ]]; do
  case $1 in
    --complete)
      USE_COMPLETE=true
      shift
      ;;
    --help|-h)
      echo "Usage: ./stop.sh [OPTIONS]"
      echo ""
      echo "Options:"
      echo "  --complete        Stop complete stack (vLLM, Pinecone, Sentence Transformers)"
      echo "  --help, -h        Show this help message"
      echo ""
      echo "Default: Stops minimal stack with Ollama, ArangoDB, and Next.js frontend"
      echo ""
      echo "Examples:"
      echo "  ./stop.sh                # Stop minimal demo"
      echo "  ./stop.sh --complete     # Stop complete stack"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Run './stop.sh --help' for usage information"
      exit 1
      ;;
  esac
done

# Check which Docker Compose version is available
DOCKER_COMPOSE_CMD=""
if docker compose version &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker compose"
elif command -v docker-compose &> /dev/null; then
  DOCKER_COMPOSE_CMD="docker-compose"
else
  echo "Error: Neither 'docker compose' nor 'docker-compose' is available"
  echo "Please install Docker Compose: https://docs.docker.com/compose/install/"
  exit 1
fi

# Check Docker daemon permissions
if ! docker info &> /dev/null; then
  echo ""
  echo "=========================================="
  echo "ERROR: Docker Permission Denied"
  echo "=========================================="
  echo ""
  echo "You don't have permission to connect to the Docker daemon."
  echo ""
  echo "To fix this, add your user to the docker group:"
  echo "  sudo usermod -aG docker \$USER"
  echo "  newgrp docker"
  echo ""
  exit 1
fi

# Build the docker-compose command
if [ "$USE_COMPLETE" = true ]; then
  CMD="$DOCKER_COMPOSE_CMD -f $(pwd)/deploy/compose/docker-compose.complete.yml"
  echo "Stopping complete stack..."
else
  CMD="$DOCKER_COMPOSE_CMD -f $(pwd)/deploy/compose/docker-compose.yml"
  echo "Stopping minimal configuration..."
fi

# Execute the command
echo "Running: $CMD down"
cd $(dirname "$0")
eval "$CMD down"

echo ""
echo "=========================================="
echo "txt2kg has been stopped"
echo "=========================================="
echo ""
echo "To start again, run: ./start.sh"
echo ""
