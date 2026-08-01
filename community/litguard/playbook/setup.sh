#!/bin/bash
set -e

echo "=== LitGuard DGX Spark Setup ==="

# Check for NVIDIA GPU
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
fi

echo "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# Check for Docker
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found. Install Docker first."
    exit 1
fi

# Check for nvidia-container-toolkit
if ! docker info 2>/dev/null | grep -q "nvidia"; then
    echo "WARNING: nvidia-container-toolkit may not be installed."
    echo "Install it with:"
    echo "  sudo apt-get install -y nvidia-container-toolkit"
    echo "  sudo systemctl restart docker"
fi

# Build and start
echo ""
echo "Starting LitGuard..."
docker compose up --build -d

echo ""
echo "=== LitGuard is starting ==="
echo "API:  http://localhost:8234"
echo "UI:   http://localhost:3000"
echo ""
echo "Models will be downloaded on first run (may take a few minutes)."
echo "Check logs: docker compose logs -f"
