#!/bin/bash
set -e

echo "=== Checking system prerequisites ==="
python3 --version
pip3 --version || true
nvidia-smi

echo "=== Creating Python virtual environment ==="
[ -d comfyui-env ] || python3 -m venv comfyui-env
source comfyui-env/bin/activate

echo "=== Installing PyTorch with CUDA 13.0 support ==="
pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu130

echo "=== Cloning ComfyUI repository ==="
[ -d ComfyUI ] || git clone --branch v0.28.2 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/

echo "=== Installing ComfyUI dependencies ==="
pip3 install -r requirements.txt

echo "=== Downloading DreamShaper 8 checkpoint ==="
cd models/checkpoints/
wget -nc https://huggingface.co/Lykon/DreamShaper/resolve/main/DreamShaper_8_pruned.safetensors
cd ../../

echo "=== Setup complete. ==="
echo "To start the server, run:"
echo "  curl -fsSL https://raw.githubusercontent.com/NVIDIA/dgx-spark-playbooks/refs/heads/main/nvidia/comfy-ui/assets/launch.sh | bash"
