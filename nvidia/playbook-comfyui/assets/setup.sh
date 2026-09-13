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
[ -d ComfyUI ] || git clone --branch v0.33.2 https://github.com/comfyanonymous/ComfyUI.git
cd ComfyUI/

echo "=== Installing ComfyUI dependencies ==="
pip3 install -r requirements.txt

echo "=== Downloading Z-Image-Turbo models (about 20 GB) ==="
BASE=https://huggingface.co/Comfy-Org/z_image_turbo/resolve/main/split_files
wget -nc -P models/diffusion_models/ "$BASE/diffusion_models/z_image_turbo_bf16.safetensors"
wget -nc -P models/text_encoders/ "$BASE/text_encoders/qwen_3_4b.safetensors"
wget -nc -P models/vae/ "$BASE/vae/ae.safetensors"

echo "=== Setup complete. ==="
echo "To start the server, run the launch.sh command from the playbook."
