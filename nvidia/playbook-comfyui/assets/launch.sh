#!/bin/bash
set -e

source comfyui-env/bin/activate
cd ComfyUI/ && python main.py --listen 0.0.0.0
