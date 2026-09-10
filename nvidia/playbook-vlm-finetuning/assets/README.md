# VLM Fine-tuning Recipes

This repository contains comprehensive fine-tuning recipes for Vision-Language Models (VLMs), supporting both **image** and **video** understanding tasks with modern models and training techniques.

## 🎯 Available Recipes

### 📸 Image VLM Fine-tuning (`ui_image/`)
- **Model**: Qwen2.5-VL-7B-Instruct
- **Task**: Wildfire detection from satellite imagery
- **Training Method**: GRPO (Generalized Reward Preference Optimization) and LoRA (Low-rank Adaptation)

### 🎥 Video VLM Fine-tuning (`ui_video/`)
- **Model**: InternVL3-8B
- **Task**: Dangerous driving detection and structured metadata generation
- **Training Method**: Supervised Fine-tuning on Multimodal Instructions

## 🚀 Quick Start

### 1. Build the Docker Container

```bash
# Build the VLM fine-tuning container
docker build --build-arg HF_TOKEN=$HF_TOKEN -t vlm_demo .
```

### 2. Launch the Container

```bash
# Enter the correct directory for building the image
cd vlm-finetuning/assets

# Run the container with GPU support
sh launch.sh

# Enter the mounted directory within the container
cd /vlm_finetuning
```

> **Note**: The same Docker container and launch commands work for both image and video VLM recipes. The container includes all necessary dependencies including FFmpeg, Decord, and optimized libraries for both workflows.

## 📚 Detailed Instructions

Each recipe includes comprehensive documentation:

- **[Image VLM README](ui_image/README.md)**: Complete guide for wildfire detection fine-tuning with Qwen2.5-VL, including dataset setup, GRPO training configuration, and interactive inference
- **[Video VLM README](ui_video/README.md)**: Full walkthrough for dangerous driving detection with InternVL3, covering video data preparation, training notebooks, and structured output generation

