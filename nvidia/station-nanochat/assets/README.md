# Nanochat Training on DGX Station

This project demonstrates training of [nanochat](https://github.com/karpathy/nanochat), "the best ChatGPT that $100 can buy," on DGX Station. The demo includes tokenization, pretraining, midtraining, supervised fine-tuning (SFT), and inference through both CLI and web UI.

## Overview

The project includes:
- **Full LLM Pipeline**: Tokenizer training, pretraining, midtraining, and SFT
- **Custom Tokenizer**: BPE tokenizer with 65K vocabulary trained on FineWeb
- **Evaluation Suite**: CORE, ARC, GSM8K, HumanEval, MMLU benchmarks
- **Interactive Inference**: Chat with your model via CLI or web UI
- **Docker Support**: Complete containerized environment with PyTorch NGC

## Contents
1. [Environment Setup](#1-environment-setup)
2. [Preparation](#2-preparation)
3. [Training](#3-training)
4. [Customization](#4-customization)
5. [Inference](#5-inference)
6. [Evaluation Results](#6-evaluation-results)
7. [Architecture Details](#7-architecture-details)

## 1. Environment Setup

### 1.1 Prerequisites

Before starting, ensure you have:
- DGX Station with driver and CUDA toolkit setup
- Docker installed on the system
- Huggingface and WandB API access

### 1.4 Enviornment Setup

For training visualization and logging, set up your W&B API key.  If you don't have a W&B account, you can create one at [wandb.ai](https://wandb.ai/). Additionally, a Huggingface token will be required for downloading certain datasets for model evaluation. Likewise, you can create a HF token by following the instructions at (huggingface.co](https://huggingface.co/docs/hub/en/security-tokens).

```bash
export WANDB_API_KEY=<YOUR_WANDB_API_KEY>
export HF_TOKEN=<YOUR_HF_TOKEN>
```

## 2. Preparation

### 2.1 Clone the repository 

Clone the current repository and change directories to the station-nanochat repository. 

```bash
cd station-nanochat
```

### 2.2 Nanochat Setup

After navigating to the assets folder and run the setup script to clone nanochat and build the Docker image on both nodes.

```bash
sh setup.sh
```

The setup script will:
- Clone the nanochat repository
- Copy the modified `speedrun_station.sh` script for training on station
- Build a custom Docker image for nanochat

Verify your directory structure after setup:

```
station-nanochat/assets/
├── Dockerfile
├── launch.sh
├── setup.sh
├── speedrun_station.sh
└── nanochat/
    ├── README.md
    ├── speedrun.sh (replaced with speedrun_station.sh)
    ├── scripts/
    ├── nanochat/
    └── ...
```

### 2.3 Verify Docker Image

Ensure the Docker image was built successfully on both nodes:

```bash
# On your system
docker images | grep nanochat
```

You should see the `nanochat` image listed on your system.

## 3. Training

### 3.1 Launch Training

Start training on DGX Station:

```bash
# Ensure that the previous environment variables are exported
# Launch training on both nodes
./ launch.sh 
```

The training script will automatically:
1. Download ~24GB of FineWeb pretraining data
2. Train a BPE tokenizer with 65K vocabulary
3. Pretrain a 561M parameter Transformer model (d20)
4. Run midtraining to teach conversation format
5. Fine-tune with supervised learning (SFT)
6. Generate evaluation reports

### 3.2 Training Duration

Expected training time on station:
- **Speedrun (d20)**: Upto 16 hours for 561M parameter model

The training uses PyTorch with:
- **Model Architecture**: GPT-style Transformer with 20 layers
- **Parameters**: 561M
- **Training Tokens**: ~11.2B tokens (Chinchilla optimal)
- **Optimizer**: Muon for pretraining, AdamW for finetuning
- **Precision**: Mixed precision (bfloat16)

### 3.3 Monitoring Training

To view the training progress via W&B, monitor any stage of nanochat training at:
```
https://wandb.ai/<your-username>/projects
```

Track key metrics:
- **Training loss**: Should decrease from ~3.5 to ~2.5
- **Validation loss**: Monitor for overfitting
- **Learning rate**: Follows cosine decay schedule
- **Throughput**: Tokens processed per second

### 3.4 Training Stages

The training pipeline consists of:

#### Stage 1: Tokenizer Training
- Downloads 2B characters from FineWeb dataset
- Trains BPE tokenizer with 65,536 vocabulary size
- Achieves ~4.8 characters per token compression

#### Stage 2: Base Model Pretraining
- Downloads 240 data shards (~24GB) from FineWeb
- Pretrains d20 model (561M params) on 11.2B tokens
- Evaluates on CORE benchmark (DCLM paper metrics)

#### Stage 3: Midtraining
- Introduces conversation special tokens (`<|im_start|>`, `<|im_end|>`)
- Trains on synthetic identity conversations
- Teaches model chat format and basic personality

#### Stage 4: Supervised Fine-tuning (SFT)
- Fine-tunes on SmolTalk dataset
- Improves conversation quality and instruction following
- Final model ready for chat inference

### 3.5 Checkpoints

Training checkpoints are automatically saved in `~/.cache/nanochat/`:
- `model_base.pt`: Pretrained base model
- `model_mid.pt`: After midtraining
- `model_sft.pt`: Final fine-tuned model
- `tokenizer.model`: Trained BPE tokenizer


## 4. Customization

For faster experimentation or testing the distributed setup, you can train a smaller model. This is useful for validating your infrastructure and workflow before committing to the full 5-day training run.

### 4.1 Remove existing nanochat installation

If you have previously run the setup, first remove the nanochat folder:

```bash
# From the assets directory
rm -rf nanochat
```

### 4.2 Modify speedrun_station.sh for minimal configuration

Edit `speedrun_station.sh` to use a smaller model configuration:

```bash
# Reduce data shards (50 shards instead of 240 for quick testing)
python -m nanochat.dataset -n 50 &

# Use depth=4 for minimal training
python -m scripts.base_train --depth=4 --device_batch_size=32
```

## 5. Inference

After training completes, you can interact with your trained model through multiple interfaces.

### 5.1 Web UI (Recommended)

Launch the ChatGPT-style web interface:

```bash

# Activate the virtual environment
cd nanochat
source ../.venv/bin/activate

# Start the web server
python -m scripts.chat_web
```

Access the UI at:
```
http://<SYSTEM_IP>:8000
```

The web UI provides:
- ChatGPT-style conversation interface
- Message history
- Real-time streaming responses
- Clean, modern design

### 5.2 Command Line Interface

For quick interactions via terminal:

```bash
# Interactive chat mode
python -m scripts.chat_cli

# Single prompt mode
python -m scripts.chat_cli -p "Why is the sky blue?"

# Specify checkpoint (base, mid, or sft)
python -m scripts.chat_cli -i sft -p "Write me a haiku about distributed training"
```

### 5.3 Sample Prompts

Try these prompts to test your model:

**Reasoning:**
```
Why do astronauts float in space?
```

**Math:**
```
A model trains for 3 epochs. Each epoch has 1000 steps and each step takes 0.5 seconds. How many minutes does the full training take?
```

**Code:**
```
Write a Python function to calculate fibonacci numbers
```

**Note:** The d20 speedrun model (561M parameters, ~4e19 FLOPs) is intentionally designed as a compact educational demonstration and has significant limitations. Expected behaviors include factual inaccuracies, hallucinations, and inconsistent reasoning. These characteristics are inherent to models trained with limited parameters and compute resources.

## 6. Evaluation Results

### 6.1 Training Report

After training completes, a comprehensive report is generated at `nanochat/report.md`. View it with:

```bash
cat nanochat/report.md
```

The report includes:
- System information and training configuration
- Training curves and loss plots
- Evaluation metrics across all benchmarks
- Sample generations at each training stage
- Total training time and cost breakdown


## 7. Architecture Details

### 7.1 Model Configuration (d20)

```
Layers: 20
Embedding Dimension: 1024
Attention Heads: 16
Context Length: 1024 tokens
Vocabulary Size: 65,536
Total Parameters: 561M
```

### 4.3 Re-run setup script

After modifying `speedrun_station.sh`, run the setup script again to deploy the changes to both nodes:

```bash

# Run setup to clone nanochat and build Docker images
sh setup.sh 
```

Then proceed with Section 3 (Training) to launch training.

## Troubleshooting

### Common Issues

**Issue**: Out of memory (OOM) errors
```
RuntimeError: CUDA out of memory
```
**Solution**: Reduce batch size in the training scripts:
```bash
--device_batch_size=16  # or 8, 4, 2, 1
```

**Issue**: Docker container not starting
**Solution**: 
- Check GPU availability: `nvidia-smi`
- Ensure no other containers using GPUs: `docker ps`
- Verify Docker has GPU access: `docker run --rm --gpus all nvidia/cuda:12.0-base nvidia-smi`

## Cleanup

### Stop Training

To stop training early, interrupt both containers:

```bash
# From the terminal running launch.sh
Ctrl+C

# Or manually stop containers
docker stop nanochat
```

### Clear Cache

To free up disk space after training:

```bash
# On both nodes
rm -rf ~/.cache/nanochat

# Clear Docker system
docker system prune -a
```

## Credits

This project is built upon:
- **[nanochat](https://github.com/karpathy/nanochat)** by Andrej Karpathy - The base LLM training framework
- **[nanoGPT](https://github.com/karpathy/nanoGPT)** by Andrej Karpathy - Inspiration for minimal LLM training
- **[modded-nanoGPT](https://github.com/KellerJordan/modded-nanogpt)** by Keller Jordan - Optimized training techniques
- **[FineWeb](https://huggingface.co/datasets/HuggingFaceFW/fineweb)** by HuggingFace - Pretraining dataset
- **[SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk)** by HuggingFace - Finetuning dataset

## License

MIT - See nanochat repository for full license details.

---

**Note**: This is an educational project demonstrating distributed LLM training. The resulting model is a micro-model suitable for learning but not production use. For state-of-the-art performance, consider using pre-trained models like GPT-4, Claude, or Llama.

