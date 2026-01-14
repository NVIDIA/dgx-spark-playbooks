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

# Launch vLLM with NVIDIA Triton Inference Server optimized build
# This should have proper support for compute capability 12.1 (DGX Spark)

# Enable unified memory usage for DGX Spark
export CUDA_MANAGED_FORCE_DEVICE_ALLOC=1
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Enable CUDA unified memory and oversubscription
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Optimized environment for performance
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

# Enable CUDA optimizations
export VLLM_USE_MODELSCOPE=false

# Enable FP8 MoE optimizations for Nemotron and other MoE models
export VLLM_USE_FLASHINFER_MOE_FP8=1
export VLLM_USE_FLASHINFER_MOE_FP4=1

# Enable FlashInfer attention backend for better performance
export VLLM_ATTENTION_BACKEND=FLASHINFER

# First, test basic CUDA functionality
echo "=== Testing CUDA functionality ==="
python3 -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version: {torch.version.cuda}')
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        props = torch.cuda.get_device_properties(i)
        print(f'GPU {i}: {props.name} (compute capability {props.major}.{props.minor})')
        # Try basic CUDA operation
        try:
            x = torch.randn(10, 10).cuda(i)
            y = torch.matmul(x, x.T)
            print(f'GPU {i}: Basic CUDA operations work')
        except Exception as e:
            print(f'GPU {i}: CUDA operation failed: {e}')
"

echo "=== Starting optimized vLLM server ==="

# Check GPU compute capability for optimal settings
COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "unknown")
echo "Detected GPU compute capability: $COMPUTE_CAPABILITY"

# Use environment variable if set, otherwise default to Qwen (not gated)
if [ -n "$VLLM_MODEL" ]; then
    MODEL_TO_USE="$VLLM_MODEL"
    echo "Using model from environment: $MODEL_TO_USE"
else
    # Default to Qwen 2.5 7B - not gated, no HuggingFace token required
    MODEL_TO_USE="Qwen/Qwen2.5-7B-Instruct"
    echo "Using default model: $MODEL_TO_USE"
fi

# Configure settings based on model size and GPU architecture
# Check if using 8B or smaller model
if [[ "$MODEL_TO_USE" == *"8B"* ]] || [[ "$MODEL_TO_USE" == *"7B"* ]] || [[ "$MODEL_TO_USE" == *"3B"* ]] || [[ "$MODEL_TO_USE" == *"1B"* ]]; then
    echo "Configuring for smaller model (8B or less)"
    QUANTIZATION_FLAG=""
    GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.9}"
    MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-8192}"
    MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-64}"
    MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}"
    CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-0}"
elif [[ "$COMPUTE_CAPABILITY" == "12.1" ]] || [[ "$COMPUTE_CAPABILITY" == "10.0" ]]; then
    # Blackwell/DGX Spark architecture with larger model - use CPU offloading
    echo "Configuring for large model on Blackwell/DGX Spark with CPU offloading"
    QUANTIZATION_FLAG=""
    GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
    MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
    MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
    CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-50}"
else
    # Other architectures with larger model
    echo "Configuring for large model on GPU architecture: $COMPUTE_CAPABILITY"
    QUANTIZATION_FLAG=""
    GPU_MEMORY_UTIL="${VLLM_GPU_MEMORY_UTILIZATION:-0.7}"
    MAX_MODEL_LEN="${VLLM_MAX_MODEL_LEN:-4096}"
    MAX_NUM_SEQS="${VLLM_MAX_NUM_SEQS:-16}"
    MAX_BATCHED_TOKENS="${VLLM_MAX_NUM_BATCHED_TOKENS:-4096}"
    CPU_OFFLOAD_GB="${VLLM_CPU_OFFLOAD_GB:-40}"
fi

echo ""
echo "=== vLLM Configuration ==="
echo "Model: $MODEL_TO_USE"
echo "GPU memory utilization: $GPU_MEMORY_UTIL"
echo "Max model length: $MAX_MODEL_LEN"
echo "Max num seqs: $MAX_NUM_SEQS"
echo "Max batched tokens: $MAX_BATCHED_TOKENS"
echo "CPU Offload: ${CPU_OFFLOAD_GB}GB"
echo "Quantization: ${QUANTIZATION_FLAG:-'none'}"
echo ""

# Build command - only add cpu-offload-gb if > 0
VLLM_CMD="vllm serve $MODEL_TO_USE \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len $MAX_MODEL_LEN \
  --max-num-seqs $MAX_NUM_SEQS \
  --gpu-memory-utilization $GPU_MEMORY_UTIL \
  --kv-cache-dtype auto \
  --trust-remote-code \
  --served-model-name $MODEL_TO_USE"

# Note: For FP8 models, vLLM auto-detects quantization from model config
# No need to specify --dtype float8 (not supported in vLLM 0.11.0)
if [[ "$MODEL_TO_USE" == *"FP8"* ]] || [[ "$MODEL_TO_USE" == *"fp8"* ]]; then
  echo "Detected FP8 model - vLLM will auto-detect FP8 quantization from model config"
fi

# Add CPU offload only for larger models
if [ "$CPU_OFFLOAD_GB" -gt 0 ] 2>/dev/null; then
  VLLM_CMD="$VLLM_CMD --cpu-offload-gb $CPU_OFFLOAD_GB"
fi

# Add quantization if specified
if [ -n "$QUANTIZATION_FLAG" ]; then
  VLLM_CMD="$VLLM_CMD $QUANTIZATION_FLAG"
fi

echo "Running: $VLLM_CMD"
exec $VLLM_CMD