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
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Enable CUDA unified memory and oversubscription
export CUDA_VISIBLE_DEVICES=0
export PYTORCH_NO_CUDA_MEMORY_CACHING=0

# Force vLLM to use CPU offloading for large models
export VLLM_CPU_OFFLOAD_GB=50
export VLLM_ALLOW_RUNTIME_LORA_UPDATES_WITH_SGD_LORA=1
export VLLM_SKIP_WARMUP=0

# Optimized environment for performance
export VLLM_LOGGING_LEVEL=INFO
export PYTHONUNBUFFERED=1

# Enable CUDA optimizations
export VLLM_USE_MODELSCOPE=false

# Enable unified memory in vLLM
export VLLM_USE_V1=0

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
# Optimized configuration for DGX Spark performance with NVFP4 quantization
# Available quantized models from NVIDIA
NVFP4_MODEL="nvidia/Llama-3.3-70B-Instruct-FP4"
NVFP8_MODEL="nvidia/Llama-3.1-8B-Instruct-FP8"
STANDARD_MODEL="meta-llama/Llama-3.1-70B-Instruct"

# Check GPU compute capability for optimal quantization
COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader,nounits 2>/dev/null || echo "unknown")
echo "Detected GPU compute capability: $COMPUTE_CAPABILITY"

# Configure quantization based on GPU architecture
if [[ "$COMPUTE_CAPABILITY" == "12.1" ]] || [[ "$COMPUTE_CAPABILITY" == "10.0" ]]; then
    # Blackwell/DGX Spark architecture - use standard 70B model with CPU offloading
    echo "Using standard Llama-3.1-70B model for Blackwell/DGX Spark with CPU offloading"
    QUANTIZATION_FLAG=""
    MODEL_TO_USE="$STANDARD_MODEL"  # Use standard 70B model
    GPU_MEMORY_UTIL="0.7"  # Lower GPU memory to allow unified memory
    MAX_MODEL_LEN="4096"   # Shorter sequences for memory efficiency
    MAX_NUM_SEQS="16"      # Lower concurrent sequences for 70B
    MAX_BATCHED_TOKENS="4096"
    CPU_OFFLOAD_GB="50"    # Offload 50GB to CPU/unified memory
elif [[ "$COMPUTE_CAPABILITY" == "9.0" ]]; then
    # Hopper architecture - use standard model
    echo "Using standard 70B model for Hopper architecture"
    QUANTIZATION_FLAG=""
    MODEL_TO_USE="$STANDARD_MODEL"
    GPU_MEMORY_UTIL="0.7"
    MAX_MODEL_LEN="4096"
    MAX_NUM_SEQS="16"
    MAX_BATCHED_TOKENS="4096"
    CPU_OFFLOAD_GB="40"
else
    # Other architectures - use standard precision
    echo "Using standard 70B model for GPU architecture: $COMPUTE_CAPABILITY"
    QUANTIZATION_FLAG=""
    MODEL_TO_USE="$STANDARD_MODEL"
    GPU_MEMORY_UTIL="0.7"
    MAX_MODEL_LEN="2048"
    MAX_NUM_SEQS="16"
    MAX_BATCHED_TOKENS="2048"
    CPU_OFFLOAD_GB="40"
fi

KV_CACHE_DTYPE="auto"

# Allow environment variables to override defaults
if [ -n "$VLLM_MODEL" ]; then
    echo "Overriding model with environment variable VLLM_MODEL: $VLLM_MODEL"
    MODEL_TO_USE="$VLLM_MODEL"
fi

if [ -n "$VLLM_MAX_MODEL_LEN" ]; then
    echo "Overriding max model len with environment variable: $VLLM_MAX_MODEL_LEN"
    MAX_MODEL_LEN="$VLLM_MAX_MODEL_LEN"
fi

if [ -n "$VLLM_GPU_MEMORY_UTILIZATION" ]; then
    echo "Overriding GPU memory utilization with environment variable: $VLLM_GPU_MEMORY_UTILIZATION"
    GPU_MEMORY_UTIL="$VLLM_GPU_MEMORY_UTILIZATION"
fi

if [ -n "$VLLM_CPU_OFFLOAD_GB" ]; then
    echo "Overriding CPU offload GB with environment variable: $VLLM_CPU_OFFLOAD_GB"
    CPU_OFFLOAD_GB="$VLLM_CPU_OFFLOAD_GB"
fi

if [ -n "$VLLM_QUANTIZATION" ]; then
    echo "Overriding quantization with environment variable: $VLLM_QUANTIZATION"
    QUANTIZATION_FLAG="--quantization $VLLM_QUANTIZATION"
fi

if [ -n "$VLLM_KV_CACHE_DTYPE" ]; then
    echo "Overriding KV cache dtype with environment variable: $VLLM_KV_CACHE_DTYPE"
    KV_CACHE_DTYPE="$VLLM_KV_CACHE_DTYPE"
fi

echo "Using model: $MODEL_TO_USE"
echo "Quantization: ${QUANTIZATION_FLAG:-'disabled'}"
echo "GPU memory utilization: $GPU_MEMORY_UTIL"

echo "CPU Offload: ${CPU_OFFLOAD_GB}GB"

vllm serve "$MODEL_TO_USE" \
  --host 0.0.0.0 \
  --port 8001 \
  --tensor-parallel-size 1 \
  --max-model-len "$MAX_MODEL_LEN" \
  --max-num-seqs "$MAX_NUM_SEQS" \
  --max-num-batched-tokens "$MAX_BATCHED_TOKENS" \
  --gpu-memory-utilization "$GPU_MEMORY_UTIL" \
  --cpu-offload-gb "$CPU_OFFLOAD_GB" \
  --kv-cache-dtype "$KV_CACHE_DTYPE" \
  --trust-remote-code \
  --served-model-name "$MODEL_TO_USE" \
  --enable-chunked-prefill \
  --disable-custom-all-reduce \
  --disable-async-output-proc \
  $QUANTIZATION_FLAG