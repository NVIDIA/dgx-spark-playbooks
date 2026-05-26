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

# vLLM startup script with NVFP4 quantization support for Llama 4 Scout
# Optimized for NVIDIA Blackwell and Hopper architectures

set -e

# Default configuration - using supported Llama 3.1 model for testing
VLLM_MODEL=${VLLM_MODEL:-"meta-llama/Llama-3.1-8B-Instruct"}
VLLM_PORT=${VLLM_PORT:-8001}
VLLM_HOST=${VLLM_HOST:-"0.0.0.0"}
VLLM_TENSOR_PARALLEL_SIZE=${VLLM_TENSOR_PARALLEL_SIZE:-2}
VLLM_MAX_MODEL_LEN=${VLLM_MAX_MODEL_LEN:-8192}
VLLM_GPU_MEMORY_UTILIZATION=${VLLM_GPU_MEMORY_UTILIZATION:-0.9}
VLLM_MAX_NUM_SEQS=${VLLM_MAX_NUM_SEQS:-128}
VLLM_MAX_NUM_BATCHED_TOKENS=${VLLM_MAX_NUM_BATCHED_TOKENS:-8192}
VLLM_KV_CACHE_DTYPE=${VLLM_KV_CACHE_DTYPE:-"auto"}

# Detect GPU compute capability and set optimizations
COMPUTE_CAPABILITY=$(nvidia-smi -i 0 --query-gpu=compute_cap --format=csv,noheader 2>/dev/null || echo "unknown")

echo "Starting vLLM service with the following configuration:"
echo "Model: $VLLM_MODEL"
echo "Port: $VLLM_PORT"
echo "Host: $VLLM_HOST"
echo "Tensor Parallel Size: $VLLM_TENSOR_PARALLEL_SIZE"
echo "Max Model Length: $VLLM_MAX_MODEL_LEN"
echo "Max Num Seqs: $VLLM_MAX_NUM_SEQS"
echo "Max Batched Tokens: $VLLM_MAX_NUM_BATCHED_TOKENS"
echo "GPU Memory Utilization: $VLLM_GPU_MEMORY_UTILIZATION"
echo "KV Cache Dtype: $VLLM_KV_CACHE_DTYPE"
echo "GPU Compute Capability: $COMPUTE_CAPABILITY"

# Set up environment variables for optimal performance based on GPU architecture
if [ "$COMPUTE_CAPABILITY" = "10.0" ]; then
    echo "Detected Blackwell architecture - enabling NVFP4 optimizations"
    # Use FlashInfer backend for attentions
    export VLLM_ATTENTION_BACKEND=FLASHINFER
    # Use FlashInfer trtllm-gen attention kernels
    export VLLM_USE_TRTLLM_ATTENTION=1
    # Use FlashInfer FP8/FP4 MoE
    export VLLM_USE_FLASHINFER_MOE_FP8=1
    export VLLM_USE_FLASHINFER_MOE_FP4=1
    # Use FlashInfer trtllm-gen MoE backend
    export VLLM_FLASHINFER_MOE_BACKEND="latency"
    # Enable async scheduling
    ASYNC_SCHEDULING_FLAG="--async-scheduling"
    # Enable FlashInfer fusions
    FUSION_FLAG='{"pass_config":{"enable_fi_allreduce_fusion":true,"enable_noop":true},"custom_ops":["+quant_fp8","+rms_norm"],"full_cuda_graph":true}'
elif [ "$COMPUTE_CAPABILITY" = "9.0" ]; then
    echo "Detected Hopper architecture - enabling FP8 optimizations"
    # Disable async scheduling on Hopper architecture due to vLLM limitations
    ASYNC_SCHEDULING_FLAG=""
    # Disable FlashInfer fusions since they are not supported on Hopper architecture
    FUSION_FLAG="{}"
else
    echo "GPU architecture not specifically optimized - using default settings"
    ASYNC_SCHEDULING_FLAG=""
    FUSION_FLAG="{}"
fi

# Check GPU availability
if ! nvidia-smi > /dev/null 2>&1; then
    echo "Warning: NVIDIA GPU not detected. vLLM may not work properly."
fi

# Create model cache directory
mkdir -p /app/models

echo "Starting vLLM's built-in OpenAI API server"

# Build vLLM command with NVFP4 optimizations
VLLM_CMD="vllm serve $VLLM_MODEL \
    --host $VLLM_HOST \
    --port $VLLM_PORT \
    --tensor-parallel-size $VLLM_TENSOR_PARALLEL_SIZE \
    --max-model-len $VLLM_MAX_MODEL_LEN \
    --max-num-seqs $VLLM_MAX_NUM_SEQS \
    --max-num-batched-tokens $VLLM_MAX_NUM_BATCHED_TOKENS \
    --gpu-memory-utilization $VLLM_GPU_MEMORY_UTILIZATION \
    --kv-cache-dtype $VLLM_KV_CACHE_DTYPE \
    --trust-remote-code \
    --served-model-name $VLLM_MODEL"

# Add async scheduling if supported
if [ -n "$ASYNC_SCHEDULING_FLAG" ]; then
    VLLM_CMD="$VLLM_CMD $ASYNC_SCHEDULING_FLAG"
fi

# Add fusion optimizations if available
if [ "$FUSION_FLAG" != "{}" ]; then
    VLLM_CMD="$VLLM_CMD --compilation-config '$FUSION_FLAG'"
fi

# Start vLLM server
exec $VLLM_CMD
