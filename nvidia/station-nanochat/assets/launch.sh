#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Lite training (default). Runs speedrun.sh, which setup copies from speedrun_lite.sh.

# Get wandb API key
export WANDB_API_KEY=$WANDB_API_KEY
if [ -z "$WANDB_API_KEY" ]; then
    echo "WANDB_API_KEY is not set"
    exit 1
fi

export WANDB_RUN=${WANDB_RUN:-speedrun}

# Get Hugging Face API key
export HF_TOKEN=$HF_TOKEN
if [ -z "$HF_TOKEN" ]; then
    echo "HF_TOKEN is not set"
    exit 1
fi

# Cleanup function to stop containers
cleanup() {
    echo
    echo "Stopping containers..."
    docker stop $(docker ps -q --filter ancestor=nanochat) 2>/dev/null || true
    echo "Interrupted training!"
    exit 0
}

workdir=$(pwd)
# DGX Station: use local cache dirs so no root paths are required
NANOCHAT_CACHE="${NANOCHAT_CACHE:-$(pwd)/nanochat_cache}"
HF_CACHE="${HF_CACHE:-$(pwd)/hf_cache}"
mkdir -p "$NANOCHAT_CACHE" "$HF_CACHE"

cmd="
mkdir -p /nanochat_cache && \
mkdir -p /hf_cache && \
chmod 777 /nanochat_cache && \
chmod 777 /hf_cache && \
docker run \
    --rm \
    --runtime=nvidia \
    --gpus all \
    --ipc=host \
    --net=host \
    --ulimit memlock=-1 \
    --ulimit stack=268435456 \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_RUN=$WANDB_RUN \
    -e HF_TOKEN=$HF_TOKEN \
    -v $(pwd)/nanochat:/workspace/nanochat \
    -v $NANOCHAT_CACHE:/root/.cache/nanochat \
    -v $HF_CACHE:/root/.cache/huggingface \
    -w /workspace/nanochat \
    nanochat \
    bash speedrun.sh"

sh -c "$cmd" &

sleep 5
while true; do
    if ! docker ps | grep -q "nanochat"; then
        echo
        echo "Training complete!"
        exit 0
    fi
    sleep 1
done