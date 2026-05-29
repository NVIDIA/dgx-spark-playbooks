#!/bin/bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#

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

# Use local cache dirs so no root paths are required
workdir=$(pwd)
NANOCHAT_CACHE="$(pwd)/nanochat_cache"
HF_CACHE="$(pwd)/hf_cache"

cleanup() {
    echo -e "\nStopping training container..."
    docker stop $(docker ps -q --filter ancestor=nanochat) 2>/dev/null
    echo "Cleanup complete."
    exit 0
}

trap cleanup SIGINT SIGTERM

# Launch Nanochat training
cmd="mkdir -p $NANOCHAT_CACHE $HF_CACHE && \
chmod u+rwx $NANOCHAT_CACHE $HF_CACHE && \
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
    bash runs/speedrun.sh"
sh -c "$cmd" &

wait
echo -e "\nTraining complete!"
