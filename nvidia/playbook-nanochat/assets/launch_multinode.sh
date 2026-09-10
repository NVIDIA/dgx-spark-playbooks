#!/bin/bash

export HOST_IP=$1
export WORKER_IP=$2

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
    ssh $USER@$WORKER_IP "docker stop \$(docker ps -q --filter ancestor=nanochat)" 2>/dev/null || true
    echo "Interrupted training!"
    exit 0
}

trap cleanup SIGINT SIGTERM

workdir=$(pwd)

# QSFP / interconnect interface for NCCL and GLOO (override if your interface differs)
export MN_IF_NAME=${MN_IF_NAME:-enp1s0f0np0}

# Checkpoint sync function - syncs checkpoints between HOST and WORKER nodes
sync_checkpoints() {
    while true; do
        for stage in "base_checkpoints" "chatsft_checkpoints" "mid_checkpoints"; do
            (
                LOCK="/tmp/rsync_${stage}.lock"
                (
                    flock -n 9 || exit 0
                    rsync -az --partial --inplace --append-verify \
                        $HOME/.cache/nanochat/${stage}/ \
                        $USER@${WORKER_IP}:$HOME/.cache/nanochat/${stage}/ 2>/dev/null
                ) 9>$LOCK
            ) &
        done
    done
    wait
    sleep 5
}

# Sync checkpoints in the background
sync_checkpoints &

cmd="
mkdir -p $HOME/.cache/nanochat && \
docker run \
    --rm \
    --gpus all \
    --ipc=host \
    --net=host \
    --group-add $(id -g) \
    --ulimit memlock=-1 \
    --ulimit stack=268435456 \
    -e NCCL_SOCKET_IFNAME=$MN_IF_NAME \
    -e GLOO_SOCKET_IFNAME=$MN_IF_NAME \
    -e NCCL_OOB_SOCKET_IFNAME=$MN_IF_NAME \
    -e NCCL_SOCKET_FAMILY=AF_INET \
    -e NCCL_IB_DISABLE=1 \
    -e WANDB_API_KEY=$WANDB_API_KEY \
    -e WANDB_RUN=$WANDB_RUN \
    -e HF_TOKEN=$HF_TOKEN \
    -e WORKER_IP=$WORKER_IP \
    -v $(pwd)/nanochat:/workspace/nanochat \
    -v $HOME/.cache/nanochat:/root/.cache/nanochat \
    -v $HOME/.cache/huggingface:/root/.cache/huggingface \
    -v $HOME/.ssh:/root/.ssh:ro \
    -w /workspace/nanochat \
    nanochat \
    bash speedrun.sh $HOST_IP"

sh -c "$cmd 0" &
ssh $USER@$WORKER_IP "$cmd 1" &

sleep 5
while true; do
    if ! docker ps | grep -q "nanochat"; then
        echo
        echo "Training complete!"
        exit 0
    fi
    sleep 1
done