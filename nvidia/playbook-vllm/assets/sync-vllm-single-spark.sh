#!/usr/bin/env bash
set -e

NAME="vllm-qwen38"

stop_vllm() {
  docker stop "${NAME}" >/dev/null 2>&1 || true
}
trap stop_vllm EXIT INT TERM HUP

# Remove the container from an earlier session, if one exists.
docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p 8000:8000 \
  -v "$HOME/.cache/huggingface/hub:/root/.cache/huggingface/hub:ro" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:qwen38 \
  vllm serve nvidia/Qwen3.8-27B-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --tensor-parallel-size 1 \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.7 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --load-format fastsafetensors \
    --reasoning-parser qwen3 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder

# Keep the custom application active while the container runs.
docker wait "${NAME}" >/dev/null
