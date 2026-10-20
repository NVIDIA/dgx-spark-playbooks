#!/usr/bin/env bash
set -euo pipefail

# Recipe commands: commands-DGX Spark | 128GB unified memory | [Qwen3.8 27B (NVFP4).md
NAME="vllm-spark-128gb-qwen38-27b"
PORT=8000
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
CONTAINER_ID=""
WAITER=""
STATUS_FILE="$(mktemp)"

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${WAITER}" ]]; then
    kill "${WAITER}" >/dev/null 2>&1 || true
    wait "${WAITER}" >/dev/null 2>&1 || true
  fi
  if [[ -n "${CONTAINER_ID}" ]]; then
    docker stop "${CONTAINER_ID}" >/dev/null 2>&1 || true
    docker rm "${CONTAINER_ID}" >/dev/null 2>&1 || true
  fi
  rm -f "${STATUS_FILE}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

mkdir -p "${HF_CACHE_DIR}" "$HOME/.cache/vllm" \
  "$HOME/.cache/flashinfer" "$HOME/.triton"

docker rm -f "${NAME}" >/dev/null 2>&1 || true

CONTAINER_ID="$(docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc host \
  --network host \
  --shm-size=16g \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -e "HF_TOKEN=${HF_TOKEN:-}" \
  -v "${HF_CACHE_DIR}:/root/.cache/huggingface" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.triton:/root/.triton" \
  vllm/vllm-openai:v0.28.0 \
  vllm serve nvidia/Qwen3.8-27B-NVFP4 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --trust-remote-code \
    --kv-cache-dtype fp8 \
    --gpu-memory-utilization 0.8 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 16384 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --speculative-config '{"method":"dflash","model": "z-lab/Qwen3.8-27B-DFlash2", "num_speculative_tokens":8}' \
    --load-format safetensors \
    --reasoning-parser qwen3 \
    --tool-call-parser qwen3_xml \
    --enable-auto-tool-choice)"

# Keep this script running so Sync can stop the container when the app closes.
docker wait "${CONTAINER_ID}" > "${STATUS_FILE}" &
WAITER=$!
wait "${WAITER}"
WAITER=""
STATUS="$(<"${STATUS_FILE}")"
if [[ ! "${STATUS}" =~ ^[0-9]+$ ]]; then
  printf 'Could not read the container exit status.\n' >&2
  exit 1
fi
exit "${STATUS}"
