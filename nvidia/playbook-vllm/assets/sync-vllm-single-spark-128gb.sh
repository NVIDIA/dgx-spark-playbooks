#!/usr/bin/env bash
set -euo pipefail

NAME="vllm-qwen36-35b-a3b"
PORT=8000
HF_CACHE_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
WAITER=""
STATUS_FILE="$(mktemp)"

cleanup() {
  local status=$?
  trap - EXIT
  if [[ -n "${WAITER}" ]]; then
    kill "${WAITER}" >/dev/null 2>&1 || true
    wait "${WAITER}" >/dev/null 2>&1 || true
  fi
  docker stop "${NAME}" >/dev/null 2>&1 || true
  docker rm "${NAME}" >/dev/null 2>&1 || true
  rm -f "${STATUS_FILE}"
  exit "${status}"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM
trap 'exit 129' HUP

mkdir -p "$HOME/.cache/vllm" "$HOME/.cache/flashinfer" \
  "$HOME/.cache/triton"
docker rm -f "${NAME}" >/dev/null 2>&1 || true

docker run -d \
  --name "${NAME}" \
  --gpus all \
  --ipc host \
  --ulimit memlock=-1 \
  --ulimit stack=67108864 \
  --entrypoint "" \
  -p "127.0.0.1:${PORT}:8000" \
  -v "$HF_CACHE_DIR:/root/.cache/huggingface:ro" \
  -v "$HOME/.cache/vllm:/root/.cache/vllm" \
  -v "$HOME/.cache/flashinfer:/root/.cache/flashinfer" \
  -v "$HOME/.cache/triton:/root/.triton" \
  -e HF_HUB_OFFLINE=1 \
  -e VLLM_USE_RUST_FRONTEND=1 \
  vllm/vllm-openai:v0.28.0 \
  vllm serve nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --served-model-name nvidia/Qwen3.6-35B-A3B-NVFP4 \
    --host 0.0.0.0 \
    --port 8000 \
    --trust-remote-code \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_coder \
    --reasoning-parser qwen3 \
    --kv-cache-dtype fp8 \
    --attention-backend flashinfer \
    --moe-backend marlin \
    --gpu-memory-utilization 0.5 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --max-num-batched-tokens 8192 \
    --enable-chunked-prefill \
    --async-scheduling \
    --enable-prefix-caching \
    --load-format fastsafetensors

docker wait "${NAME}" > "${STATUS_FILE}" &
WAITER=$!
wait "${WAITER}"
WAITER=""
STATUS="$(<"${STATUS_FILE}")"
if [[ ! "${STATUS}" =~ ^[0-9]+$ ]]; then
  printf 'Could not read the container exit status.\n' >&2
  exit 1
fi
exit "${STATUS}"
