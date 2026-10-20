#!/usr/bin/env bash
set -euo pipefail

# Recipe commands: https://localai-inference-recipes-4170b6.gitlab-master-pages.nvidia.com/hardware/dgx-station/index.html?model=qwen%2Fqwen3.8-flash-next&os=linux&backend=vllm&memory=&precision=nvfp4&variant=&memoryConfig=748
NAME="vllm-station-qwen38-flash-next"
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

GPU_UUID="$(nvidia-smi --query-gpu=uuid,name --format=csv,noheader \
  | awk -F, 'tolower($2) ~ /gb300/ { gsub(/[[:space:]]/, "", $1); print $1; exit }')"
if [[ -z "${GPU_UUID}" ]]; then
  printf 'Could not find a GB300 GPU with nvidia-smi.\n' >&2
  exit 1
fi

docker rm -f "${NAME}" >/dev/null 2>&1 || true

CONTAINER_ID="$(docker run -d \
  --name "${NAME}" \
  --gpus "device=${GPU_UUID}" \
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
  vllm/vllm-openai:qwen38-flash-next \
  vllm serve Inferact/Qwen3.8-Flash-Next-NVFP4 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --no-enable-flashinfer-autotune \
    --max-num-seqs 32 \
    --gpu-memory-utilization 0.95 \
    --max-num-batched-tokens 8192 \
    --enable-prefix-caching \
    --distributed-executor-backend mp \
    --compilation-config '{"mode": "none", "cudagraph_mode": "full_decode_only"}' \
    --tensor-parallel-size 1 \
    --enable-auto-tool-choice \
    --tool-call-parser qwen3_xml \
    --reasoning-parser qwen3)"

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
