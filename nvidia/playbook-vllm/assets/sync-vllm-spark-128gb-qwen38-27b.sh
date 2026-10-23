#!/usr/bin/env bash
set -euo pipefail

# Recipe: https://github.com/NVIDIA/local-ai-inference-recipes/blob/7d1162cfb965847d9412cef9bd5c5d4ade53d957/models/qwen/qwen3.8-27b/dgx-spark/linux/vllm/128gb/nvfp4/recipe.yaml
# Image digest recorded in the sibling performance.yaml; recipe validated 2026-10-19.
# Sync uses loopback binding and manages the container's lifetime.
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

# Preserve the image's prebuilt kernel/autotune caches; mount only the HF cache.
mkdir -p "${HF_CACHE_DIR}"

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
  nvcr.io/nvidia/vllm:26.10.pre1-py3 \
  vllm serve nvidia/Qwen3.8-27B-NVFP4 \
    --host 127.0.0.1 \
    --port "${PORT}" \
    --kv-cache-dtype fp8 \
    --max-model-len 262144 \
    --max-num-seqs 8 \
    --async-scheduling \
    --enable-flashinfer-autotune \
    --gpu-memory-utilization 0.60 \
    --kernel-config '{"linear_backend":"nvfp4_dynamic","nvfp4_dynamic_max_tokens":16,"unquantized_linear_backend":"flashinfer_cudnn","linear_backend_per_quant":{"fp8_w8a8":"cutlass"}}' \
    --speculative-config '{"method":"dflash","model":"z-lab/Qwen3.8-27B-DFlash2","num_speculative_tokens":7,"draft_sample_method":"probabilistic","rejection_sample_method":"standard"}' \
    --compilation-config '{"pass_config":{"fuse_act_quant":true},"max_cudagraph_capture_size":64}' \
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
