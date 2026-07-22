#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Per-rail RDMA or GPUDirect bandwidth test wrapper.
#
# Server example on station A:
#   ./assets/run_perftest.sh --server --rail 0
#
# Client example on station B:
#   ROLE=station-b ./assets/run_perftest.sh --client --rail 0
#
# Add --gdr on both sides for GPUDirect/Data Direct.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

MODE=""
RAIL=""
GDR=0
SIZE="${SIZE:-1048576}"
DURATION="${DURATION:-20}"
GPU_BDF="${GPU_BDF:-}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --server) MODE="server" ;;
    --client) MODE="client" ;;
    --rail)
      [[ -n "${2:-}" ]] || die "--rail requires 0 or 1"
      RAIL="$2"
      shift
      ;;
    --gdr) GDR=1 ;;
    --size)
      [[ -n "${2:-}" ]] || die "--size requires bytes"
      SIZE="$2"
      shift
      ;;
    --duration)
      [[ -n "${2:-}" ]] || die "--duration requires seconds"
      DURATION="$2"
      shift
      ;;
    --gpu-bdf)
      [[ -n "${2:-}" ]] || die "--gpu-bdf requires a PCI bus ID"
      GPU_BDF="$2"
      shift
      ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done

[[ "${MODE}" == "server" || "${MODE}" == "client" ]] || die "Use --server or --client"
[[ "${RAIL}" == "0" || "${RAIL}" == "1" ]] || die "Use --rail 0 or --rail 1"
if [[ "${MODE}" == "client" ]]; then
  need_role
fi
ib_write_bw_path="$(need_ib_write_bw)"
detect_netifs

hca="$(hca_for_rail "${RAIL}")"
if [[ -n "${ROLE}" ]]; then
  ensure_rail_ip_present "${RAIL}"
fi

args=(-d "${hca}" -F --report_gbits -s "${SIZE}" -D "${DURATION}")

if [[ "${GDR}" == "1" ]]; then
  help_text="$("${ib_write_bw_path}" --help 2>&1 || true)"
  grep -q -- '--use_cuda_dmabuf' <<<"${help_text}" || \
    die "${ib_write_bw_path} does not support --use_cuda_dmabuf; install a perftest build with CUDA DMA-BUF support"
  grep -q -- '--use_data_direct' <<<"${help_text}" || \
    die "${ib_write_bw_path} does not support --use_data_direct; install a perftest build with Data Direct support"

  if lsmod | grep -E '^nvidia_peermem' >/dev/null; then
    echo "INFO: nvidia_peermem is loaded"
  else
    echo "INFO: nvidia_peermem is not loaded; continuing with CUDA DMA-BUF/Data Direct GDR path"
  fi

  if [[ -n "${GPU_BDF}" ]]; then
    grep -q -- '--use_cuda_bus_id' <<<"${help_text}" || \
      die "GPU_BDF was set, but ib_write_bw does not support --use_cuda_bus_id"
    args+=(--use_cuda_bus_id="${GPU_BDF}")
  else
    args+=(--use_cuda=0)
  fi
  args+=(--use_cuda_dmabuf --use_data_direct)
fi

if [[ "${MODE}" == "server" ]]; then
  # Step 8 redirects server output to a file and waits for perftest's readiness
  # banner before launching the client. Force line buffering so that banner is
  # observable immediately instead of remaining in the stdio buffer.
  need_cmd stdbuf
  echo "Using ib_write_bw: ${ib_write_bw_path}"
  echo "Starting ib_write_bw server on ${hca}; start client on peer."
  exec stdbuf -oL "${ib_write_bw_path}" "${args[@]}"
else
  peer="$(peer_ip_for_rail "${RAIL}")"
  echo "Using ib_write_bw: ${ib_write_bw_path}"
  echo "Starting ib_write_bw client on ${hca}; peer=${peer}"
  exec "${ib_write_bw_path}" "${args[@]}" "${peer}"
fi
