#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configure RoCEv2 QoS and GPUDirect RDMA basics. Run on both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

SKIP_PFC=0
if [[ "${1:-}" == "--skip-pfc" ]]; then
  SKIP_PFC=1
elif [[ $# -gt 0 ]]; then
  die "Usage: sudo $0 [--skip-pfc]"
fi

detect_netifs
print_effective_config

warn_count=0
peermem_ready=0
dmabuf_data_direct_ready=0
ib_write_bw_path="$(need_ib_write_bw)"
warn() {
  warn_count=$((warn_count + 1))
  echo "WARN: $*"
}

info() {
  echo "INFO: $*"
}

if command -v mlnx_qos >/dev/null 2>&1; then
  for rail in 0 1; do
    netif="$(netif_for_rail "${rail}")"
    note "Configuring QoS on ${netif}"
    mlnx_qos -i "${netif}" --trust dscp >/dev/null
    if [[ "${SKIP_PFC}" == "0" ]]; then
      mlnx_qos -i "${netif}" --pfc 0,0,0,1,0,0,0,0 >/dev/null || \
        warn "PFC command failed on ${netif}; continue for back-to-back lossy RoCE"
    fi
    mlnx_qos -i "${netif}" 2>&1 | grep -E 'Priority trust state|PFC configuration|enabled|buffer' || true
  done
else
  warn "mlnx_qos not found; skipped QoS setup"
fi

note "Loading RDMA CM and GPUDirect kernel modules"
modprobe rdma_cm || warn "rdma_cm did not load; cma_roce_tos may be unavailable"

if command -v cma_roce_tos >/dev/null 2>&1; then
  for dev in "${RAIL0_DEV}" "${RAIL1_DEV}"; do
    note "Setting RoCE ToS ${ROCE_TOS} on ${dev}"
    cma_roce_tos -d "${dev}" -t "${ROCE_TOS}" || \
      warn "cma_roce_tos failed on ${dev}; continue and let validation/perftest prove RDMA path"
  done
else
  warn "cma_roce_tos not found; skipped RoCE ToS setup"
fi

info "Checking ib_write_bw at ${ib_write_bw_path}"
if "${ib_write_bw_path}" --help 2>&1 | grep -q -- '--use_cuda_dmabuf' && \
   "${ib_write_bw_path}" --help 2>&1 | grep -q -- '--use_data_direct'; then
  dmabuf_data_direct_ready=1
  echo "PASS: CUDA DMA-BUF/Data Direct GPUDirect path is available for Step 8 --gdr"
else
  warn "${ib_write_bw_path} does not advertise CUDA DMA-BUF/Data Direct flags for Step 8 --gdr"
fi

if lsmod | grep -E '^nvidia_peermem' >/dev/null; then
  peermem_ready=1
  echo "INFO: nvidia_peermem is loaded"
elif [[ "${TRY_NVIDIA_PEERMEM}" == "1" ]]; then
  peermem_modprobe_output=""
  if peermem_modprobe_output="$(modprobe nvidia_peermem 2>&1)"; then
    lsmod | grep -E '^nvidia_peermem' >/dev/null && peermem_ready=1
  else
    warn "nvidia_peermem failed to load: ${peermem_modprobe_output}"
    warn "nvidia_peermem-based GPUDirect RDMA is not ready, but basic RoCE can still be validated"
    echo "### recent kernel messages for nvidia_peermem"
    dmesg -T 2>/dev/null | grep -Ei 'nvidia_peermem|peer.?mem' | tail -20 || true
  fi
else
  info "nvidia_peermem is not loaded; not required for the CUDA DMA-BUF/Data Direct path"
  info "Set CX8_TRY_NVIDIA_PEERMEM=1 in 00_env.local only if you need to test the peermem path"
fi

if [[ "${warn_count}" == "0" ]]; then
  if [[ "${dmabuf_data_direct_ready}" == "1" || "${peermem_ready}" == "1" ]]; then
    echo "PASS: RoCEv2 and GPUDirect runtime basics configured"
  else
    echo "PASS: RoCEv2 basics configured; GPUDirect path not proven by this step"
  fi
else
  echo "PASS: RoCEv2 basics configured with ${warn_count} warning(s); continue to Step 7 basic validation"
fi
