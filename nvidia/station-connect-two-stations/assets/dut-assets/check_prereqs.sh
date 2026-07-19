#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Read-only prerequisite and current-state check. Run on both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

CX8_PRIVILEGED="${CX8_PRIVILEGED:-0}"

echo "### host"
date -u '+utc=%Y-%m-%dT%H:%M:%SZ'
hostname -f 2>/dev/null || hostname
uname -a
cat /etc/os-release 2>/dev/null || true
test -r /etc/dgx-release && cat /etc/dgx-release || true

echo
echo "### required commands"
for cmd in lspci nvidia-smi ibdev2netdev ibv_devinfo ip ethtool; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "ok ${cmd}=$(command -v "${cmd}")"
  else
    echo "missing ${cmd}"
  fi
done
for cmd in show_gids rdma_topo flint mlxconfig mlnx_qos cma_roce_tos ib_write_bw; do
  if command -v "${cmd}" >/dev/null 2>&1; then
    echo "optional_ok ${cmd}=$(command -v "${cmd}")"
  else
    echo "optional_missing ${cmd}"
  fi
done

echo
echo "### gpu"
nvidia-smi -L 2>&1 || true
nvidia-smi --query-gpu=index,uuid,name,pci.bus_id,driver_version,memory.total \
  --format=csv,noheader 2>&1 || true
nvidia-smi topo -m 2>&1 || true

echo
echo "### cx8 pci"
lspci -Dnn | grep -Ei 'ConnectX|Mellanox|NVIDIA.*Ethernet|Infiniband' || true

echo
echo "### cx8 ibdev2netdev"
ibdev2netdev 2>&1 || true

echo
echo "### effective config"
print_effective_config 2>&1 || true

echo
echo "### port state"
for dev in "${RAIL0_DEV}" "${RAIL1_DEV}"; do
  echo "--- ${dev}"
  ibv_devinfo -d "${dev}" 2>&1 | grep -E 'hca_id|transport|firmware|node_guid|phys_port_cnt|port:|state:|link_layer|active_mtu|active_speed' || true
done

echo
echo "### link and address state"
ip -br link 2>&1 | grep -E "(${NETIF0:-__unset__}|${NETIF1:-__unset__}|mlx|ib|enp|ens)" || true
ip -o -4 addr show 2>&1 || true
ip route 2>&1 || true

echo
echo "### roce gids"
show_gids 2>&1 | grep -E 'mlx5_|RoCE|v2|IPv4|GID' || true

echo
echo "### firmware visibility"
BDF="$(lspci -D 2>/dev/null | awk 'tolower($0) ~ /connectx/ {print $1; exit}' || true)"
if [[ -n "${BDF}" ]] && command -v flint >/dev/null 2>&1; then
  if [[ "${CX8_PRIVILEGED}" == "1" ]]; then
    sudo flint -d "${BDF}" q 2>&1 | grep -E 'FW Version|Product Version|PSID|Description' || true
  else
    echo "flint query skipped by default; rerun with CX8_PRIVILEGED=1 if OS-side firmware visibility is required"
  fi
fi

echo
echo "### gpudirect"
lsmod | grep -E '^nvidia_peermem' || echo "nvidia_peermem not loaded"
if command -v rdma_topo >/dev/null 2>&1; then
  if [[ "${CX8_PRIVILEGED}" == "1" ]]; then
    sudo rdma_topo topo 2>&1 || true
  else
    rdma_topo topo 2>&1 || echo "rdma_topo topo did not complete without sudo; skipped by default"
  fi
fi

echo
echo "### recent errors"
dmesg -T 2>&1 | grep -Ei 'mlx5|rdma|infiniband|roce|nvrm|xid|sxid|aer|pcie bus error|iommu|dma|fatal|timeout|link down' | tail -n 120 || true

echo
echo "PASS: read-only prerequisite check completed"
