#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Revert runtime CX8 rail setup on one DUT. Called by ../99_cleanup_runtime.sh.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

REMOVE_PERSIST=0
DOWN_INTERFACES=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-persist)
      REMOVE_PERSIST=1
      ;;
    --down)
      DOWN_INTERFACES=1
      ;;
    *)
      die "Usage: sudo env ROLE=station-a|station-b $0 [--remove-persist] [--down]"
      ;;
  esac
  shift
done

need_role
detect_netifs

note "Cleaning CX8 runtime setup for ${ROLE}"
print_effective_config

for rail in 0 1; do
  netif="$(netif_for_rail "${rail}")"
  cidr="$(local_cidr_for_rail "${rail}")"
  note "rail${rail}: removing ${cidr} from ${netif}"
  ip addr del "${cidr}" dev "${netif}" >/dev/null 2>&1 || \
    echo "WARN: ${cidr} was not present on ${netif}"
  ip link set dev "${netif}" mtu 1500
  if [[ "${DOWN_INTERFACES}" == "1" ]]; then
    ip link set dev "${netif}" down
  fi
  echo "--- ${netif}"
  ip -brief addr show dev "${netif}" || true
done

if [[ "${REMOVE_PERSIST}" == "1" ]]; then
  note "Removing persistent netplan file if present"
  rm -f /etc/netplan/60-cx8-fabric.yaml
  if command -v netplan >/dev/null 2>&1; then
    netplan apply
  else
    echo "WARN: netplan command not found; removed file but did not apply netplan"
  fi
fi

echo "PASS: CX8 runtime cleanup completed for ${ROLE}"
