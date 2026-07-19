#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configure the two CX8 rail IP addresses and MTU. Run once per station.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

PERSIST=0
if [[ "${1:-}" == "--persist" ]]; then
  PERSIST=1
elif [[ $# -gt 0 ]]; then
  die "Usage: sudo env ROLE=station-a|station-b $0 [--persist]"
fi

need_role
detect_netifs

note "Configuring ${ROLE}"
print_effective_config

for rail in 0 1; do
  netif="$(netif_for_rail "${rail}")"
  cidr="$(local_cidr_for_rail "${rail}")"
  note "rail${rail}: ${netif} ${cidr}"
  ip link set dev "${netif}" mtu "${MTU}"
  ip link set dev "${netif}" up
  ip addr replace "${cidr}" dev "${netif}"
done

if [[ "${PERSIST}" == "1" ]]; then
  mac0="$(cat "/sys/class/net/${NETIF0}/address")"
  mac1="$(cat "/sys/class/net/${NETIF1}/address")"
  cidr0="$(local_cidr_for_rail 0)"
  cidr1="$(local_cidr_for_rail 1)"
  tmp="$(mktemp)"
  trap 'rm -f "${tmp:-}"' EXIT
  cat >"${tmp}" <<EOF
network:
  version: 2
  ethernets:
    cx8-rail0:
      match:
        macaddress: "${mac0}"
      set-name: cx8r0
      mtu: ${MTU}
      dhcp4: false
      dhcp6: false
      addresses: [${cidr0}]
      optional: true
    cx8-rail1:
      match:
        macaddress: "${mac1}"
      set-name: cx8r1
      mtu: ${MTU}
      dhcp4: false
      dhcp6: false
      addresses: [${cidr1}]
      optional: true
EOF
  install -m 600 -o root -g root "${tmp}" /etc/netplan/60-cx8-fabric.yaml
  rm -f "${tmp}"
  trap - EXIT
  netplan apply
  unset NETIF0 NETIF1
  detect_netifs
  note "Wrote and applied /etc/netplan/60-cx8-fabric.yaml"
else
  note "Temporary configuration only; rerun with --persist to write netplan"
fi

for rail in 0 1; do
  netif="$(netif_for_rail "${rail}")"
  echo "--- ${netif}"
  ip addr show dev "${netif}"
  ethtool "${netif}" | grep -E 'Speed|Link detected' || true
done

echo "PASS: CX8 rail addressing configured"
