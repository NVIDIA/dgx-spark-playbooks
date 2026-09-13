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

if networkmanager_is_active; then
  need_cmd nmcli
fi

note "Configuring ${ROLE}"
print_effective_config

configure_runtime_rail() {
  local rail="$1"
  local netif="$2"
  local cidr="$3"
  local con_name=""

  if networkmanager_manages_interface "${netif}"; then
    con_name="$(runtime_connection_name "${rail}")"
    note "${netif} is managed by NetworkManager; activating temporary profile ${con_name}"

    # The profile is deliberately in-memory only (save no), so this remains a
    # runtime configuration and disappears on NetworkManager restart or reboot.
    # Replace our prior in-memory profile to make reruns deterministic.
    if nmcli -g connection.uuid connection show "${con_name}" >/dev/null 2>&1; then
      nmcli connection delete "${con_name}" >/dev/null
    fi
    nmcli connection add \
      type ethernet \
      ifname "${netif}" \
      con-name "${con_name}" \
      autoconnect no \
      save no \
      mtu "${MTU}" \
      ip4 "${cidr}" \
      -- \
      ipv4.never-default yes \
      ipv6.method disabled
    nmcli connection up "${con_name}" ifname "${netif}"
  else
    note "${netif} is not managed by NetworkManager; applying runtime settings with ip"
    ip link set dev "${netif}" mtu "${MTU}"
    ip link set dev "${netif}" up
    ip addr replace "${cidr}" dev "${netif}"
  fi
}

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
  for rail in 0 1; do
    netif="$(netif_for_rail "${rail}")"
    cidr="$(local_cidr_for_rail "${rail}")"
    note "rail${rail}: ${netif} ${cidr}"
    configure_runtime_rail "${rail}" "${netif}" "${cidr}"
  done
  note "Temporary configuration only; NetworkManager profiles are in-memory and are not saved"
fi

for rail in 0 1; do
  netif="$(netif_for_rail "${rail}")"
  echo "--- ${netif}"
  ip addr show dev "${netif}"
  ethtool "${netif}" | grep -E 'Speed|Link detected' || true
done

echo "PASS: CX8 rail addressing configured"
