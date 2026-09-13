#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check software-visible CX8 cable/link presence on one DUT.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

detect_netifs

for rail in 0 1; do
  hca="$(hca_for_rail "${rail}")"
  netif="$(netif_for_rail "${rail}")"
  devinfo="$(ibv_devinfo -d "${hca}" 2>&1 || true)"
  ethtool_out="$(ethtool "${netif}" 2>&1 || true)"
  port_state="$(printf '%s\n' "${devinfo}" | awk -F: '/state:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
  link_layer="$(printf '%s\n' "${devinfo}" | awk -F: '/link_layer:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
  speed="$(printf '%s\n' "${ethtool_out}" | awk -F: '/Speed:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
  link_detected="$(printf '%s\n' "${ethtool_out}" | awk -F: '/Link detected:/ {gsub(/^[ \t]+|[ \t]+$/, "", $2); print $2; exit}')"
  carrier="$(cat "/sys/class/net/${netif}/carrier" 2>/dev/null || echo unknown)"
  operstate="$(cat "/sys/class/net/${netif}/operstate" 2>/dev/null || echo unknown)"
  cable_visible="no"
  if [[ "${port_state}" == *"PORT_ACTIVE"* && "${link_detected}" == "yes" ]]; then
    cable_visible="yes"
  fi
  speed_ok="unknown"
  if [[ "${cable_visible}" == "yes" ]]; then
    if [[ "${speed}" == "400000Mb/s" || "${speed}" == "400Gb/s" ]]; then
      speed_ok="yes"
    else
      speed_ok="no"
      printf 'ACTION rail=%s speed="%s" is below expected 400G; limited-speed validation can continue, but check cable capability, cable orientation, module temperature, and port/cable seating before claiming 800G setup\n' \
        "${rail}" "${speed:-unknown}"
    fi
  fi
  printf 'CHECK rail=%s hca=%s netif=%s port_state="%s" link_layer="%s" ethtool_link=%s speed="%s" carrier=%s operstate=%s cable_visible=%s speed_ok=%s\n' \
    "${rail}" "${hca}" "${netif}" "${port_state:-unknown}" "${link_layer:-unknown}" \
    "${link_detected:-unknown}" "${speed:-unknown}" "${carrier}" "${operstate}" "${cable_visible}" "${speed_ok}"
done

if command -v dmesg >/dev/null 2>&1; then
  dmesg -T 2>/dev/null | grep -Ei 'port_module:.*(Cable error|High Temperature)|Cable error|High Temperature' | tail -12 | \
    sed 's/^/EVENT /' || true
fi
