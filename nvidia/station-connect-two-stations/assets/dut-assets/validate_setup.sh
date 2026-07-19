#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validate the basic two-station CX8 setup. Run on both stations after setup.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

need_role
detect_netifs

tmp_dir="$(mktemp -d)"
trap 'rm -rf "${tmp_dir}"' EXIT

print_effective_config

speed_warn_count=0
for rail in 0 1; do
  netif="$(netif_for_rail "${rail}")"
  hca="$(hca_for_rail "${rail}")"
  cidr="$(local_cidr_for_rail "${rail}")"
  peer="$(peer_ip_for_rail "${rail}")"
  route_log="${tmp_dir}/route-rail${rail}.txt"
  ethtool_log="${tmp_dir}/ethtool-rail${rail}.txt"
  ethtool_err="${tmp_dir}/ethtool-rail${rail}.stderr"
  ibv_log="${tmp_dir}/ibv-devinfo-rail${rail}.txt"

  echo
  echo "### rail${rail}: ${hca} ${netif}"
  ip addr show dev "${netif}" | grep -F "${cidr}" >/dev/null || \
    die "${netif} missing ${cidr}; rerun Step 5 temporary rail IP/MTU setup"
  [[ "$(cat "/sys/class/net/${netif}/mtu")" == "${MTU}" ]] || die "${netif} MTU is not ${MTU}"
  ip route get "${peer}" | tee "${route_log}"
  grep -F " dev ${netif} " "${route_log}" >/dev/null || \
    die "route to ${peer} does not use ${netif}"
  ethtool "${netif}" >"${ethtool_log}" 2>"${ethtool_err}" || true
  if [[ -s "${ethtool_err}" ]]; then
    sed 's/^/WARN: ethtool stderr: /' "${ethtool_err}"
  fi
  grep -E 'Speed:|Link detected:|Port:|Transceiver:' "${ethtool_log}" || true
  grep -F 'Link detected: yes' "${ethtool_log}" >/dev/null || \
    die "${netif} link is not detected; rerun Step 4 cable presence check and inspect rail${rail} cable/port"
  if ! grep -E 'Speed: 400000Mb/s|Speed: 400Gb/s' "${ethtool_log}" >/dev/null; then
    echo "WARN: ${netif} link is detected but speed is not 400G; continuing limited-speed validation only"
    speed_warn_count=$((speed_warn_count + 1))
  fi
  ping -M do -s 8972 -c 3 -I "${netif}" "${peer}"
  ibv_devinfo -d "${hca}" | tee "${ibv_log}"
  grep -E 'link_layer:.*Ethernet' "${ibv_log}" >/dev/null || \
    die "${hca} link layer is not Ethernet"
  grep -E 'state:.*PORT_ACTIVE' "${ibv_log}" >/dev/null || \
    die "${hca} port is not active"
  if grep -E 'active_speed:' "${ibv_log}" >/dev/null; then
    grep -E 'active_speed:.*400|active_speed:.*NDR' "${ibv_log}" >/dev/null || \
      die "${hca} verbs active_speed is present but is not 400G/NDR"
  else
    echo "WARN: ${hca} verbs active_speed not reported; using ethtool 400G link speed as authority"
  fi
done

if lsmod | grep -E '^nvidia_peermem' >/dev/null; then
  echo "PASS: nvidia_peermem loaded for GPUDirect RDMA"
else
  echo "INFO: nvidia_peermem not loaded; basic rail validation can continue"
  echo "INFO: Step 8 --gdr can use CUDA DMA-BUF/Data Direct if ib_write_bw supports those flags"
fi

echo
echo "### rdma counters"
for hca in "${RAIL0_DEV}" "${RAIL1_DEV}"; do
  for c in port_rcv_errors port_xmit_discards symbol_error link_error_recovery link_downed; do
    path="/sys/class/infiniband/${hca}/ports/1/counters/${c}"
    [[ -r "${path}" ]] && echo "${hca} ${c}=$(cat "${path}")"
  done
done

echo
if [[ "${speed_warn_count}" != "0" ]]; then
  echo "WARN: ${speed_warn_count} rail(s) linked below 400G; this is not an 800G setup pass"
  echo "PASS: basic CX8 two-station setup validated for ${ROLE} with limited-speed warning(s)"
else
  echo "PASS: basic CX8 two-station setup validated for ${ROLE}"
fi
