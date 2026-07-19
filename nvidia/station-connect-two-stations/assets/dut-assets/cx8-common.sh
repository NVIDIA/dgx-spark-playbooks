#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the DGX Station CX8 two-station playbook scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [[ -n "${CX8_ENV:-}" && -r "${CX8_ENV}" ]]; then
  # shellcheck disable=SC1090
  source "${CX8_ENV}"
elif [[ -r "${SCRIPT_DIR}/cx8.env" ]]; then
  # shellcheck disable=SC1091
  source "${SCRIPT_DIR}/cx8.env"
fi

ROLE="${ROLE:-}"
RAIL0_DEV="${RAIL0_DEV:-mlx5_0}"
RAIL1_DEV="${RAIL1_DEV:-mlx5_1}"
MTU="${MTU:-9000}"
ROCE_TOS="${ROCE_TOS:-106}"
TRY_NVIDIA_PEERMEM="${TRY_NVIDIA_PEERMEM:-0}"
PERFTEST_PREFIX="${PERFTEST_PREFIX:-/usr/local}"

if [[ -n "${PERFTEST_PREFIX}" && -d "${PERFTEST_PREFIX}/bin" ]]; then
  case ":${PATH}:" in
    *":${PERFTEST_PREFIX}/bin:"*) ;;
    *) export PATH="${PERFTEST_PREFIX}/bin:${PATH}" ;;
  esac
fi

STATION_A_RAIL0_CIDR="${STATION_A_RAIL0_CIDR:-192.168.100.1/24}"
STATION_A_RAIL1_CIDR="${STATION_A_RAIL1_CIDR:-192.168.101.1/24}"
STATION_B_RAIL0_CIDR="${STATION_B_RAIL0_CIDR:-192.168.100.2/24}"
STATION_B_RAIL1_CIDR="${STATION_B_RAIL1_CIDR:-192.168.101.2/24}"

STATION_A_RAIL0_IP="${STATION_A_RAIL0_CIDR%/*}"
STATION_A_RAIL1_IP="${STATION_A_RAIL1_CIDR%/*}"
STATION_B_RAIL0_IP="${STATION_B_RAIL0_CIDR%/*}"
STATION_B_RAIL1_IP="${STATION_B_RAIL1_CIDR%/*}"

die() {
  echo "ERROR: $*" >&2
  exit 1
}

note() {
  echo "==> $*"
}

need_cmd() {
  command -v "$1" >/dev/null 2>&1 || die "Missing command: $1"
}

need_role() {
  [[ "${ROLE}" == "station-a" || "${ROLE}" == "station-b" ]] || \
    die "Set ROLE=station-a or ROLE=station-b"
}

netdev_for_hca() {
  local hca="$1"
  ibdev2netdev | awk -v hca="${hca}" '
    $1 == hca && netdev == "" {netdev=$5}
    END {if (netdev != "") print netdev}
  '
}

detect_netifs() {
  need_cmd ibdev2netdev
  NETIF0="${NETIF0:-$(netdev_for_hca "${RAIL0_DEV}")}"
  NETIF1="${NETIF1:-$(netdev_for_hca "${RAIL1_DEV}")}"
  [[ -n "${NETIF0}" ]] || die "Could not map ${RAIL0_DEV} to a netdev"
  [[ -n "${NETIF1}" ]] || die "Could not map ${RAIL1_DEV} to a netdev"
}

local_cidr_for_rail() {
  need_role
  local rail="$1"
  if [[ "${ROLE}" == "station-a" && "${rail}" == "0" ]]; then
    echo "${STATION_A_RAIL0_CIDR}"
  elif [[ "${ROLE}" == "station-a" && "${rail}" == "1" ]]; then
    echo "${STATION_A_RAIL1_CIDR}"
  elif [[ "${ROLE}" == "station-b" && "${rail}" == "0" ]]; then
    echo "${STATION_B_RAIL0_CIDR}"
  elif [[ "${ROLE}" == "station-b" && "${rail}" == "1" ]]; then
    echo "${STATION_B_RAIL1_CIDR}"
  else
    die "Invalid rail: ${rail}"
  fi
}

peer_ip_for_rail() {
  need_role
  local rail="$1"
  if [[ "${ROLE}" == "station-a" && "${rail}" == "0" ]]; then
    echo "${STATION_B_RAIL0_IP}"
  elif [[ "${ROLE}" == "station-a" && "${rail}" == "1" ]]; then
    echo "${STATION_B_RAIL1_IP}"
  elif [[ "${ROLE}" == "station-b" && "${rail}" == "0" ]]; then
    echo "${STATION_A_RAIL0_IP}"
  elif [[ "${ROLE}" == "station-b" && "${rail}" == "1" ]]; then
    echo "${STATION_A_RAIL1_IP}"
  else
    die "Invalid rail: ${rail}"
  fi
}

hca_for_rail() {
  case "$1" in
    0) echo "${RAIL0_DEV}" ;;
    1) echo "${RAIL1_DEV}" ;;
    *) die "Invalid rail: $1" ;;
  esac
}

netif_for_rail() {
  detect_netifs
  case "$1" in
    0) echo "${NETIF0}" ;;
    1) echo "${NETIF1}" ;;
    *) die "Invalid rail: $1" ;;
  esac
}

ensure_rail_ip_present() {
  local rail="$1"
  local netif
  local cidr
  netif="$(netif_for_rail "${rail}")"
  cidr="$(local_cidr_for_rail "${rail}")"
  ip -o -4 addr show dev "${netif}" | grep -F " ${cidr} " >/dev/null || \
    die "${netif} missing ${cidr}; rerun Step 5 rail IP/MTU setup, or use Step 5 --persist when the rail IPs must survive reboot/network-manager refresh"
}

print_effective_config() {
  detect_netifs
  cat <<EOF
ROLE=${ROLE:-not-set}
RAIL0_DEV=${RAIL0_DEV}
RAIL1_DEV=${RAIL1_DEV}
NETIF0=${NETIF0}
NETIF1=${NETIF1}
MTU=${MTU}
STATION_A_RAIL0_CIDR=${STATION_A_RAIL0_CIDR}
STATION_A_RAIL1_CIDR=${STATION_A_RAIL1_CIDR}
STATION_B_RAIL0_CIDR=${STATION_B_RAIL0_CIDR}
STATION_B_RAIL1_CIDR=${STATION_B_RAIL1_CIDR}
EOF
}
