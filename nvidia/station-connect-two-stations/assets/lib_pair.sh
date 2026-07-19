#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Shared helpers for the DGX Station CX8 setup manual scripts.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/00_env.sh"

timestamp() {
  date -u '+%Y%m%dT%H%M%SZ'
}

die() {
  echo "ERROR: $*" >&2
  exit 1
}

need_asset_src() {
  [[ -d "${CX8_ASSET_SRC}" ]] || die "Missing asset source: ${CX8_ASSET_SRC}"
}

make_log_dir() {
  local name="$1"
  local ts
  ts="$(timestamp)"
  local dir="${CX8_LOCAL_LOG_ROOT}/${ts}_${name}"
  mkdir -p "${dir}"
  echo "${dir}"
}

ssh_plain() {
  local host="$1"
  shift
  ssh "${SSH_OPTS[@]}" "${host}" "$@"
}

ssh_tty() {
  local host="$1"
  shift
  ssh -tt "${SSH_OPTS[@]}" "${host}" "$@"
}

remote_cmd_header() {
  local label="$1"
  local host="$2"
  echo
  echo "### ${label}: ${host}"
  echo
}

remote_runtime_env() {
  printf '%q=%q ' RAIL0_DEV "${RAIL0_DEV}"
  printf '%q=%q ' RAIL1_DEV "${RAIL1_DEV}"
  printf '%q=%q ' MTU "${MTU}"
  printf '%q=%q ' ROCE_TOS "${ROCE_TOS}"
  printf '%q=%q ' GPU_BDF "${CX8_GPU_BDF}"
  printf '%q=%q ' TRY_NVIDIA_PEERMEM "${CX8_TRY_NVIDIA_PEERMEM}"
  printf '%q=%q ' PERFTEST_REPO "${CX8_PERFTEST_REPO}"
  printf '%q=%q ' PERFTEST_REF "${CX8_PERFTEST_REF}"
  printf '%q=%q ' PERFTEST_PREFIX "${CX8_PERFTEST_PREFIX}"
  printf '%q=%q ' STATION_A_RAIL0_CIDR "${STATION_A_RAIL0_CIDR}"
  printf '%q=%q ' STATION_A_RAIL1_CIDR "${STATION_A_RAIL1_CIDR}"
  printf '%q=%q ' STATION_B_RAIL0_CIDR "${STATION_B_RAIL0_CIDR}"
  printf '%q=%q ' STATION_B_RAIL1_CIDR "${STATION_B_RAIL1_CIDR}"
}
