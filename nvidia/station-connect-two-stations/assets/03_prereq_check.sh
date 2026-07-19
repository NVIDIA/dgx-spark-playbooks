#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Run read-only prerequisite/current-state checks on both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

log_dir="$(make_log_dir prereq_check)"
summary="${log_dir}/summary.md"
: >"${summary}"

if [[ -t 1 ]]; then
  color_green=$'\033[32m'
  color_yellow=$'\033[33m'
  color_red=$'\033[31m'
  color_reset=$'\033[0m'
else
  color_green=""
  color_yellow=""
  color_red=""
  color_reset=""
fi

summarize_one() {
  local label="$1"
  local log="$2"
  local rc="$3"
  local required_missing optional_missing setup_missing mlx0 mlx1 port0 port1 peermem topo_status status status_console

  required_missing="$(grep -E '^missing ' "${log}" | sed 's/^missing //' | xargs 2>/dev/null || true)"
  optional_missing="$(grep -E '^optional_missing ' "${log}" | sed 's/^optional_missing //' | xargs 2>/dev/null || true)"
  setup_missing=""
  for cmd in ${optional_missing}; do
    case "${cmd}" in
      flint|mlxconfig) ;;
      *) setup_missing="${setup_missing} ${cmd}" ;;
    esac
  done
  setup_missing="$(echo "${setup_missing}" | xargs 2>/dev/null || true)"
  mlx0="$(grep -m1 '^mlx5_0 port' "${log}" || true)"
  mlx1="$(grep -m1 '^mlx5_1 port' "${log}" || true)"

  if grep -A8 '^--- mlx5_0' "${log}" | grep -q 'PORT_ACTIVE'; then
    port0="PORT_ACTIVE"
  elif grep -A8 '^--- mlx5_0' "${log}" | grep -q 'PORT_DOWN'; then
    port0="PORT_DOWN"
  else
    port0="unknown"
  fi

  if grep -A8 '^--- mlx5_1' "${log}" | grep -q 'PORT_ACTIVE'; then
    port1="PORT_ACTIVE"
  elif grep -A8 '^--- mlx5_1' "${log}" | grep -q 'PORT_DOWN'; then
    port1="PORT_DOWN"
  else
    port1="unknown"
  fi

  if grep -q '^nvidia_peermem[[:space:]]' "${log}"; then
    peermem="loaded"
  elif grep -q 'nvidia_peermem not loaded' "${log}"; then
    peermem="not loaded (optional; CUDA DMA-BUF/Data Direct may be used for Step 8 --gdr)"
  else
    peermem="unknown"
  fi

  if grep -q '^RDMA NIC=' "${log}"; then
    topo_status="visible"
  elif grep -q 'rdma_topo topo did not complete without sudo' "${log}"; then
    topo_status="available, privileged details skipped"
  else
    topo_status="not shown"
  fi

  if [[ "${rc}" != "0" ]]; then
    if grep -q "bash\\\\r" "${log}"; then
      status="FAIL: remote helper has CRLF line endings; rerun Step 2 to repush normalized assets"
    elif grep -q 'No such file or directory' "${log}"; then
      status="FAIL: remote helper missing; rerun Step 2 to copy assets"
    else
      status="FAIL: remote prereq command returned ${rc}"
    fi
  elif [[ -n "${required_missing}" ]]; then
    status="FAIL: missing required command(s): ${required_missing}"
  elif [[ -n "${setup_missing}" ]]; then
    status="FAIL: missing setup command(s): ${setup_missing}"
  elif [[ -z "${mlx0}" || -z "${mlx1}" ]]; then
    status="FAIL: did not find both CX8 rails in ibdev2netdev output"
  else
    status="PASS: prerequisite data collected"
  fi

  case "${status}" in
    PASS:*) status_console="${color_green}${status}${color_reset}" ;;
    WARN:*) status_console="${color_yellow}${status}${color_reset}" ;;
    FAIL:*) status_console="${color_red}${status}${color_reset}" ;;
    *) status_console="${status}" ;;
  esac

  {
    echo "### ${label}"
    echo
    echo "- Status: ${status}"
    echo "- Required missing commands: ${required_missing:-none}"
    echo "- Optional missing commands: ${optional_missing:-none}"
    echo "- Missing setup commands other than flint/mlxconfig: ${setup_missing:-none}"
    echo "- CX8 rail map:"
    echo "  - ${mlx0:-mlx5_0 not found}"
    echo "  - ${mlx1:-mlx5_1 not found}"
    echo "- Verbs port state: mlx5_0=${port0}, mlx5_1=${port1}"
    echo "- GPUDirect module: ${peermem}"
    echo "- RDMA topology detail: ${topo_status}"
    echo "- Full log: ${log}"
    echo
  } >>"${summary}"

  {
    echo "### ${label}"
    echo
    printf -- "- Status: %s\n" "${status_console}"
    echo "- Required missing commands: ${required_missing:-none}"
    echo "- Optional missing commands: ${optional_missing:-none}"
    echo "- Missing setup commands other than flint/mlxconfig: ${setup_missing:-none}"
    echo "- CX8 rail map:"
    echo "  - ${mlx0:-mlx5_0 not found}"
    echo "  - ${mlx1:-mlx5_1 not found}"
    echo "- Verbs port state: mlx5_0=${port0}, mlx5_1=${port1}"
    echo "- GPUDirect module: ${peermem}"
    echo "- RDMA topology detail: ${topo_status}"
    echo "- Full log: ${log}"
    echo
  }

  [[ "${status}" != FAIL:* ]]
}

check_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_prereq_check.log"
  remote_cmd_header "${label}" "${host}"
  set +e
  ssh_plain "${host}" "cd '${CX8_REMOTE_BASE}' && CX8_PRIVILEGED=0 ./assets/check_prereqs.sh 2>&1" >"${log}" 2>&1
  local rc=$?
  set -e
  summarize_one "${label}" "${log}" "${rc}"
}

overall_rc=0
check_one "${CX8_A_NAME}" "${remote_a}" || overall_rc=$?
check_one "${CX8_B_NAME}" "${remote_b}" || overall_rc=$?

echo
echo "Summary: ${summary}"
echo "Logs: ${log_dir}"
exit "${overall_rc}"
