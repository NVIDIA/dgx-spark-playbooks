#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Validate basic two-rail setup on both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

log_dir="$(make_log_dir validate_setup)"

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

colorize_status_stream() {
  awk -v green="${color_green}" -v yellow="${color_yellow}" -v red="${color_red}" -v reset="${color_reset}" '
    /^PASS:/ { print green $0 reset; fflush(); next }
    /^WARN:/ { print yellow $0 reset; fflush(); next }
    /^mlx5_[0-9]+ (port_rcv_errors|port_xmit_discards|symbol_error|link_error_recovery|link_downed)=[1-9][0-9]*$/ {
      print red $0 reset; fflush(); next
    }
    /^FAIL:/ || /^ERROR:/ ||
      /[Ff]ailed/ ||
      /does not support/ ||
      /command not found/ ||
      /No such file/ ||
      /Permission denied/ { print red $0 reset; fflush(); next }
    { print; fflush(); next }
  '
}

validate_one() {
  local label="$1"
  local host="$2"
  local role="$3"
  local log="${log_dir}/${label}_validate_setup.log"
  local env_assign
  env_assign="$(remote_runtime_env)"
  remote_cmd_header "${label}" "${host}" | tee "${log}"
  set +e
  ssh_tty "${host}" "cd '${CX8_REMOTE_BASE}' && env ROLE='${role}' ${env_assign} ./assets/validate_setup.sh" 2>&1 | tee -a "${log}" | colorize_status_stream
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" != "0" ]]; then
    printf '%s\n' "${color_red}FAIL: ${label} basic validation failed with exit code ${rc}${color_reset}"
    if grep -Eq 'missing .*/[0-9]+; rerun Step 5' "${log}"; then
      printf '%s\n' "${color_yellow}ACTION: rail IP/MTU runtime config is missing on ${label}; rerun ./05_configure_rails_runtime.sh, then rerun ./07_validate_setup.sh${color_reset}"
    elif grep -Eiq 'Overheat|High Temperature|Cable error' "${log}"; then
      printf '%s\n' "${color_yellow}ACTION: ${label} reports QSFP/CX8 thermal or cable event; stop software setup, let the module cool, verify airflow/fan, reseat or swap the failed rail cable, then rerun ./04_check_cable_presence.sh${color_reset}"
    elif grep -Eq 'link is not detected|Link detected: no|Speed: Unknown' "${log}"; then
      printf '%s\n' "${color_yellow}ACTION: ${label} has a CX8 rail link-down condition; rerun ./04_check_cable_presence.sh and inspect the reported rail cable/port before rerunning Step 7${color_reset}"
    elif grep -Eq 'link is detected but speed is not 400G|Speed: 200000Mb/s|Speed: 200Gb/s' "${log}"; then
      printf '%s\n' "${color_yellow}ACTION: ${label} has a CX8 rail link running below the expected 400G rate; rerun ./04_check_cable_presence.sh, check the reported rail/cable speed and module temperature, and reseat/swap the cable or port before rerunning Step 7${color_reset}"
    fi
    return "${rc}"
  fi
}

overall_rc=0
validate_one "${CX8_A_NAME}" "${remote_a}" "${CX8_A_ROLE}" || overall_rc=$?
validate_one "${CX8_B_NAME}" "${remote_b}" "${CX8_B_ROLE}" || overall_rc=$?

echo
echo "Logs: ${log_dir}"
exit "${overall_rc}"
