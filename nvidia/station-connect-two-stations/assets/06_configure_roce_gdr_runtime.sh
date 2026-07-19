#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configure runtime RoCEv2/GPUDirect basics on both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

log_dir="$(make_log_dir configure_roce_gdr)"

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
    /^INFO:/ { print; fflush(); next }
    /^FAIL:/ || /^ERROR:/ ||
      /[Ee][Rr][Rr][Oo][Rr]/ ||
      /[Ff]ailed/ ||
      /does not support/ ||
      /command not found/ ||
      /No such file/ ||
      /Permission denied/ { print red $0 reset; fflush(); next }
    { print; fflush(); next }
  '
}

configure_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_configure_roce_gdr.log"
  local env_assign
  env_assign="$(remote_runtime_env)"
  remote_cmd_header "${label}" "${host}" | tee "${log}"
  echo "Prompt note: enter the OS SSH password first; if sudo prompts, enter the same DUT password again. Password input is hidden."
  set +e
  ssh_tty "${host}" "cd '${CX8_REMOTE_BASE}' && sudo -p '[sudo on ${label}] password for %u: ' env ${env_assign} ./assets/configure_roce_gdr.sh" 2>&1 | tee -a "${log}" | colorize_status_stream
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" != "0" ]]; then
    printf '%s\n' "${color_red}FAIL: ${label} RoCE/GPUDirect configuration failed with exit code ${rc}${color_reset}"
    return "${rc}"
  fi
}

overall_rc=0
configure_one "${CX8_A_NAME}" "${remote_a}" || overall_rc=$?
configure_one "${CX8_B_NAME}" "${remote_b}" || overall_rc=$?

echo
echo "Logs: ${log_dir}"
exit "${overall_rc}"
