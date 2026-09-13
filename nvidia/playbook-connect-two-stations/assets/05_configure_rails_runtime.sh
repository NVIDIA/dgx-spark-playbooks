#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Configure CX8 rail IP/MTU on both stations.
#
# Default is temporary runtime config. Pass --persist to write netplan.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

persist_arg=""
if [[ "${1:-}" == "--persist" ]]; then
  persist_arg="--persist"
elif [[ $# -gt 0 ]]; then
  die "Usage: $0 [--persist]"
fi

log_dir="$(make_log_dir configure_rails)"

if [[ -t 1 ]]; then
  color_green=$'\033[32m'
  color_red=$'\033[31m'
  color_reset=$'\033[0m'
else
  color_green=""
  color_red=""
  color_reset=""
fi

colorize_status_stream() {
  awk -v green="${color_green}" -v red="${color_red}" -v reset="${color_reset}" '
    /^PASS:/ { print green $0 reset; fflush(); next }
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
  local role="$3"
  local log="${log_dir}/${label}_configure_rails.log"
  local env_assign
  env_assign="$(remote_runtime_env)"
  remote_cmd_header "${label}" "${host}" | tee "${log}"
  echo "Prompt note: enter the OS SSH password first; if sudo prompts, enter the same DUT password again. Password input is hidden."
  set +e
  ssh_tty "${host}" "cd '${CX8_REMOTE_BASE}' && sudo -p '[sudo on ${label}] password for %u: ' env ROLE='${role}' ${env_assign} ./assets/configure_rails.sh ${persist_arg}" 2>&1 | tee -a "${log}" | colorize_status_stream
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" != "0" ]]; then
    printf '%s\n' "${color_red}FAIL: ${label} rail IP/MTU configuration failed with exit code ${rc}${color_reset}"
    return "${rc}"
  fi
}

overall_rc=0
configure_one "${CX8_A_NAME}" "${remote_a}" "${CX8_A_ROLE}" || overall_rc=$?
configure_one "${CX8_B_NAME}" "${remote_b}" "${CX8_B_ROLE}" || overall_rc=$?

echo
echo "Logs: ${log_dir}"
exit "${overall_rc}"
