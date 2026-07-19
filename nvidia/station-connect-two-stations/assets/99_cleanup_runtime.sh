#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Cleanup temporary CX8 rail setup on both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

remove_persist=0
down_interfaces=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --remove-persist)
      remove_persist=1
      ;;
    --down)
      down_interfaces=1
      ;;
    *)
      die "Usage: $0 [--remove-persist] [--down]"
      ;;
  esac
  shift
done

log_dir="$(make_log_dir cleanup_runtime)"

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
    /^FAIL:/ || /^ERROR:/ ||
      /[Ee][Rr][Rr][Oo][Rr]/ ||
      /[Ff]ailed/ ||
      /command not found/ ||
      /No such file/ ||
      /Permission denied/ { print red $0 reset; fflush(); next }
    { print; fflush(); next }
  '
}

cleanup_one() {
  local label="$1"
  local host="$2"
  local role="$3"
  local log="${log_dir}/${label}_cleanup_runtime.log"
  local env_assign args
  env_assign="$(remote_runtime_env)"
  args=""
  [[ "${remove_persist}" == "1" ]] && args="${args} --remove-persist"
  [[ "${down_interfaces}" == "1" ]] && args="${args} --down"

  remote_cmd_header "${label}" "${host}" | tee "${log}"
  echo "Prompt note: enter the OS SSH password first; if sudo prompts, enter the same DUT password again. Password input is hidden."
  set +e
  ssh_tty "${host}" "cd '${CX8_REMOTE_BASE}' && sudo -p '[sudo on ${label}] password for %u: ' env ROLE='${role}' ${env_assign} ./assets/cleanup_runtime_on_dut.sh ${args}" 2>&1 | tee -a "${log}" | colorize_status_stream
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" != "0" ]]; then
    printf '%s\n' "${color_red}FAIL: ${label} cleanup failed with exit code ${rc}${color_reset}"
    return "${rc}"
  fi
}

overall_rc=0
cleanup_one "${CX8_A_NAME}" "${remote_a}" "${CX8_A_ROLE}" || overall_rc=$?
cleanup_one "${CX8_B_NAME}" "${remote_b}" "${CX8_B_ROLE}" || overall_rc=$?

echo
echo "Logs: ${log_dir}"
exit "${overall_rc}"
