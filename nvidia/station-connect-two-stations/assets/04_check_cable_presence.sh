#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Check software-visible CX8 cable/link presence after physical cabling.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

log_dir="$(make_log_dir cable_presence)"
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
  local status status_console ok_count event_count slow_count

  ok_count="$(grep -c ' cable_visible=yes' "${log}" 2>/dev/null || true)"
  slow_count="$(grep -c ' speed_ok=no' "${log}" 2>/dev/null || true)"
  event_count="$(grep -c '^EVENT ' "${log}" 2>/dev/null || true)"

  if [[ "${slow_count}" != "0" ]]; then
    status="WARN: one or more CX8 rails are linked below expected 400G; you may continue for limited-speed functional validation, but this is not an 800G setup pass"
  elif [[ "${rc}" != "0" ]]; then
    status="FAIL: remote cable check returned ${rc}"
  elif [[ "${ok_count}" == "2" ]]; then
    if [[ "${event_count}" == "0" ]]; then
      status="PASS: both CX8 rails show software-visible link"
    else
      status="WARN: both CX8 rails link now, but recent cable/high-temperature event(s) were found"
    fi
  else
    status="FAIL: one or more CX8 rails do not show link yet"
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
    grep '^CHECK ' "${log}" | sed 's/^CHECK /- /' || true
    grep '^ACTION ' "${log}" | sed 's/^ACTION /- action: /' || true
    grep '^EVENT ' "${log}" | sed 's/^EVENT /- recent event: /' || true
    echo "- Full log: ${log}"
    echo
  } >>"${summary}"

  {
    echo "### ${label}"
    echo
    printf -- "- Status: %s\n" "${status_console}"
    grep '^CHECK ' "${log}" | sed 's/^CHECK /- /' || true
    grep '^ACTION ' "${log}" | sed 's/^ACTION /- action: /' || true
    grep '^EVENT ' "${log}" | sed 's/^EVENT /- recent event: /' || true
    echo "- Full log: ${log}"
    echo
  }

  [[ "${status}" != FAIL:* ]]
}

check_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_cable_presence.log"
  local env_assign
  env_assign="$(remote_runtime_env)"
  remote_cmd_header "${label}" "${host}"
  set +e
  ssh_plain "${host}" "cd '${CX8_REMOTE_BASE}' && env ${env_assign} ./assets/check_cable_presence_on_dut.sh" >"${log}" 2>&1
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
