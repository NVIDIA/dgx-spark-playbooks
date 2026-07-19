#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optional remediation: install or build a perftest ib_write_bw that supports
# CUDA DMA-BUF and Data Direct flags for Step 8 --gdr.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

check_only=0
source_only=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check-only)
      check_only=1
      extra_args+=("--check-only")
      ;;
    --source-only)
      source_only=1
      extra_args+=("--source-only")
      ;;
    --ref)
      [[ -n "${2:-}" ]] || die "--ref requires a git ref"
      CX8_PERFTEST_REF="$2"
      extra_args+=("--ref" "$2")
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--check-only] [--source-only] [--ref GIT_REF]"
      exit 0
      ;;
    *)
      die "Usage: $0 [--check-only] [--source-only] [--ref GIT_REF]"
      ;;
  esac
  shift
done

log_dir="$(make_log_dir install_perftest_gdr)"

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

quote_args() {
  local out=""
  local arg
  for arg in "$@"; do
    printf -v out '%s %q' "${out}" "${arg}"
  done
  echo "${out}"
}

install_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_install_perftest_gdr.log"
  local env_assign
  local remote_extra
  env_assign="$(remote_runtime_env)"
  remote_extra="$(quote_args "${extra_args[@]}")"
  remote_cmd_header "${label}" "${host}" | tee "${log}"
  if [[ "${check_only}" == "1" ]]; then
    echo "Checking whether installed ib_write_bw already supports CUDA DMA-BUF/Data Direct."
  elif [[ "${source_only}" == "1" ]]; then
    echo "Building perftest from source without first trying OS package upgrade."
  else
    echo "Trying OS perftest package upgrade first; falling back to source build if needed."
  fi
  echo "Prompt note: enter the OS SSH password first; if sudo prompts, enter the same DUT password again. Password input is hidden."
  set +e
  ssh_tty "${host}" "cd '${CX8_REMOTE_BASE}' && sudo -p '[sudo on ${label}] password for %u: ' env ${env_assign} ./assets/install_perftest_gdr.sh${remote_extra}" 2>&1 | tee -a "${log}" | colorize_status_stream
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" != "0" ]]; then
    printf '%s\n' "${color_red}FAIL: ${label} perftest GDR install/check failed with exit code ${rc}${color_reset}"
    return "${rc}"
  fi
}

overall_rc=0
install_one "${CX8_A_NAME}" "${remote_a}" || overall_rc=$?
install_one "${CX8_B_NAME}" "${remote_b}" || overall_rc=$?

echo
echo "Logs: ${log_dir}"
exit "${overall_rc}"
