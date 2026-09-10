#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Inspect GDR ACS/Data Direct readiness or write the required GRUB config.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

apply=0
extra_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --apply)
      apply=1
      extra_args+=("--apply")
      ;;
    -h|--help)
      echo "Usage: $0 [--apply]"
      echo
      echo "Without --apply, inspect rdma_topo topology/check only."
      echo "With --apply, write ACS/Data Direct GRUB config on both DUTs; reboot is required."
      echo "After reboot, rerun ./05_configure_rails_runtime.sh, ./06_configure_roce_gdr_runtime.sh, ./07_validate_setup.sh, and this inspection before both ./08_run_perftest_pair.sh GDR rail tests."
      exit 0
      ;;
    *)
      die "Usage: $0 [--apply]"
      ;;
  esac
  shift
done

log_dir="$(make_log_dir configure_acs_grub)"

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
    /^WARN:/ || /^CHANGE REQUIRED:/ { print yellow $0 reset; fflush(); next }
    /^FAIL:/ || /^ERROR:/ || /^NOT READY:/ ||
      /FAIL[[:space:]]+ACS/ ||
      /FAIL[[:space:]]+Kernel iommu_group/ ||
      /incorrect values/ { print red $0 reset; fflush(); next }
    /^OK[[:space:]]/ { print green $0 reset; fflush(); next }
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

configure_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_configure_acs_grub.log"
  local remote_extra=""
  if [[ "${#extra_args[@]}" -gt 0 ]]; then
    remote_extra="$(quote_args "${extra_args[@]}")"
  fi

  remote_cmd_header "${label}" "${host}" | tee "${log}"
  if [[ "${apply}" == "1" ]]; then
    echo "Applying ACS/Data Direct GRUB config. This script does not reboot either DUT; a reboot is required before retesting."
    echo "Before rebooting, confirm both DUTs are idle and downtime/workload-owner approval is in place."
    echo "After reboot, rerun ./05_configure_rails_runtime.sh, ./06_configure_roce_gdr_runtime.sh, ./07_validate_setup.sh, and ./11_configure_acs_grub.sh before both ./08_run_perftest_pair.sh GDR rail tests."
  else
    echo "Inspecting ACS/Data Direct prerequisites only. No system changes will be written."
  fi
  echo "Prompt note: enter the OS SSH password first; if sudo prompts, enter the same DUT password again. Password input is hidden."

  set +e
  ssh_tty "${host}" "cd '${CX8_REMOTE_BASE}' && sudo -p '[sudo on ${label}] password for %u: ' ./assets/configure_acs.sh${remote_extra}" 2>&1 | tee -a "${log}" | colorize_status_stream
  local rc=${PIPESTATUS[0]}
  set -e
  if [[ "${rc}" != "0" ]]; then
    if [[ "${apply}" == "1" ]]; then
      printf '%s\n' "${color_red}FAIL: ${label} ACS/Data Direct GRUB apply failed with exit code ${rc}${color_reset}"
    else
      printf '%s\n' "${color_red}NOT READY: ${label} ACS/Data Direct prerequisites did not pass${color_reset}"
    fi
    return "${rc}"
  fi
}

overall_rc=0
configure_one "${CX8_A_NAME}" "${remote_a}" || overall_rc=$?
configure_one "${CX8_B_NAME}" "${remote_b}" || overall_rc=$?

if [[ "${apply}" == "1" && "${overall_rc}" == "0" ]]; then
  cat <<'EOF'

Next after both DUTs reboot:
  Confirm both DUTs are idle and downtime/workload-owner approval is in place.
  ./05_configure_rails_runtime.sh
  ./06_configure_roce_gdr_runtime.sh
  ./07_validate_setup.sh
  ./11_configure_acs_grub.sh
  ./08_run_perftest_pair.sh --rail 0 --gdr
  ./08_run_perftest_pair.sh --rail 1 --gdr

The prerequisite/current-state check and cable/link presence check are optional
after reboot unless OS/packages changed or cable state is uncertain.
Use ./05_configure_rails_runtime.sh --persist instead only when the pair must
retain rail IP/MTU settings across later reboots or repeated performance runs.
EOF
elif [[ "${apply}" == "1" ]]; then
  printf '%s\n' "${color_red}ACTION: ACS/Data Direct GRUB apply did not complete on both DUTs. Do not reboot or continue the pair until the failed DUT is fixed and ./11_configure_acs_grub.sh --apply passes on both systems.${color_reset}" >&2
elif [[ "${overall_rc}" == "0" ]]; then
  printf '%s\n' "${color_green}PASS: both DUTs satisfy the ACS/Data Direct prerequisite for GDR.${color_reset}"
else
  printf '%s\n' "${color_red}STOP: do not run ./08_run_perftest_pair.sh --gdr until ./11_configure_acs_grub.sh passes on both DUTs.${color_reset}" >&2
fi

echo
echo "Logs: ${log_dir}"
exit "${overall_rc}"
