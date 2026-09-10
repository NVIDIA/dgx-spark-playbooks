#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Inspect or apply rdma_topo ACS configuration for GPUDirect Data Direct.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/cx8-common.sh"

APPLY=0
if [[ "${1:-}" == "--apply" ]]; then
  APPLY=1
elif [[ $# -gt 0 ]]; then
  die "Usage: sudo $0 [--apply]"
fi

need_cmd rdma_topo

echo "### rdma_topo topo"
rdma_topo topo

echo
echo "### rdma_topo check"
check_rc=0
check_output="$(rdma_topo check 2>&1)" || check_rc=$?

acs_ready=1
if [[ "${check_rc}" != "0" ]] || \
   grep -Eq '^(FAIL|ERROR)([[:space:]]|:)' <<<"${check_output}"; then
  acs_ready=0
fi

remediable_mismatch=0
unexpected_check_failure=0
if grep -Eq '^FAIL[[:space:]]+(ACS|Kernel iommu_group)' <<<"${check_output}"; then
  remediable_mismatch=1
fi
if grep -Eq '^ERROR([[:space:]]|:)' <<<"${check_output}" || \
   grep -E '^FAIL([[:space:]]|:)' <<<"${check_output}" \
     | grep -Ev '^FAIL[[:space:]]+(ACS|Kernel iommu_group)' >/dev/null; then
  unexpected_check_failure=1
elif [[ "${check_rc}" != "0" && "${remediable_mismatch}" != "1" ]]; then
  unexpected_check_failure=1
fi

if [[ "${APPLY}" == "1" ]]; then
  # A remediable ACS/IOMMU mismatch is the reason --apply exists. Label that
  # pre-change state without making a successful apply look like a failed run.
  # Unknown FAIL/ERROR output remains unchanged and visible.
  sed -E \
    '/^FAIL[[:space:]]+(ACS|Kernel iommu_group)/s/^FAIL[[:space:]]+/CHANGE REQUIRED: /' \
    <<<"${check_output}"
else
  printf '%s\n' "${check_output}"
fi

if [[ "${APPLY}" == "1" ]]; then
  if [[ "${unexpected_check_failure}" == "1" ]]; then
    echo "ERROR: rdma_topo check failed for a reason that write-grub-acs cannot safely remediate. No GRUB configuration was written." >&2
    if [[ "${check_rc}" == "0" ]]; then
      check_rc=1
    fi
    exit "${check_rc}"
  fi
  echo
  note "Applying ACS grub configuration"
  rdma_topo write-grub-acs
  echo "PASS: ACS grub configuration written. Reboot is required."
  echo "After reboot, rerun the control-host rail configuration, RoCE/GPUDirect configuration, basic validation, and ACS inspection before both GDR rail tests."
else
  echo
  if [[ "${acs_ready}" == "1" ]]; then
    echo "PASS: ACS/Data Direct prerequisites are configured for GDR."
    echo "READ-ONLY: no ACS changes written."
  else
    echo "NOT READY: ACS/Data Direct prerequisites must be configured before running a GDR test."
    echo "Run the control-host Script 11 action with --apply to configure both DUTs."
    exit 2
  fi
fi
