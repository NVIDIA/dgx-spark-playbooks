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
rdma_topo check || true

if [[ "${APPLY}" == "1" ]]; then
  echo
  note "Applying ACS grub configuration"
  rdma_topo write-grub-acs
  echo "PASS: ACS grub configuration written. Reboot is required."
  echo "After reboot, rerun control-host Steps 5, 6, and 7 before Step 8 --gdr."
else
  echo
  echo "DRY-RUN: no ACS changes written."
  echo "To apply: sudo $0 --apply"
fi
