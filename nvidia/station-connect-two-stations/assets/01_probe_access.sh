#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# OS access and command-availability probe.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

log_dir="$(make_log_dir probe_access)"

probe_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_probe_access.log"
  remote_cmd_header "${label}" "${host}" | tee "${log}"
  ssh_plain "${host}" '
    set -u
    echo "utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    echo "host=$(hostname -f 2>/dev/null || hostname)"
    echo "user=$(id -un)"
    echo "kernel=$(uname -r)"
    test -r /etc/os-release && . /etc/os-release && echo "os=${PRETTY_NAME}"
    echo "[required commands]"
    for cmd in bash sudo tar lspci nvidia-smi ibdev2netdev ibv_devinfo ip ethtool show_gids rdma_topo mlnx_qos cma_roce_tos ib_write_bw; do
      if command -v "${cmd}" >/dev/null 2>&1; then
        echo "ok ${cmd}=$(command -v "${cmd}")"
      else
        echo "missing ${cmd}"
      fi
    done
    echo "[optional MFT commands]"
    for cmd in flint mlxconfig; do
      if command -v "${cmd}" >/dev/null 2>&1; then
        echo "optional_ok ${cmd}=$(command -v "${cmd}")"
      else
        echo "optional_missing ${cmd}"
      fi
    done
    if sudo -n true >/dev/null 2>&1; then
      echo "sudo_status=passwordless-or-cached"
    else
      echo "sudo_status=will-prompt-or-require-password"
    fi
  ' 2>&1 | tee -a "${log}"
}

probe_one "${CX8_A_NAME}" "${remote_a}"
probe_one "${CX8_B_NAME}" "${remote_b}"

echo
echo "Logs: ${log_dir}"
