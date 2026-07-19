#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Non-secret environment for a two-DGX Station CX8 setup.

set -euo pipefail

CX8_PAIR_NAME="${CX8_PAIR_NAME:-dgx-station-cx8-setup}"
CX8_OS_USER="${CX8_OS_USER:-nvidia}"

CX8_A_NAME="${CX8_A_NAME:-station-a}"
CX8_A_HOST="${CX8_A_HOST:-}"
CX8_A_ROLE="${CX8_A_ROLE:-station-a}"

CX8_B_NAME="${CX8_B_NAME:-station-b}"
CX8_B_HOST="${CX8_B_HOST:-}"
CX8_B_ROLE="${CX8_B_ROLE:-station-b}"

RAIL0_DEV="${RAIL0_DEV:-mlx5_0}"
RAIL1_DEV="${RAIL1_DEV:-mlx5_1}"
MTU="${MTU:-9000}"
ROCE_TOS="${ROCE_TOS:-106}"
CX8_GPU_BDF="${CX8_GPU_BDF:-}"
CX8_TRY_NVIDIA_PEERMEM="${CX8_TRY_NVIDIA_PEERMEM:-0}"
CX8_PERFTEST_REPO="${CX8_PERFTEST_REPO:-https://github.com/linux-rdma/perftest.git}"
CX8_PERFTEST_REF="${CX8_PERFTEST_REF:-26.04.17}"
CX8_PERFTEST_PREFIX="${CX8_PERFTEST_PREFIX:-/usr/local}"
CX8_PERFTEST_SERVER_READY_TIMEOUT="${CX8_PERFTEST_SERVER_READY_TIMEOUT:-30}"

STATION_A_RAIL0_CIDR="${STATION_A_RAIL0_CIDR:-192.168.100.1/24}"
STATION_A_RAIL1_CIDR="${STATION_A_RAIL1_CIDR:-192.168.101.1/24}"
STATION_B_RAIL0_CIDR="${STATION_B_RAIL0_CIDR:-192.168.100.2/24}"
STATION_B_RAIL1_CIDR="${STATION_B_RAIL1_CIDR:-192.168.101.2/24}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="${REPO_ROOT:-${SCRIPT_DIR}}"
CX8_ASSET_SRC="${CX8_ASSET_SRC:-${REPO_ROOT}/dut-assets}"
CX8_LOCAL_LOG_ROOT="${CX8_LOCAL_LOG_ROOT:-${REPO_ROOT}/logs}"

LOCAL_ENV="${SCRIPT_DIR}/00_env.local"
if [[ -f "${LOCAL_ENV}" ]]; then
  # 00_env.local is intentionally git-ignored. Use it for lab-specific
  # hostnames/IPs or user overrides that must not be committed.
  # shellcheck source=/dev/null
  source "${LOCAL_ENV}"
fi

# Derive the remote workspace after 00_env.local is loaded, so a local
# CX8_OS_USER override also changes the default remote home directory.
CX8_REMOTE_BASE="${CX8_REMOTE_BASE:-/home/${CX8_OS_USER}/cx8-two-station}"

# Default to OpenSSH's interactive host-key confirmation instead of trusting
# first use automatically. Labs that intentionally want TOFU automation can set
# CX8_SSH_STRICT_HOST_KEY_CHECKING=accept-new in 00_env.local.
CX8_SSH_STRICT_HOST_KEY_CHECKING="${CX8_SSH_STRICT_HOST_KEY_CHECKING:-ask}"
case "${CX8_SSH_STRICT_HOST_KEY_CHECKING}" in
  yes|ask|accept-new) ;;
  *)
    echo "ERROR: CX8_SSH_STRICT_HOST_KEY_CHECKING must be one of: yes, ask, accept-new" >&2
    return 2 2>/dev/null || exit 2
    ;;
esac

SSH_OPTS=(
  -o "StrictHostKeyChecking=${CX8_SSH_STRICT_HOST_KEY_CHECKING}"
  -o ServerAliveInterval=30
  -o ServerAliveCountMax=4
)

remote_a="${CX8_OS_USER}@${CX8_A_HOST}"
remote_b="${CX8_OS_USER}@${CX8_B_HOST}"

if [[ -z "${CX8_A_HOST}" || -z "${CX8_B_HOST}" ]]; then
  echo "ERROR: set CX8_A_HOST and CX8_B_HOST in 00_env.local or the environment before running this setup." >&2
  return 2 2>/dev/null || exit 2
fi
