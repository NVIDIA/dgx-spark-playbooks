#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Collect local run logs into one tarball. Does not touch DUT state.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

mkdir -p "${CX8_LOCAL_LOG_ROOT}"
ts="$(timestamp)"
tarball="${CX8_LOCAL_LOG_ROOT}_manual_logs_${ts}.tar.gz"

tar -C "${CX8_LOCAL_LOG_ROOT}" -czf "${tarball}" .
sha256sum "${tarball}" >"${tarball}.sha256" 2>/dev/null || shasum -a 256 "${tarball}" >"${tarball}.sha256"

echo "Tarball: ${tarball}"
echo "SHA256:  ${tarball}.sha256"
