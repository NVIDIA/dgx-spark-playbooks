#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Copy generic CX8 setup scripts to both stations.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

need_asset_src
log_dir="$(make_log_dir push_assets)"
tmp_assets="$(mktemp -d)"
trap 'rm -rf "${tmp_assets}"' EXIT

prepare_assets() {
  rm -f "${tmp_assets}/"*.sh
  cp "${CX8_ASSET_SRC}/"*.sh "${tmp_assets}/"
  # Guard against CRLF checkouts. A CR in the shebang makes remote Linux fail
  # with: /usr/bin/env: 'bash\r': No such file or directory
  perl -pi -e 's/\r$//' "${tmp_assets}/"*.sh
  chmod +x "${tmp_assets}/"*.sh
}

prepare_assets

push_one() {
  local label="$1"
  local host="$2"
  local log="${log_dir}/${label}_push_assets.log"
  local remote_script
  printf -v remote_script '%s\n' \
    "set -euo pipefail" \
    "mkdir -p '${CX8_REMOTE_BASE}/assets' '${CX8_REMOTE_BASE}/logs'" \
    "tar -C '${CX8_REMOTE_BASE}/assets' -xf -" \
    "chmod +x '${CX8_REMOTE_BASE}/assets/'*.sh" \
    "ls -l '${CX8_REMOTE_BASE}/assets'"
  remote_cmd_header "${label}" "${host}" | tee "${log}"
  (
    cd "${tmp_assets}"
    COPYFILE_DISABLE=1 tar --format ustar -cf - ./*.sh
  ) | ssh_plain "${host}" "bash -lc $(printf '%q' "${remote_script}")" 2>&1 | tee -a "${log}"
}

push_one "${CX8_A_NAME}" "${remote_a}"
push_one "${CX8_B_NAME}" "${remote_b}"

echo
echo "Logs: ${log_dir}"
