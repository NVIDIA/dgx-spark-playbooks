#!/usr/bin/env bash
#
# SPDX-FileCopyrightText: Copyright (c) 1993-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Optional pair RDMA/GPUDirect bandwidth smoke test.

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck disable=SC1091
source "${SCRIPT_DIR}/lib_pair.sh"

rail=""
gdr=0
duration="${DURATION:-20}"
size="${SIZE:-1048576}"
gpu_bdf="${CX8_GPU_BDF:-}"
server_ready_timeout="${CX8_PERFTEST_SERVER_READY_TIMEOUT:-30}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rail)
      [[ -n "${2:-}" ]] || die "--rail requires 0 or 1"
      rail="$2"
      shift
      ;;
    --gdr) gdr=1 ;;
    --duration)
      [[ -n "${2:-}" ]] || die "--duration requires seconds"
      duration="$2"
      shift
      ;;
    --size)
      [[ -n "${2:-}" ]] || die "--size requires bytes"
      size="$2"
      shift
      ;;
    --gpu-bdf)
      [[ -n "${2:-}" ]] || die "--gpu-bdf requires a PCI bus ID"
      gpu_bdf="$2"
      shift
      ;;
    *) die "Unknown argument: $1" ;;
  esac
  shift
done

[[ "${rail}" == "0" || "${rail}" == "1" ]] || die "Usage: $0 --rail 0|1 [--gdr] [--duration SEC] [--size BYTES] [--gpu-bdf PCI_BUS_ID]"
[[ "${server_ready_timeout}" =~ ^[0-9]+$ ]] || die "CX8_PERFTEST_SERVER_READY_TIMEOUT must be seconds"
CX8_GPU_BDF="${gpu_bdf}"
server_hca="${RAIL0_DEV}"
if [[ "${rail}" == "1" ]]; then
  server_hca="${RAIL1_DEV}"
fi

gdr_arg=""
gdr_name="hostmem"
if [[ "${gdr}" == "1" ]]; then
  gdr_arg="--gdr"
  gdr_name="gdr"
fi

log_dir="$(make_log_dir "perftest_rail${rail}_${gdr_name}")"
run_id="$(timestamp)_rail${rail}_${gdr_name}"
remote_server_log="${CX8_REMOTE_BASE}/logs/${run_id}_server.log"
remote_server_pid="${CX8_REMOTE_BASE}/logs/${run_id}_server.pid"
env_assign="$(remote_runtime_env)"
ssh_mux_enabled=1
case "$(uname -s 2>/dev/null || echo unknown)" in
  MINGW*|MSYS*|CYGWIN*) ssh_mux_enabled=0 ;;
esac
if [[ "${CX8_DISABLE_SSH_MUX:-0}" == "1" ]]; then
  ssh_mux_enabled=0
fi
# OpenSSH ControlPath uses a Unix-domain socket on macOS/Linux. The socket path
# must be short (roughly <104 bytes on many systems), so keep it out of the
# long per-run log directory and macOS's long TMPDIR path.
if [[ "${ssh_mux_enabled}" == "1" ]]; then
  if command -v shasum >/dev/null 2>&1; then
    control_hash="$(printf '%s' "${log_dir}-${run_id}" | shasum -a 1 | awk '{print substr($1,1,10)}')"
  else
    control_hash="$(printf '%s' "${log_dir}-${run_id}" | cksum | awk '{print $1}')"
  fi
  control_parent="${CX8_SSH_CONTROL_TMP:-/tmp}"
  control_dir="$(mktemp -d "${control_parent%/}/cx8s-${control_hash}.XXXXXX")"
else
  control_dir=""
  echo "SSH multiplexing disabled for this control host; additional password prompts may appear"
fi

control_path_for_host() {
  local safe_host
  safe_host="$(printf '%s' "$1" | cksum | awk '{print $1}')"
  echo "${control_dir}/${safe_host}.s"
}

ssh_mux_plain() {
  local host="$1"
  shift
  if [[ "${ssh_mux_enabled}" != "1" ]]; then
    ssh "${SSH_OPTS[@]}" "${host}" "$@"
    return
  fi
  ssh "${SSH_OPTS[@]}" -o ControlMaster=auto -o ControlPersist=120 -o ControlPath="$(control_path_for_host "${host}")" "${host}" "$@"
}

ssh_mux_tty() {
  local host="$1"
  shift
  if [[ "${ssh_mux_enabled}" != "1" ]]; then
    ssh -tt "${SSH_OPTS[@]}" "${host}" "$@"
    return
  fi
  ssh -tt "${SSH_OPTS[@]}" -o ControlMaster=auto -o ControlPersist=120 -o ControlPath="$(control_path_for_host "${host}")" "${host}" "$@"
}

cleanup_mux() {
  [[ "${ssh_mux_enabled}" == "1" ]] || return 0
  ssh -O exit "${SSH_OPTS[@]}" -o ControlPath="$(control_path_for_host "${remote_a}")" "${remote_a}" >/dev/null 2>&1 || true
  ssh -O exit "${SSH_OPTS[@]}" -o ControlPath="$(control_path_for_host "${remote_b}")" "${remote_b}" >/dev/null 2>&1 || true
  [[ -n "${control_dir}" && -d "${control_dir}" ]] && rm -rf "${control_dir}"
}
trap cleanup_mux EXIT

echo "Starting server on ${CX8_A_NAME} (${remote_a}), rail ${rail}"
ssh_mux_plain "${remote_a}" "mkdir -p '${CX8_REMOTE_BASE}/logs'; pkill -f '[i]b_write_bw.*-d ${server_hca}' >/dev/null 2>&1 || true; cd '${CX8_REMOTE_BASE}'; nohup env ROLE='${CX8_A_ROLE}' ${env_assign} ./assets/run_perftest.sh --server --rail '${rail}' ${gdr_arg} --duration '${duration}' --size '${size}' >'${remote_server_log}' 2>&1 & server_pid=\$!; echo \${server_pid} >'${remote_server_pid}'; start=\$(date +%s); timeout='${server_ready_timeout}'; while :; do if grep -Eq 'Waiting for client to connect|local address:' '${remote_server_log}' 2>/dev/null; then exit 0; fi; if ! kill -0 \${server_pid} >/dev/null 2>&1; then echo 'ERROR: perftest server exited before accepting a client'; cat '${remote_server_log}' 2>/dev/null || true; exit 1; fi; now=\$(date +%s); if [ \$((now - start)) -ge \${timeout} ]; then echo \"ERROR: perftest server was not ready within \${timeout}s\"; cat '${remote_server_log}' 2>/dev/null || true; exit 1; fi; sleep 1; done"
echo "Server is ready on ${CX8_A_NAME}; starting client"

echo "Starting client on ${CX8_B_NAME} (${remote_b}), rail ${rail}"
client_log="${log_dir}/${CX8_B_NAME}_client_rail${rail}_${gdr_name}.log"
set +e
ssh_mux_tty "${remote_b}" "cd '${CX8_REMOTE_BASE}' && env ROLE='${CX8_B_ROLE}' ${env_assign} ./assets/run_perftest.sh --client --rail '${rail}' ${gdr_arg} --duration '${duration}' --size '${size}'" 2>&1 | tee "${client_log}"
client_rc="${PIPESTATUS[0]}"
set -e

echo "Collecting server log"
server_log="${log_dir}/${CX8_A_NAME}_server_rail${rail}_${gdr_name}.log"
ssh_mux_plain "${remote_a}" "cat '${remote_server_log}' 2>/dev/null || true" 2>&1 | tee "${server_log}"

echo "Cleaning up server if still running"
ssh_mux_plain "${remote_a}" "if test -s '${remote_server_pid}'; then pid=\$(cat '${remote_server_pid}'); kill \"\${pid}\" >/dev/null 2>&1 || true; fi" >/dev/null 2>&1 || true

echo
echo "Logs: ${log_dir}"
exit "${client_rc}"
