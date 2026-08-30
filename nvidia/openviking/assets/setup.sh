#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly INSTALL_ROOT="${HOME}/.local/share/openviking-cuvs"
readonly VENV="${INSTALL_ROOT}/.venv"
readonly CONFIG_DIR="${HOME}/.openviking"
readonly CONFIG_PATH="${CONFIG_DIR}/ov.conf"
readonly USER_UNIT_DIR="${HOME}/.config/systemd/user"
readonly USER_UNIT_PATH="${USER_UNIT_DIR}/openviking-cuvs.service"
readonly PYPI_INDEX_URL="${OPENVIKING_PYPI_INDEX_URL:-https://pypi.org/simple}"
readonly NVIDIA_PYPI_INDEX_URL="${OPENVIKING_NVIDIA_PYPI_INDEX_URL:-https://pypi.nvidia.com}"
readonly REQUIRED_OLLAMA_VERSION="0.33.2"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "This lock and setup script require Linux aarch64 (DGX Spark)." >&2
  exit 1
fi

for command_name in curl nvidia-smi systemctl; do
  command -v "${command_name}" >/dev/null || {
    echo "${command_name} is required. Install current DGX Spark system updates first." >&2
    exit 1
  }
done

nvidia-smi >/dev/null || {
  echo "nvidia-smi could not access the GPU. Fix the driver before continuing." >&2
  exit 1
}

systemctl --user show-environment >/dev/null || {
  echo "A working systemd user session is required. Log in locally or through SSH first." >&2
  exit 1
}

command -v python3 >/dev/null || {
  echo "Python 3.12 is required." >&2
  exit 1
}

python3 - <<'PY'
import platform
import sys

if sys.version_info[:2] != (3, 12):
    raise SystemExit(f"Python 3.12 is required; found {sys.version.split()[0]}")

libc_name, libc_version = platform.libc_ver()
libc_parts = tuple(int(part) for part in libc_version.split(".")[:2])
if libc_name != "glibc" or libc_parts < (2, 39):
    raise SystemExit(
        f"The lock targets Ubuntu 24.04 / glibc 2.39 or newer; found {libc_name} {libc_version}"
    )
PY

command -v ollama >/dev/null || {
  echo "Install the pinned Ollama release and pull both models before running setup.sh." >&2
  exit 1
}

ollama_version="$(ollama --version 2>&1)"
if [[ "${ollama_version}" != *"${REQUIRED_OLLAMA_VERSION}"* ]]; then
  echo "Ollama ${REQUIRED_OLLAMA_VERSION} is required; found: ${ollama_version}" >&2
  exit 1
fi

ollama_server_json="$(
  curl --fail --silent --show-error http://127.0.0.1:11434/api/version
)" || {
  echo "Ollama is not reachable at http://127.0.0.1:11434." >&2
  exit 1
}

ollama_server_version="$(
  python3 -c 'import json, sys; print(json.load(sys.stdin)["version"])' \
    <<<"${ollama_server_json}"
)"
if [[ "${ollama_server_version}" != "${REQUIRED_OLLAMA_VERSION}" ]]; then
  echo "Ollama server ${REQUIRED_OLLAMA_VERSION} is required; found ${ollama_server_version}." >&2
  exit 1
fi

if [[ -e "${CONFIG_PATH}" ]] && ! cmp --silent "${SCRIPT_DIR}/ov.conf" "${CONFIG_PATH}"; then
  echo "Refusing to overwrite existing ${CONFIG_PATH}. Back it up or merge the playbook settings." >&2
  exit 1
fi

if [[ -e "${USER_UNIT_PATH}" ]] && ! cmp --silent \
  "${SCRIPT_DIR}/openviking-cuvs.service" "${USER_UNIT_PATH}"; then
  echo "Refusing to overwrite existing ${USER_UNIT_PATH}. Back it up or merge it first." >&2
  exit 1
fi

mkdir -p "${INSTALL_ROOT}" "${CONFIG_DIR}" "${USER_UNIT_DIR}"
python3 -m venv "${VENV}"
"${VENV}/bin/python" -m pip install \
  --index-url "${PYPI_INDEX_URL}" \
  --upgrade "pip==26.2.1"
"${VENV}/bin/python" -m pip install \
  --require-hashes \
  --only-binary=:all: \
  --index-url "${PYPI_INDEX_URL}" \
  --extra-index-url "${NVIDIA_PYPI_INDEX_URL}" \
  --requirement "${SCRIPT_DIR}/requirements-cu13.lock"
"${VENV}/bin/python" -m pip check

install -m 0600 "${SCRIPT_DIR}/ov.conf" "${CONFIG_PATH}"
install -m 0644 "${SCRIPT_DIR}/openviking-cuvs.service" "${USER_UNIT_PATH}"

OPENVIKING_CONFIG_FILE="${CONFIG_PATH}" "${VENV}/bin/openviking-server" doctor

systemctl --user daemon-reload
systemctl --user enable --now openviking-cuvs.service

for _ in $(seq 1 60); do
  if curl --fail --silent http://127.0.0.1:1933/health >/dev/null; then
    echo "OpenViking is healthy at http://127.0.0.1:1933."
    exit 0
  fi
  sleep 2
done

echo "OpenViking did not become healthy. Inspect: journalctl --user -u openviking-cuvs -n 100" >&2
exit 1
