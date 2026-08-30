#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly PINS_PATH="${SCRIPT_DIR}/pins.json"
readonly INSTALL_ROOT="${HOME}/.local/share/openviking-cuvs"
readonly RELEASES_DIR="${INSTALL_ROOT}/releases"
readonly VENV="${INSTALL_ROOT}/.venv"
readonly ROLLBACK_VENV="${INSTALL_ROOT}/.venv-rollback"
readonly CONFIG_DIR="${HOME}/.openviking"
readonly CONFIG_PATH="${CONFIG_DIR}/ov.conf"
readonly USER_UNIT_DIR="${HOME}/.config/systemd/user"
readonly USER_UNIT_PATH="${USER_UNIT_DIR}/openviking-cuvs.service"
readonly PYPI_INDEX_URL="${OPENVIKING_PYPI_INDEX_URL:-https://pypi.org/simple}"
readonly NVIDIA_PYPI_INDEX_URL="${OPENVIKING_NVIDIA_PYPI_INDEX_URL:-https://pypi.nvidia.com}"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "This lock and setup script require Linux aarch64 (DGX Spark)." >&2
  exit 1
fi

for command_name in cmp cp curl date grep install ln mkdir mktemp mv nvidia-smi python3 \
  readlink rm seq sleep ss systemctl tr uname; do
  command -v "${command_name}" >/dev/null || {
    echo "${command_name} is required. Install current DGX Spark system updates first." >&2
    exit 1
  }
done

readarray -t deployment_pins < <(
  python3 - "${PINS_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    pins = json.load(file)
print(pins["openviking_version"])
print(pins["minimum_nvidia_driver_version"])
print(pins["ollama"]["version"])
PY
)
readonly REQUIRED_OPENVIKING_VERSION="${deployment_pins[0]}"
readonly MINIMUM_NVIDIA_DRIVER_VERSION="${deployment_pins[1]}"
readonly REQUIRED_OLLAMA_VERSION="${deployment_pins[2]}"
readonly EXPECTED_OLLAMA_BINARY="/opt/ollama/${REQUIRED_OLLAMA_VERSION}/bin/ollama"

nvidia-smi >/dev/null || {
  echo "nvidia-smi could not access the GPU. Fix the driver before continuing." >&2
  exit 1
}

driver_versions="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader)"
printf '%s\n' "${driver_versions}" | python3 -c '
import re
import sys

minimum_text = sys.argv[1]
minimum = tuple(int(part) for part in minimum_text.split("."))
versions = [line.strip() for line in sys.stdin if line.strip()]
if not versions:
    raise SystemExit("nvidia-smi returned no driver versions")
unsupported = []
for version in versions:
    if re.fullmatch(r"[0-9]+(?:\.[0-9]+)+", version) is None:
        unsupported.append(version)
        continue
    actual = tuple(int(part) for part in version.split("."))
    width = max(len(actual), len(minimum))
    if actual + (0,) * (width - len(actual)) < minimum + (0,) * (width - len(minimum)):
        unsupported.append(version)
if unsupported:
    raise SystemExit(
        f"RAPIDS 26.06 on CUDA 13 requires NVIDIA driver {minimum_text} or newer; "
        f"found {unsupported}"
    )
' "${MINIMUM_NVIDIA_DRIVER_VERSION}"

systemctl --user show-environment >/dev/null || {
  echo "A working systemd user session is required. Log in locally or through SSH first." >&2
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
  echo "Run ./install-ollama.sh and pull both pinned models before setup.sh." >&2
  exit 1
}
ollama_command="$(command -v ollama)"
if [[ "$(readlink -f "${ollama_command}")" != "${EXPECTED_OLLAMA_BINARY}" ]]; then
  echo "Ollama command must resolve to ${EXPECTED_OLLAMA_BINARY}; run ./install-ollama.sh." >&2
  exit 1
fi

ollama_client_output="$(ollama --version 2>&1)"
printf '%s\n' "${ollama_client_output}" | python3 -c '
import re
import sys

required = sys.argv[1]
versions = re.findall(r"(?<![0-9.])(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?![0-9.])", sys.stdin.read())
if not versions or versions[-1] != required:
    raise SystemExit(f"Expected exact Ollama client version {required}; found {versions}")
' "${REQUIRED_OLLAMA_VERSION}"

ollama_server_json="$(
  curl --fail --silent --show-error http://127.0.0.1:11434/api/version
)" || {
  echo "Ollama is not reachable at http://127.0.0.1:11434." >&2
  exit 1
}
printf '%s\n' "${ollama_server_json}" | python3 -c '
import json
import sys

required = sys.argv[1]
actual = json.load(sys.stdin).get("version")
if actual != required:
    raise SystemExit(f"Expected Ollama server {required}; found {actual}")
' "${REQUIRED_OLLAMA_VERSION}"

ollama_listeners="$(ss -H -ltn 'sport = :11434')"
printf '%s\n' "${ollama_listeners}" | python3 -c '
import ipaddress
import sys

lines = [line for line in sys.stdin.read().splitlines() if line.strip()]
if not lines:
    raise SystemExit("No TCP listener found on port 11434")
for line in lines:
    fields = line.split()
    if len(fields) < 4:
        raise SystemExit(f"Could not parse listener: {line!r}")
    host = fields[3].rsplit(":", 1)[0].strip("[]")
    try:
        address = ipaddress.ip_address(host)
    except ValueError as error:
        raise SystemExit(f"Ollama must listen only on loopback; found {fields[3]}") from error
    if not address.is_loopback:
        raise SystemExit(f"Ollama must listen only on loopback; found {fields[3]}")
'

ollama_tags_json="$(curl --fail --silent --show-error http://127.0.0.1:11434/api/tags)"
printf '%s\n' "${ollama_tags_json}" | python3 -c '
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    expected = json.load(file)["ollama"]["models"]
models = json.load(sys.stdin).get("models", [])
actual = {
    str(model.get("name", "")): str(model.get("digest", "")).removeprefix("sha256:").lower()
    for model in models
}
problems = {
    name: {"expected": digest, "actual": actual.get(name)}
    for name, digest in expected.items()
    if actual.get(name) != digest
}
if problems:
    raise SystemExit(
        "Ollama model tag/digest mismatch. Tags are mutable; update pins.json only after "
        f"intentional model review. Mismatches: {problems}"
    )
' "${PINS_PATH}"

if [[ -L "${CONFIG_PATH}" ]]; then
  echo "Refusing symlinked configuration ${CONFIG_PATH}." >&2
  exit 1
fi
if [[ -e "${CONFIG_PATH}" ]] && ! cmp --silent "${SCRIPT_DIR}/ov.conf" "${CONFIG_PATH}"; then
  echo "Refusing to overwrite existing ${CONFIG_PATH}. Back it up or merge the playbook settings." >&2
  exit 1
fi

if [[ -L "${USER_UNIT_PATH}" ]]; then
  echo "Refusing symlinked user unit ${USER_UNIT_PATH}." >&2
  exit 1
fi
if [[ -e "${USER_UNIT_PATH}" && ! -f "${USER_UNIT_PATH}" ]]; then
  echo "Refusing non-file user unit ${USER_UNIT_PATH}." >&2
  exit 1
fi
unit_had_previous=false
unit_needs_update=true
if [[ -e "${USER_UNIT_PATH}" ]]; then
  unit_had_previous=true
  if cmp --silent "${SCRIPT_DIR}/openviking-cuvs.service" "${USER_UNIT_PATH}"; then
    unit_needs_update=false
  else
    if ! grep -Fq 'openviking-cuvs/.venv/bin/openviking-server' "${USER_UNIT_PATH}"; then
      echo "Refusing to overwrite unrelated ${USER_UNIT_PATH}. Back it up or merge it first." >&2
      exit 1
    fi
  fi
fi

if [[ -L "${INSTALL_ROOT}" || -L "${RELEASES_DIR}" ]]; then
  echo "Refusing symlinked OpenViking install or releases directory." >&2
  exit 1
fi
mkdir -p "${INSTALL_ROOT}" "${RELEASES_DIR}" "${CONFIG_DIR}" "${USER_UNIT_DIR}"
if [[ -e "${ROLLBACK_VENV}" || -L "${ROLLBACK_VENV}" ]]; then
  echo "Refusing to continue while ${ROLLBACK_VENV} exists. Inspect the prior rollback first." >&2
  exit 1
fi

staged_venv="$(mktemp -d "${RELEASES_DIR}/release.XXXXXX")"
next_link="${INSTALL_ROOT}/.venv-next.$$"
cutover_started=false
runtime_verified=false
had_previous=false
old_was_active=false
old_was_enabled=false
old_release=""
unit_changed=false
unit_backup=""
service_touched=false

cleanup() {
  status=$?
  trap - EXIT
  set +e
  rollback_failed=false

  if [[ ${status} -ne 0 && ( \
    "${cutover_started}" == true || \
    "${unit_changed}" == true || \
    "${service_touched}" == true \
  ) && "${runtime_verified}" == false ]]; then
    systemctl --user stop openviking-cuvs.service >/dev/null 2>&1
    if [[ -L "${VENV}" && "$(readlink -f "${VENV}")" == "${staged_venv}" ]]; then
      if ! rm -- "${VENV}"; then
        echo "Failed to remove the failed OpenViking environment link ${VENV}." >&2
        rollback_failed=true
      fi
    fi
    if [[ -e "${ROLLBACK_VENV}" || -L "${ROLLBACK_VENV}" ]]; then
      if mv "${ROLLBACK_VENV}" "${VENV}"; then
        echo "Restored the previous OpenViking environment." >&2
      else
        echo "Failed to restore ${VENV} from ${ROLLBACK_VENV}." >&2
        rollback_failed=true
      fi
    fi

    if [[ "${unit_changed}" == true ]]; then
      if [[ "${unit_had_previous}" == true ]]; then
        rm -f -- "${USER_UNIT_PATH}"
        if ! mv -- "${unit_backup}" "${USER_UNIT_PATH}"; then
          echo "Failed to restore ${USER_UNIT_PATH} from ${unit_backup}." >&2
          rollback_failed=true
        fi
      elif ! rm -f -- "${USER_UNIT_PATH}"; then
        echo "Failed to remove newly installed ${USER_UNIT_PATH}." >&2
        rollback_failed=true
      fi
    fi

    if ! systemctl --user daemon-reload; then
      echo "Failed to reload user systemd while restoring OpenViking." >&2
      rollback_failed=true
    fi
    if [[ "${old_was_enabled}" == true ]]; then
      if ! systemctl --user enable openviking-cuvs.service; then
        echo "Failed to restore the enabled state of openviking-cuvs.service." >&2
        rollback_failed=true
      fi
    else
      systemctl --user disable openviking-cuvs.service >/dev/null 2>&1 || true
    fi
    if [[ "${old_was_active}" == true ]]; then
      if ! systemctl --user restart openviking-cuvs.service; then
        echo "Failed to restart the previous OpenViking service." >&2
        rollback_failed=true
      fi
    else
      systemctl --user stop openviking-cuvs.service >/dev/null 2>&1 || true
    fi

    if [[ "${cutover_started}" == true ]]; then
      echo "Preserved the failed staged environment at ${staged_venv}." >&2
    fi
    if [[ "${rollback_failed}" == true ]]; then
      echo "OpenViking rollback was incomplete; inspect the paths above before retrying." >&2
    else
      echo "Previous OpenViking unit and service state restored." >&2
    fi
  fi
  if [[ -L "${next_link}" ]]; then
    rm -- "${next_link}"
  fi
  if [[ ${status} -ne 0 && "${cutover_started}" == false && -d "${staged_venv}" ]]; then
    rm -rf -- "${staged_venv}"
  fi
  exit "${status}"
}
trap cleanup EXIT

python3 -m venv "${staged_venv}"
"${staged_venv}/bin/python" -m pip --version
"${staged_venv}/bin/python" -m pip install \
  --require-hashes \
  --only-binary=:all: \
  --no-deps \
  --index-url "${PYPI_INDEX_URL}" \
  --requirement "${SCRIPT_DIR}/requirements-bootstrap.lock"
"${staged_venv}/bin/python" -m pip install \
  --require-hashes \
  --only-binary=:all: \
  --index-url "${PYPI_INDEX_URL}" \
  --extra-index-url "${NVIDIA_PYPI_INDEX_URL}" \
  --requirement "${SCRIPT_DIR}/requirements-cu13.lock"
"${staged_venv}/bin/python" -m pip check
"${staged_venv}/bin/python" "${SCRIPT_DIR}/validate-lock.py" \
  "${SCRIPT_DIR}/requirements-bootstrap.lock" \
  "${SCRIPT_DIR}/requirements-cu13.lock"

if systemctl --user is-active --quiet openviking-cuvs.service; then
  old_was_active=true
fi
if systemctl --user is-enabled --quiet openviking-cuvs.service; then
  old_was_enabled=true
fi

install -m 0600 "${SCRIPT_DIR}/ov.conf" "${CONFIG_PATH}"
if [[ "${unit_needs_update}" == true ]]; then
  if [[ "${unit_had_previous}" == true ]]; then
    unit_backup="${USER_UNIT_PATH}.backup-$(date -u +%Y%m%dT%H%M%SZ).$$"
    cp -a "${USER_UNIT_PATH}" "${unit_backup}"
    echo "Backed up the previous OpenViking unit to ${unit_backup}."
  fi
  unit_changed=true
  install -m 0644 "${SCRIPT_DIR}/openviking-cuvs.service" "${USER_UNIT_PATH}"
fi
OPENVIKING_CONFIG_FILE="${CONFIG_PATH}" \
  "${staged_venv}/bin/openviking-server" doctor

old_pid="$(systemctl --user show openviking-cuvs.service --property MainPID --value 2>/dev/null || true)"
service_touched=true
systemctl --user stop openviking-cuvs.service >/dev/null 2>&1 || true
ln -s "${staged_venv}" "${next_link}"
if [[ -e "${VENV}" || -L "${VENV}" ]]; then
  had_previous=true
  if [[ -L "${VENV}" ]]; then
    old_release="$(readlink -f "${VENV}")"
  fi
  mv "${VENV}" "${ROLLBACK_VENV}"
fi
cutover_started=true
mv "${next_link}" "${VENV}"

systemctl --user daemon-reload
systemctl --user enable openviking-cuvs.service
systemctl --user restart openviking-cuvs.service

health_json=""
for _ in $(seq 1 60); do
  if health_json="$(curl --fail --silent --show-error \
    http://127.0.0.1:1933/health 2>/dev/null)"; then
    break
  fi
  sleep 2
done
if [[ -z "${health_json}" ]]; then
  echo "OpenViking did not become healthy. Inspect: journalctl --user -u openviking-cuvs -n 100" >&2
  exit 1
fi

auth_mode="$(printf '%s\n' "${health_json}" | python3 -c '
import json
import sys

required = sys.argv[1]
health = json.load(sys.stdin)
if health.get("status") != "ok" or health.get("healthy") is not True:
    raise SystemExit(f"OpenViking health response is not healthy: {health}")
actual = health.get("version")
if actual != required:
    raise SystemExit(f"Expected OpenViking runtime {required}; found {actual}")
auth_mode = health.get("auth_mode", "unknown")
if auth_mode != "dev":
    raise SystemExit(f"Expected the supplied loopback-only dev auth mode; found {auth_mode}")
print(auth_mode)
' "${REQUIRED_OPENVIKING_VERSION}")"

new_pid="$(systemctl --user show openviking-cuvs.service --property MainPID --value)"
if [[ ! "${new_pid}" =~ ^[1-9][0-9]*$ ]]; then
  echo "OpenViking service has no live MainPID after restart: ${new_pid}" >&2
  exit 1
fi
if [[ "${old_pid}" =~ ^[1-9][0-9]*$ && "${new_pid}" == "${old_pid}" ]]; then
  echo "OpenViking restart reused the old MainPID ${old_pid}; refusing unverified cutover." >&2
  exit 1
fi
runtime_cmdline="$(tr '\0' ' ' <"/proc/${new_pid}/cmdline")"
active_release="$(readlink -f "${VENV}")"
if [[ "${active_release}" != "${staged_venv}" ]]; then
  echo "OpenViking environment link does not point at the verified release." >&2
  exit 1
fi
if [[ "${runtime_cmdline}" != *"${staged_venv}/bin/"* || \
  "${runtime_cmdline}" != *"openviking-server"* ]]; then
  echo "OpenViking MainPID ${new_pid} is not running the installed staged environment." >&2
  exit 1
fi

runtime_verified=true
if [[ "${had_previous}" == true ]]; then
  if [[ -L "${ROLLBACK_VENV}" ]]; then
    if ! rm -- "${ROLLBACK_VENV}"; then
      echo "Warning: could not remove ${ROLLBACK_VENV}; inspect it before the next setup run." >&2
    fi
  else
    if ! rm -rf -- "${ROLLBACK_VENV}"; then
      echo "Warning: could not remove ${ROLLBACK_VENV}; inspect it before the next setup run." >&2
    fi
  fi
  if [[ "${old_release}" == "${RELEASES_DIR}"/release.* && \
    "${old_release}" != "${staged_venv}" && -d "${old_release}" ]]; then
    if ! rm -rf -- "${old_release}"; then
      echo "Warning: could not remove retired release ${old_release}." >&2
    fi
  fi
fi
trap - EXIT

echo "OpenViking ${REQUIRED_OPENVIKING_VERSION} is running from the new environment as PID ${new_pid}."
echo "Health reports auth_mode=${auth_mode}; this playbook relies on loopback and SSH access control."
