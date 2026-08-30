#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly PINS_PATH="${SCRIPT_DIR}/pins.json"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "This installer requires Linux aarch64 (DGX Spark)." >&2
  exit 1
fi

for command_name in curl getent id ln python3 readlink sha256sum ss stat sudo systemctl tar unzstd; do
  command -v "${command_name}" >/dev/null || {
    echo "${command_name} is required before installing Ollama." >&2
    exit 1
  }
done

readarray -t ollama_pins < <(
  python3 - "${PINS_PATH}" <<'PY'
import json
import sys

with open(sys.argv[1], encoding="utf-8") as file:
    ollama = json.load(file)["ollama"]
print(ollama["version"])
print(ollama["archive_url"])
print(ollama["archive_size_bytes"])
print(ollama["archive_sha256"])
PY
)
readonly REQUIRED_OLLAMA_VERSION="${ollama_pins[0]}"
readonly OLLAMA_ARCHIVE_URL="${ollama_pins[1]}"
readonly OLLAMA_ARCHIVE_SIZE_BYTES="${ollama_pins[2]}"
readonly OLLAMA_ARCHIVE_SHA256="${ollama_pins[3]}"
readonly INSTALL_PARENT="/opt/ollama"
readonly INSTALL_PATH="${INSTALL_PARENT}/${REQUIRED_OLLAMA_VERSION}"
readonly SYSTEM_STAGE="${INSTALL_PARENT}/.${REQUIRED_OLLAMA_VERSION}.stage.$$"
readonly OLLAMA_LINK="/usr/local/bin/ollama"
readonly SYSTEM_UNIT_PATH="/etc/systemd/system/ollama.service"

work_dir="$(mktemp -d)"
system_stage_created=false
cleanup() {
  if [[ "${system_stage_created}" == true ]]; then
    sudo rm -rf -- "${SYSTEM_STAGE}"
  fi
  rm -rf -- "${work_dir}"
}
trap cleanup EXIT

archive_path="${work_dir}/ollama-linux-arm64.tar.zst"
extract_path="${work_dir}/extract"
mkdir -p "${extract_path}"

echo "Downloading Ollama ${REQUIRED_OLLAMA_VERSION} without elevated privileges..."
curl --fail --location --show-error \
  "${OLLAMA_ARCHIVE_URL}" \
  --output "${archive_path}"
if [[ "$(stat --format='%s' "${archive_path}")" != "${OLLAMA_ARCHIVE_SIZE_BYTES}" ]]; then
  echo "Downloaded Ollama archive size does not match pins.json." >&2
  exit 1
fi
echo "${OLLAMA_ARCHIVE_SHA256}  ${archive_path}" | sha256sum --check

tar --use-compress-program=unzstd -tf "${archive_path}" | python3 -c '
import pathlib
import sys

members = [line.rstrip("\n") for line in sys.stdin]
for member in members:
    path = pathlib.PurePosixPath(member)
    if path.is_absolute() or ".." in path.parts:
        raise SystemExit(f"Unsafe archive member: {member!r}")
if not any(member.lstrip("./") == "bin/ollama" for member in members):
    raise SystemExit("Archive does not contain bin/ollama")
'
tar --use-compress-program=unzstd -xf "${archive_path}" -C "${extract_path}"

client_output="$(OLLAMA_HOST=127.0.0.1:1 "${extract_path}/bin/ollama" --version 2>&1)"
printf '%s\n' "${client_output}" | python3 -c '
import re
import sys

required = sys.argv[1]
versions = re.findall(r"(?<![0-9.])(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)\.(?:0|[1-9][0-9]*)(?![0-9.])", sys.stdin.read())
if not versions or versions[-1] != required:
    raise SystemExit(f"Expected exact Ollama client version {required}; found {versions}")
' "${REQUIRED_OLLAMA_VERSION}"

if sudo test -L "${INSTALL_PARENT}"; then
  echo "Refusing symlinked installation parent ${INSTALL_PARENT}." >&2
  exit 1
fi
if sudo test -L "${INSTALL_PATH}"; then
  echo "Refusing symlinked installation target ${INSTALL_PATH}." >&2
  exit 1
elif sudo test -e "${INSTALL_PATH}"; then
  if ! sudo test -d "${INSTALL_PATH}"; then
    echo "Refusing non-directory installation target ${INSTALL_PATH}." >&2
    exit 1
  fi
  installed_hash="$(sudo sed -n '1p' "${INSTALL_PATH}/.archive-sha256" 2>/dev/null || true)"
  if [[ "${installed_hash}" != "${OLLAMA_ARCHIVE_SHA256}" ]]; then
    echo "Refusing to reuse ${INSTALL_PATH}: its verified archive marker is missing or different." >&2
    exit 1
  fi
  if ! sudo cmp --silent "${extract_path}/bin/ollama" "${INSTALL_PATH}/bin/ollama"; then
    echo "Refusing to reuse ${INSTALL_PATH}: its Ollama executable differs from the verified archive." >&2
    exit 1
  fi
else
  sudo install -d -m 0755 "${INSTALL_PARENT}"
  sudo install -d -m 0755 "${SYSTEM_STAGE}"
  system_stage_created=true
  sudo cp -a "${extract_path}/." "${SYSTEM_STAGE}/"
  printf '%s\n' "${OLLAMA_ARCHIVE_SHA256}" >"${work_dir}/archive-sha256"
  sudo install -m 0644 "${work_dir}/archive-sha256" "${SYSTEM_STAGE}/.archive-sha256"
  sudo mv "${SYSTEM_STAGE}" "${INSTALL_PATH}"
  system_stage_created=false
fi

if sudo test -d "${OLLAMA_LINK}" && ! sudo test -L "${OLLAMA_LINK}"; then
  echo "Refusing to replace directory ${OLLAMA_LINK}. Move it and rerun the installer." >&2
  exit 1
fi
if sudo test -e "${OLLAMA_LINK}" || sudo test -L "${OLLAMA_LINK}"; then
  current_link="$(sudo readlink -f "${OLLAMA_LINK}")"
  if [[ "${current_link}" != "${INSTALL_PATH}/bin/ollama" ]]; then
    link_backup="${OLLAMA_LINK}.backup-$(date -u +%Y%m%dT%H%M%SZ)"
    sudo cp -a "${OLLAMA_LINK}" "${link_backup}"
    echo "Backed up the previous Ollama command to ${link_backup}."
  fi
fi
sudo install -d -m 0755 /usr/local/bin
sudo ln -sfn "${INSTALL_PATH}/bin/ollama" "${OLLAMA_LINK}"
if [[ "$(sudo readlink -f "${OLLAMA_LINK}")" != "${INSTALL_PATH}/bin/ollama" ]]; then
  echo "Ollama command link does not resolve to the verified installation." >&2
  exit 1
fi

if ! getent group ollama >/dev/null; then
  sudo groupadd --system ollama
fi
if ! id -u ollama >/dev/null 2>&1; then
  sudo useradd --system --gid ollama --home-dir /var/lib/ollama \
    --shell /usr/sbin/nologin ollama
fi
sudo install -d -o ollama -g ollama -m 0750 /var/lib/ollama /var/lib/ollama/models
for gpu_group in render video; do
  if getent group "${gpu_group}" >/dev/null; then
    sudo usermod -aG "${gpu_group}" ollama
  fi
done

if sudo test -L "${SYSTEM_UNIT_PATH}"; then
  echo "Refusing symlinked system unit ${SYSTEM_UNIT_PATH}. Move it and rerun the installer." >&2
  exit 1
fi
if sudo test -e "${SYSTEM_UNIT_PATH}" && ! sudo cmp --silent \
  "${SCRIPT_DIR}/ollama.service" "${SYSTEM_UNIT_PATH}"; then
  unit_backup="${SYSTEM_UNIT_PATH}.backup-$(date -u +%Y%m%dT%H%M%SZ)"
  sudo cp -a "${SYSTEM_UNIT_PATH}" "${unit_backup}"
  echo "Backed up the previous Ollama unit to ${unit_backup}."
fi
sudo install -m 0644 "${SCRIPT_DIR}/ollama.service" "${SYSTEM_UNIT_PATH}"
sudo systemctl daemon-reload
sudo systemctl enable ollama.service
sudo systemctl restart ollama.service

for _ in $(seq 1 60); do
  if server_json="$(curl --fail --silent --show-error \
    http://127.0.0.1:11434/api/version 2>/dev/null)"; then
    break
  fi
  sleep 2
done
if [[ -z "${server_json:-}" ]]; then
  echo "Ollama did not become ready. Inspect: sudo journalctl -u ollama -n 100" >&2
  exit 1
fi

printf '%s\n' "${server_json}" | python3 -c '
import json
import sys

required = sys.argv[1]
actual = json.load(sys.stdin).get("version")
if actual != required:
    raise SystemExit(f"Expected Ollama server {required}; found {actual}")
' "${REQUIRED_OLLAMA_VERSION}"

listeners="$(ss -H -ltn 'sport = :11434')"
printf '%s\n' "${listeners}" | python3 -c '
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
        raise SystemExit(f"Non-loopback Ollama listener: {fields[3]}") from error
    if not address.is_loopback:
        raise SystemExit(f"Non-loopback Ollama listener: {fields[3]}")
'

echo "Ollama ${REQUIRED_OLLAMA_VERSION} is verified and listening only on loopback."
