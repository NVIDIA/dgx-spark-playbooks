#!/usr/bin/env bash
set -euo pipefail
umask 022

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
readonly PINS_PATH="${SCRIPT_DIR}/pins.json"

if [[ "$(uname -s)" != "Linux" || "$(uname -m)" != "aarch64" ]]; then
  echo "This installer requires Linux aarch64 (DGX Spark)." >&2
  exit 1
fi

for command_name in chmod chown cmp cp curl date getent groupadd id install ln mkdir mktemp mv \
  python3 readlink rm sed sha256sum sleep ss stat sudo systemctl tar uname unzstd useradd \
  usermod; do
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
link_changed=false
link_had_previous=false
link_backup=""
unit_changed=false
unit_had_previous=false
unit_backup=""
service_touched=false
service_was_active=false
service_was_enabled=false
cleanup() {
  exit_status=$?
  trap - EXIT
  set +e

  rollback_failed=false
  if [[ "${exit_status}" -ne 0 && ( \
    "${link_changed}" == true || \
    "${unit_changed}" == true || \
    "${service_touched}" == true \
  ) ]]; then
    echo "Installation failed; restoring the previous Ollama command, unit, and service state." >&2
    sudo systemctl stop ollama.service >/dev/null 2>&1 || true

    if [[ "${unit_changed}" == true ]]; then
      if [[ "${unit_had_previous}" == true ]]; then
        sudo rm -f -- "${SYSTEM_UNIT_PATH}"
        if ! sudo mv -- "${unit_backup}" "${SYSTEM_UNIT_PATH}"; then
          echo "Failed to restore ${SYSTEM_UNIT_PATH} from ${unit_backup}." >&2
          rollback_failed=true
        fi
      elif ! sudo rm -f -- "${SYSTEM_UNIT_PATH}"; then
        echo "Failed to remove newly installed ${SYSTEM_UNIT_PATH}." >&2
        rollback_failed=true
      fi
    fi

    if [[ "${link_changed}" == true ]]; then
      sudo rm -f -- "${OLLAMA_LINK}"
      if [[ "${link_had_previous}" == true ]] && \
        ! sudo mv -- "${link_backup}" "${OLLAMA_LINK}"; then
        echo "Failed to restore ${OLLAMA_LINK} from ${link_backup}." >&2
        rollback_failed=true
      fi
    fi

    if ! sudo systemctl daemon-reload; then
      echo "Failed to reload systemd while restoring the previous Ollama service." >&2
      rollback_failed=true
    fi
    if [[ "${service_was_enabled}" == true ]]; then
      if ! sudo systemctl enable ollama.service; then
        echo "Failed to restore the enabled state of ollama.service." >&2
        rollback_failed=true
      fi
    elif ! sudo systemctl disable ollama.service >/dev/null 2>&1; then
      # A previously absent or static unit may not support disable.
      true
    fi
    if [[ "${service_was_active}" == true ]]; then
      if ! sudo systemctl restart ollama.service; then
        echo "Failed to restart the previous ollama.service." >&2
        rollback_failed=true
      fi
    else
      sudo systemctl stop ollama.service >/dev/null 2>&1 || true
    fi

    if [[ "${rollback_failed}" == true ]]; then
      echo "Ollama rollback was incomplete; inspect the backup paths above before retrying." >&2
    else
      echo "Previous Ollama command, unit, and service state restored." >&2
    fi
  fi

  if [[ "${system_stage_created}" == true ]]; then
    sudo rm -rf -- "${SYSTEM_STAGE}"
  fi
  rm -rf -- "${work_dir}"
  exit "${exit_status}"
}
trap cleanup EXIT

archive_path="${work_dir}/ollama-linux-arm64.tar.zst"
extract_path="${work_dir}/extract"
mkdir -p "${extract_path}"
chmod 0755 "${extract_path}"

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
import sys

required = sys.argv[1]
prefix = "Warning: client version is "
versions = [line.removeprefix(prefix).strip() for line in sys.stdin if line.startswith(prefix)]
if versions != [required]:
    raise SystemExit(f"Expected exact Ollama client version {required}; found {versions}")
' "${REQUIRED_OLLAMA_VERSION}"

verify_installed_tree() {
  sudo python3 - "${extract_path}" "${INSTALL_PATH}" "${OLLAMA_ARCHIVE_SHA256}" <<'PY'
from __future__ import annotations

import hashlib
import os
import stat
import sys
from pathlib import Path

expected_root = Path(sys.argv[1])
installed_root = Path(sys.argv[2])
expected_archive_hash = sys.argv[3]
marker_name = ".archive-sha256"


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def collect(root: Path, *, installed: bool) -> dict[str, tuple[str, int, int, int, str | None]]:
    root_stat = os.lstat(root)
    if not stat.S_ISDIR(root_stat.st_mode):
        raise SystemExit(f"Expected a directory, found {root}")

    entries: dict[str, tuple[str, int, int, int, str | None]] = {
        ".": ("directory", stat.S_IMODE(root_stat.st_mode), root_stat.st_uid, root_stat.st_gid, None)
    }
    for directory, directory_names, file_names in os.walk(root, followlinks=False):
        for name in sorted(directory_names + file_names):
            path = Path(directory, name)
            relative = path.relative_to(root).as_posix()
            if installed and relative == marker_name:
                continue
            metadata = os.lstat(path)
            mode = stat.S_IMODE(metadata.st_mode)
            if stat.S_ISLNK(metadata.st_mode):
                kind = "symlink"
                payload = os.readlink(path)
            elif stat.S_ISDIR(metadata.st_mode):
                kind = "directory"
                payload = None
            elif stat.S_ISREG(metadata.st_mode):
                kind = "file"
                payload = digest(path)
            else:
                raise SystemExit(f"Unsupported installed payload entry: {relative}")
            entries[relative] = (kind, mode, metadata.st_uid, metadata.st_gid, payload)
    return entries


expected = collect(expected_root, installed=False)
installed = collect(installed_root, installed=True)
missing = sorted(expected.keys() - installed.keys())
unexpected = sorted(installed.keys() - expected.keys())
if missing or unexpected:
    raise SystemExit(f"Installed Ollama tree differs from archive: missing={missing}, unexpected={unexpected}")

for relative, expected_entry in expected.items():
    expected_kind, expected_mode, _, _, expected_payload = expected_entry
    actual_kind, actual_mode, actual_uid, actual_gid, actual_payload = installed[relative]
    normalized_mode = expected_mode if expected_kind == "symlink" else expected_mode & ~0o022
    if actual_kind != expected_kind or actual_payload != expected_payload:
        raise SystemExit(f"Installed Ollama payload differs at {relative}")
    if expected_kind != "symlink" and actual_mode != normalized_mode:
        raise SystemExit(
            f"Installed Ollama mode differs at {relative}: {actual_mode:04o} != {normalized_mode:04o}"
        )
    if (actual_uid, actual_gid) != (0, 0):
        raise SystemExit(f"Installed Ollama entry is not root-owned: {relative}")
    if expected_kind != "symlink" and actual_mode & 0o022:
        raise SystemExit(f"Installed Ollama entry is writable by group or other: {relative}")

marker = installed_root / marker_name
marker_stat = os.lstat(marker)
if not stat.S_ISREG(marker_stat.st_mode):
    raise SystemExit(f"Installed archive marker is not a regular file: {marker}")
if (marker_stat.st_uid, marker_stat.st_gid) != (0, 0):
    raise SystemExit(f"Installed archive marker is not root-owned: {marker}")
if stat.S_IMODE(marker_stat.st_mode) != 0o644:
    raise SystemExit(f"Installed archive marker mode is not 0644: {marker}")
if marker.read_bytes() != f"{expected_archive_hash}\n".encode():
    raise SystemExit(f"Installed archive marker content is invalid: {marker}")

print(f"Verified {len(installed)} root-owned Ollama payload entries against the pinned archive.")
PY
}

if sudo test -L "${INSTALL_PARENT}"; then
  echo "Refusing symlinked installation parent ${INSTALL_PARENT}." >&2
  exit 1
fi
if sudo test -e "${INSTALL_PARENT}"; then
  if ! sudo test -d "${INSTALL_PARENT}"; then
    echo "Refusing non-directory installation parent ${INSTALL_PARENT}." >&2
    exit 1
  fi
else
  sudo install -d -o root -g root -m 0755 "${INSTALL_PARENT}"
fi
install_parent_metadata="$(sudo stat --format='%u:%g:%a' "${INSTALL_PARENT}")"
if [[ "${install_parent_metadata}" != "0:0:755" ]]; then
  echo "Refusing insecure installation parent ${INSTALL_PARENT}: ${install_parent_metadata}." >&2
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
else
  sudo install -d -o root -g root -m 0755 "${SYSTEM_STAGE}"
  system_stage_created=true
  sudo cp -a "${extract_path}/." "${SYSTEM_STAGE}/"
  sudo chown -R --no-dereference root:root "${SYSTEM_STAGE}"
  sudo chmod -R go-w "${SYSTEM_STAGE}"
  printf '%s\n' "${OLLAMA_ARCHIVE_SHA256}" >"${work_dir}/archive-sha256"
  sudo install -o root -g root -m 0644 \
    "${work_dir}/archive-sha256" "${SYSTEM_STAGE}/.archive-sha256"
  sudo mv "${SYSTEM_STAGE}" "${INSTALL_PATH}"
  system_stage_created=false
fi
verify_installed_tree

if sudo systemctl is-active --quiet ollama.service; then
  service_was_active=true
fi
if sudo systemctl is-enabled --quiet ollama.service; then
  service_was_enabled=true
fi

if (sudo test -e "${OLLAMA_LINK}" || sudo test -L "${OLLAMA_LINK}") && \
  ! sudo test -L "${OLLAMA_LINK}" && ! sudo test -f "${OLLAMA_LINK}"; then
  echo "Refusing non-file command path ${OLLAMA_LINK}. Move it and rerun the installer." >&2
  exit 1
fi
link_needs_update=true
if sudo test -e "${OLLAMA_LINK}" || sudo test -L "${OLLAMA_LINK}"; then
  link_had_previous=true
  current_link="$(sudo readlink -f "${OLLAMA_LINK}" 2>/dev/null || true)"
  if sudo test -L "${OLLAMA_LINK}" && [[ "${current_link}" == "${INSTALL_PATH}/bin/ollama" ]]; then
    link_needs_update=false
  else
    link_backup="${OLLAMA_LINK}.backup-$(date -u +%Y%m%dT%H%M%SZ).$$"
    sudo cp -a "${OLLAMA_LINK}" "${link_backup}"
    echo "Backed up the previous Ollama command to ${link_backup}."
  fi
fi
if sudo test -L /usr/local/bin; then
  echo "Refusing symlinked command directory /usr/local/bin." >&2
  exit 1
elif sudo test -e /usr/local/bin; then
  if ! sudo test -d /usr/local/bin; then
    echo "Refusing non-directory command path /usr/local/bin." >&2
    exit 1
  fi
else
  sudo install -d -o root -g root -m 0755 /usr/local/bin
fi
command_dir_metadata="$(sudo stat --format='%u:%g:%a' /usr/local/bin)"
if [[ "${command_dir_metadata}" != "0:0:755" ]]; then
  echo "Refusing insecure command directory /usr/local/bin: ${command_dir_metadata}." >&2
  exit 1
fi
if [[ "${link_needs_update}" == true ]]; then
  link_changed=true
  sudo ln -sfn "${INSTALL_PATH}/bin/ollama" "${OLLAMA_LINK}"
fi
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
ollama_uid="$(id -u ollama)"
ollama_gid="$(id -g ollama)"
service_touched=true
sudo systemctl stop ollama.service >/dev/null 2>&1 || true
if sudo systemctl is-active --quiet ollama.service; then
  echo "Could not stop ollama.service before validating its writable state directories." >&2
  exit 1
fi
sudo python3 - "${ollama_uid}" "${ollama_gid}" <<'PY'
import errno
import os
import stat
import sys

ollama_uid = int(sys.argv[1])
ollama_gid = int(sys.argv[2])
directory_flags = os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW | os.O_CLOEXEC


def ensure_directory(parent_fd: int, name: str, uid: int, gid: int, mode: int) -> int:
    created = False
    try:
        descriptor = os.open(name, directory_flags, dir_fd=parent_fd)
    except FileNotFoundError:
        os.mkdir(name, mode=mode, dir_fd=parent_fd)
        created = True
        descriptor = os.open(name, directory_flags, dir_fd=parent_fd)
    except OSError as error:
        if error.errno in {errno.ELOOP, errno.ENOTDIR}:
            raise SystemExit(f"Refusing non-directory or symlinked Ollama state path: {name}")
        raise

    metadata = os.fstat(descriptor)
    if created:
        os.fchown(descriptor, uid, gid)
        os.fchmod(descriptor, mode)
        metadata = os.fstat(descriptor)
    actual = (metadata.st_uid, metadata.st_gid, stat.S_IMODE(metadata.st_mode))
    expected = (uid, gid, mode)
    if actual != expected:
        os.close(descriptor)
        raise SystemExit(
            f"Refusing incompatible Ollama state directory {name}: {actual}; expected {expected}"
        )
    return descriptor


var_lib_fd = os.open("/var/lib", directory_flags)
try:
    ollama_fd = ensure_directory(var_lib_fd, "ollama", ollama_uid, ollama_gid, 0o750)
    try:
        models_fd = ensure_directory(ollama_fd, "models", ollama_uid, ollama_gid, 0o750)
        os.close(models_fd)
    finally:
        os.close(ollama_fd)
finally:
    os.close(var_lib_fd)
PY
for gpu_group in render video; do
  if getent group "${gpu_group}" >/dev/null; then
    sudo usermod -aG "${gpu_group}" ollama
  fi
done

if sudo test -L "${SYSTEM_UNIT_PATH}"; then
  echo "Refusing symlinked system unit ${SYSTEM_UNIT_PATH}. Move it and rerun the installer." >&2
  exit 1
fi
if sudo test -e "${SYSTEM_UNIT_PATH}" && ! sudo test -f "${SYSTEM_UNIT_PATH}"; then
  echo "Refusing non-file system unit ${SYSTEM_UNIT_PATH}. Move it and rerun the installer." >&2
  exit 1
fi
unit_needs_update=true
if sudo test -e "${SYSTEM_UNIT_PATH}"; then
  unit_had_previous=true
  if sudo cmp --silent "${SCRIPT_DIR}/ollama.service" "${SYSTEM_UNIT_PATH}"; then
    unit_needs_update=false
  else
    unit_backup="${SYSTEM_UNIT_PATH}.backup-$(date -u +%Y%m%dT%H%M%SZ).$$"
    sudo cp -a "${SYSTEM_UNIT_PATH}" "${unit_backup}"
    echo "Backed up the previous Ollama unit to ${unit_backup}."
  fi
fi
if [[ "${unit_needs_update}" == true ]]; then
  unit_changed=true
  sudo install -o root -g root -m 0644 \
    "${SCRIPT_DIR}/ollama.service" "${SYSTEM_UNIT_PATH}"
fi
service_touched=true
sudo systemctl daemon-reload
sudo systemctl enable ollama.service
sudo systemctl restart ollama.service

server_json=""
ollama_main_pid=""
ollama_listeners=""
readiness_deadline=$((SECONDS + 120))
while ((SECONDS < readiness_deadline)); do
  candidate_active="$(sudo systemctl show ollama.service --property=ActiveState --value \
    2>/dev/null || true)"
  candidate_substate="$(sudo systemctl show ollama.service --property=SubState --value \
    2>/dev/null || true)"
  candidate_pid="$(sudo systemctl show ollama.service --property=MainPID --value \
    2>/dev/null || true)"
  if [[ "${candidate_active}" == "active" && "${candidate_substate}" == "running" && \
    "${candidate_pid}" =~ ^[1-9][0-9]*$ ]]; then
    candidate_listeners="$(sudo ss -H -ltnp 'sport = :11434' 2>/dev/null || true)"
    if printf '%s\n' "${candidate_listeners}" | python3 "${SCRIPT_DIR}/validate_listener.py" \
      --port 11434 --label Ollama --expected-pid "${candidate_pid}" >/dev/null 2>&1 && \
      candidate_json="$(curl --fail --silent --show-error --connect-timeout 2 --max-time 5 \
        http://127.0.0.1:11434/api/version 2>/dev/null)"; then
      confirmed_active="$(sudo systemctl show ollama.service --property=ActiveState --value \
        2>/dev/null || true)"
      confirmed_substate="$(sudo systemctl show ollama.service --property=SubState --value \
        2>/dev/null || true)"
      confirmed_pid="$(sudo systemctl show ollama.service --property=MainPID --value \
        2>/dev/null || true)"
      confirmed_listeners="$(sudo ss -H -ltnp 'sport = :11434' 2>/dev/null || true)"
      if [[ "${confirmed_active}" == "active" && "${confirmed_substate}" == "running" && \
        "${confirmed_pid}" == "${candidate_pid}" ]] && \
        printf '%s\n' "${confirmed_listeners}" | \
          python3 "${SCRIPT_DIR}/validate_listener.py" --port 11434 --label Ollama \
            --expected-pid "${confirmed_pid}" >/dev/null 2>&1; then
        server_json="${candidate_json}"
        ollama_main_pid="${confirmed_pid}"
        ollama_listeners="${confirmed_listeners}"
        break
      fi
    fi
  fi
  sleep 2
done
if [[ -z "${server_json}" ]]; then
  failed_active="$(sudo systemctl show ollama.service --property=ActiveState --value \
    2>/dev/null || true)"
  failed_substate="$(sudo systemctl show ollama.service --property=SubState --value \
    2>/dev/null || true)"
  failed_pid="$(sudo systemctl show ollama.service --property=MainPID --value \
    2>/dev/null || true)"
  failed_listeners="$(sudo ss -H -ltnp 'sport = :11434' 2>/dev/null || true)"
  echo "Ollama did not become ready as its own loopback listener " \
    "(ActiveState=${failed_active}, SubState=${failed_substate}, MainPID=${failed_pid})." >&2
  printf '%s\n' "${failed_listeners}" >&2
  echo "Inspect: sudo journalctl -u ollama -n 100" >&2
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

final_active="$(sudo systemctl show ollama.service --property=ActiveState --value)"
final_substate="$(sudo systemctl show ollama.service --property=SubState --value)"
final_pid="$(sudo systemctl show ollama.service --property=MainPID --value)"
ollama_listeners="$(sudo ss -H -ltnp 'sport = :11434')"
if [[ "${final_active}" != "active" || "${final_substate}" != "running" || \
  "${final_pid}" != "${ollama_main_pid}" ]]; then
  echo "Ollama service changed after readiness verification: " \
    "ActiveState=${final_active}, SubState=${final_substate}, MainPID=${final_pid}." >&2
  exit 1
fi
printf '%s\n' "${ollama_listeners}" | python3 "${SCRIPT_DIR}/validate_listener.py" \
  --port 11434 --label Ollama --expected-pid "${ollama_main_pid}"

echo "Ollama ${REQUIRED_OLLAMA_VERSION} is verified as MainPID ${ollama_main_pid} on loopback."
