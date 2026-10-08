"""Build a clean release from a reviewed, hash-pinned file list. Python 3.11+.

No discovery, live-state snapshot, credentials import, or directory copying.
The checked-in inventory intentionally refuses a full runtime release today.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import stat
import tempfile
from urllib.parse import urlsplit
import zipfile


class ReleaseError(ValueError):
    pass


REQUIRED_GATES = {
    "fresh_household_initialization", "no_seeded_property_records",
    "own_ha_account_and_devices", "own_google_scope_and_oauth",
    "portable_health_collector", "native_openclaw_configuration",
    "own_maintenance_incident_ingestion", "runtime_dependencies_pinned",
    "fresh_host_validation",
}
PRIVATE_PARTS = {
    ".git", ".storage", ".ssh", ".env", "state", "sessions", "session",
    "browser_profiles", "browser-profile", "cookies", "credentials", "secrets",
    "property", "memory", "cache", "logs", "work", "relocation-backup",
}
PRIVATE_NAMES = {"auth.json", "openclaw.json", "config.yaml", "secrets.yaml", "household.json",
                 "home-settings.json", "health-config.json", "google-scope.json"}
PRIVATE_SUFFIXES = {".db", ".sqlite", ".sqlite3", ".pem", ".key", ".pfx", ".p12", ".log"}
SHA256 = re.compile(r"[0-9a-fA-F]{64}")
WINDOWS_DEVICES = re.compile(r"(?i)(CON|PRN|AUX|NUL|COM[0-9]|LPT[0-9])(?:\..*)?")
SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bya29\.[A-Za-z0-9_-]{20,}"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}"),
    re.compile(rb"\beyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}"),
    re.compile(rb"(?i)[\"']?(?:access_token|refresh_token|client_secret|ha_token|discord_token)[\"']?\s*[:=]\s*[\"']?[A-Za-z0-9_-]{20,}"),
]


def relative_path(value: str) -> str:
    if not isinstance(value, str) or not value or "\\" in value or value.startswith("/"):
        raise ReleaseError("Paths must be nonempty forward-slash relative paths")
    for part in value.split("/"):
        if (part in {"", ".", ".."} or part.endswith((" ", "."))
                or any(ord(c) < 32 or c in '<>:"|?*' for c in part)
                or WINDOWS_DEVICES.fullmatch(part)):
            raise ReleaseError("Unsafe Windows relative path")
    return value


def clean_path(value: str) -> str:
    value = relative_path(value)
    parts = value.lower().split("/")
    name = parts[-1]
    if (any(p in PRIVATE_PARTS for p in parts) or name in PRIVATE_NAMES
            or name.startswith(".env") or re.search(r"(?:token|credential|cookie|session|secret)", name)
            or Path(name).suffix in PRIVATE_SUFFIXES or name.endswith(("-wal", "-shm"))):
        raise ReleaseError("Private-state path rejected: " + value)
    return value


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _checked_file(root: Path, name: str) -> Path:
    current = root
    for part in clean_path(name).split("/"):
        current = current / part
        info = current.lstat()
        if stat.S_ISLNK(info.st_mode) or getattr(info, "st_file_attributes", 0) & 0x400:
            raise ReleaseError("Symlinks/junctions/reparse points are not release inputs")
    if not current.is_file() or not current.resolve().is_relative_to(root):
        raise ReleaseError("Release input must be an ordinary file under source root")
    return current


def inventory_files(inventory: dict, root: Path) -> list[tuple[str, Path, str]]:
    if inventory.get("schemaVersion") != 1 or inventory.get("architecture") != "arm64":
        raise ReleaseError("Inventory requires schemaVersion 1 and architecture arm64")
    version = inventory.get("releaseVersion")
    if not isinstance(version, str) or not re.fullmatch(r"[A-Za-z0-9_.-]{1,80}", version):
        raise ReleaseError("Invalid releaseVersion")
    blockers = inventory.get("blockers")
    if not isinstance(blockers, list) or blockers:
        raise ReleaseError("Runtime release blocked; resolve every recorded blocker before building")
    gates = inventory.get("readiness")
    if not isinstance(gates, dict) or set(gates) != REQUIRED_GATES or any(v is not True for v in gates.values()):
        raise ReleaseError("Runtime readiness gates are incomplete")
    entries = inventory.get("files")
    if not isinstance(entries, list) or not entries:
        raise ReleaseError("A nonempty explicit reviewed file inventory is required")
    files, destinations = [], set()
    for row in entries:
        if not isinstance(row, dict) or set(row) != {"source", "path", "sha256", "classification", "reviewed"}:
            raise ReleaseError("Each inventory entry requires source/path/sha256/classification/reviewed")
        if row["reviewed"] is not True or row["classification"] not in {"code", "runtime", "license", "clean-template"}:
            raise ReleaseError("Every exact file requires an explicit clean-content review")
        name = clean_path(row["path"])
        if name.casefold() in destinations:
            raise ReleaseError("Duplicate Windows destination")
        destinations.add(name.casefold())
        if not isinstance(row["sha256"], str) or not SHA256.fullmatch(row["sha256"]):
            raise ReleaseError("Every release input must have a SHA256 pin")
        source = _checked_file(root, row["source"])
        files.append((name, source, row["sha256"].lower()))
    # Windows cannot extract a file and a directory at the same case-insensitive path.
    for name, _, _ in files:
        parts = name.split("/")
        if any("/".join(parts[:i]).casefold() in destinations for i in range(1, len(parts))):
            raise ReleaseError("File/directory destination collision")
    entry = clean_path(inventory.get("entryPoint"))
    if Path(entry).suffix.lower() not in {".ps1", ".exe", ".cmd", ".bat"} or entry.casefold() not in destinations:
        raise ReleaseError("entryPoint must reference a packaged Windows launcher")
    # Enforce exact casing, since the bootstrapper also validates ZIP contents.
    if entry not in {row[0] for row in files}:
        raise ReleaseError("entryPoint casing must match its inventory path")
    return sorted(files)


def models(inventory: dict, destinations: set[str]) -> list[dict]:
    rows = inventory.get("models")
    if not isinstance(rows, list):
        raise ReleaseError("models must be an explicit list (possibly empty)")
    answer = []
    for row in rows:
        if not isinstance(row, dict) or set(row) != {"path", "url", "size", "sha256"}:
            raise ReleaseError("Each model requires path/url/size/sha256")
        name = clean_path(row["path"])
        if name.casefold() in destinations or any(
                p.startswith(name.casefold() + "/") or name.casefold().startswith(p + "/")
                for p in destinations):
            raise ReleaseError("Model/payload path collision")
        destinations.add(name.casefold())
        parsed = urlsplit(row["url"])
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
            raise ReleaseError("Model URLs require HTTPS without embedded credentials")
        if type(row["size"]) is not int or row["size"] <= 0 or not SHA256.fullmatch(row["sha256"]):
            raise ReleaseError("Model size/hash pins are required")
        answer.append(dict(row, path=name, sha256=row["sha256"].lower()))
    return answer


def build(inventory: dict, source_root: Path, output: Path, payload_url: str | None = None) -> dict:
    root = source_root.resolve(strict=True)
    files = inventory_files(inventory, root)
    model_rows = models(inventory, {name.casefold() for name, _, _ in files})
    if payload_url:
        parsed = urlsplit(payload_url)
        if parsed.scheme != "https" or not parsed.netloc or parsed.username or parsed.password or parsed.fragment:
            raise ReleaseError("Published payload URL requires HTTPS without embedded credentials")
    output = output.absolute()
    output.parent.mkdir(parents=True, exist_ok=True)
    # Output is new and immutable; never replace an already reviewed build.
    output.mkdir()
    try:
        with tempfile.TemporaryDirectory(prefix="clean-stage-", dir=output) as staging:
            stage = Path(staging)
            unpacked = 0
            for name, source, expected in files:
                target = stage / name
                target.parent.mkdir(parents=True, exist_ok=True)
                h, carry = hashlib.sha256(), b""
                with source.open("rb") as reader, target.open("xb") as writer:
                    for block in iter(lambda: reader.read(1024 * 1024), b""):
                        sample = carry + block
                        if any(pattern.search(sample) for pattern in SECRET_PATTERNS):
                            # Never print secret values or matching snippets.
                            raise ReleaseError("Credential-shaped content rejected: " + name)
                        carry = sample[-4096:]
                        h.update(block)
                        writer.write(block)
                        unpacked += len(block)
                if h.hexdigest() != expected:
                    raise ReleaseError("Reviewed source hash mismatch: " + name)
            archive = output / "living-home-arm64.zip"
            with zipfile.ZipFile(archive, "x", zipfile.ZIP_DEFLATED, allowZip64=True) as z:
                for name, _, _ in files:
                    z.write(stage / name, name)
            manifest = {
                "schemaVersion": 1, "releaseVersion": inventory["releaseVersion"], "architecture": "arm64",
                "payload": {"url": payload_url or archive.name, "size": archive.stat().st_size,
                            "sha256": digest(archive), "unpackedBytes": unpacked, "entryPoint": inventory["entryPoint"]},
                "models": model_rows,
                # Download + extracted payload + model bytes + 1 GiB working space.
                "minimumFreeBytes": archive.stat().st_size + unpacked + sum(r["size"] for r in model_rows) + 1024**3,
            }
            (output / "reviewed-inventory.json").write_text(json.dumps(inventory, indent=2) + "\n", encoding="utf-8")
            (output / "release.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
        return manifest
    except Exception:
        if (output / "release.json").exists():
            (output / "release.json").unlink()
        # The caller can inspect partial local output; it never contains a release.json
        # until the entire archive has passed review checks. No recursive cleanup.
        (output / "BUILD_FAILED.txt").write_text("Incomplete build. Do not publish this directory.\n", encoding="utf-8")
        raise


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inventory", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--payload-url", help="Optional exact future HTTPS archive URL; no upload is performed")
    args = parser.parse_args()
    try:
        inventory = json.loads(args.inventory.read_text("utf-8-sig"))
        result = build(inventory, args.source_root, args.output, args.payload_url)
    except (ReleaseError, OSError, ValueError, TypeError) as exc:
        print("Build refused: " + str(exc))
        return 2
    print(f"Built {result['releaseVersion']}: {args.output / 'release.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
