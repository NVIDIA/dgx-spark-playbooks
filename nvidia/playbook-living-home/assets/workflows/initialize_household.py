"""Preview/create fresh local household configuration; starts no service.

Use own HA setup/account/token. No credential/device/account import occurs.
--write saves only into a new household data directory; --prompt-token asks via
getpass and stores it in that directory's protected .env without printing it.
"""
from __future__ import annotations

import argparse
import getpass
import json
import os
from pathlib import Path
import re
import secrets
import subprocess
import sys
from urllib.parse import urlsplit
import uuid

from household_state import HouseholdRunStore


def validate(raw: dict) -> dict:
    fields = {"schemaVersion", "haMode", "haUrl", "entityIds", "timeZone", "openclawProfile", "google"}
    if not isinstance(raw, dict) or set(raw) != fields or raw["schemaVersion"] != 1:
        raise ValueError("Provide the documented household config schemaVersion 1")
    if raw["haMode"] not in {"existing", "new"}:
        raise ValueError("haMode must be existing or new")
    if not isinstance(raw["haUrl"], str):
        raise ValueError("Provide the own HA origin as a string")
    parsed = urlsplit(raw["haUrl"])
    if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment or parsed.path not in {"", "/"}:
        raise ValueError("haUrl must be the own HA HTTP(S) origin without embedded credentials")
    if raw["haMode"] == "new" and parsed.hostname not in {"127.0.0.1", "localhost", "::1"}:
        raise ValueError("New local HA uses its local origin; finish HA account/device setup yourself first")
    ids = raw["entityIds"]
    if not isinstance(ids, list) or not ids or len(ids) > 500 or any(
            not isinstance(eid, str) or not re.fullmatch(r"[a-z_][a-z0-9_]*\.[a-z0-9_]+", eid) for eid in ids) or len(set(ids)) != len(ids):
        raise ValueError("entityIds must explicitly list 1–500 unique observed own HA entities")
    if not isinstance(raw["timeZone"], str) or not re.fullmatch(r"(?:UTC|[A-Za-z_+-]+(?:/[A-Za-z0-9_+-]+){1,3})", raw["timeZone"]):
        raise ValueError("Provide an IANA timeZone")
    if not isinstance(raw["openclawProfile"], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", raw["openclawProfile"]):
        raise ValueError("Provide an own OpenClaw profile name")
    google = raw["google"]
    if google != {"enabled": False}:
        if not isinstance(google, dict) or set(google) != {"enabled", "expectedAccount", "gmailLabel", "driveFolderId"} or google["enabled"] is not True:
            raise ValueError("Optional Google config requires explicit expectedAccount/gmailLabel/driveFolderId")
        if not isinstance(google["expectedAccount"], str) or not re.fullmatch(r"[^\s@]+@[^\s@]+\.[^\s@]+", google["expectedAccount"]):
            raise ValueError("Provide the own Google account email")
        if not isinstance(google["gmailLabel"], str) or not google["gmailLabel"].strip() or len(google["gmailLabel"]) > 225 or any(ord(c) < 32 for c in google["gmailLabel"]):
            raise ValueError("Provide the own Gmail label")
        if not isinstance(google["driveFolderId"], str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,200}", google["driveFolderId"]):
            raise ValueError("Provide the own Drive folder ID")
    return raw


def protect_private_directory(root: Path) -> None:
    if os.name != "nt":
        root.chmod(0o700)
        return
    # Determine current user SID without reading/printing any credentials.
    result = subprocess.run(["whoami.exe", "/user", "/fo", "csv", "/nh"],
                            capture_output=True, text=True, timeout=15, shell=False,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    match = re.search(r"S-1-[0-9-]+", result.stdout)
    if result.returncode or not match:
        raise ValueError("Could not determine the current account SID; credentials were not saved")
    access = subprocess.run(["icacls.exe", str(root), "/inheritance:r", "/grant:r", "*" + match.group() + ":(OI)(CI)F"],
                            capture_output=True, timeout=15, shell=False,
                            creationflags=subprocess.CREATE_NO_WINDOW)
    if access.returncode:
        raise ValueError("Could not protect the private household directory; credentials were not saved")


def initialize(raw: dict, root: Path, token: str | None = None) -> dict:
    cfg = validate(raw)
    if token is not None and (not isinstance(token, str) or not token.strip() or any(c.isspace() for c in token.strip()) or len(token) > 16000):
        raise ValueError('Enter a valid Home Assistant access token')
    for candidate in (root.absolute(), *root.absolute().parents):
        if candidate.is_symlink() or (hasattr(candidate, 'is_junction') and candidate.is_junction()):
            raise ValueError('Choose a data directory without links or junctions')
    root = root.resolve(strict=False)
    original = Path(r"C:\LivingHome").resolve(strict=False)
    if os.name == "nt" and (root == original or root.is_relative_to(original)):
        raise ValueError("Choose a fresh data directory outside the current C:\\LivingHome installation")
    # New directory only. Existing state can be read but never rewritten/reset.
    root.mkdir(parents=True, exist_ok=False)
    protect_private_directory(root)
    property_id = "home-" + uuid.uuid4().hex
    state = HouseholdRunStore(root / "household.json").initialize(property_id)
    settings = dict(cfg, propertyId=property_id, stateFile="household.json", propertyRoot="property",
                    haAutomationWriteMode="not_configured", runtimeWired=False)
    (root / "home-settings.json").write_text(json.dumps(settings, indent=2) + "\n", encoding="utf-8")
    (root / "health-config.json").write_text(json.dumps({"entity_ids": cfg["entityIds"], "timezone": cfg["timeZone"]}, indent=2) + "\n", encoding="utf-8")
    (root / "property").mkdir()
    (root / "api-token.txt").write_text(secrets.token_urlsafe(32) + "\n", encoding="utf-8")
    if cfg["google"]["enabled"]:
        g = cfg["google"]
        (root / "google-scope.json").write_text(json.dumps({"expected_account": g["expectedAccount"],
            "gmail_label": g["gmailLabel"], "drive_folder_id": g["driveFolderId"]}, indent=2) + "\n", encoding="utf-8")
    if token is not None:
        # Plain local .env is intentionally private to the current OS account.
        # Values never enter argv, logs, release metadata or console output.
        (root / ".env").write_text("HA_URL=" + cfg["haUrl"].rstrip("/") + "\nHA_TOKEN=" + token.strip() + "\n", encoding="utf-8")
    return {"created": True, "dataDirectory": str(root), "propertyId": property_id,
            "deviceProfile": state["device_profile"], "credentialSaved": token is not None,
            "servicesStarted": False, "runtimeWired": False,
            "nextStep": "Wire the clean backend and health plugin to this data directory; do not use demo reset."}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--stdin", action="store_true", help="Trusted first-run UI JSON bridge; token stays off argv")
    parser.add_argument("--write", action="store_true")
    parser.add_argument("--prompt-token", action="store_true", help="Read the own HA token privately; requires --write")
    args = parser.parse_args()
    try:
        token = None
        if args.stdin:
            if args.config or args.data_dir or args.prompt_token or not args.write:
                raise ValueError("--stdin requires --write and forbids config/data-dir/prompt-token overrides")
            incoming = sys.stdin.read(256 * 1024 + 1)
            if len(incoming) > 256 * 1024:
                raise ValueError("First-run JSON exceeds its limit")
            envelope = json.loads(incoming)
            if not isinstance(envelope, dict) or set(envelope) != {"schemaVersion", "configuration", "dataDirectory", "haToken"} or envelope["schemaVersion"] != 1:
                raise ValueError("Invalid first-run JSON envelope")
            if not isinstance(envelope["dataDirectory"], str) or not Path(envelope["dataDirectory"]).is_absolute():
                raise ValueError("First-run data directory must be absolute")
            if not isinstance(envelope["haToken"], str):
                raise ValueError("First-run HA token must be a string")
            args.data_dir = Path(envelope["dataDirectory"])
            token = envelope["haToken"]
            cfg = validate(envelope["configuration"])
        else:
            if not args.config or not args.data_dir:
                raise ValueError("Provide --config and --data-dir, or the trusted --stdin bridge")
            cfg = validate(json.loads(args.config.read_text("utf-8-sig")))
        if args.prompt_token and not args.write:
            raise ValueError("--prompt-token requires --write")
        if args.write:
            if args.prompt_token:
                token = getpass.getpass("Own Home Assistant long-lived access token (hidden): ")
            result = initialize(cfg, args.data_dir, token)
        else:
            result = {"previewOnly": True, "dataDirectory": str(args.data_dir.absolute()), "configuration": cfg,
                      "deviceProfile": "physical", "seededRecords": 0, "servicesStarted": False, "runtimeWired": False}
        print(json.dumps(dict(result, ok=True) if args.stdin else result, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        if args.stdin:
            print(json.dumps({"ok": False, "error": str(exc)}))
        else:
            print("Household initialization refused: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
