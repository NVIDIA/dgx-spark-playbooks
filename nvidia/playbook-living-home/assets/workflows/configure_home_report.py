"""Preview or configure an own-profile OpenClaw status-report cron job.

Preview is offline. --apply calls the chosen local OpenClaw Gateway, creates a
disabled job, or updates and disables changed configuration. It never runs a job.
CLI contract verified against installed OpenClaw 2026.9.4 help/source.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
from pathlib import Path
import re
import subprocess
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError, available_timezones

DECLARATION = "living-home-device-status-v1"
NAME = "Living Home Device Status Report"
DESCRIPTION = "Read selected Home Assistant device evidence and save an agent-authored local health report."


def report_message(delivery_mode: str) -> str:
    text = (
        "Prepare a fresh home device status report for this household. "
        "Call living_home_property with operation=health_snapshot. "
        "Use only returned evidence from the user's configured Home Assistant entity scope. "
        "Preserve collected_at, observed/source timestamps, coverage, failed reads, missing entities, "
        "unknown states and availability. A device being off is not proof of a fault. "
        "Do not infer physical hardware failure or physical actuation from an entity-state response. "
        "Do not invent measurements, incidents, diagnoses, sample records or remediation. "
        "If collection fails or returns ok=false, report that failure clearly; do not save a successful "
        "health report based on prior evidence. Otherwise write concise, original evidence-grounded prose "
        "and call living_home_property operation=report with title, body, report_type=health, "
        "publish_to_drive=false and format=markdown in the same call; omit incident_id. "
        "The report belongs in this household's local property store. "
        "Read it back with living_home_property operation=report_read, report_id=latest, "
        "report_type=health. Check the returned report ID matches the just-saved report; "
        "if another report became latest, retrieve the exact observed saved ID instead. "
        "Only claim saved after a confirmed write and matching readback. "
        "Return a short result with the observed report ID/local link and collection time. "
        "Do not control devices, alter automations, send email, publish to Google, or call share. "
    )
    if delivery_mode == "none":
        return text + "Keep the result local; do not post to Discord or any other external channel."
    return text + "The cron runner alone handles the operator-configured delivery of the final result."


def config(raw: dict) -> dict:
    expected = {"schemaVersion", "openclawProfile", "gatewayUrl", "agentId", "cron", "timeZone", "timeoutSeconds", "delivery"}
    if not isinstance(raw, dict) or set(raw) != expected or raw["schemaVersion"] != 1:
        raise ValueError("Configuration requires the documented schemaVersion 1 fields")
    for key in ("openclawProfile", "agentId"):
        if not isinstance(raw[key], str) or not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", raw[key]):
            raise ValueError("Invalid explicit own " + key)
    if not isinstance(raw["gatewayUrl"], str):
        raise ValueError("Provide the own local Gateway WebSocket origin")
    url = urlsplit(raw["gatewayUrl"])
    if (url.scheme not in {"ws", "wss"} or url.hostname not in {"127.0.0.1", "::1"}
            or not url.port or url.username or url.password or url.query or url.fragment or url.path not in {"", "/"}):
        raise ValueError("Gateway URL must be an explicit loopback WebSocket origin and port")
    expr = raw["cron"]
    if not isinstance(expr, str) or len(expr.split()) not in {5, 6} or len(expr) > 160 or not re.fullmatch(r"[0-9A-Za-z*/?,#L W-]+", expr):
        raise ValueError("Provide a 5- or 6-field cron expression")
    if not isinstance(raw["timeZone"], str) or not re.fullmatch(r"(?:UTC|[A-Za-z_+-]+(?:/[A-Za-z0-9_+-]+){1,3})", raw["timeZone"]):
        raise ValueError("Provide an IANA timeZone, for example America/Los_Angeles")
    try:
        ZoneInfo(raw["timeZone"])
    except ZoneInfoNotFoundError as exc:
        # Standard Windows Python has no tzdata. The selected Node runtime's
        # Intl validates the zone before --apply makes any Gateway call.
        if available_timezones():
            raise ValueError("Unknown IANA timeZone") from exc
    if type(raw["timeoutSeconds"]) is not int or not 30 <= raw["timeoutSeconds"] <= 3600:
        raise ValueError("timeoutSeconds must be between 30 and 3600")
    delivery = raw["delivery"]
    if delivery == {"mode": "none"}:
        return copy.deepcopy(raw)
    if not isinstance(delivery, dict) or set(delivery) - {"mode", "channel", "to", "accountId"} or not {"mode", "channel", "to"} <= set(delivery) or delivery["mode"] != "announce":
        raise ValueError("delivery is {mode:none} or explicit announce with channel/to and optional accountId")
    for key in ("channel", "to", "accountId"):
        if key not in delivery:
            continue
        if not isinstance(delivery[key], str) or not delivery[key].strip() or len(delivery[key]) > 200 or any(ord(c) < 32 for c in delivery[key]):
            raise ValueError("Invalid delivery " + key)
    if delivery["channel"] == "last":
        raise ValueError("Delivery must name an explicit user-configured channel, never last")
    return copy.deepcopy(raw)


def desired(cfg: dict) -> dict:
    return {
        "name": NAME, "description": DESCRIPTION, "declarationKey": DECLARATION,
        "agentId": cfg["agentId"], "sessionTarget": "isolated", "wakeMode": "now",
        "schedule": {"kind": "cron", "expr": cfg["cron"], "tz": cfg["timeZone"], "staggerMs": 0},
        "payload": {"kind": "agentTurn", "message": report_message(cfg["delivery"]["mode"]),
                    "timeoutSeconds": cfg["timeoutSeconds"]},
        "delivery": cfg["delivery"],
    }


def matches(job: dict, spec: dict) -> bool:
    if any(job.get(key) != spec[key] for key in ("name", "description", "declarationKey", "agentId", "sessionTarget", "wakeMode")):
        return False
    schedule, payload, delivery = job.get("schedule", {}), job.get("payload", {}), job.get("delivery", {})
    if any(schedule.get(key) != value for key, value in spec["schedule"].items()):
        return False
    if any(payload.get(key) != value for key, value in spec["payload"].items()):
        return False
    if (payload.get("model") or payload.get("thinking") or payload.get("fallbacks") or payload.get("tools")
            or payload.get("lightContext") or job.get("sessionKey") or job.get("trigger") or job.get("pacing")
            or job.get("failureAlert", {}).get("enabled")):
        # An inherited arbitrary override may alter report execution.
        return False
    if delivery.get("mode") != spec["delivery"]["mode"]:
        return False
    if spec["delivery"]["mode"] == "none":
        return not any(delivery.get(k) for k in ("channel", "to", "accountId", "threadId"))
    return all(delivery.get(key) == spec["delivery"].get(key) for key in ("channel", "to", "accountId")) and not delivery.get("threadId")


def mutation_args(spec: dict, existing: dict | None) -> list[str]:
    action = ["cron", "edit", existing["id"]] if existing else ["cron", "add", "--declaration-key", DECLARATION]
    action += ["--name", NAME, "--description", DESCRIPTION, "--agent", spec["agentId"],
               "--session", "isolated", "--wake", "now", "--cron", spec["schedule"]["expr"],
               "--tz", spec["schedule"]["tz"], "--exact", "--message", spec["payload"]["message"],
               "--timeout-seconds", str(spec["payload"]["timeoutSeconds"]), "--json"]
    if existing:
        action += ["--disable", "--clear-channel", "--clear-to", "--clear-account", "--clear-thread-id",
                   "--clear-session-key", "--clear-model", "--clear-thinking", "--clear-fallbacks", "--clear-trigger", "--clear-pacing",
                   "--clear-tools", "--no-light-context", "--no-best-effort-deliver", "--no-failure-alert"]
    else:
        action += ["--disabled"]
    delivery = spec["delivery"]
    if delivery["mode"] == "none":
        action += ["--no-deliver"]
        if not existing:
            # OpenClaw add defaults channel to 'last'; an explicit empty value
            # normalizes to absent and prevents a relative external target.
            action += ["--channel", ""]
    else:
        # A changed target replaces old targets; clear flags must not mask new values.
        action = [x for x in action if x not in {"--clear-channel", "--clear-to"}]
        action += ["--announce", "--channel", delivery["channel"], "--to", delivery["to"]]
        if delivery.get("accountId"):
            action = [x for x in action if x != "--clear-account"]
            action += ["--account", delivery["accountId"]]
    return action


def ensure(cfg: dict, invoke) -> dict:
    spec = desired(cfg)
    listing = invoke(["cron", "list", "--all", "--json"])
    if not isinstance(listing, dict) or not isinstance(listing.get("jobs"), list):
        raise ValueError("OpenClaw cron list did not return its documented JSON jobs array")
    # Read-only CLI list fetches all pages in the installed version.
    candidates = [j for j in listing["jobs"] if j.get("declarationKey") == DECLARATION]
    collisions = [j for j in listing["jobs"] if j.get("name") == NAME and j.get("declarationKey") != DECLARATION]
    if collisions or len(candidates) > 1:
        raise ValueError("Ambiguous/foreign job identity; resolve explicitly before setup")
    existing = candidates[0] if candidates else None
    if existing and not isinstance(existing.get("id"), str):
        raise ValueError("Existing cron job lacks an observed ID")
    if existing and existing.get("payload", {}).get("kind") != "agentTurn":
        raise ValueError("Existing declaration is not an agentTurn; refusing to repurpose it")
    changed = not existing or not matches(existing, spec)
    if changed:
        response = invoke(mutation_args(spec, existing))
        row = response.get("job", response) if isinstance(response, dict) else None
        job_id = existing["id"] if existing else row.get("id") if isinstance(row, dict) else None
        if not isinstance(job_id, str) or not job_id:
            raise ValueError("Creation did not return an observed job ID; inspect own cron list before retry")
    else:
        job_id = existing["id"]
    readback = invoke(["cron", "show", job_id, "--json"])
    job = readback.get("job", readback) if isinstance(readback, dict) else None
    if not isinstance(job, dict) or job.get("id") != job_id or not matches(job, spec):
        raise ValueError("Saved cron job failed exact schedule/timezone/payload/delivery verification")
    if changed and job.get("enabled") is not False:
        raise ValueError("Changed cron job was not saved disabled; disable and inspect before continuing")
    return {"jobId": job_id, "changed": changed, "enabled": job.get("enabled"),
            "schedule": job["schedule"], "deliveryMode": spec["delivery"]["mode"], "verified": True,
            "nextStep": "Run once and verify the saved local report before explicitly enabling the schedule."}


def subprocess_runner(node: Path, entry: Path, profile: str, time_zone: str, gateway_url: str):
    if not node.is_file() or not entry.is_file() or node.suffix.lower() != ".exe" or entry.suffix.lower() != ".mjs":
        raise ValueError("Provide the own installation's node.exe and OpenClaw openclaw.mjs")
    prefix = [str(node.resolve()), str(entry.resolve()), "--profile", profile]
    env = dict(os.environ)
    # Explicit profile must not silently inherit a private installation's state.
    if env.get("OPENCLAW_STATE_DIR") or env.get("OPENCLAW_CONFIG_PATH"):
        raise ValueError("Unset OPENCLAW_STATE_DIR/OPENCLAW_CONFIG_PATH before using an explicit own profile")
    validation = subprocess.run([str(node.resolve()), "-e",
        "new Intl.DateTimeFormat('en', {timeZone:process.argv[1]}).format()", time_zone],
        capture_output=True, timeout=15, shell=False, env=env,
        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if validation.returncode:
        raise ValueError("Selected Node runtime rejected timeZone; no Gateway call was made")
    def invoke(args):
        proc = subprocess.run(prefix + args + ["--url", gateway_url], capture_output=True, text=True, encoding="utf-8",
                              timeout=60, shell=False, env=env,
                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        if proc.returncode:
            # CLI errors may contain configuration/auth values. Do not echo them.
            raise ValueError("OpenClaw CLI failed; inspect that profile locally (output withheld)")
        try:
            return json.loads(proc.stdout)
        except ValueError as exc:
            raise ValueError("OpenClaw did not return pure JSON; verify compatible CLI/version") from exc
    return invoke


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--node", type=Path, help="Own installation's node.exe; required with --apply")
    parser.add_argument("--openclaw", type=Path, help="Own installation's openclaw.mjs; required with --apply")
    parser.add_argument("--apply", action="store_true", help="Create/update on the selected own local profile; does not run or enable")
    args = parser.parse_args()
    try:
        cfg = config(json.loads(args.config.read_text("utf-8-sig")))
        if args.apply:
            if not args.node or not args.openclaw:
                raise ValueError("--apply requires explicit --node and --openclaw paths")
            output = ensure(cfg, subprocess_runner(args.node, args.openclaw, cfg["openclawProfile"], cfg["timeZone"], cfg["gatewayUrl"]))
        else:
            output = {"previewOnly": True, "openclawProfile": cfg["openclawProfile"], "gatewayUrl": cfg["gatewayUrl"],
                      "desiredJob": dict(desired(cfg), enabled=False), "cronArguments": mutation_args(desired(cfg), None),
                      "timeZoneVerification": "Node Intl verifies IANA timezone before apply; Gateway readback verifies saved timezone.",
                      "requires": "Own configured Gateway, local model, Living Home plugin and working fresh household backend."}
        print(json.dumps(output, indent=2))
        return 0
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        print("Cron setup refused: " + str(exc))
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
