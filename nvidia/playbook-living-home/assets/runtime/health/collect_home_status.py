"""Read-only, remote Home Assistant status evidence for a Living Home install.

Only explicitly selected entities are retained. No entity registry, local HA
files, service calls, physical device tests, or cloud reporting are performed.
"""
from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import re
import tempfile
from typing import Any
import urllib.error
import urllib.parse
import urllib.request
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError


INSTALL_ROOT = Path(__file__).resolve().parents[2]
MAX_SELECTED_ENTITIES = 500
MAX_RESPONSE_BYTES = 8 * 1024 * 1024
CONTROL_DOMAINS = frozenset({
    "light", "switch", "lock", "climate", "cover", "fan", "media_player",
    "vacuum", "camera", "humidifier", "alarm_control_panel", "water_heater",
})
INTEGRATION_FAILURE_STATES = frozenset({
    "setup_retry", "setup_error", "migration_error", "failed_unload",
})
ENTITY_ID_RE = re.compile(r"^[a-z_][a-z0-9_]*\.[a-z0-9_]+$")


class ConfigurationError(ValueError):
    """A safe configuration error whose message never includes user values."""


@dataclass(frozen=True)
class HealthConfig:
    entity_ids: tuple[str, ...] = ()
    timezone: str | None = None
    battery_warning_threshold: float = 20
    check_integrations: bool = False
    timeout_seconds: float = 8

    @classmethod
    def from_mapping(cls, value: Any) -> "HealthConfig":
        if not isinstance(value, dict):
            raise ConfigurationError("Health configuration must be a JSON object")
        allowed = {
            "entity_ids", "timezone", "battery_warning_threshold",
            "check_integrations", "timeout_seconds",
        }
        if set(value) - allowed:
            raise ConfigurationError("Health configuration contains an unsupported field")
        selected = value.get("entity_ids", [])
        if not isinstance(selected, list) or len(selected) > MAX_SELECTED_ENTITIES:
            raise ConfigurationError("entity_ids must be a list with at most 500 entries")
        if any(not isinstance(eid, str) or not ENTITY_ID_RE.fullmatch(eid) for eid in selected):
            raise ConfigurationError("entity_ids must contain exact Home Assistant entity IDs")
        if len(set(selected)) != len(selected):
            raise ConfigurationError("entity_ids must not contain duplicates")
        tz = value.get("timezone")
        if tz is not None and (not isinstance(tz, str) or not tz.strip()):
            raise ConfigurationError("timezone must be an IANA timezone name or null")
        threshold = value.get("battery_warning_threshold", 20)
        timeout = value.get("timeout_seconds", 8)
        if not _number_in_range(threshold, 0, 100):
            raise ConfigurationError("battery_warning_threshold must be between 0 and 100")
        if not _number_in_range(timeout, 0.1, 60):
            raise ConfigurationError("timeout_seconds must be between 0.1 and 60")
        integrations = value.get("check_integrations", False)
        if not isinstance(integrations, bool):
            raise ConfigurationError("check_integrations must be a boolean")
        return cls(tuple(selected), tz, float(threshold), integrations, float(timeout))


def _number_in_range(value: Any, lower: float, upper: float) -> bool:
    return (isinstance(value, (int, float)) and not isinstance(value, bool)
            and math.isfinite(value) and lower <= value <= upper)


def load_config(path: Path) -> HealthConfig:
    try:
        content = path.read_text(encoding="utf-8-sig")
        return HealthConfig.from_mapping(json.loads(content))
    except (OSError, UnicodeError, json.JSONDecodeError):
        raise ConfigurationError("Health configuration file is missing or invalid JSON") from None


def load_credentials(env_file: Path | None = None) -> tuple[str, str]:
    """Read only HA_URL/HA_TOKEN. Process environment takes precedence.

    This is a literal dotenv reader, not a shell interpreter: no expansion or
    sourcing of the credential file occurs. No credential values are logged.
    """
    values: dict[str, str] = {}
    if env_file is not None:
        try:
            for line in env_file.read_text(encoding="utf-8-sig").splitlines():
                key, separator, raw = line.strip().partition("=")
                if separator and key in {"HA_URL", "HA_TOKEN"}:
                    raw = raw.strip()
                    if len(raw) >= 2 and raw[0] == raw[-1] and raw[0] in {"'", '"'}:
                        raw = raw[1:-1]
                    values[key] = raw
        except FileNotFoundError:
            pass
        except (OSError, UnicodeError):
            raise ConfigurationError("Home Assistant credential file cannot be read") from None
    for key in ("HA_URL", "HA_TOKEN"):
        if key in os.environ:
            values[key] = os.environ[key].strip()
    return values.get("HA_URL", ""), values.get("HA_TOKEN", "")


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        # Do not forward the Authorization header to a redirected address.
        return None


@dataclass(frozen=True)
class APIResult:
    value: Any = None
    status: str = "unavailable"
    error: str | None = None
    http_status: int | None = None


@dataclass
class HomeAssistantClient:
    url: str
    token: str = field(repr=False)
    timeout_seconds: float = 8

    def __post_init__(self):
        if not self.url:
            raise ConfigurationError("HA_URL is required; there is no default Home Assistant address")
        try:
            parsed = urllib.parse.urlsplit(self.url)
            valid = (parsed.scheme in {"http", "https"} and parsed.hostname
                     and parsed.username is None and parsed.password is None
                     and not parsed.query and not parsed.fragment)
            parsed.port  # Validate an explicitly supplied port without displaying it.
        except ValueError:
            valid = False
        if not valid or any(char.isspace() for char in self.url):
            raise ConfigurationError("HA_URL must be an HTTP(S) base URL without credentials or query parameters")
        if "\r" in self.token or "\n" in self.token:
            raise ConfigurationError("HA_TOKEN must be a single-line bearer token")
        self.url = self.url.rstrip("/")

    def get(self, path: str) -> APIResult:
        if not self.token:
            return APIResult(error="Home Assistant credential unavailable")
        request = urllib.request.Request(
            self.url + path, method="GET",
            headers={"Authorization": "Bearer " + self.token, "Accept": "application/json"},
        )
        try:
            opener = urllib.request.build_opener(_NoRedirect())
            with opener.open(request, timeout=self.timeout_seconds) as response:
                raw = response.read(MAX_RESPONSE_BYTES + 1)
                if len(raw) > MAX_RESPONSE_BYTES:
                    return APIResult(error="API response exceeds the collection limit")
                result = json.loads(raw)
            return APIResult(result, "collected")
        except urllib.error.HTTPError as exc:
            status = "unsupported" if exc.code in {404, 405, 501} else "unavailable"
            message = ("Home Assistant authentication rejected" if exc.code in {401, 403}
                       else "API redirect refused" if 300 <= exc.code < 400
                       else "HTTP " + str(exc.code))
            code = exc.code
            exc.close()
            return APIResult(status=status, error=message, http_status=code)
        except (ValueError, UnicodeError):
            return APIResult(error="API response is not valid JSON")
        except Exception:
            # urllib exceptions can contain URLs, tokens or server-supplied text.
            return APIResult(error="Home Assistant request failed or timed out")


def _validate_result(result: APIResult, expected: type) -> APIResult:
    if result.status == "collected" and not isinstance(result.value, expected):
        return APIResult(error="API response has an unexpected JSON shape")
    if (result.status == "collected" and expected is list
            and any(not isinstance(item, dict) for item in result.value)):
        return APIResult(error="API response has an unexpected JSON shape")
    return result


def _string(value: Any) -> str | None:
    return value if isinstance(value, str) else None


def _entity_item(entity_id: str, observed: dict | None, states_collected: bool) -> dict:
    attributes = observed.get("attributes") if observed else {}
    attributes = attributes if isinstance(attributes, dict) else {}
    state = _string(observed.get("state")) if observed else None
    if observed is None:
        reason = "missing_entity" if states_collected else "states_not_collected"
        availability = "unknown"
    elif state in {"unknown", "unavailable"}:
        reason, availability = "reported_" + state, state
    elif state is None:
        reason, availability = "invalid_reported_state", "unknown"
    else:
        reason, availability = "reported_state", "available"
    return {
        "entity_id": entity_id,
        "name": _string(attributes.get("friendly_name")) or entity_id,
        "state": state if state is not None else "unknown",
        "availability": availability,
        "availability_reason": reason,
        "present_in_states": observed is not None if states_collected else None,
        "last_changed": _string(observed.get("last_changed")) if observed else None,
        "last_updated": _string(observed.get("last_updated")) if observed else None,
        "device_class": _string(attributes.get("device_class")),
        "unit_of_measurement": _string(attributes.get("unit_of_measurement")),
        "source": "home_assistant_entity",
        "platform": None,
        "physical_operation": "not_tested",
    }


def _battery_warning(item: dict, threshold: float) -> dict | None:
    if item["device_class"] != "battery" or item["availability"] != "available":
        return None
    common = {"entity_id": item["entity_id"], "name": item["name"]}
    if item["entity_id"].startswith("binary_sensor.") and item["state"] == "on":
        return {**common, "percent": None, "reason": "reported_low_battery"}
    if item["unit_of_measurement"] != "%":
        return None
    try:
        level = float(item["state"])
    except (TypeError, ValueError):
        return None
    if math.isfinite(level) and 0 <= level <= threshold:
        return {**common, "percent": level, "threshold_percent": threshold,
                "reason": "reported_percentage_at_or_below_threshold"}
    return None


def _redact(value: Any, secret: str) -> Any:
    if isinstance(value, str):
        return value.replace(secret, "[redacted]") if secret else value
    if isinstance(value, dict):
        return {key: _redact(item, secret) for key, item in value.items()}
    if isinstance(value, list):
        return [_redact(item, secret) for item in value]
    return value


def collect(config: HealthConfig | dict | None = None, *, env_file: Path | None = None,
            url: str | None = None, token: str | None = None) -> dict:
    """Collect fresh status. Explicit url/token are convenient for embedding/tests."""
    if isinstance(config, HealthConfig):
        config = HealthConfig.from_mapping({
            "entity_ids": list(config.entity_ids), "timezone": config.timezone,
            "battery_warning_threshold": config.battery_warning_threshold,
            "check_integrations": config.check_integrations,
            "timeout_seconds": config.timeout_seconds,
        })
    else:
        config = HealthConfig.from_mapping({} if config is None else config)
    credentials = load_credentials(env_file)
    client = HomeAssistantClient(
        credentials[0] if url is None else url,
        credentials[1] if token is None else token,
        config.timeout_seconds,
    )
    started = datetime.now(timezone.utc)
    paths = {"config": "/api/config", "states": "/api/states"}
    if config.check_integrations:
        paths["integrations"] = "/api/config/config_entries/entry"
    with ThreadPoolExecutor(max_workers=len(paths)) as pool:
        futures = {name: pool.submit(client.get, path) for name, path in paths.items()}
        results = {name: _validate_result(future.result(), dict if name == "config" else list)
                   for name, future in futures.items()}
    errors = {name: result.error for name, result in results.items() if result.error}
    ha_config = results["config"].value if results["config"].status == "collected" else {}
    states_collected = results["states"].status == "collected"
    states = results["states"].value if states_collected else []
    by_id = {state.get("entity_id"): state for state in states
             if isinstance(state.get("entity_id"), str)}
    inventory = [_entity_item(eid, by_id.get(eid), states_collected) for eid in config.entity_ids]
    unavailable = [item for item in inventory if item["availability"] != "available"]
    controls = [item for item in inventory if item["entity_id"].split(".", 1)[0] in CONTROL_DOMAINS]
    unavailable_controls = [item for item in controls if item["availability"] != "available"]
    batteries = [warning for item in inventory
                 if (warning := _battery_warning(item, config.battery_warning_threshold))]
    integrations = results.get("integrations", APIResult(status="not_requested"))
    integration_entries = integrations.value if integrations.status == "collected" else []
    issues = [{"domain": _string(entry.get("domain")), "state": _string(entry.get("state"))}
              for entry in integration_entries if _string(entry.get("state")) in INTEGRATION_FAILURE_STATES]
    integration_counts = dict(Counter(_string(entry.get("state")) or "unknown"
                                     for entry in integration_entries))
    configured_timezone = _string(ha_config.get("time_zone"))
    tz_name = config.timezone or configured_timezone or "UTC"
    tz_source = "health_config" if config.timezone else "home_assistant" if configured_timezone else "utc_fallback"
    try:
        zone = timezone.utc if tz_name in {"UTC", "Etc/UTC"} else ZoneInfo(tz_name)
    except (ZoneInfoNotFoundError, ValueError):
        zone = timezone.utc
        tz_name, tz_source = "UTC", "utc_fallback"
        errors["timezone"] = "Requested timezone is unavailable; timestamps use UTC"
    finished = datetime.now(timezone.utc)
    limitations = [
        "Availability is Home Assistant's reported entity state; physical operation was not tested.",
        "Off, idle, closed and locked are normal states, not availability failures.",
        "Coverage includes only entity_ids explicitly selected in health-config.json.",
        "Entity totals are not a count of physical products; entity/device registries were not collected.",
        "Battery warnings cover selected battery-class percentage sensors and binary battery sensors only.",
        "A fresh snapshot does not prove that the underlying device last reported recently; entity timestamps are included when reported.",
        "Integration observations do not establish cause, persistence or downstream service impact.",
    ]
    if integrations.status != "collected":
        limitations.append("Integration status coverage is incomplete; an empty issue list does not establish integration availability.")
    if not states_collected:
        limitations.append("Entity states were not collected; selected entity availability and battery status are unknown.")
    if not config.entity_ids:
        limitations.append("No entities are selected; configure entity_ids before relying on entity or battery coverage.")
    # An unsupported optional endpoint reduces coverage, but is not evidence of
    # an unhealthy integration. Its collection error remains visible separately.
    required_errors = any(name in errors for name in ("config", "states", "timezone"))
    integration_failure = config.check_integrations and integrations.status not in {"collected", "unsupported"}
    status = ("attention" if required_errors or integration_failure or unavailable or batteries or issues
              else "unknown" if not config.entity_ids else "available")
    snapshot = {
        "schema_version": 1,
        "checked_at": finished.astimezone(zone).isoformat(timespec="seconds"),
        "started_at": started.isoformat(timespec="seconds"),
        "completed_at": finished.isoformat(timespec="seconds"),
        "timezone": tz_name,
        "timezone_source": tz_source,
        "status": status,
        "home_assistant": {
            "api_reachable": any(result.status == "collected" or result.http_status is not None
                                 for result in results.values()),
            "api_authenticated": any(result.status == "collected" for result in results.values()),
            "config_collected": results["config"].status == "collected",
            "states_collected": states_collected,
            "version": _string(ha_config.get("version")),
            "state": _string(ha_config.get("state")),
            "configured_timezone": configured_timezone,
        },
        "summary": {
            "selected_entities": len(config.entity_ids),
            "tracked_entities": len(inventory),
            "observed_selected_entities": sum(item["present_in_states"] is True for item in inventory) if states_collected else None,
            "available_selected_entities": sum(item["availability"] == "available" for item in inventory) if states_collected else None,
            "unavailable_or_unknown_entities": len(unavailable) if states_collected else None,
            "missing_selected_entities": sum(item["availability_reason"] == "missing_entity" for item in inventory) if states_collected else None,
            "tracked_control_entities": len(controls) if states_collected else None,
            "source_counts": {"home_assistant_entity": len(controls)} if states_collected else None,
            "unavailable_or_unknown_controls": len(unavailable_controls) if states_collected else None,
            "reported_low_batteries": len(batteries) if states_collected else None,
            "failing_integrations": len(issues) if integrations.status == "collected" else None,
        },
        "inventory": inventory,
        "unavailable_entities": unavailable,
        "unavailable_controls": unavailable_controls,
        "battery_attention": batteries,
        "integration_issues": issues,
        "integration_state_counts": integration_counts if integrations.status == "collected" else None,
        "integration_evidence": {
            "source": paths.get("integrations"),
            "collection_status": integrations.status,
            "observation_scope": "home_assistant_configuration_entries",
            "persistence": "not_measured", "cause": "not_established",
            "downstream_service_impact": "not_established", "external_runtime_health": "not_checked",
        },
        "coverage": {
            "entity_selection": "explicit_entity_ids", "selection_configured": bool(config.entity_ids),
            "entity_states": results["states"].status,
            "entity_registry": "not_collected", "device_registry": "not_collected",
            "physical_devices": "not_tested", "battery_scope": "selected_reporting_entities",
            "integrations": integrations.status,
        },
        "check_errors": errors,
        "evidence": [{"endpoint": path, "collection_status": results[name].status,
                      "http_status": results[name].http_status} for name, path in paths.items()],
        "limitations": limitations,
    }
    return _redact(snapshot, client.token)


def render_summary(snapshot: dict) -> str:
    lines = ["Living Home — Home Assistant entity status", "Checked: " + snapshot["checked_at"],
             "Status: " + snapshot["status"], "Coverage: selected entities only; physical operation not tested."]
    for item in snapshot["inventory"]:
        lines.append(f"{item['name']} ({item['entity_id']}): {item['state']} [{item['availability']}; {item['availability_reason']}]")
    for item in snapshot["battery_attention"]:
        value = "low battery reported" if item["percent"] is None else str(item["percent"]) + "% battery reported"
        lines.append(f"Battery attention: {item['name']} ({item['entity_id']}): {value}")
    lines.append("Integration coverage: " + snapshot["coverage"]["integrations"])
    if snapshot["check_errors"]:
        lines.append("Collection issues: " + json.dumps(snapshot["check_errors"], ensure_ascii=False))
    if not snapshot["coverage"]["selection_configured"]:
        lines.append("Select your entity_ids in health-config.json before relying on entity coverage.")
    return "\n".join(lines) + "\n"


def _atomic_write(path: Path, content: str) -> None:
    descriptor, temporary = tempfile.mkstemp(prefix="." + path.name + ".", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as handle:
            handle.write(content)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass


def save_snapshot(snapshot: dict, directory: Path) -> Path:
    directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
    content = json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False) + "\n"
    path = directory / (stamp + ".json")
    _atomic_write(path, content)
    _atomic_write(directory / "latest.json", content)
    _atomic_write(directory / "latest-summary.txt", render_summary(snapshot))
    return path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=INSTALL_ROOT / "health-config.json")
    parser.add_argument("--env-file", type=Path, default=INSTALL_ROOT / ".env")
    parser.add_argument("--output-dir", type=Path, default=INSTALL_ROOT / "health" / "snapshots")
    args = parser.parse_args(argv)
    try:
        snapshot = collect(load_config(args.config), env_file=args.env_file)
    except ConfigurationError as exc:
        print(json.dumps({"checked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                          "status": "configuration_error", "check_errors": {"configuration": str(exc)}}))
        return 2
    try:
        save_snapshot(snapshot, args.output_dir)
    except OSError:
        snapshot["status"] = "attention"
        snapshot["check_errors"]["evidence_files"] = "Snapshot evidence files could not be saved"
        print(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False))
        return 3
    print(json.dumps(snapshot, ensure_ascii=False, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
