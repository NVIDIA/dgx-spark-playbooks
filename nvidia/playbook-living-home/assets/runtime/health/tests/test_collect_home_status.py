"""Remote HA contract tests: every network call targets a temporary fake server."""
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import importlib.util
import io
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import unittest
from unittest.mock import patch


SCRIPT = Path(__file__).resolve().parents[1] / "collect_home_status.py"
spec = importlib.util.spec_from_file_location("living_home_health", SCRIPT)
health = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = health
spec.loader.exec_module(health)
TEST_TOKEN = "fake-test-bearer-token-123456789"


def entity(entity_id, state, **attributes):
    return {"entity_id": entity_id, "state": state, "attributes": attributes,
            "last_changed": "2026-01-01T01:00:00+00:00", "last_updated": "2026-01-01T02:00:00+00:00"}


@contextmanager
def fake_ha(states=None, *, config=None, extra=None):
    responses = {
        "/api/config": (200, config if config is not None else {"version": "test-version", "state": "RUNNING", "time_zone": "UTC"}),
        "/api/states": (200, states if states is not None else []),
    }
    responses.update(extra or {})
    calls = []
    call_lock = threading.Lock()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def do_GET(self):
            with call_lock:
                calls.append((self.command, self.path, self.headers.get("Authorization")))
            status, value = responses.get(self.path, (404, {"message": "unsupported"}))
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if 300 <= status < 400:
                self.send_header("Location", "/should-not-receive-token")
            self.end_headers()
            self.wfile.write(value if isinstance(value, bytes) else json.dumps(value).encode())

        def do_POST(self):
            with call_lock:
                calls.append((self.command, self.path, self.headers.get("Authorization")))
            self.send_error(405)

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    worker = threading.Thread(target=server.serve_forever, daemon=True)
    worker.start()
    try:
        yield "http://127.0.0.1:" + str(server.server_port), calls
    finally:
        server.shutdown()
        server.server_close()
        worker.join(timeout=2)


class CollectorTests(unittest.TestCase):
    def collect(self, url, selected, **config):
        return health.collect({"entity_ids": selected, **config}, url=url, token=TEST_TOKEN)

    def test_remote_explicit_selection_normal_states_and_missing_entities(self):
        states = [entity("light.owned", "off", friendly_name="Owned Lamp"),
                  entity("lock.owned", "locked", friendly_name="Owned Lock"),
                  entity("media_player.owned", "idle"), entity("cover.owned", "closed"),
                  entity("light.failed", "unavailable"), entity("switch.uncertain", "unknown"),
                  entity("light.unselected", "unavailable", friendly_name="Private unselected name")]
        selected = ["light.owned", "lock.owned", "media_player.owned", "cover.owned",
                    "light.failed", "switch.uncertain", "light.missing"]
        before = datetime.now(timezone.utc)
        with fake_ha(states) as (url, calls):
            result = self.collect(url, selected)
        after = datetime.now(timezone.utc)
        self.assertEqual([item["entity_id"] for item in result["inventory"]], selected)
        self.assertEqual([item["availability"] for item in result["inventory"]],
                         ["available"] * 4 + ["unavailable", "unknown", "unknown"])
        self.assertEqual(result["inventory"][-1]["availability_reason"], "missing_entity")
        self.assertEqual(result["inventory"][-1]["present_in_states"], False)
        self.assertEqual(result["summary"]["missing_selected_entities"], 1)
        self.assertEqual(result["summary"]["available_selected_entities"], 4)
        self.assertEqual(result["summary"]["unavailable_or_unknown_controls"], 3)
        self.assertEqual(result["summary"]["source_counts"], {"home_assistant_entity": 7})
        self.assertEqual(result["status"], "attention")
        self.assertNotIn("Private unselected name", json.dumps(result))
        self.assertEqual(result["inventory"][0]["last_updated"], "2026-01-01T02:00:00+00:00")
        self.assertTrue(all(item["physical_operation"] == "not_tested" for item in result["inventory"]))
        completed = datetime.fromisoformat(result["completed_at"])
        self.assertLessEqual(before - timedelta(seconds=1), completed)
        self.assertLessEqual(completed, after)
        self.assertEqual({path for _, path, _ in calls}, {"/api/config", "/api/states"})
        self.assertTrue(all(method == "GET" and auth == "Bearer " + TEST_TOKEN for method, _, auth in calls))

    def test_battery_warnings_require_selected_reported_percentage_or_binary(self):
        states = [entity("sensor.low", "20", device_class="battery", unit_of_measurement="%"),
                  entity("sensor.high", "21", device_class="battery", unit_of_measurement="%"),
                  entity("sensor.voltage", "2.5", device_class="battery", unit_of_measurement="V"),
                  entity("sensor.invalid", "-1", device_class="battery", unit_of_measurement="%"),
                  entity("sensor.nan", "nan", device_class="battery", unit_of_measurement="%"),
                  entity("sensor.failed", "unavailable", device_class="battery", unit_of_measurement="%"),
                  entity("binary_sensor.low", "on", device_class="battery"),
                  entity("binary_sensor.fine", "off", device_class="battery"),
                  entity("sensor.unselected", "1", device_class="battery", unit_of_measurement="%")]
        selected = [item["entity_id"] for item in states[:-1]]
        with fake_ha(states) as (url, _):
            result = self.collect(url, selected)
        self.assertEqual([item["entity_id"] for item in result["battery_attention"]], ["sensor.low", "binary_sensor.low"])
        self.assertEqual(result["battery_attention"][0]["percent"], 20)
        self.assertIsNone(result["battery_attention"][1]["percent"])
        self.assertEqual(result["summary"]["reported_low_batteries"], 2)
        self.assertEqual(result["coverage"]["battery_scope"], "selected_reporting_entities")

    def test_battery_threshold_is_configurable(self):
        with fake_ha([entity("sensor.battery", "15", device_class="battery", unit_of_measurement="%")]) as (url, _):
            result = self.collect(url, ["sensor.battery"], battery_warning_threshold=10)
        self.assertEqual(result["battery_attention"], [])
        self.assertEqual(result["status"], "available")

    def test_no_sample_defaults_and_no_selection_is_unknown(self):
        template = health.load_config(SCRIPT.parent / "health-config.example.json")
        self.assertEqual(template.entity_ids, ())
        with fake_ha([entity("light.unselected", "on")]) as (url, _):
            result = health.collect(template, url=url, token=TEST_TOKEN)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["inventory"], [])
        self.assertFalse(result["coverage"]["selection_configured"])
        self.assertEqual(result["summary"]["selected_entities"], 0)

    def test_selection_limits_validated_before_network(self):
        invalid = [
            {"entity_ids": ["light.x"] * 2},
            {"entity_ids": [f"sensor.x_{i}" for i in range(501)]},
            {"entity_ids": ["light.*"]}, {"entity_ids": "all"},
            {"entity_ids": ["/api/services/light/turn_on"]},
            {"entity_ids": [], "battery_warning_threshold": True},
            {"entity_ids": [], "battery_warning_threshold": float("nan")},
            {"entity_ids": [], "timeout_seconds": 0},
            {"entity_ids": [], "check_integrations": "yes"},
            {"entity_ids": [], "typo": "value"}, [],
        ]
        with patch.object(health.HomeAssistantClient, "get", side_effect=AssertionError("Network must not run")):
            for config in invalid:
                with self.subTest(config_type=type(config).__name__):
                    with self.assertRaises(health.ConfigurationError):
                        health.collect(config, url="http://127.0.0.1:1", token=TEST_TOKEN)
        valid = health.HealthConfig.from_mapping({"entity_ids": [f"sensor.x_{i}" for i in range(500)]})
        self.assertEqual(len(valid.entity_ids), 500)

    def test_optional_unsupported_integrations_leave_explicit_coverage(self):
        for code in (404, 405, 501):
            with self.subTest(code=code), fake_ha([entity("light.owned", "off")], extra={
                "/api/config/config_entries/entry": (code, {"error": TEST_TOKEN}),
            }) as (url, _):
                result = self.collect(url, ["light.owned"], check_integrations=True)
            self.assertEqual(result["status"], "available")
            self.assertEqual(result["coverage"]["integrations"], "unsupported")
            self.assertIsNone(result["summary"]["failing_integrations"])
            self.assertEqual(result["integration_issues"], [])
            self.assertIn("integrations", result["check_errors"])
            self.assertNotIn(TEST_TOKEN, json.dumps(result))

    def test_integration_failures_report_only_domain_state(self):
        entries = [{"domain": "owned_integration", "state": "setup_retry", "data": {"token": TEST_TOKEN}},
                   {"domain": "other", "state": "loaded"}, {"domain": "malformed", "state": []}]
        with fake_ha([entity("light.owned", "on")], extra={"/api/config/config_entries/entry": (200, entries)}) as (url, _):
            result = self.collect(url, ["light.owned"], check_integrations=True)
        self.assertEqual(result["status"], "attention")
        self.assertEqual(result["integration_issues"], [{"domain": "owned_integration", "state": "setup_retry"}])
        self.assertEqual(result["summary"]["failing_integrations"], 1)
        self.assertEqual(result["integration_state_counts"], {"setup_retry": 1, "loaded": 1, "unknown": 1})
        self.assertNotIn(TEST_TOKEN, json.dumps(result))

    def test_api_auth_failure_does_not_claim_missing_entities_or_leak(self):
        with fake_ha(extra={"/api/states": (401, {"error": "Bearer " + TEST_TOKEN})}) as (url, _):
            result = self.collect(url, ["light.owned"])
        self.assertEqual(result["status"], "attention")
        self.assertEqual(result["inventory"][0]["availability_reason"], "states_not_collected")
        self.assertIsNone(result["inventory"][0]["present_in_states"])
        self.assertIsNone(result["summary"]["missing_selected_entities"])
        self.assertIsNone(result["summary"]["reported_low_batteries"])
        self.assertEqual(result["check_errors"]["states"], "Home Assistant authentication rejected")
        self.assertNotIn(TEST_TOKEN, json.dumps(result))

    def test_missing_credentials_make_no_network_calls(self):
        with fake_ha() as (url, calls):
            result = health.collect({"entity_ids": ["light.owned"]}, url=url, token="")
        self.assertEqual(calls, [])
        self.assertEqual(result["status"], "attention")
        self.assertFalse(result["home_assistant"]["api_reachable"])
        self.assertFalse(result["home_assistant"]["api_authenticated"])
        self.assertEqual(result["inventory"][0]["availability_reason"], "states_not_collected")

    def test_api_echoed_secrets_redacted_in_snapshot_and_summary(self):
        state = entity("light.owned", "off", friendly_name="Echo " + TEST_TOKEN, unrelated_secret=TEST_TOKEN)
        with fake_ha([state], config={"time_zone": "UTC", "version": TEST_TOKEN}) as (url, _):
            result = self.collect(url, ["light.owned"])
        self.assertNotIn(TEST_TOKEN, json.dumps(result))
        self.assertNotIn(TEST_TOKEN, health.render_summary(result))
        self.assertIn("[redacted]", result["inventory"][0]["name"])
        self.assertNotIn(url, json.dumps(result))

    def test_redirect_refused_without_forwarding_authorization(self):
        with fake_ha(extra={"/api/states": (302, {})}) as (url, calls):
            result = self.collect(url, ["light.owned"])
        self.assertEqual(result["check_errors"]["states"], "API redirect refused")
        self.assertNotIn("/should-not-receive-token", [path for _, path, _ in calls])

    def test_malformed_api_json_and_schema_are_safe_failures(self):
        for payload in (b"not-json", {"unexpected": "shape"}, ["bad-state"]):
            with self.subTest(payload_type=type(payload).__name__), fake_ha(extra={"/api/states": (200, payload)}) as (url, _):
                result = self.collect(url, ["light.owned"])
            self.assertEqual(result["status"], "attention")
            self.assertEqual(result["coverage"]["entity_states"], "unavailable")
            self.assertEqual(result["inventory"][0]["availability_reason"], "states_not_collected")

    def test_home_assistant_timezone_and_explicit_override(self):
        with fake_ha([entity("light.owned", "off")], config={"time_zone": "Asia/Kolkata"}) as (url, _):
            configured = self.collect(url, ["light.owned"])
            override = self.collect(url, ["light.owned"], timezone="UTC")
        self.assertEqual(configured["timezone"], "Asia/Kolkata")
        self.assertEqual(configured["timezone_source"], "home_assistant")
        self.assertEqual(datetime.fromisoformat(configured["checked_at"]).utcoffset(), timedelta(hours=5, minutes=30))
        self.assertEqual(override["timezone"], "UTC")
        self.assertEqual(override["timezone_source"], "health_config")
        self.assertEqual(datetime.fromisoformat(override["checked_at"]).utcoffset(), timedelta(0))

    def test_invalid_timezone_falls_back_with_evidence(self):
        with fake_ha([entity("light.owned", "off")]) as (url, _):
            result = self.collect(url, ["light.owned"], timezone="Invalid/Not_A_Zone")
        self.assertEqual(result["timezone"], "UTC")
        self.assertEqual(result["timezone_source"], "utc_fallback")
        self.assertIn("timezone", result["check_errors"])

    def test_utc_does_not_require_installed_iana_data(self):
        with patch.object(health, "ZoneInfo", side_effect=health.ZoneInfoNotFoundError("Missing data")):
            with fake_ha([entity("light.owned", "off")]) as (url, _):
                result = self.collect(url, ["light.owned"])
        self.assertEqual(result["timezone"], "UTC")
        self.assertEqual(result["status"], "available")
        self.assertNotIn("timezone", result["check_errors"])

    def test_atomic_write_failure_preserves_previous_latest_and_cleans_temp(self):
        with tempfile.TemporaryDirectory() as directory:
            latest = Path(directory) / "latest.json"
            latest.write_text("previous-evidence", encoding="utf-8")
            with patch.object(health.os, "replace", side_effect=OSError("Test filesystem failure")):
                with self.assertRaises(OSError):
                    health._atomic_write(latest, "replacement-evidence")
            self.assertEqual(latest.read_text(encoding="utf-8"), "previous-evidence")
            self.assertEqual(list(Path(directory).glob("*.tmp")), [])

    def test_credentials_literal_file_environment_precedence_and_no_default_url(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / ".env"
            path.write_text('HA_URL="http://127.0.0.1:1"\nHA_TOKEN=\'literal-token\'\nOTHER=unused\n', encoding="utf-8")
            with patch.dict(os.environ, {}, clear=True):
                self.assertEqual(health.load_credentials(path), ("http://127.0.0.1:1", "literal-token"))
            with patch.dict(os.environ, {"HA_URL": "http://127.0.0.1:2", "HA_TOKEN": "environment-token"}, clear=True):
                self.assertEqual(health.load_credentials(path), ("http://127.0.0.1:2", "environment-token"))
            with patch.dict(os.environ, {}, clear=True):
                with self.assertRaises(health.ConfigurationError):
                    health.collect({"entity_ids": []})

    def test_url_credentials_queries_fragments_and_header_injection_rejected(self):
        for url in ["http://user:secret@localhost:8123", "http://localhost:8123?token=secret",
                    "http://localhost:8123#secret", "file:///private", "http://localhost:bad"]:
            with self.subTest(url_shape=url.split(":", 1)[0]):
                with self.assertRaises(health.ConfigurationError) as raised:
                    health.HomeAssistantClient(url, TEST_TOKEN)
                self.assertNotIn("secret", str(raised.exception))
        with self.assertRaises(health.ConfigurationError):
            health.HomeAssistantClient("http://127.0.0.1:1", "bad\r\nheader")
        self.assertNotIn(TEST_TOKEN, repr(health.HomeAssistantClient("http://127.0.0.1:1", TEST_TOKEN)))

    def test_snapshot_files_atomic_replacement_and_cli_structured_json(self):
        with tempfile.TemporaryDirectory() as directory, fake_ha([entity("light.owned", "off", friendly_name="Owned Lamp")]) as (url, calls):
            root = Path(directory)
            config_path = root / "health-config.json"
            env_path = root / ".env"
            output = root / "snapshots"
            config_path.write_text(json.dumps({"entity_ids": ["light.owned"]}), encoding="utf-8")
            env_path.write_text("HA_URL=" + url + "\nHA_TOKEN=" + TEST_TOKEN + "\n", encoding="utf-8")
            result = self.collect(url, ["light.owned"])
            first = health.save_snapshot(result, output)
            health.save_snapshot({**result, "status": "unknown"}, output)
            # Preserve Windows/Python runtime variables but prohibit the host's
            # HA credentials from overriding this temporary fake server file.
            child_env = {key: value for key, value in os.environ.items() if key not in {"HA_URL", "HA_TOKEN"}}
            process = subprocess.run([sys.executable, str(SCRIPT), "--config", str(config_path),
                                      "--env-file", str(env_path), "--output-dir", str(output)],
                                     capture_output=True, text=True, check=False, timeout=10, env=child_env)
            self.assertEqual(process.returncode, 0, process.stderr)
            stdout = json.loads(process.stdout)
            saved = json.loads((output / "latest.json").read_text(encoding="utf-8"))
            self.assertEqual(stdout, saved)
            self.assertEqual(saved["status"], "available", saved["check_errors"])
            self.assertEqual(json.loads(first.read_text(encoding="utf-8")), result)
            self.assertIn("Owned Lamp", (output / "latest-summary.txt").read_text(encoding="utf-8"))
            self.assertEqual(len(list(output.glob("*.json"))), 4)
            self.assertEqual(list(output.glob("*.tmp")), [])
            self.assertNotIn(TEST_TOKEN, process.stdout)
            self.assertEqual(process.stderr, "")
            self.assertTrue(all(method == "GET" for method, _, _ in calls))

    def test_cli_missing_configuration_is_structured_and_never_connects(self):
        stream = io.StringIO()
        with patch("sys.stdout", stream), patch.object(health.HomeAssistantClient, "get", side_effect=AssertionError("No network")):
            code = health.main(["--config", str(Path(tempfile.gettempdir()) / "not-present-health-config-7264.json")])
        result = json.loads(stream.getvalue())
        self.assertEqual(code, 2)
        self.assertEqual(result["status"], "configuration_error")
        self.assertNotIn("7264", json.dumps(result))


if __name__ == "__main__":
    unittest.main()
