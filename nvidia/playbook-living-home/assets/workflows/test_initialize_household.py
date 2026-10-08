import json
import io
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from contextlib import redirect_stdout

from household_state import HouseholdRunStore
import initialize_household as setup


class FirstRunTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="living-home-household-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "new-home"
        self.cfg = {"schemaVersion": 1, "haMode": "existing", "haUrl": "http://127.0.0.1:8123",
                    "entityIds": ["light.own_test_entity"], "timeZone": "UTC", "openclawProfile": "own-home-test",
                    "google": {"enabled": False}}

    def initialize(self, token=None):
        with patch.object(setup, "protect_private_directory") as protect:
            result = setup.initialize(self.cfg, self.root, token)
            protect.assert_called_once()
        return result

    def test_creates_new_physical_context_without_seeded_records(self):
        result = self.initialize()
        state = HouseholdRunStore(self.root / "household.json")._load()
        self.assertFalse(state["demo"])
        self.assertEqual(state["device_profile"], "physical")
        self.assertEqual(state["state_scope"], "household")
        self.assertNotIn("maintenance", state)
        self.assertNotIn("devices", state)
        self.assertNotIn("calendar", state)
        self.assertEqual(list((self.root / "property").iterdir()), [])
        self.assertFalse(result["runtimeWired"])
        self.assertFalse(result["servicesStarted"])
        self.assertFalse((self.root / ".env").exists())
        self.assertGreaterEqual(len((self.root / "api-token.txt").read_text().strip()), 40)
        self.assertEqual(json.loads((self.root / "health-config.json").read_text())["entity_ids"], self.cfg["entityIds"])

    def test_own_token_saved_only_in_private_env_not_result_or_settings(self):
        token = "local-unit-test-token-value"
        result = self.initialize(token)
        self.assertIn(token, (self.root / ".env").read_text())
        self.assertNotIn(token, json.dumps(result))
        self.assertNotIn(token, (self.root / "home-settings.json").read_text())

    def test_refuses_overwrite_and_preserves_identity_and_token(self):
        self.initialize("first-test-token")
        before = (self.root / "household.json").read_bytes()
        with self.assertRaises(FileExistsError):
            self.initialize("new-test-token")
        self.assertEqual(before, (self.root / "household.json").read_bytes())
        self.assertIn("first-test-token", (self.root / ".env").read_text())

    def test_context_initialization_idempotent_without_reset(self):
        self.initialize()
        store = HouseholdRunStore(self.root / "household.json")
        initial = store._load()
        self.assertEqual(store.initialize(initial["property_id"]), initial)
        with self.assertRaises(ValueError):
            store.initialize("home-another-test")
        self.assertEqual(store._load(), initial)

    def test_rejects_implicit_scope_or_credential_in_url(self):
        for changes in ({"entityIds": []}, {"entityIds": ["light.a", "light.a"]},
                        {"entityIds": [{}]}, {"haUrl": "http://user:password@127.0.0.1:8123"},
                        {"haMode": "new", "haUrl": "https://existing.example.invalid"}):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                setup.validate(dict(self.cfg, **changes))

    def test_own_google_scope_is_saved_without_bootstrap_or_oauth(self):
        self.cfg["google"] = {"enabled": True, "expectedAccount": "owner@example.invalid",
                              "gmailLabel": "MyHome", "driveFolderId": "own_test_folder"}
        self.initialize()
        scope = json.loads((self.root / "google-scope.json").read_text())
        self.assertEqual(scope["gmail_label"], "MyHome")
        self.assertEqual(scope["expected_account"], "owner@example.invalid")
        self.assertFalse((self.root / "google_token.json").exists())
        self.assertEqual(list((self.root / "property").iterdir()), [])

    def test_future_ui_stdin_bridge_keeps_token_off_stdout(self):
        token = "hidden-stdin-fixture-value"
        envelope = {"schemaVersion": 1, "configuration": self.cfg,
                    "dataDirectory": str(self.root), "haToken": token}
        output = io.StringIO()
        with patch("sys.argv", ["initialize_household.py", "--stdin", "--write"]), \
                patch("sys.stdin", io.StringIO(json.dumps(envelope))), \
                patch.object(setup, "protect_private_directory"), redirect_stdout(output):
            self.assertEqual(setup.main(), 0)
        response = json.loads(output.getvalue())
        self.assertTrue(response["ok"])
        self.assertNotIn(token, output.getvalue())


if __name__ == "__main__":
    unittest.main()
