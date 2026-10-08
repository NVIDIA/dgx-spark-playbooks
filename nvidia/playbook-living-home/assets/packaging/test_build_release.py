import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from build_release import ReleaseError, REQUIRED_GATES, build, clean_path, inventory_files


class CleanReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(prefix="living-home-release-test-")
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / "source"
        self.root.mkdir()
        self.launcher = b"Write-Output 'synthetic offline test only'\n"
        (self.root / "Setup.ps1").write_bytes(self.launcher)
        self.inventory = {
            "schemaVersion": 1, "architecture": "arm64", "releaseVersion": "test-only",
            "entryPoint": "Setup.ps1", "blockers": [], "readiness": {key: True for key in REQUIRED_GATES},
            "files": [{"source": "Setup.ps1", "path": "Setup.ps1", "sha256": hashlib.sha256(self.launcher).hexdigest(),
                       "classification": "code", "reviewed": True}], "models": [],
        }

    def test_explicit_inventory_omits_unlisted_private_files(self):
        (self.root / ".env").write_text("private fixture, never copied")
        (self.root / "unreviewed.txt").write_text("private fixture")
        output = Path(self.temp.name) / "release"
        result = build(self.inventory, self.root, output)
        with zipfile.ZipFile(output / "living-home-arm64.zip") as z:
            self.assertEqual(z.namelist(), ["Setup.ps1"])
            self.assertEqual(result["payload"]["unpackedBytes"], sum(i.file_size for i in z.infolist()))
        self.assertEqual(result["payload"]["url"], "living-home-arm64.zip")
        self.assertEqual(result["payload"]["sha256"], hashlib.sha256((output / "living-home-arm64.zip").read_bytes()).hexdigest())
        self.assertEqual(json.loads((output / "release.json").read_text()), result)

    def test_denies_private_and_unsafe_paths(self):
        for path in (".env", "infra/.env.backup", "ha/.storage/core.config", "openclaw/state/settings.json",
                     "property/report.md", "secrets.yaml", "auth.json", "google_token.json", "a.sqlite",
                     "a.db-wal", "a/key.pem", "../escape.ps1", "C:/file.ps1", "a\\file.ps1",
                     "NUL.ps1", "a/COM1", "a/name.", "a/name ", "a//b", "a/file:ads"):
            with self.subTest(path=path), self.assertRaises(ReleaseError):
                clean_path(path)

    def test_source_private_path_cannot_hide_behind_clean_destination(self):
        row = self.inventory["files"][0]
        row["source"] = ".env"
        with self.assertRaises(ReleaseError):
            inventory_files(self.inventory, self.root)

    def test_blockers_and_unmet_gates_refuse_release(self):
        self.inventory["blockers"] = ["unresolved fixture dependency"]
        with self.assertRaises(ReleaseError):
            inventory_files(self.inventory, self.root)
        self.inventory["blockers"] = []
        self.inventory["readiness"]["fresh_household_initialization"] = False
        with self.assertRaises(ReleaseError):
            inventory_files(self.inventory, self.root)

    def test_unreviewed_or_unpinned_input_rejected(self):
        for key, value in (("reviewed", False), ("sha256", ""), ("classification", "account-data")):
            inventory = copy.deepcopy(self.inventory)
            inventory["files"][0][key] = value
            with self.subTest(key=key), self.assertRaises(ReleaseError):
                inventory_files(inventory, self.root)

    def test_hash_mismatch_prevents_manifest(self):
        (self.root / "Setup.ps1").write_text("changed after review")
        output = Path(self.temp.name) / "release"
        with self.assertRaises(ReleaseError):
            build(self.inventory, self.root, output)
        self.assertFalse((output / "release.json").exists())

    def test_secret_content_rejected_without_echo(self):
        content = b"access_token='" + b"z" * 40 + b"'"
        (self.root / "Setup.ps1").write_bytes(content)
        self.inventory["files"][0]["sha256"] = hashlib.sha256(content).hexdigest()
        output = Path(self.temp.name) / "release"
        with self.assertRaises(ReleaseError) as error:
            build(self.inventory, self.root, output)
        self.assertNotIn("z" * 40, str(error.exception))
        self.assertFalse((output / "release.json").exists())

    def test_case_insensitive_path_collision_rejected(self):
        row = dict(self.inventory["files"][0], path="setup.PS1")
        self.inventory["files"].append(row)
        with self.assertRaises(ReleaseError):
            inventory_files(self.inventory, self.root)

    def test_file_directory_collision_rejected(self):
        self.inventory["files"].append(dict(self.inventory["files"][0], path="Setup.ps1/a.txt"))
        with self.assertRaises(ReleaseError):
            inventory_files(self.inventory, self.root)

    def test_models_require_real_pins_and_https(self):
        self.inventory["models"] = [{"path": "models/test.gguf", "url": "http://invalid.test/model", "size": 4, "sha256": "0" * 64}]
        with self.assertRaises(ReleaseError):
            build(self.inventory, self.root, Path(self.temp.name) / "release")

    def test_model_payload_collision_rejected(self):
        self.inventory["models"] = [{"path": "Setup.ps1/sub.gguf", "url": "https://invalid.test/model", "size": 4, "sha256": "0" * 64}]
        with self.assertRaises(ReleaseError):
            build(self.inventory, self.root, Path(self.temp.name) / "release")

    def test_never_overwrites_existing_build(self):
        output = Path(self.temp.name) / "release"
        build(self.inventory, self.root, output)
        before = (output / "release.json").read_bytes()
        with self.assertRaises(FileExistsError):
            build(self.inventory, self.root, output)
        self.assertEqual(before, (output / "release.json").read_bytes())


if __name__ == "__main__":
    unittest.main()
