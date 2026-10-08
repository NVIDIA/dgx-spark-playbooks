from pathlib import Path
import tempfile
import unittest
import zipfile

from build_development import build_development
from build_release import ReleaseError


class DevelopmentBundleTests(unittest.TestCase):
    def test_only_declared_source_and_clear_nonrelease_contract(self):
        with tempfile.TemporaryDirectory(prefix="living-home-dev-test-") as temp:
            root = Path(temp) / "source"
            root.mkdir()
            (root / "app.py").write_text("print('synthetic source test')\n")
            (root / ".env").write_text("private fixture")
            approved = {"schemaVersion": 1, "kind": "development-source", "files": [
                {"path": "app.py", "classification": "code", "optional": False},
                {"path": "future.py", "classification": "code", "optional": True}]}
            output = Path(temp) / "output"
            result = build_development(root, approved, output)
            self.assertFalse(result["runnableStack"])
            self.assertFalse((output / "release.json").exists())
            self.assertEqual(result["missingDeclaredComponents"], ["future.py"])
            with zipfile.ZipFile(output / "development-source.zip") as z:
                self.assertEqual(z.namelist(), ["app.py"])

    def test_private_path_in_declared_list_refused(self):
        with tempfile.TemporaryDirectory(prefix="living-home-dev-test-") as temp:
            approved = {"schemaVersion": 1, "kind": "development-source", "files": [
                {"path": ".env", "classification": "code", "optional": False}]}
            with self.assertRaises(ReleaseError):
                build_development(Path(temp), approved, Path(temp) / "output")


if __name__ == "__main__":
    unittest.main()
