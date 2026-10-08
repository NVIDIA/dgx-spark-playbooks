"""A fresh household run/profile provider, independent of demo/reset records.

This is an injectable interface for PlanStore. Wiring it into a clean backend,
excluding demo routes and replacing demo-specific capabilities remains required.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
import tempfile
import threading
import uuid


class HouseholdRunStore:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.lock = threading.RLock()

    def initialize(self, property_id: str) -> dict:
        with self.lock:
            if self.path.exists():
                current = self._load()
                if current["property_id"] != property_id:
                    raise ValueError("Household already exists with a different identity; no reset performed")
                return current
            if not isinstance(property_id, str) or not property_id.startswith("home-"):
                raise ValueError("Use the generated own household identity")
            current = {"schemaVersion": 1, "property_id": property_id, "run_id": uuid.uuid4().hex,
                       "device_profile": "physical", "state_scope": "household", "demo": False}
            self._write(current)
            return current

    def _load(self) -> dict:
        with self.path.open(encoding="utf-8") as stream:
            value = json.load(stream)
        if (not isinstance(value, dict) or value.get("schemaVersion") != 1
                or value.get("device_profile") != "physical" or value.get("demo") is not False
                or value.get("state_scope") != "household" or not isinstance(value.get("property_id"), str)):
            raise ValueError("Invalid household state; no seeded fallback or reset is available")
        uuid.UUID(hex=value["run_id"])
        return value

    def _write(self, value: dict) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=self.path.parent, delete=False) as stream:
                temporary = Path(stream.name)
                json.dump(value, stream, indent=2)
                stream.write("\n")
                stream.flush()
                os.fsync(stream.fileno())
            temporary.replace(self.path)
        finally:
            if temporary and temporary.exists():
                temporary.unlink()
