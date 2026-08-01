"""In-memory metrics collector for litguard using multiprocessing-safe shared state."""

import json
import os
import time
import fcntl
from dataclasses import dataclass
from pathlib import Path


METRICS_FILE = Path(os.environ.get("LITGUARD_METRICS_DIR", "/tmp")) / "litguard_metrics.jsonl"
COUNTERS_FILE = Path(os.environ.get("LITGUARD_METRICS_DIR", "/tmp")) / "litguard_counters.json"


@dataclass
class ClassificationRecord:
    timestamp: float
    input_text: str
    model: str
    label: str
    score: float
    latency_ms: float


class MetricsCollector:
    """File-backed metrics that work across LitServe's multiprocess workers."""

    def __init__(self, max_history: int = 1000):
        self._max_history = max_history
        # Reset on startup
        METRICS_FILE.write_text("")
        COUNTERS_FILE.write_text(json.dumps({
            "total_requests": 0,
            "total_latency_ms": 0.0,
            "injection_count": 0,
            "benign_count": 0,
        }))

    def record(self, record: ClassificationRecord):
        entry = json.dumps({
            "timestamp": record.timestamp,
            "input_text": record.input_text[:120],
            "model": record.model,
            "label": record.label,
            "score": round(record.score, 4),
            "latency_ms": round(record.latency_ms, 2),
        })

        # Append to history file (atomic with file lock)
        with open(METRICS_FILE, "a") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            f.write(entry + "\n")
            fcntl.flock(f, fcntl.LOCK_UN)

        # Update counters
        with open(COUNTERS_FILE, "r+") as f:
            fcntl.flock(f, fcntl.LOCK_EX)
            try:
                counters = json.load(f)
            except (json.JSONDecodeError, ValueError):
                counters = {"total_requests": 0, "total_latency_ms": 0.0,
                            "injection_count": 0, "benign_count": 0}
            counters["total_requests"] += 1
            counters["total_latency_ms"] += record.latency_ms
            if record.label == "injection":
                counters["injection_count"] += 1
            else:
                counters["benign_count"] += 1
            f.seek(0)
            f.truncate()
            json.dump(counters, f)
            fcntl.flock(f, fcntl.LOCK_UN)

    def get_history(self, limit: int = 1000) -> list[dict]:
        try:
            with open(METRICS_FILE, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                lines = f.readlines()
                fcntl.flock(f, fcntl.LOCK_UN)
        except FileNotFoundError:
            return []

        records = []
        for line in lines[-limit:]:
            line = line.strip()
            if line:
                try:
                    r = json.loads(line)
                    records.append({
                        "timestamp": r["timestamp"],
                        "input_preview": r["input_text"],
                        "model": r["model"],
                        "label": r["label"],
                        "score": r["score"],
                        "latency_ms": r["latency_ms"],
                    })
                except (json.JSONDecodeError, KeyError):
                    continue
        return records

    def get_metrics(self) -> dict:
        try:
            with open(COUNTERS_FILE, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                counters = json.load(f)
                fcntl.flock(f, fcntl.LOCK_UN)
        except (FileNotFoundError, json.JSONDecodeError):
            counters = {"total_requests": 0, "total_latency_ms": 0.0,
                        "injection_count": 0, "benign_count": 0}

        total = counters["total_requests"]
        avg_latency = counters["total_latency_ms"] / total if total > 0 else 0.0

        # Count recent requests for RPS
        try:
            with open(METRICS_FILE, "r") as f:
                fcntl.flock(f, fcntl.LOCK_SH)
                lines = f.readlines()
                fcntl.flock(f, fcntl.LOCK_UN)
        except FileNotFoundError:
            lines = []

        now = time.time()
        recent_count = 0
        for line in reversed(lines):
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                if now - r["timestamp"] < 60:
                    recent_count += 1
                else:
                    break
            except (json.JSONDecodeError, KeyError):
                continue

        rps = recent_count / 60.0

        return {
            "total_requests": total,
            "requests_per_second": round(rps, 2),
            "avg_latency_ms": round(avg_latency, 2),
            "injection_count": counters["injection_count"],
            "benign_count": counters["benign_count"],
        }


# Global singleton
metrics = MetricsCollector()
