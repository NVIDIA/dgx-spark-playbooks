#!/usr/bin/env python3
"""End-to-end verification for the DGX Spark OpenViking cuVS playbook."""

from __future__ import annotations

import importlib.metadata
import json
import os
import platform
import re
import sys
import time
import urllib.error
import urllib.request
from typing import Any


OPENVIKING_URL = os.environ.get("OPENVIKING_URL", "http://127.0.0.1:1933").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
API_KEY = os.environ.get("OPENVIKING_API_KEY", "")
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
VLM_MODEL = "qwen3.8:27b"
SMOKE_URI = "viking://resources/dgx-spark-openviking-cuvs-smoke.md"
MARKER = "heliotrope-viking-7319 confirms the DGX Spark cuVS route"
RED_SQUARE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAIAAACQkWg2AAAAF0lEQVR4nGP4z8BAEiJN9aiG"
    "UQ1DSgMAkPn/Afnh+ngAAAAASUVORK5CYII="
)


class CheckError(RuntimeError):
    pass


def request_json(
    method: str,
    url: str,
    payload: dict[str, Any] | None = None,
    *,
    openviking: bool = False,
    timeout: float = 600,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if openviking and API_KEY:
        headers["X-API-Key"] = API_KEY
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise CheckError(f"{method} {url} returned HTTP {exc.code}: {body}") from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise CheckError(f"{method} {url} failed: {exc}") from exc


def require_ok(envelope: dict[str, Any], label: str) -> dict[str, Any]:
    if envelope.get("status") != "ok":
        raise CheckError(f"{label} returned a non-ok envelope: {envelope}")
    return envelope


def check_gpu_runtime() -> dict[str, str]:
    if platform.system() != "Linux" or platform.machine() != "aarch64":
        raise CheckError(
            f"expected Linux aarch64, found {platform.system()} {platform.machine()}"
        )

    try:
        import cupy
        import cuvs
    except ImportError as exc:
        raise CheckError(f"GPU Python package import failed: {exc}") from exc

    try:
        if not getattr(cuvs, "__file__", None):
            raise CheckError("cuVS imported without a module path")
        device = cupy.cuda.runtime.getDeviceProperties(0)
        name = device.get("name", device.get(b"name", "unknown"))
        if isinstance(name, bytes):
            name = name.decode("utf-8")
        cupy.asarray([1.0], dtype=cupy.float32).sum().item()
    except Exception as exc:
        raise CheckError(f"CUDA device smoke failed: {exc}") from exc

    return {
        "device": str(name),
        "cuvs": importlib.metadata.version("cuvs-cu13"),
        "cupy": cupy.__version__,
        "openviking": importlib.metadata.version("openviking"),
        "python": platform.python_version(),
    }


def check_ollama_models() -> dict[str, Any]:
    tags = request_json("GET", f"{OLLAMA_URL}/api/tags")
    names = {str(model.get("name", "")) for model in tags.get("models", [])}
    missing = {EMBEDDING_MODEL, VLM_MODEL} - names
    if missing:
        raise CheckError(
            f"Ollama is missing required models: {sorted(missing)}; found {sorted(names)}"
        )

    embedded = request_json(
        "POST",
        f"{OLLAMA_URL}/api/embed",
        {"model": EMBEDDING_MODEL, "input": MARKER, "keep_alive": "10m"},
    )
    embeddings = embedded.get("embeddings")
    if not isinstance(embeddings, list) or not embeddings or len(embeddings[0]) != 1024:
        raise CheckError(
            "Ollama embedding probe did not return one 1024-dimensional vector"
        )

    chat = request_json(
        "POST",
        f"{OLLAMA_URL}/api/chat",
        {
            "model": VLM_MODEL,
            "messages": [
                {
                    "role": "user",
                    "content": "What is the dominant color of this square? Reply with one color word.",
                    "images": [RED_SQUARE_PNG],
                }
            ],
            "stream": False,
            "think": False,
            "keep_alive": "10m",
            "options": {"num_ctx": 16384, "temperature": 0},
        },
    )
    content = str(chat.get("message", {}).get("content", ""))
    if re.search(r"\bred\b", content, flags=re.IGNORECASE) is None:
        raise CheckError(f"Ollama VLM probe returned unexpected content: {content!r}")

    return {"embedding_dimension": len(embeddings[0]), "vlm_response": content.strip()}


def check_openviking_and_cuvs_route() -> dict[str, Any]:
    require_ok(
        request_json("GET", f"{OPENVIKING_URL}/health", openviking=True, timeout=30),
        "OpenViking health",
    )
    models = require_ok(
        request_json(
            "GET",
            f"{OPENVIKING_URL}/api/v1/observer/models",
            openviking=True,
            timeout=30,
        ),
        "OpenViking model observer",
    )
    if not models.get("result", {}).get("is_healthy"):
        raise CheckError(f"OpenViking model observer is unhealthy: {models}")

    write = require_ok(
        request_json(
            "POST",
            f"{OPENVIKING_URL}/api/v1/content/write",
            {
                "uri": SMOKE_URI,
                "content": f"# DGX Spark cuVS smoke\n\n{MARKER}.\n",
                "mode": "upsert",
                "wait": True,
                "timeout": 300,
                "processing_mode": "vectors_only",
            },
            openviking=True,
        ),
        "OpenViking content write",
    )

    deadline = time.monotonic() + 180
    last_routes: dict[str, int] = {}
    last_hits: list[str] = []
    while time.monotonic() < deadline:
        found = require_ok(
            request_json(
                "POST",
                f"{OPENVIKING_URL}/api/v1/search/find",
                {
                    "query": MARKER,
                    "limit": 5,
                    "read_content": True,
                    "telemetry": {"summary": True},
                },
                openviking=True,
            ),
            "OpenViking semantic find",
        )
        resources = found.get("result", {}).get("resources", [])
        last_hits = [
            str(hit.get("uri", "")) for hit in resources if isinstance(hit, dict)
        ]
        known_result = any(
            hit.get("uri") == SMOKE_URI or MARKER in str(hit.get("content", ""))
            for hit in resources
            if isinstance(hit, dict)
        )
        summary = found.get("telemetry", {}).get("summary", {})
        cuvs_summary = summary.get("vector", {}).get("cuvs", {})
        last_routes = cuvs_summary.get("routes", {})
        if known_result and int(last_routes.get("cuvs", 0)) > 0:
            index_size = int(cuvs_summary.get("index_size_max", 0))
            if index_size < 1:
                raise CheckError(
                    f"cuVS route reported an invalid index size: {cuvs_summary}"
                )
            return {
                "smoke_uri": SMOKE_URI,
                "routes": last_routes,
                "index_size": index_size,
                "write_status": write.get("result", {}).get("status", "ok"),
            }
        time.sleep(2)

    raise CheckError(
        "known-result search never used the cuVS route within 180 seconds; "
        f"last routes={last_routes}, last hits={last_hits}"
    )


def main() -> int:
    report = {
        "gpu_runtime": check_gpu_runtime(),
        "ollama": check_ollama_models(),
        "openviking": check_openviking_and_cuvs_route(),
    }
    print(json.dumps(report, indent=2, sort_keys=True))
    print("OPENVIKING_CUVS_E2E_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except CheckError as exc:
        print(f"OPENVIKING_CUVS_E2E_FAILED: {exc}", file=sys.stderr)
        raise SystemExit(1)
