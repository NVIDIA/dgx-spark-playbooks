#!/usr/bin/env python3
"""End-to-end verification for the DGX Spark OpenViking cuVS playbook."""

from __future__ import annotations

import importlib.metadata
import ipaddress
import json
import os
import platform
import re
import subprocess
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


OPENVIKING_URL = os.environ.get("OPENVIKING_URL", "http://127.0.0.1:1933").rstrip("/")
OLLAMA_URL = os.environ.get("OLLAMA_URL", "http://127.0.0.1:11434").rstrip("/")
API_KEY = os.environ.get("OPENVIKING_API_KEY", "")
PINS = json.loads(Path(__file__).with_name("pins.json").read_text(encoding="utf-8"))
OPENVIKING_VERSION = str(PINS["openviking_version"])
EMBEDDING_MODEL = "qwen3-embedding:0.6b"
VLM_MODEL = "qwen3.8:27b"
MODEL_DIGESTS = PINS["ollama"]["models"]
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
    allow_not_found: bool = False,
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
        if allow_not_found and exc.code == 404:
            return {"status": "ok", "result": {"already_absent": True}}
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

    installed_openviking = importlib.metadata.version("openviking")
    if installed_openviking != OPENVIKING_VERSION:
        raise CheckError(
            f"expected OpenViking package {OPENVIKING_VERSION}, found {installed_openviking}"
        )

    return {
        "device": str(name),
        "cuvs": importlib.metadata.version("cuvs-cu13"),
        "cupy": cupy.__version__,
        "openviking": installed_openviking,
        "python": platform.python_version(),
    }


def check_ollama_models(marker: str) -> dict[str, Any]:
    parsed_url = urllib.parse.urlsplit(OLLAMA_URL)
    try:
        ollama_address = ipaddress.ip_address(parsed_url.hostname or "")
    except ValueError as exc:
        raise CheckError(
            f"OLLAMA_URL must use a loopback IP address: {OLLAMA_URL}"
        ) from exc
    if not ollama_address.is_loopback or parsed_url.port != 11434:
        raise CheckError(f"OLLAMA_URL must be loopback port 11434: {OLLAMA_URL}")

    try:
        listeners = subprocess.run(
            ["ss", "-H", "-ltn", "sport = :11434"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout.splitlines()
    except (OSError, subprocess.CalledProcessError) as exc:
        raise CheckError(f"Could not inspect the Ollama TCP listener: {exc}") from exc
    if not listeners:
        raise CheckError("No TCP listener found on port 11434")
    listener_addresses: list[str] = []
    for line in listeners:
        fields = line.split()
        if len(fields) < 4:
            raise CheckError(f"Could not parse Ollama listener: {line!r}")
        listener = fields[3]
        host = listener.rsplit(":", 1)[0].strip("[]")
        try:
            address = ipaddress.ip_address(host)
        except ValueError as exc:
            raise CheckError(f"Non-loopback Ollama listener: {listener}") from exc
        if not address.is_loopback:
            raise CheckError(f"Non-loopback Ollama listener: {listener}")
        listener_addresses.append(listener)

    tags = request_json("GET", f"{OLLAMA_URL}/api/tags")
    models = {
        str(model.get("name", "")): str(model.get("digest", ""))
        .removeprefix("sha256:")
        .lower()
        for model in tags.get("models", [])
        if isinstance(model, dict)
    }
    mismatches = {
        name: {"expected": digest, "actual": models.get(name)}
        for name, digest in MODEL_DIGESTS.items()
        if models.get(name) != digest
    }
    if mismatches:
        raise CheckError(
            "Ollama model tag/digest mismatch. Tags are mutable; change pins.json only "
            f"after intentional model review. Mismatches: {mismatches}"
        )

    embedded = request_json(
        "POST",
        f"{OLLAMA_URL}/api/embed",
        {"model": EMBEDDING_MODEL, "input": marker, "keep_alive": "10m"},
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

    return {
        "embedding_dimension": len(embeddings[0]),
        "listeners": listener_addresses,
        "model_digests": {name: models[name] for name in MODEL_DIGESTS},
        "vlm_response": content.strip(),
    }


def read_generated_semantic(uri: str, label: str) -> str:
    query = urllib.parse.urlencode({"uri": uri})
    response = require_ok(
        request_json(
            "GET",
            f"{OPENVIKING_URL}/api/v1/content/read?{query}",
            openviking=True,
            timeout=30,
        ),
        label,
    )
    content = response.get("result")
    if not isinstance(content, str) or len(content.strip()) < 20:
        raise CheckError(
            f"{label} did not contain generated semantic output: {content!r}"
        )
    placeholder_phrases = ("is not generated", "is not ready")
    if any(phrase in content.lower() for phrase in placeholder_phrases):
        raise CheckError(f"{label} is still a placeholder: {content!r}")
    return content.strip()


def cleanup_smoke_root(smoke_root: str) -> dict[str, Any]:
    query = urllib.parse.urlencode(
        {
            "uri": smoke_root,
            "recursive": "true",
            "wait": "true",
            "timeout": "300",
        }
    )
    response = require_ok(
        request_json(
            "DELETE",
            f"{OPENVIKING_URL}/api/v1/fs?{query}",
            openviking=True,
            timeout=330,
            allow_not_found=True,
        ),
        "OpenViking smoke cleanup",
    )
    result = response.get("result", {})
    if result.get("already_absent") is True:
        return {"status": "already_absent"}
    if result.get("uri") != smoke_root:
        raise CheckError(
            f"OpenViking cleanup returned the wrong URI: expected {smoke_root}, got {result}"
        )
    if result.get("semantic_status") != "complete":
        raise CheckError(
            f"OpenViking cleanup semantic refresh did not complete: {result}"
        )
    return {
        "status": "deleted",
        "semantic_status": result["semantic_status"],
    }


def run_openviking_e2e(smoke_root: str, smoke_uri: str, marker: str) -> dict[str, Any]:
    health = require_ok(
        request_json("GET", f"{OPENVIKING_URL}/health", openviking=True, timeout=30),
        "OpenViking health",
    )
    if health.get("healthy") is not True or health.get("version") != OPENVIKING_VERSION:
        raise CheckError(
            f"Expected healthy OpenViking {OPENVIKING_VERSION}; received {health}"
        )

    write = require_ok(
        request_json(
            "POST",
            f"{OPENVIKING_URL}/api/v1/content/write",
            {
                "uri": smoke_uri,
                "content": (
                    "# DGX Spark cuVS verification\n\n"
                    f"The unique verification marker is {marker}.\n\n"
                    "This document checks all-local semantic processing with the configured "
                    "Ollama vision-language model and dense-vector retrieval through cuVS.\n"
                ),
                "mode": "create",
                "wait": True,
                "timeout": 600,
                "processing_mode": "semantic_and_vectors",
            },
            openviking=True,
            timeout=630,
        ),
        "OpenViking content write",
    )
    write_result = write.get("result", {})
    expected_write = {
        "uri": smoke_uri,
        "mode": "create",
        "semantic_status": "complete",
        "vector_status": "complete",
    }
    mismatches = {
        key: {"expected": expected, "actual": write_result.get(key)}
        for key, expected in expected_write.items()
        if write_result.get(key) != expected
    }
    if mismatches:
        raise CheckError(f"OpenViking write completion mismatch: {mismatches}")
    queue_status = write_result.get("queue_status") or {}
    for queue_name in ("Semantic", "Embedding"):
        queue = queue_status.get(queue_name, {})
        if int(queue.get("error_count", 0) or 0) != 0:
            raise CheckError(f"OpenViking {queue_name} queue reported errors: {queue}")

    abstract = read_generated_semantic(
        f"{smoke_root}/.abstract.md", "OpenViking generated abstract"
    )
    overview = read_generated_semantic(
        f"{smoke_root}/.overview.md", "OpenViking generated overview"
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
                    "query": marker,
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
            hit.get("uri") == smoke_uri or marker in str(hit.get("content", ""))
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
                "auth_mode": health.get("auth_mode"),
                "smoke_uri": smoke_uri,
                "routes": last_routes,
                "index_size": index_size,
                "semantic_status": write_result["semantic_status"],
                "vector_status": write_result["vector_status"],
                "vlm": {
                    "abstract_characters": len(abstract),
                    "overview_characters": len(overview),
                },
            }
        time.sleep(2)

    raise CheckError(
        "known-result search never used the cuVS route within 180 seconds; "
        f"last routes={last_routes}, last hits={last_hits}"
    )


def check_openviking_and_cuvs_route(run_id: str, marker: str) -> dict[str, Any]:
    smoke_root = f"viking://resources/dgx-spark-openviking-cuvs-smoke-{run_id}"
    smoke_uri = f"{smoke_root}/document.md"
    primary_error: BaseException | None = None
    result: dict[str, Any] | None = None
    cleanup: dict[str, Any] | None = None

    try:
        result = run_openviking_e2e(smoke_root, smoke_uri, marker)
    except BaseException as exc:
        primary_error = exc

    try:
        cleanup = cleanup_smoke_root(smoke_root)
    except BaseException as cleanup_error:
        if primary_error is not None:
            raise CheckError(
                f"E2E failed: {primary_error}; cleanup also failed: {cleanup_error}"
            ) from primary_error
        raise

    if primary_error is not None:
        raise primary_error
    if result is None or cleanup is None:
        raise CheckError("OpenViking E2E returned no result")
    result["cleanup"] = cleanup
    return result


def main() -> int:
    run_id = uuid.uuid4().hex
    marker = f"heliotrope-viking-7319-{run_id} confirms the DGX Spark cuVS route"
    report = {
        "gpu_runtime": check_gpu_runtime(),
        "ollama": check_ollama_models(marker),
        "openviking": check_openviking_and_cuvs_route(run_id, marker),
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
