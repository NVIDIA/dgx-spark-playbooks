#!/usr/bin/env python3
"""Verify an OpenViking semantic write and a real cuVS search route."""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import Any


BASE_URL = os.environ.get("OPENVIKING_URL", "http://127.0.0.1:1933").rstrip("/")
API_KEY = os.environ.get("OPENVIKING_API_KEY", "")
EXPECTED_VERSION = "0.4.17.1"


class VerificationError(RuntimeError):
    pass


def request_json(
    method: str,
    path: str,
    payload: dict[str, Any] | None = None,
    *,
    timeout: float = 30,
) -> dict[str, Any]:
    data = None if payload is None else json.dumps(payload).encode()
    headers = {"Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
    if API_KEY:
        headers["X-API-Key"] = API_KEY

    request = urllib.request.Request(
        f"{BASE_URL}{path}", data=data, headers=headers, method=method
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode())
    except urllib.error.HTTPError as exc:
        body = exc.read().decode(errors="replace")
        raise VerificationError(
            f"{method} {path} returned HTTP {exc.code}: {body}"
        ) from exc
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
        raise VerificationError(f"{method} {path} failed: {exc}") from exc


def require_ok(response: dict[str, Any], label: str) -> dict[str, Any]:
    if response.get("status") != "ok":
        raise VerificationError(f"{label} returned: {response}")
    return response


def read_semantic_sidecar(uri: str) -> str:
    query = urllib.parse.urlencode({"uri": uri})
    response = require_ok(request_json("GET", f"/api/v1/content/read?{query}"), uri)
    content = response.get("result")
    if not isinstance(content, str) or len(content.strip()) < 20:
        raise VerificationError(f"{uri} has no generated semantic content")
    if "is not generated" in content.lower() or "is not ready" in content.lower():
        raise VerificationError(f"{uri} is still a placeholder")
    return content.strip()


def delete_smoke_root(root_uri: str) -> None:
    query = urllib.parse.urlencode(
        {"uri": root_uri, "recursive": "true", "wait": "true", "timeout": "300"}
    )
    require_ok(
        request_json("DELETE", f"/api/v1/fs?{query}", timeout=330),
        "smoke cleanup",
    )


def verify(root_uri: str, document_uri: str, marker: str) -> dict[str, Any]:
    health = require_ok(request_json("GET", "/health"), "health")
    if health.get("healthy") is not True:
        raise VerificationError(f"OpenViking is not healthy: {health}")
    if health.get("version") != EXPECTED_VERSION:
        raise VerificationError(
            f"expected OpenViking {EXPECTED_VERSION}, got {health.get('version')}"
        )

    write = require_ok(
        request_json(
            "POST",
            "/api/v1/content/write",
            {
                "uri": document_uri,
                "content": (
                    "# DGX Spark cuVS verification\n\n"
                    f"The unique verification marker is {marker}.\n\n"
                    "This resource checks local semantic processing and dense retrieval.\n"
                ),
                "mode": "create",
                "wait": True,
                "timeout": 600,
                "processing_mode": "semantic_and_vectors",
            },
            timeout=630,
        ),
        "content write",
    )
    write_result = write.get("result", {})
    for field in ("semantic_status", "vector_status"):
        if write_result.get(field) != "complete":
            raise VerificationError(f"write did not complete {field}: {write_result}")

    abstract = read_semantic_sidecar(f"{root_uri}/.abstract.md")
    overview = read_semantic_sidecar(f"{root_uri}/.overview.md")

    deadline = time.monotonic() + 180
    last_routes: dict[str, int] = {}
    last_hits: list[str] = []
    while time.monotonic() < deadline:
        response = require_ok(
            request_json(
                "POST",
                "/api/v1/search/find",
                {
                    "query": marker,
                    "limit": 5,
                    "read_content": True,
                    "telemetry": {"summary": True},
                },
                timeout=30,
            ),
            "semantic search",
        )
        resources = response.get("result", {}).get("resources", [])
        last_hits = [
            str(item.get("uri", "")) for item in resources if isinstance(item, dict)
        ]
        found_marker = any(
            item.get("uri") == document_uri or marker in str(item.get("content", ""))
            for item in resources
            if isinstance(item, dict)
        )
        cuvs = (
            response.get("telemetry", {})
            .get("summary", {})
            .get("vector", {})
            .get("cuvs", {})
        )
        last_routes = cuvs.get("routes", {})
        if found_marker and int(last_routes.get("cuvs", 0)) > 0:
            index_size = int(cuvs.get("index_size_max", 0))
            if index_size < 1:
                raise VerificationError(f"cuVS reported invalid index size: {cuvs}")
            return {
                "version": health.get("version"),
                "semantic_status": write_result["semantic_status"],
                "vector_status": write_result["vector_status"],
                "abstract_characters": len(abstract),
                "overview_characters": len(overview),
                "routes": last_routes,
                "index_size": index_size,
            }
        time.sleep(2)

    raise VerificationError(
        "known-result search did not use cuVS within 180 seconds; "
        f"last routes={last_routes}, last hits={last_hits}"
    )


def main() -> int:
    run_id = uuid.uuid4().hex
    root_uri = f"viking://resources/dgx-spark-cuvs-smoke-{run_id}"
    document_uri = f"{root_uri}/document.md"
    marker = f"heliotrope-viking-{run_id}"
    result: dict[str, Any] | None = None
    primary_error: BaseException | None = None

    try:
        result = verify(root_uri, document_uri, marker)
    except BaseException as exc:
        primary_error = exc

    try:
        delete_smoke_root(root_uri)
    except BaseException as cleanup_error:
        if primary_error is not None:
            raise VerificationError(
                f"verification failed: {primary_error}; cleanup failed: {cleanup_error}"
            ) from primary_error
        raise

    if primary_error is not None:
        raise primary_error
    if result is None:
        raise VerificationError("verification returned no result")

    result["cleanup"] = "complete"
    print(json.dumps(result, indent=2, sort_keys=True))
    print("OPENVIKING_CUVS_E2E_OK")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except VerificationError as exc:
        print(f"OPENVIKING_CUVS_E2E_FAILED: {exc}")
        raise SystemExit(1)
