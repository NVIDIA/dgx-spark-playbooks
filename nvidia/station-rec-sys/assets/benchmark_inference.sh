#!/usr/bin/env bash
# Run the recommendation throughput benchmark.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

exec uv run --project "$REPO_DIR" python "$SCRIPT_DIR/benchmark_throughput.py" "$@"
