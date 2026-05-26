#!/usr/bin/env bash
# Launch the FastAPI web UI and start Ollama if needed.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
OLLAMA_URL="${OLLAMA_URL:-http://localhost:11434}"

if ! curl -sf -m 3 "$OLLAMA_URL/api/tags" >/dev/null 2>&1; then
    if command -v ollama >/dev/null 2>&1; then
        echo "Starting Ollama..."
        ollama serve >/tmp/rec-sys-ollama.log 2>&1 &
        sleep 3
    else
        echo "Ollama is not installed; explanations will use fallback text."
    fi
fi

exec uv run --project "$REPO_DIR" python "$SCRIPT_DIR/app.py" "$@"
