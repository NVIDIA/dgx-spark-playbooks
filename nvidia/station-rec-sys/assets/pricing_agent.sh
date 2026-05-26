#!/usr/bin/env bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
: "${PLAYBOOK_WORKSPACE:=$HOME}"
export PLAYBOOK_WORKSPACE
exec uv run --project "$REPO_DIR" python "$SCRIPT_DIR/pricing_agent.py" "$@"
