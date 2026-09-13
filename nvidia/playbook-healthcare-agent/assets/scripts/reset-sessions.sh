#!/usr/bin/env bash
# Flush all OpenClaw session history and reset token counts.
# Wipes all conversation data, subagent runs, and memory files.
set -euo pipefail

OPENCLAW_DIR="$HOME/.openclaw"

if [ ! -d "$OPENCLAW_DIR" ]; then
  echo "Nothing to reset: $OPENCLAW_DIR does not exist."
  exit 0
fi

# Abort if any openclaw processes are still running
if pgrep -f openclaw >/dev/null 2>&1; then
  echo "ERROR: openclaw processes are still running. Stop them first:" >&2
  pgrep -af openclaw >&2
  exit 1
fi

echo "This will delete all session history, subagent runs, and memory files under $OPENCLAW_DIR."
read -r -p "Continue? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
  echo "Aborted."
  exit 0
fi

AGENTS_DIR="$OPENCLAW_DIR/agents"

for agent_dir in "$AGENTS_DIR"/*/sessions; do
  [ -d "$agent_dir" ] || continue
  find "$agent_dir" -type f -name "*.jsonl*" -delete
  # Reset sessions.json to empty
  echo '{}' > "$agent_dir/sessions.json"
  echo "Cleared $(basename "$(dirname "$agent_dir")")"
done

# Clear subagent run history
mkdir -p "$OPENCLAW_DIR/subagents"
echo '[]' > "$OPENCLAW_DIR/subagents/runs.json"

# Clear accumulated memory files
[ -d "$OPENCLAW_DIR/workspace/memory/" ] && rm -rf "$OPENCLAW_DIR/workspace/memory/" || true
for ws in "$OPENCLAW_DIR/workspaces"/*/; do
  [ -d "${ws}memory/" ] && rm -rf "${ws}memory/" || true
done

echo "All sessions reset. Do /new in the dashboard to start fresh."
