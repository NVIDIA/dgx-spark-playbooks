#!/usr/bin/env bash
# Remove dgx-spark-playbooks symlinks from both skill-mode and plugin-mode install targets.
# Only removes symlinks — never touches real directories.

set -euo pipefail

SKILLS_DIR="${CLAUDE_SKILLS:-$HOME/.claude/skills}"
PLUGINS_DIR="${CLAUDE_PLUGINS:-$HOME/.claude/plugins}"

count=0

if [ -d "$SKILLS_DIR" ]; then
  for link in "$SKILLS_DIR/dgx-spark" "$SKILLS_DIR/dgx-spark-"*; do
    [ -L "$link" ] || continue
    rm "$link"
    count=$((count + 1))
  done
fi

plugin_link="$PLUGINS_DIR/dgx-spark-playbooks"
if [ -L "$plugin_link" ]; then
  rm "$plugin_link"
  count=$((count + 1))
fi

echo "✓ Removed $count symlinks"
