#!/usr/bin/env bash
# Install dgx-spark-playbooks into Claude Code.
#
# Modes:
#   --skills  (default) Symlink each generated skill into ~/.claude/skills/
#   --plugin            Symlink the whole repo into ~/.claude/plugins/ as a plugin
#
# Env overrides:
#   CLAUDE_SKILLS   target for individual skills (default: ~/.claude/skills)
#   CLAUDE_PLUGINS  target for plugin install    (default: ~/.claude/plugins)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SKILLS_DIR="${CLAUDE_SKILLS:-$HOME/.claude/skills}"
PLUGINS_DIR="${CLAUDE_PLUGINS:-$HOME/.claude/plugins}"

MODE="skills"
for arg in "$@"; do
  case "$arg" in
    --skills) MODE="skills" ;;
    --plugin) MODE="plugin" ;;
    -h|--help)
      sed -n '2,11p' "$0" | sed 's/^# \{0,1\}//'
      exit 0
      ;;
    *)
      echo "unknown argument: $arg" >&2
      exit 1
      ;;
  esac
done

# Regenerate skills/ if Node is available; otherwise use what's already committed.
if command -v node >/dev/null 2>&1; then
  NODE_MAJOR=$(node -p "process.versions.node.split('.')[0]")
  if [ "$NODE_MAJOR" -ge 18 ]; then
    echo "→ Regenerating skills/ from overrides/ + nvidia/*/README.md"
    node "$REPO/scripts/generate.mjs"
  else
    echo "  node >= 18 required to regenerate (have $(node -v)) — using existing skills/"
  fi
else
  echo "  node not found — using existing skills/"
fi

if [ ! -d "$REPO/skills" ]; then
  echo "error: $REPO/skills does not exist and could not be regenerated" >&2
  exit 1
fi

# Clean previous installs from BOTH targets so switching modes stays clean.
cleanup_skills() {
  [ -d "$SKILLS_DIR" ] || return 0
  for link in "$SKILLS_DIR/dgx-spark" "$SKILLS_DIR/dgx-spark-"*; do
    [ -L "$link" ] && rm "$link"
  done
}
cleanup_plugin() {
  local link="$PLUGINS_DIR/dgx-spark-playbooks"
  [ -L "$link" ] && rm "$link"
  return 0
}
cleanup_skills
cleanup_plugin

if [ "$MODE" = "plugin" ]; then
  mkdir -p "$PLUGINS_DIR"
  ln -s "$REPO" "$PLUGINS_DIR/dgx-spark-playbooks"
  echo "✓ Installed as plugin: $PLUGINS_DIR/dgx-spark-playbooks → $REPO"
else
  mkdir -p "$SKILLS_DIR"
  count=0
  for dir in "$REPO/skills"/*/; do
    name=$(basename "$dir")
    link="$SKILLS_DIR/$name"
    if [ -e "$link" ] && [ ! -L "$link" ]; then
      echo "  ! $name already exists as a real directory — skipping (remove manually to replace)"
      continue
    fi
    ln -s "$dir" "$link"
    count=$((count + 1))
  done
  echo "✓ Installed $count skills: $SKILLS_DIR/dgx-spark-*"
fi

echo ""
echo "Update:    cd $REPO && git pull && ./scripts/install.sh --$MODE"
echo "Uninstall: $REPO/scripts/uninstall.sh"
