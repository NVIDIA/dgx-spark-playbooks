#!/bin/sh
# install.sh — Install DGX Station AI Skills into a project for a chosen coding agent.
#
# Usage: ./install.sh <harness> [target-dir] [--force]
#   harness:    claude | codex | gemini | cursor | all
#   target-dir: where to install (default: current directory)
#   --force:    overwrite existing context files (AGENTS.md, CLAUDE.md, GEMINI.md)
#
# Layout produced per harness:
#   claude  -> CLAUDE.md           + .claude/skills/<name>/SKILL.md
#   codex   -> AGENTS.md           + $CODEX_HOME/skills/<name>/SKILL.md
#   gemini  -> GEMINI.md           + .gemini/commands/<name>.md
#   cursor  -> AGENTS.md           + .cursor/rules/<name>.mdc
#   all     -> all of the above

set -eu

usage() {
    cat <<EOF
Usage: $0 <harness> [target-dir] [--force]

Harnesses:
  claude   Claude Code        -> CLAUDE.md  + .claude/skills/<name>/SKILL.md
  codex    OpenAI Codex CLI   -> AGENTS.md  + \$CODEX_HOME/skills/<name>/SKILL.md
  gemini   Gemini CLI         -> GEMINI.md  + .gemini/commands/<name>.md
  cursor   Cursor             -> AGENTS.md  + .cursor/rules/<name>.mdc
  all      Install for all four

Options:
  --force  Overwrite existing context files instead of erroring
EOF
}

if [ $# -lt 1 ]; then
    usage >&2
    exit 2
fi

case "$1" in
    -h|--help) usage; exit 0 ;;
esac

HARNESS="$1"
shift

TARGET="."
FORCE=0
while [ $# -gt 0 ]; do
    case "$1" in
        --force) FORCE=1 ;;
        -h|--help) usage; exit 0 ;;
        *) TARGET="$1" ;;
    esac
    shift
done

case "$HARNESS" in
    claude|codex|gemini|cursor|all) ;;
    *) printf 'Error: unknown harness "%s"\n\n' "$HARNESS" >&2; usage >&2; exit 2 ;;
esac

ASSETS="$(cd "$(dirname "$0")" && pwd)"
SKILLS_DIR="$ASSETS/skills"
AGENTS_MD="$ASSETS/AGENTS.md"

if [ ! -f "$AGENTS_MD" ]; then
    printf 'Error: %s not found\n' "$AGENTS_MD" >&2
    exit 1
fi

if [ ! -d "$SKILLS_DIR" ]; then
    printf 'Error: %s not found\n' "$SKILLS_DIR" >&2
    exit 1
fi

mkdir -p "$TARGET"

SKILL_NAMES="vllm-setup sglang-setup mig-configure dgx-diagnose"

# write_context <target-filename>
# Copies AGENTS.md to <target-dir>/<target-filename>, refusing to overwrite without --force.
write_context() {
    fname="$1"
    dest="$TARGET/$fname"
    if [ -e "$dest" ] && [ "$FORCE" -ne 1 ]; then
        printf '  SKIP %s (exists; pass --force to overwrite)\n' "$dest" >&2
        return 1
    fi
    cp "$AGENTS_MD" "$dest"
    printf '  WROTE %s\n' "$dest"
}

# strip_frontmatter <src> <dest>
# Emits the SKILL.md body (everything after the closing `---`) to <dest>.
# Note: POSIX sh has no local vars; use unique names to avoid clobbering callers.
strip_frontmatter() {
    _sf_src="$1"
    _sf_dest="$2"
    awk 'BEGIN { in_fm=0; past_fm=0 }
         past_fm == 1 { print; next }
         /^---$/ && in_fm == 0 { in_fm=1; next }
         /^---$/ && in_fm == 1 { past_fm=1; next }
         in_fm == 0 && past_fm == 0 { past_fm=1; print }' "$_sf_src" > "$_sf_dest"
}

# write_cursor_rule <src> <dest> <name> <description>
# Writes a Cursor .mdc rule: replaces Anthropic frontmatter with Cursor's shape, keeps the body.
write_cursor_rule() {
    _wc_src="$1"
    _wc_dest="$2"
    _wc_name="$3"
    _wc_desc="$4"
    {
        printf -- '---\n'
        printf 'description: %s\n' "$_wc_desc"
        printf 'globs: ["**/*"]\n'
        printf 'alwaysApply: false\n'
        printf -- '---\n\n'
    } > "$_wc_dest"
    strip_frontmatter "$_wc_src" "$_wc_dest.body"
    cat "$_wc_dest.body" >> "$_wc_dest"
    rm -f "$_wc_dest.body"
}

# extract_description <skill-name>
# Reads the description: line from the skill's SKILL.md frontmatter.
extract_description() {
    _ed_name="$1"
    awk '/^description: / { sub(/^description: /, ""); print; exit }' "$SKILLS_DIR/$_ed_name/SKILL.md"
}

install_claude() {
    printf 'Installing for Claude Code into %s/\n' "$TARGET"
    write_context "CLAUDE.md" || true
    for name in $SKILL_NAMES; do
        dest_dir="$TARGET/.claude/skills/$name"
        dest="$dest_dir/SKILL.md"
        mkdir -p "$dest_dir"
        if [ -e "$dest" ]; then
            printf '  SKIP %s (exists)\n' "$dest" >&2
            continue
        fi
        cp "$SKILLS_DIR/$name/SKILL.md" "$dest"
        printf '  WROTE %s\n' "$dest"
    done
    printf 'Next: cd %s && claude   (type "/" to see vllm-setup, sglang-setup, mig-configure, dgx-diagnose)\n' "$TARGET"
}

install_codex() {
    printf 'Installing for OpenAI Codex CLI into %s/\n' "$TARGET"
    write_context "AGENTS.md" || true
    codex_home="${CODEX_HOME:-$HOME/.codex}"
    codex_skills="$codex_home/skills"
    mkdir -p "$codex_skills"
    for name in $SKILL_NAMES; do
        dest_dir="$codex_skills/$name"
        dest="$dest_dir/SKILL.md"
        if [ -e "$dest" ] && [ "$FORCE" -ne 1 ]; then
            printf '  SKIP %s (exists)\n' "$dest" >&2
            continue
        fi
        mkdir -p "$dest_dir"
        cp -R "$SKILLS_DIR/$name/." "$dest_dir/"
        printf '  WROTE %s\n' "$dest_dir"
    done
    printf 'Next: cd %s && codex   (mention $vllm-setup or "use vllm-setup"; restart Codex if it was already running)\n' "$TARGET"
}

install_gemini() {
    printf 'Installing for Gemini CLI into %s/\n' "$TARGET"
    write_context "GEMINI.md" || true
    mkdir -p "$TARGET/.gemini/commands"
    for name in $SKILL_NAMES; do
        dest="$TARGET/.gemini/commands/$name.md"
        if [ -e "$dest" ]; then
            printf '  SKIP %s (exists)\n' "$dest" >&2
            continue
        fi
        strip_frontmatter "$SKILLS_DIR/$name/SKILL.md" "$dest"
        printf '  WROTE %s\n' "$dest"
    done
    printf 'Next: cd %s && gemini   (type /<name> to invoke a skill)\n' "$TARGET"
}

install_cursor() {
    printf 'Installing for Cursor into %s/\n' "$TARGET"
    write_context "AGENTS.md" || true
    mkdir -p "$TARGET/.cursor/rules"
    for name in $SKILL_NAMES; do
        dest="$TARGET/.cursor/rules/$name.mdc"
        if [ -e "$dest" ]; then
            printf '  SKIP %s (exists)\n' "$dest" >&2
            continue
        fi
        desc="$(extract_description "$name")"
        write_cursor_rule "$SKILLS_DIR/$name/SKILL.md" "$dest" "$name" "$desc"
        printf '  WROTE %s\n' "$dest"
    done
    printf 'Next: open %s in Cursor   (reference rules by name in chat, e.g. "use the vllm-setup rule")\n' "$TARGET"
}

case "$HARNESS" in
    claude)  install_claude  ;;
    codex)   install_codex   ;;
    gemini)  install_gemini  ;;
    cursor)  install_cursor  ;;
    all)
        install_claude
        printf '\n'
        install_codex
        printf '\n'
        install_gemini
        printf '\n'
        install_cursor
        ;;
esac

printf '\nDone.\n'
