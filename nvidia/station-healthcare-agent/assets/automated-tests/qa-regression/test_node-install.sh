#!/usr/bin/env bash
# Regression test: Node.js install must NOT pipe the NodeSource setup script
# directly into (sudo) bash, and must handle the "command not found" case
# (some DGX Station configs ship with no Node.js at all).
#
# Usage: test_node-install.sh [path-to-playbook-dir]
# Exit 0 = pass, non-zero = fail.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
INSTRUCTIONS="$PB_DIR/instructions.md"
FAILURES=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

if [ ! -d "$PB_DIR" ]; then
    fail "$PB_DIR: playbook directory not found"
    exit 1
fi

# Any line that references the NodeSource setup script and pipes it into
# bash (covers plain '|', markdown-escaped '\|', and 'sudo -E bash -').
PIPE_PATTERN='deb\.nodesource\.com/setup[^|]*\|.*bash'

# ---------------------------------------------------------------------------
# Group 1: instructions.md Step 1 — Node.js install method
# ---------------------------------------------------------------------------
if [ ! -f "$INSTRUCTIONS" ]; then
    fail "$INSTRUCTIONS: file not found"
else
    G1_OK=1

    # 1a. Must not pipe the NodeSource script into bash.
    if grep -En "$PIPE_PATTERN" "$INSTRUCTIONS" >/dev/null; then
        fail "instructions.md: pipes deb.nodesource.com/setup directly into bash (curl | sudo bash anti-pattern)"
        G1_OK=0
    fi

    # 1b. Must download the setup script to a file first (curl ... -o <file>).
    if ! grep -E 'curl[^|]*deb\.nodesource\.com/setup[^|]*[[:space:]]-o[[:space:]]' "$INSTRUCTIONS" >/dev/null; then
        fail "instructions.md: NodeSource setup script is not downloaded to a file first (no 'curl ... deb.nodesource.com/setup... -o <file>')"
        G1_OK=0
    fi

    # 1c. The downloaded file must then be executed with sudo bash <file>.
    if ! grep -E 'sudo([[:space:]]+-E)?[[:space:]]+bash[[:space:]]+[^|<-][^|]*nodesource' "$INSTRUCTIONS" >/dev/null; then
        fail "instructions.md: downloaded NodeSource script is not executed via 'sudo bash <file>'"
        G1_OK=0
    fi

    # 1d. Must cover the no-Node-at-all case ('command not found'), not just
    #     an old v18 install.
    if ! grep -E 'node --version.*command not found' "$INSTRUCTIONS" >/dev/null; then
        fail "instructions.md: Node.js install trigger does not cover the 'command not found' case (systems shipping with no Node.js)"
        G1_OK=0
    fi

    if [ "$G1_OK" -eq 1 ]; then
        echo "PASS: instructions.md downloads NodeSource setup script to a file (no curl|sudo bash) and covers 'command not found'"
    fi
fi

# ---------------------------------------------------------------------------
# Group 2: whole-playbook sweep for the curl|bash anti-pattern
# ---------------------------------------------------------------------------
SWEEP_HITS=0
# bash 3.2 compatible: iterate find output line by line (no mapfile).
while IFS= read -r f; do
    HITS=$(grep -En "$PIPE_PATTERN" "$f" 2>/dev/null)
    if [ -n "$HITS" ]; then
        REL="${f#"$PB_DIR"/}"
        # one FAIL line per matching line
        while IFS= read -r line; do
            fail "$REL: NodeSource setup script piped into bash -> $line"
            SWEEP_HITS=$((SWEEP_HITS + 1))
        done <<EOF
$HITS
EOF
    fi
done <<EOF2
$(find "$PB_DIR" \( -name '*.md' -o -name '*.sh' \) -type f | grep -v '/assets/automated-tests/')
EOF2

if [ "$SWEEP_HITS" -eq 0 ]; then
    echo "PASS: no file in the playbook pipes deb.nodesource.com/setup into bash"
fi

# ---------------------------------------------------------------------------
if [ "$FAILURES" -gt 0 ]; then
    echo "RESULT: $FAILURES failure(s)"
    exit 1
fi
echo "RESULT: all checks passed"
exit 0
