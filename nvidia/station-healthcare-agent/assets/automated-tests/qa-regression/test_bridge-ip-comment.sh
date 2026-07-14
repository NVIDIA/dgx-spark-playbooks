#!/usr/bin/env bash
# Regression test for QA Issue 5: setup_sandbox.sh header comment claimed the
# GB300 Docker bridge IP is 172.18.0.1, but on DGX GB300WS it is 172.17.0.1.
# The auto-detect logic (ip -4 addr show docker0) was always correct; only the
# comment was wrong. Fix replaced the machine-specific IP mapping comment with
# a note that the bridge IP is auto-detected.
#
# Assertions:
#   1. setup_sandbox.sh contains no 172.18.0.1 claim (comment or otherwise).
#   2. setup_sandbox.sh contains no "GB300 ... Docker bridge <IP>" style
#      machine-to-IP mapping comment.
#   3. setup_sandbox.sh retains the auto-detect logic: ip -4 addr show docker0.
#   4. Sweep: no other file in the playbook claims 172.18.0.1 as a bridge IP.
#
# Usage: bash test_bridge-ip-comment.sh [playbook_dir]
# Exit 0 = pass, non-zero = fail. Portable to macOS bash 3.2.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

if [ ! -d "$PB_DIR" ]; then
    echo "FAIL: $PB_DIR: playbook directory not found"
    exit 2
fi

SETUP="$PB_DIR/assets/scripts/setup_sandbox.sh"
FAILURES=0

# --- Assertion group 1: no 172.18.0.1 claim in setup_sandbox.sh -------------
if [ ! -f "$SETUP" ]; then
    echo "FAIL: $SETUP: file not found"
    exit 2
fi

if grep -n '172\.18\.0\.1' "$SETUP" >/dev/null 2>&1; then
    grep -n '172\.18\.0\.1' "$SETUP" | while IFS= read -r line; do
        echo "FAIL: $SETUP: claims incorrect bridge IP 172.18.0.1 (actual on GB300WS is 172.17.0.1): $line"
    done
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: setup_sandbox.sh contains no 172.18.0.1 claim"
fi

# --- Assertion group 2: no hardcoded machine->bridge-IP mapping comment -----
# Matches lines like "#   GB300:  Docker bridge 172.x.0.1" i.e. a comment that
# pins a specific bridge IP to a machine type instead of auto-detecting.
if grep -niE '^[[:space:]]*#.*(GB300|new station).*(docker[[:space:]]+bridge|bridge[[:space:]]+ip).*[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$SETUP" >/dev/null 2>&1; then
    grep -niE '^[[:space:]]*#.*(GB300|new station).*(docker[[:space:]]+bridge|bridge[[:space:]]+ip).*[0-9]+\.[0-9]+\.[0-9]+\.[0-9]+' "$SETUP" | while IFS= read -r line; do
        echo "FAIL: $SETUP: comment pins a specific bridge IP to a machine type (should say auto-detected): $line"
    done
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: setup_sandbox.sh has no machine-specific bridge IP mapping comment"
fi

# --- Assertion group 3: auto-detect logic present ----------------------------
if grep -E 'ip[[:space:]]+-4[[:space:]]+addr[[:space:]]+show[[:space:]]+docker0' "$SETUP" >/dev/null 2>&1; then
    echo "PASS: setup_sandbox.sh auto-detects bridge IP via 'ip -4 addr show docker0'"
else
    echo "FAIL: $SETUP: auto-detect logic 'ip -4 addr show docker0' is missing"
    FAILURES=$((FAILURES + 1))
fi

# --- Assertion group 4: sweep rest of playbook for the same wrong claim -----
SWEEP_HITS=0
SWEEP_TMP="$(mktemp)"
# Text files only; skip this test directory itself and any .git dir.
find "$PB_DIR" -type f \
    ! -path '*/.git/*' \
    ! -path '*/automated-tests/qa-regression/*' \
    \( -name '*.sh' -o -name '*.py' -o -name '*.md' -o -name '*.yaml' \
       -o -name '*.yml' -o -name '*.json' -o -name '*.txt' -o -name 'Makefile' \
       -o -name '*.toml' -o -name '*.cfg' -o -name '*.env' -o -name '*.env.example' \) \
    -print > "$SWEEP_TMP"

while IFS= read -r f; do
    if grep -n '172\.18\.0\.1' "$f" >/dev/null 2>&1; then
        grep -n '172\.18\.0\.1' "$f" | while IFS= read -r line; do
            echo "FAIL: $f: claims incorrect bridge IP 172.18.0.1 (GB300WS bridge is 172.17.0.1): $line"
        done
        SWEEP_HITS=$((SWEEP_HITS + 1))
    fi
done < "$SWEEP_TMP"
rm -f "$SWEEP_TMP"

if [ "$SWEEP_HITS" -eq 0 ]; then
    echo "PASS: no other playbook file claims bridge IP 172.18.0.1"
else
    FAILURES=$((FAILURES + 1))
fi

# --- Result ------------------------------------------------------------------
if [ "$FAILURES" -gt 0 ]; then
    echo "RESULT: FAIL ($FAILURES assertion group(s) failed)"
    exit 1
fi
echo "RESULT: PASS (all assertion groups passed)"
exit 0
