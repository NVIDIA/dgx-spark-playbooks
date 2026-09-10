#!/usr/bin/env bash
# Regression test: QA Issue 4 — NemoClaw's openclaw-gateway.service (systemd --user,
# port 18789) must be stopped AND disabled before 'openshell sandbox create', with a
# fuser -k fallback; instructions.md Step 1 must document the manual stop commands.
#
# Usage: test_port-18789-systemd.sh [playbook-dir]
# Exit 0 = pass, non-zero = fail. macOS bash 3.2 compatible.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SETUP="$PB_DIR/assets/scripts/setup_sandbox.sh"
INSTR="$PB_DIR/instructions.md"

FAILURES=0
fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}
pass() {
    echo "PASS: $1"
}

# --- Preconditions: files exist ---
[ -f "$SETUP" ] || { fail "$SETUP: file not found"; }
[ -f "$INSTR" ] || { fail "$INSTR: file not found"; }
if [ "$FAILURES" -ne 0 ]; then
    echo "FAIL: required playbook files missing under $PB_DIR"
    exit 1
fi

# ============================================================
# Group 1: setup_sandbox.sh contains stop + disable + fuser fallback
# ============================================================
G1_OK=1

STOP_LINE=$(grep -nE 'systemctl[[:space:]]+--user[[:space:]]+stop[[:space:]]+openclaw-gateway\.service' "$SETUP" | head -1 | cut -d: -f1)
if [ -z "$STOP_LINE" ]; then
    fail "setup_sandbox.sh: missing 'systemctl --user stop openclaw-gateway.service'"
    G1_OK=0
fi

DISABLE_LINE=$(grep -nE 'systemctl[[:space:]]+--user[[:space:]]+disable[[:space:]]+openclaw-gateway\.service' "$SETUP" | head -1 | cut -d: -f1)
if [ -z "$DISABLE_LINE" ]; then
    fail "setup_sandbox.sh: missing 'systemctl --user disable openclaw-gateway.service'"
    G1_OK=0
fi

FUSER_LINE=$(grep -nE 'fuser[[:space:]]+-k[[:space:]]+.*(\$\{?PORT\}?|18789)' "$SETUP" | head -1 | cut -d: -f1)
if [ -z "$FUSER_LINE" ]; then
    fail "setup_sandbox.sh: missing 'fuser -k' fallback for the gateway port"
    G1_OK=0
fi

if [ "$G1_OK" -eq 1 ]; then
    pass "setup_sandbox.sh stops+disables openclaw-gateway.service and has fuser -k fallback"
fi

# ============================================================
# Group 2: the stop/disable/fuser block runs BEFORE 'openshell sandbox create'
# ============================================================
G2_OK=1

CREATE_LINE=$(grep -nE 'openshell[[:space:]]+sandbox[[:space:]]+create' "$SETUP" | head -1 | cut -d: -f1)
if [ -z "$CREATE_LINE" ]; then
    fail "setup_sandbox.sh: no 'openshell sandbox create' invocation found"
    G2_OK=0
else
    for pair in "stop:$STOP_LINE" "disable:$DISABLE_LINE" "fuser:$FUSER_LINE"; do
        name="${pair%%:*}"
        line="${pair#*:}"
        if [ -n "$line" ] && [ "$line" -ge "$CREATE_LINE" ]; then
            fail "setup_sandbox.sh: '$name' step (line $line) does not precede 'openshell sandbox create' (line $CREATE_LINE)"
            G2_OK=0
        fi
    done
fi

if [ "$G2_OK" -eq 1 ] && [ "$G1_OK" -eq 1 ]; then
    pass "gateway stop/disable/fuser block precedes 'openshell sandbox create' (lines $STOP_LINE/$DISABLE_LINE/$FUSER_LINE < $CREATE_LINE)"
fi

# ============================================================
# Group 3: instructions.md Step 1 documents the manual stop/disable commands
# ============================================================
G3_OK=1

# Extract Step 1 section (from the Step 1 heading up to the Step 2 heading).
STEP1=$(awk '/^#+[[:space:]]*Step 1[.:]/{flag=1} /^#+[[:space:]]*Step 2[.:]/{flag=0} flag' "$INSTR")

if [ -z "$STEP1" ]; then
    fail "instructions.md: could not locate a 'Step 1' section"
    G3_OK=0
else
    if ! printf '%s\n' "$STEP1" | grep -qE 'systemctl[[:space:]]+--user[[:space:]]+stop[[:space:]]+openclaw-gateway\.service'; then
        fail "instructions.md: Step 1 missing 'systemctl --user stop openclaw-gateway.service'"
        G3_OK=0
    fi
    if ! printf '%s\n' "$STEP1" | grep -qE 'systemctl[[:space:]]+--user[[:space:]]+disable[[:space:]]+openclaw-gateway\.service'; then
        fail "instructions.md: Step 1 missing 'systemctl --user disable openclaw-gateway.service'"
        G3_OK=0
    fi
    if ! printf '%s\n' "$STEP1" | grep -q '18789'; then
        fail "instructions.md: Step 1 does not mention port 18789"
        G3_OK=0
    fi
fi

if [ "$G3_OK" -eq 1 ]; then
    pass "instructions.md Step 1 documents stop/disable of openclaw-gateway.service and port 18789"
fi

# ============================================================
if [ "$FAILURES" -ne 0 ]; then
    echo "FAIL: $FAILURES assertion(s) failed (playbook: $PB_DIR)"
    exit 1
fi
echo "PASS: all port-18789 systemd gateway regression checks passed"
exit 0
