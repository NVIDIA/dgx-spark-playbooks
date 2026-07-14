#!/usr/bin/env bash
# Regression test: OpenShell >= 0.0.44 removed 'openshell gateway start'
# (the old k3s-based flow). The playbook must use the standalone
# 'openshell-gateway' binary + 'openshell gateway add' flow (port 17670),
# and contain no stale k3s-era guidance or gateway port 8080 references.
#
# Usage: test_gateway-start-removed.sh [playbook_dir]
# Exit 0 = pass, non-zero = fail.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

if [ ! -d "$PB_DIR" ]; then
    echo "FAIL: $PB_DIR: playbook directory does not exist"
    exit 2
fi
if [ ! -f "$PB_DIR/instructions.md" ]; then
    echo "FAIL: $PB_DIR: instructions.md not found (not a playbook dir?)"
    exit 2
fi

FAILURES=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

# Search all regular files in the playbook, excluding .git and this test's
# own directory (the test file quotes the forbidden strings).
# grep_playbook <grep-args...> ; prints matches, returns grep's status.
grep_playbook() {
    find "$PB_DIR" -type d -name .git -prune -o \
         -type d -name qa-regression -prune -o \
         -type f -print 2>/dev/null \
    | while IFS= read -r f; do
        grep -H "$@" "$f" 2>/dev/null
      done
}

# --- Assertion group 1: no stale 'openshell gateway start' command ----------
# Match the literal command only, not prose like "gateway started"
# ('openshell gateway start' followed by a non-letter or end of line).
MATCHES=$(grep_playbook -E 'openshell gateway start([^a-zA-Z]|$)')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        fail "stale 'openshell gateway start' command: $line"
    done
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: no 'openshell gateway start' command anywhere in playbook"
fi

# --- Assertion group 2: no k3s-era tokens ------------------------------------
GROUP2=0
for token in 'OPENSHELL_K3S_ARGS' 'cgroup-driver' 'kubelet-arg'; do
    MATCHES=$(grep_playbook -F "$token")
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES" | while IFS= read -r line; do
            fail "stale k3s-era token '$token': $line"
        done
        GROUP2=1
    fi
done
# 'k3s' as a standalone token (word boundary), any case
MATCHES=$(grep_playbook -iE '(^|[^a-zA-Z0-9])k3s([^a-zA-Z0-9]|$)')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        fail "stale k3s reference: $line"
    done
    GROUP2=1
fi
if [ "$GROUP2" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: no k3s-era tokens (OPENSHELL_K3S_ARGS, cgroup-driver, kubelet-arg, k3s)"
fi

# --- Assertion group 3: no 'gateway destroy' subcommand ----------------------
MATCHES=$(grep_playbook -E 'gateway destroy([^a-zA-Z]|$)')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        fail "stale 'gateway destroy' subcommand: $line"
    done
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: no 'gateway destroy' subcommand anywhere in playbook"
fi

# --- Assertion group 4: no gateway port 8080 references ----------------------
MATCHES=$(grep_playbook -E '(^|[^0-9])8080([^0-9]|$)')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        fail "stale gateway port 8080 reference: $line"
    done
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: no port 8080 references anywhere in playbook"
fi

# --- Assertion group 5: instructions.md documents the new canonical flow -----
INSTR="$PB_DIR/instructions.md"
GROUP5=0
if ! grep -q 'openshell-gateway' "$INSTR"; then
    fail "$INSTR: missing standalone 'openshell-gateway' binary invocation"
    GROUP5=1
fi
if ! grep -q -- '--drivers docker' "$INSTR"; then
    fail "$INSTR: missing '--drivers docker' flag for openshell-gateway"
    GROUP5=1
fi
if ! grep -q 'openshell gateway add' "$INSTR"; then
    fail "$INSTR: missing 'openshell gateway add' registration step"
    GROUP5=1
fi
if [ "$GROUP5" -eq 0 ]; then
    echo "PASS: instructions.md documents openshell-gateway --drivers docker + openshell gateway add"
fi

# --- Result -------------------------------------------------------------------
if [ "$FAILURES" -ne 0 ]; then
    echo "RESULT: FAIL ($FAILURES assertion group(s) failed) for $PB_DIR"
    exit 1
fi
echo "RESULT: PASS for $PB_DIR"
exit 0
