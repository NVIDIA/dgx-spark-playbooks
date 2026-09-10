#!/usr/bin/env bash
# Regression test: modern OpenShell uses the package-managed gateway service.
# The playbook must not revive the removed k3s lifecycle or replace the
# package service with a plaintext background process.
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

# --- Assertion group 5: docs document the canonical secure flow -------------
GROUP5=0
for DOC in "$PB_DIR/instructions.md" "$PB_DIR/assets/SETUP-GUIDE.md"; do
    if ! grep -q 'ensure_openshell_gateway.sh' "$DOC"; then
        fail "$DOC: missing package-gateway setup helper"
        GROUP5=1
    fi
    if ! grep -q 'systemctl --user' "$DOC"; then
        fail "$DOC: missing systemd user-service lifecycle"
        GROUP5=1
    fi
    if ! grep -q 'https://127\.0\.0\.1:17670' "$DOC"; then
        fail "$DOC: missing local HTTPS endpoint"
        GROUP5=1
    fi
done
if [ "$GROUP5" -eq 0 ]; then
    echo "PASS: docs document package-managed HTTPS/mTLS flow"
fi

# --- Assertion group 6: no insecure/manual gateway lifecycle ----------------
GROUP6=0
for pattern in \
    '^[[:space:]]*nohup[[:space:]]+openshell-gateway' \
    '^[[:space:]]*openshell-gateway.*--disable-tls' \
    '^[[:space:]]*openshell gateway add[[:space:]]+http://127\.0\.0\.1:17670' \
    '^[[:space:]]*pkill.*openshell-gateway'; do
    MATCHES=$(grep_playbook -E "$pattern")
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES"
        fail "insecure or manually owned gateway lifecycle matches: $pattern"
        GROUP6=1
    fi
done
if [ "$GROUP6" -eq 0 ]; then
    echo "PASS: no plaintext or manually backgrounded OpenShell gateway flow"
fi

# --- Assertion group 7: native config is fail-closed -------------------------
CONFIG="$PB_DIR/assets/openshell-gateway.toml"
GROUP7=0
for required in \
    'compute_drivers = \["docker"\]' \
    'disable_tls = false' \
    'allow_unauthenticated_users = false' \
    'enabled = true' \
    'socket_path = "/var/run/docker.sock"'; do
    if ! grep -qE "$required" "$CONFIG" 2>/dev/null; then
        fail "$CONFIG: missing required setting: $required"
        GROUP7=1
    fi
done
if [ "$GROUP7" -eq 0 ]; then
    echo "PASS: native OpenShell config pins Docker, TLS, and mTLS auth"
fi

# --- Result -------------------------------------------------------------------
if [ "$FAILURES" -ne 0 ]; then
    echo "RESULT: FAIL ($FAILURES assertion group(s) failed) for $PB_DIR"
    exit 1
fi
echo "RESULT: PASS for $PB_DIR"
exit 0
