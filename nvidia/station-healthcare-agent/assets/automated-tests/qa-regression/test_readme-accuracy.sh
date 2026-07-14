#!/usr/bin/env bash
# Regression test: README/doc accuracy fixes from QA.
#   1) DGX OS 7.5.0 ships Node v22, NOT v18 — the playbook must not claim
#      "DGX Station ships with v18" (or v18.19.1) anywhere.
#   2) The documented 'make status' expected output must match the actual
#      format emitted by the Makefile status target:
#         "Ollama (port 11434):     ✓ healthy"
#         "OpenFold3 (port 8000):  ✓ healthy"
#      The old stale format ("Ollama:    ✓ healthy", no port) must be gone.
#
# Usage: test_readme-accuracy.sh [playbook_dir]
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
# grep_playbook <grep-args...> ; prints "<file>:<line>" matches.
grep_playbook() {
    find "$PB_DIR" -type d -name .git -prune -o \
         -type d -name qa-regression -prune -o \
         -type f -print 2>/dev/null \
    | while IFS= read -r f; do
        grep -Hn "$@" "$f" 2>/dev/null
      done
}

# --- Assertion group 1: assets/README.md makes no Node v18 ship claim -------
README="$PB_DIR/assets/README.md"
GROUP1=0
if [ ! -f "$README" ]; then
    fail "$README: file not found"
    GROUP1=1
else
    MATCHES=$(grep -in 'ships with \(node\.js \)\{0,1\}v18' "$README"; \
              grep -n 'v18\.19\.1' "$README")
    if [ -n "$MATCHES" ]; then
        echo "$MATCHES" | while IFS= read -r line; do
            echo "FAIL: $README: claims DGX Station ships Node v18: $line"
        done
        GROUP1=1
    fi
fi
if [ "$GROUP1" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: assets/README.md has no 'DGX Station ships with v18' claim"
fi

# --- Assertion group 2: authoritative 'make status' format in Makefile ------
# Guard: the Makefile status target must emit the per-service health lines
# in the "(port N): ✓ healthy" format. If this ever changes, the doc
# assertions below must be updated too.
MAKEFILE="$PB_DIR/assets/Makefile"
GROUP2=0
if [ ! -f "$MAKEFILE" ]; then
    fail "$MAKEFILE: file not found"
    GROUP2=1
else
    for svc in Ollama OpenFold3; do
        if ! grep -E "echo \"  $svc \(port .*✓ healthy\"" "$MAKEFILE" > /dev/null; then
            fail "$MAKEFILE: status target does not emit '$svc (port ...): ✓ healthy' (authoritative format changed or missing)"
            GROUP2=1
        fi
    done
fi
if [ "$GROUP2" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: Makefile status target emits 'Ollama/OpenFold3 (port N): ✓ healthy' (authoritative format)"
fi

# --- Assertion group 3: documented 'make status' output matches Makefile ----
# instructions.md documents the expected output of 'make status'; it must
# use the current format (service + port + checkmark), with the default
# ports from the Makefile (11434 / 8000).
INSTR="$PB_DIR/instructions.md"
GROUP3=0
if ! grep -E 'Ollama \(port 11434\): +✓ healthy' "$INSTR" > /dev/null; then
    fail "$INSTR: documented 'make status' output missing 'Ollama (port 11434):     ✓ healthy'"
    GROUP3=1
fi
if ! grep -E 'OpenFold3 \(port 8000\): +✓ healthy' "$INSTR" > /dev/null; then
    fail "$INSTR: documented 'make status' output missing 'OpenFold3 (port 8000):  ✓ healthy'"
    GROUP3=1
fi
# No doc anywhere may still show the stale port-less format
# ("Ollama:    ✓ healthy" / "OpenFold3: ✓ healthy").
MATCHES=$(grep_playbook -E '(Ollama|OpenFold3): +✓')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        echo "FAIL: stale port-less 'make status' expected output: $line"
    done
    GROUP3=1
fi
if [ "$GROUP3" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: documented 'make status' expected output matches Makefile format"
fi

# --- Assertion group 4: playbook-wide sweep for Node v18 ship claims ---------
# No file anywhere in the playbook may claim the DGX Station ships Node v18
# (DGX OS 7.5.0 ships v22). Covers overview.md, troubleshooting.md,
# assets/SETUP-GUIDE.md, etc.
GROUP4=0
MATCHES=$(grep_playbook -iE 'ships with (node\.js )?v18')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        echo "FAIL: residual claim that DGX Station ships Node v18: $line"
    done
    GROUP4=1
fi
MATCHES=$(grep_playbook -E 'v18\.19\.1')
if [ -n "$MATCHES" ]; then
    echo "$MATCHES" | while IFS= read -r line; do
        echo "FAIL: residual Node v18.19.1 version claim: $line"
    done
    GROUP4=1
fi
if [ "$GROUP4" -ne 0 ]; then
    FAILURES=$((FAILURES + 1))
else
    echo "PASS: no 'DGX Station ships Node v18' claim anywhere in playbook"
fi

# --- Result -------------------------------------------------------------------
if [ "$FAILURES" -ne 0 ]; then
    echo "RESULT: FAIL ($FAILURES assertion group(s) failed) for $PB_DIR"
    exit 1
fi
echo "RESULT: PASS for $PB_DIR"
exit 0
