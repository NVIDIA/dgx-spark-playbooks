#!/usr/bin/env bash
# Regression test: openshell sandbox upload destination must be /sandbox/
#
# openshell >= 0.0.44 changed `sandbox upload SRC DEST` to place SRC *inside*
# DEST (like cp -r), instead of copying its contents. Uploading the repo to
# /sandbox/clinical-intelligence therefore nested the tree at
# /sandbox/clinical-intelligence/clinical-intelligence. The fix (Step 4 of
# setup_sandbox.sh) uploads to /sandbox/ so the source directory lands at
# /sandbox/clinical-intelligence.
#
# This test statically asserts that every non-comment `openshell sandbox
# upload` invocation in setup_sandbox.sh uses /sandbox/ (or /sandbox) as the
# destination argument, and NOT /sandbox/clinical-intelligence.
#
# Usage: bash test_upload-nesting.sh [PLAYBOOK_DIR]
# Exit 0 = pass, non-zero = fail. Portable to macOS bash 3.2 and Linux bash.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
SCRIPT_REL="assets/scripts/setup_sandbox.sh"
SCRIPT="$PB_DIR/$SCRIPT_REL"
FAILURES=0

fail() {
    echo "FAIL: $SCRIPT_REL: $1"
    FAILURES=$((FAILURES + 1))
}

if [ ! -f "$SCRIPT" ]; then
    echo "FAIL: $SCRIPT_REL: file not found under $PB_DIR"
    exit 1
fi

# Collect non-comment lines containing an `openshell sandbox upload` command.
# Comments in the script mention "sandbox upload" in prose, so strip lines
# whose first non-whitespace character is '#'.
UPLOAD_LINES=$(grep -n 'openshell[[:space:]]\{1,\}sandbox[[:space:]]\{1,\}upload' "$SCRIPT" \
    | grep -v -E '^[0-9]+:[[:space:]]*#' || true)

if [ -z "$UPLOAD_LINES" ]; then
    fail "no 'openshell sandbox upload' invocation found (expected exactly at least one in Step 4)"
    exit 1
fi

COUNT=0
# Iterate line-by-line without mapfile (bash 3.2 compatible).
while IFS= read -r ENTRY; do
    [ -n "$ENTRY" ] || continue
    COUNT=$((COUNT + 1))
    LINENO_TAG=${ENTRY%%:*}
    LINE=${ENTRY#*:}

    # Destination is the last whitespace-separated token of the invocation.
    # Strip trailing comments, then take the last field and remove quotes.
    DEST=$(printf '%s\n' "$LINE" \
        | sed -e 's/[[:space:]]#.*$//' \
        | awk '{print $NF}' \
        | sed -e 's/^"//' -e 's/"$//' -e "s/^'//" -e "s/'\$//")

    case "$DEST" in
        /sandbox/clinical-intelligence|/sandbox/clinical-intelligence/*)
            fail "line $LINENO_TAG: upload destination is '$DEST'; openshell >= 0.0.44 nests SRC inside DEST, producing /sandbox/clinical-intelligence/clinical-intelligence. Destination must be /sandbox/"
            ;;
        /sandbox|/sandbox/)
            : # correct
            ;;
        *)
            fail "line $LINENO_TAG: unexpected upload destination '$DEST' (expected /sandbox/ or /sandbox)"
            ;;
    esac
done <<EOF_LINES
$UPLOAD_LINES
EOF_LINES

if [ "$FAILURES" -eq 0 ]; then
    echo "PASS: $COUNT 'openshell sandbox upload' invocation(s) target /sandbox/ (no /sandbox/clinical-intelligence nesting destination)"
    exit 0
fi

echo "FAIL: $FAILURES assertion(s) failed in $SCRIPT_REL"
exit 1
