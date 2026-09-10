#!/usr/bin/env bash
# Regression test for QA Issue 3: _sandbox() hard-coded '--gateway-name openshell'
# in the 'openshell ssh-proxy' ProxyCommand, which fails with
# "Unknown gateway 'openshell'" when the active gateway has another name
# (e.g. 'nemoclaw' from a prior NemoClaw playbook install).
#
# Fix under test: a _gw_name() helper that resolves the gateway name with the
# precedence: OPENSHELL_GATEWAY env var -> parse 'openshell status' -> 'openshell'.
#
# Usage: bash test_gateway-name-hardcoded.sh [path-to-playbook-dir]
# Exit 0 = pass, non-zero = fail.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

FAILURES=0

fail() {
    # fail <file> <reason>
    echo "FAIL: $1: $2"
    FAILURES=$((FAILURES + 1))
}

pass() {
    echo "PASS: $1"
}

if [ ! -d "$PB_DIR" ]; then
    fail "$PB_DIR" "playbook directory not found"
    exit 1
fi

SETUP_SH="$PB_DIR/assets/scripts/setup_sandbox.sh"

# ---------------------------------------------------------------------------
# 1. Static: no hard-coded '--gateway-name openshell' at executable call sites
#    (Makefile + shell scripts). Docs may legitimately show example output, so
#    only executable assets are scanned.
# ---------------------------------------------------------------------------
static_ok=1
call_sites=""
if [ -f "$PB_DIR/assets/Makefile" ]; then
    call_sites="$PB_DIR/assets/Makefile"
fi
for f in "$PB_DIR"/assets/scripts/*.sh; do
    [ -f "$f" ] || continue
    call_sites="$call_sites $f"
done

if [ -z "$call_sites" ]; then
    fail "$PB_DIR" "no executable call sites found (assets/Makefile, assets/scripts/*.sh)"
    static_ok=0
else
    for f in $call_sites; do
        if grep -n -- '--gateway-name openshell' "$f" >/dev/null 2>&1; then
            line=$(grep -n -- '--gateway-name openshell' "$f" | head -1)
            fail "$f" "hard-coded '--gateway-name openshell' still present (line ${line%%:*}); must use the _gw_name helper / resolved gateway name"
            static_ok=0
        fi
    done
fi
if [ "$static_ok" -eq 1 ]; then
    pass "no hard-coded '--gateway-name openshell' at executable call sites"
fi

# ---------------------------------------------------------------------------
# 2. Static: _gw_name helper exists and references OPENSHELL_GATEWAY
# ---------------------------------------------------------------------------
helper_ok=1
if [ ! -f "$SETUP_SH" ]; then
    fail "$SETUP_SH" "setup_sandbox.sh not found"
    helper_ok=0
elif ! grep -E '^[[:space:]]*_gw_name\(\)' "$SETUP_SH" >/dev/null 2>&1; then
    fail "$SETUP_SH" "_gw_name() helper not defined"
    helper_ok=0
else
    if ! grep -E 'OPENSHELL_GATEWAY' "$SETUP_SH" >/dev/null 2>&1; then
        fail "$SETUP_SH" "_gw_name helper does not reference OPENSHELL_GATEWAY env var"
        helper_ok=0
    fi
    if ! grep -E 'openshell status' "$SETUP_SH" >/dev/null 2>&1; then
        fail "$SETUP_SH" "_gw_name helper does not consult 'openshell status' for the active gateway"
        helper_ok=0
    fi
    # The _sandbox ProxyCommand must use the resolved name, not a literal.
    if ! grep -E -- '--gateway-name +"?\$\{?GW_NAME' "$SETUP_SH" >/dev/null 2>&1; then
        fail "$SETUP_SH" "ssh-proxy ProxyCommand does not use the resolved \$GW_NAME"
        helper_ok=0
    fi
fi
if [ "$helper_ok" -eq 1 ]; then
    pass "_gw_name helper exists, references OPENSHELL_GATEWAY, parses 'openshell status', and ProxyCommand uses \$GW_NAME"
fi

# ---------------------------------------------------------------------------
# 3. Behavioral: extract _gw_name and verify the resolution order with a
#    mocked 'openshell' shim on PATH. Skipped if the helper is absent
#    (already reported as a static failure above).
# ---------------------------------------------------------------------------
if [ "$helper_ok" -eq 1 ]; then
    TMPDIR_T=$(mktemp -d "${TMPDIR:-/tmp}/gwname-test.XXXXXX") || {
        fail "$0" "could not create temp dir"
        exit 1
    }
    trap 'rm -rf "$TMPDIR_T"' EXIT

    # Extract the helper function body (from '_gw_name() {' to the first '}'
    # at column 1 or matching indent).
    awk '/^[[:space:]]*_gw_name\(\)[[:space:]]*\{/ {infn=1} infn {print} infn && /^[[:space:]]*\}[[:space:]]*$/ && !/\(\)/ {exit}' \
        "$SETUP_SH" > "$TMPDIR_T/gw_name_fn.sh"

    if ! grep -E '_gw_name\(\)' "$TMPDIR_T/gw_name_fn.sh" >/dev/null 2>&1; then
        fail "$SETUP_SH" "could not extract _gw_name() function body for behavioral test"
    else
        # Mock 'openshell' that reports a NemoClaw gateway in 'status'.
        mkdir -p "$TMPDIR_T/bin"
        cat > "$TMPDIR_T/bin/openshell" <<'MOCK'
#!/usr/bin/env bash
if [ "${1:-}" = "status" ]; then
    echo "OpenShell status"
    echo "  Gateway: nemoclaw (active)"
    exit 0
fi
exit 1
MOCK
        chmod +x "$TMPDIR_T/bin/openshell"

        # Mock 'openshell' that fails entirely (gateway detection unavailable).
        mkdir -p "$TMPDIR_T/bin-broken"
        cat > "$TMPDIR_T/bin-broken/openshell" <<'MOCK'
#!/usr/bin/env bash
exit 1
MOCK
        chmod +x "$TMPDIR_T/bin-broken/openshell"

        run_gw_name() {
            # run_gw_name <PATH-prefix> <OPENSHELL_GATEWAY value or ''>
            env PATH="$1:/usr/bin:/bin" OPENSHELL_GATEWAY="$2" \
                bash -c 'unset OPENSHELL_GATEWAY_UNSET
                         if [ -z "${OPENSHELL_GATEWAY}" ]; then unset OPENSHELL_GATEWAY; fi
                         . "$1"; _gw_name' _ "$TMPDIR_T/gw_name_fn.sh" 2>/dev/null
        }

        # (a) Env var wins even when 'openshell status' reports another name.
        got=$(run_gw_name "$TMPDIR_T/bin" "custom-gw")
        if [ "$got" = "custom-gw" ]; then
            pass "behavioral: OPENSHELL_GATEWAY env var wins ('custom-gw')"
        else
            fail "$SETUP_SH" "_gw_name ignored OPENSHELL_GATEWAY env var (got '$got', want 'custom-gw')"
        fi

        # (b) Env unset: parses mocked 'openshell status' -> 'nemoclaw'.
        got=$(run_gw_name "$TMPDIR_T/bin" "")
        if [ "$got" = "nemoclaw" ]; then
            pass "behavioral: parses 'openshell status' active gateway ('nemoclaw')"
        else
            fail "$SETUP_SH" "_gw_name did not parse gateway from 'openshell status' (got '$got', want 'nemoclaw')"
        fi

        # (c) Env unset + 'openshell' unusable: falls back to 'openshell'.
        got=$(run_gw_name "$TMPDIR_T/bin-broken" "")
        if [ "$got" = "openshell" ]; then
            pass "behavioral: falls back to 'openshell' when detection unavailable"
        else
            fail "$SETUP_SH" "_gw_name fallback wrong (got '$got', want 'openshell')"
        fi
    fi
fi

# ---------------------------------------------------------------------------
echo ""
if [ "$FAILURES" -gt 0 ]; then
    echo "RESULT: $FAILURES failure(s) in $PB_DIR"
    exit 1
fi
echo "RESULT: all gateway-name regression checks passed in $PB_DIR"
exit 0
