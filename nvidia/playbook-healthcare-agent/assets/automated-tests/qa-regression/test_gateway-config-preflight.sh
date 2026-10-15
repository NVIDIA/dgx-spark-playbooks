#!/usr/bin/env bash
# Exercise the shipped TOML with a real OpenShell gateway parser. This test
# never downloads a binary, starts a gateway, or changes the user's config.
# Usage: OPENSHELL_GATEWAY_TEST_BINARY=/path/to/openshell-gateway bash "$0"
set -euo pipefail

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
GATEWAY_BINARY="${OPENSHELL_GATEWAY_TEST_BINARY:-$(command -v openshell-gateway || true)}"
[[ -n "$GATEWAY_BINARY" && -x "$GATEWAY_BINARY" ]] || {
    echo "FAIL: real openshell-gateway binary required; set OPENSHELL_GATEWAY_TEST_BINARY to its executable path" >&2
    exit 2
}
CONFIG="$PB_DIR/assets/openshell-gateway.toml"
[[ -f "$CONFIG" ]] || { echo "FAIL: missing $CONFIG" >&2; exit 2; }

TMP_DIR="$(mktemp -d)"
trap 'rm -rf "$TMP_DIR"' EXIT
cp "$CONFIG" "$TMP_DIR/shipped.toml"
cp "$CONFIG" "$TMP_DIR/explicit-mtls.toml"
printf '\n[openshell.gateway.mtls_auth]\nenabled = true\n' >>"$TMP_DIR/explicit-mtls.toml"
printf '[openshell]\nversion = [\n' >"$TMP_DIR/invalid.toml"

# An empty environment prevents operator-provided certificate paths or config
# overrides from hiding missing fields in the fixture.
if ! env -i "$GATEWAY_BINARY" config preflight --path "$TMP_DIR/shipped.toml" \
    >"$TMP_DIR/output" 2>&1; then
    cat "$TMP_DIR/output" >&2
    echo "FAIL: shipped gateway configuration failed real parser preflight" >&2
    exit 1
fi
cmp -s "$CONFIG" "$TMP_DIR/shipped.toml" || {
    echo "FAIL: preflight modified the shipped configuration" >&2
    exit 1
}
echo "PASS: shipped gateway configuration passes real parser preflight with an empty environment"

for fixture in explicit-mtls invalid; do
    cp "$TMP_DIR/$fixture.toml" "$TMP_DIR/preimage.toml"
    if env -i "$GATEWAY_BINARY" config preflight --path "$TMP_DIR/$fixture.toml" \
        >"$TMP_DIR/output" 2>&1; then
        echo "FAIL: real parser accepted $fixture configuration" >&2
        exit 1
    fi
    grep -Fq 'category=malformed' "$TMP_DIR/output" || {
        cat "$TMP_DIR/output" >&2
        echo "FAIL: $fixture was not rejected as malformed configuration" >&2
        exit 1
    }
    cmp -s "$TMP_DIR/preimage.toml" "$TMP_DIR/$fixture.toml" || {
        echo "FAIL: preflight modified the rejected $fixture configuration" >&2
        exit 1
    }
    echo "PASS: real parser rejects $fixture configuration without modifying it"
done
