#!/usr/bin/env bash
# Behavioral regression test for QA ticket 6261462 (Issue 3), ANSI variant.
#
# _gw_name() must resolve a non-default active gateway even when `openshell
# status` colorizes its output. The real CLI prints the line as
#   "  <ESC>[2mGateway:<ESC>[0m nemoclaw"
# and the reset code between "Gateway:" and the name defeated the grep, so
# _gw_name silently fell back to 'openshell' -> "Unknown gateway 'openshell'"
# (the exact failure Issue 3 was meant to fix). Found on a real DGX Station
# (aarch64 GB300) whose active OpenShell gateway was named 'nemoclaw'.
#
# This test mocks a color-emitting `openshell` and asserts the fix. It fails on
# any version of test-lib.sh that lacks the ANSI-stripping step (pre-fix and the
# first _gw_name implementation), and passes once the strip is present.
#
# Exit 0 = pass, non-zero = fail.
# macOS bash 3.2 and Linux bash compatible: no mapfile/declare -A/grep -P.
# The ANSI strip uses a literal ESC so it works under both GNU and BSD sed.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
TESTLIB="$PB_DIR/assets/scripts/test-lib.sh"

fail() { echo "FAIL: $1"; [ -n "${MOCKDIR:-}" ] && rm -rf "$MOCKDIR"; exit 1; }

[ -f "$TESTLIB" ] || fail "test-lib.sh not found at $TESTLIB"

# Mock `openshell` whose `status` output is colorized like the real CLI and
# reports a NON-default active gateway.
MOCKDIR=$(mktemp -d)
cat > "$MOCKDIR/openshell" <<'MOCK'
#!/usr/bin/env bash
if [ "${1:-}" = "status" ]; then
  printf '\033[1m\033[36mServer Status\033[39m\033[0m\n\n'
  printf '  \033[2mGateway:\033[0m nemoclaw\n'
  printf '  \033[2mServer:\033[0m http://127.0.0.1:8080\n'
  exit 0
fi
exit 0
MOCK
chmod +x "$MOCKDIR/openshell"

# 1) With env unset, _gw_name must parse the colorized status and return 'nemoclaw'.
GW=$(PATH="$MOCKDIR:$PATH"; unset OPENSHELL_GATEWAY GW_NAME; set +u; . "$TESTLIB" >/dev/null 2>&1; printf '%s' "${GW_NAME:-}")
[ "$GW" = "nemoclaw" ] \
  || fail "_gw_name resolved active gateway to '$GW', expected 'nemoclaw' (ANSI-colored 'openshell status' not parsed)"
echo "PASS: _gw_name parses ANSI-colored 'openshell status' and resolves active gateway 'nemoclaw'"

# 2) OPENSHELL_GATEWAY env override must still win.
GW2=$(PATH="$MOCKDIR:$PATH"; export OPENSHELL_GATEWAY=custom-xyz; unset GW_NAME; set +u; . "$TESTLIB" >/dev/null 2>&1; printf '%s' "${GW_NAME:-}")
[ "$GW2" = "custom-xyz" ] \
  || fail "OPENSHELL_GATEWAY override not honored (got '$GW2', expected 'custom-xyz')"
echo "PASS: OPENSHELL_GATEWAY override honored over status parsing"

rm -rf "$MOCKDIR"
echo "PASS: ticket 6261462 Issue-3 gateway-name resolution robust to ANSI-colored output"
exit 0
