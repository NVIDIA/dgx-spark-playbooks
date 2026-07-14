#!/usr/bin/env bash
# Regression test for TICKET 6376168A
#
# BUG: OpenFold3 and NemoClaw's nemoclaw-vllm both use host port 8000, so
#      running OpenFold3 after the NemoClaw playbook fails to bind 8000.
# FIX: OPENFOLD_PORT is documented as the override (=8001):
#      - docker-compose.yml maps the OpenFold3 host port via ${OPENFOLD_PORT:-8000}
#      - instructions.md Step 3 documents the nemoclaw-vllm conflict, the
#        OPENFOLD_PORT=8001 override, and the `docker stop nemoclaw-vllm` remedy
#      - troubleshooting.md has a row mentioning port 8000 + nemoclaw-vllm
#      - .env.example has an OPENFOLD_PORT entry with a NemoClaw/8000 comment
#
# Exit 0 = pass, non-zero = fail.
# macOS bash 3.2 and Linux bash compatible: no mapfile, no declare -A,
# grep -E (not grep -P). No network / docker / openshell.

set -u

# Script lives 3 levels below the playbook root:
#   <root>/assets/automated-tests/qa-regression/test_port-8000-conflict.sh
PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

COMPOSE="$PB_DIR/assets/docker-compose.yml"
INSTRUCTIONS="$PB_DIR/instructions.md"
TROUBLESHOOTING="$PB_DIR/troubleshooting.md"
ENV_EXAMPLE="$PB_DIR/assets/.env.example"

fail() { echo "FAIL: $1: $2"; exit 1; }

# ─── Assertion group 1: docker-compose.yml parameterizes the host port ───────
[ -f "$COMPOSE" ] || fail "$COMPOSE" "file not found"

# The OpenFold3 host port must be parameterized via ${OPENFOLD_PORT:-8000},
# not a hard-coded 8000:8000 mapping.
if ! grep -Eq '"\$\{OPENFOLD_PORT:-8000\}:8000"' "$COMPOSE"; then
  fail "$COMPOSE" 'OpenFold3 host port not mapped via ${OPENFOLD_PORT:-8000}:8000'
fi
# Guard against a regression to a hard-coded host-port mapping.
if grep -Eq '^[[:space:]]*-[[:space:]]*"?8000:8000"?[[:space:]]*$' "$COMPOSE"; then
  fail "$COMPOSE" 'OpenFold3 uses a hard-coded 8000:8000 host-port mapping'
fi
echo "PASS: docker-compose.yml maps OpenFold3 host port via \${OPENFOLD_PORT:-8000}"

# ─── Assertion group 2: instructions.md Step 3 documents the conflict ────────
[ -f "$INSTRUCTIONS" ] || fail "$INSTRUCTIONS" "file not found"

# Extract the Step 3 section (from "# Step 3." up to the next "# Step " heading).
STEP3="$(awk '
  /^# Step 3\./ { grab=1; print; next }
  grab && /^# Step / { exit }
  grab { print }
' "$INSTRUCTIONS")"

[ -n "$STEP3" ] || fail "$INSTRUCTIONS" "could not locate Step 3 section"

# 2a. mentions the nemoclaw-vllm port-8000 conflict
if ! printf '%s\n' "$STEP3" | grep -q 'nemoclaw-vllm'; then
  fail "$INSTRUCTIONS" "Step 3 does not mention the nemoclaw-vllm port-8000 conflict"
fi
if ! printf '%s\n' "$STEP3" | grep -q '8000'; then
  fail "$INSTRUCTIONS" "Step 3 nemoclaw-vllm conflict does not reference port 8000"
fi
# 2b. documents the OPENFOLD_PORT=8001 override
if ! printf '%s\n' "$STEP3" | grep -Eq 'OPENFOLD_PORT=8001'; then
  fail "$INSTRUCTIONS" "Step 3 does not document the OPENFOLD_PORT=8001 override"
fi
# 2c. documents the `docker stop nemoclaw-vllm` remedy
if ! printf '%s\n' "$STEP3" | grep -Eq 'docker stop nemoclaw-vllm'; then
  fail "$INSTRUCTIONS" "Step 3 does not document the 'docker stop nemoclaw-vllm' remedy"
fi
echo "PASS: instructions.md Step 3 documents nemoclaw-vllm conflict, OPENFOLD_PORT=8001 override, and docker stop remedy"

# ─── Assertion group 3: troubleshooting.md has a port-8000 + nemoclaw row ────
[ -f "$TROUBLESHOOTING" ] || fail "$TROUBLESHOOTING" "file not found"

# A table row (starts with '|') mentioning both port 8000 and nemoclaw-vllm.
if ! grep -E '^\|' "$TROUBLESHOOTING" | grep '8000' | grep -q 'nemoclaw-vllm'; then
  fail "$TROUBLESHOOTING" "no table row mentioning port 8000 and nemoclaw-vllm"
fi
echo "PASS: troubleshooting.md has a table row for the port-8000 nemoclaw-vllm conflict"

# ─── Assertion group 4: .env.example OPENFOLD_PORT entry + NemoClaw comment ──
[ -f "$ENV_EXAMPLE" ] || fail "$ENV_EXAMPLE" "file not found"

# There must be an OPENFOLD_PORT assignment.
if ! grep -Eq '^OPENFOLD_PORT=' "$ENV_EXAMPLE"; then
  fail "$ENV_EXAMPLE" "no OPENFOLD_PORT= entry"
fi
# A comment line (starts with '#') referencing the NemoClaw/8000 conflict.
if ! grep -E '^#' "$ENV_EXAMPLE" | grep -Ei 'nemoclaw' | grep -q '8000'; then
  fail "$ENV_EXAMPLE" "OPENFOLD_PORT has no comment referencing the NemoClaw/8000 conflict"
fi
echo "PASS: .env.example has OPENFOLD_PORT with a NemoClaw/8000 conflict comment"

# ─── Assertion group 5: bundled scripts don't hard-code -p 8000:8000 ─────────
MOLVIEW="$PB_DIR/assets/scripts/molecular_viewer.py"
if [ -f "$MOLVIEW" ]; then
  if grep -Eq -- '-p[[:space:]]+"?8000:8000"?' "$MOLVIEW"; then
    fail "$MOLVIEW" "docstring hard-codes '-p 8000:8000' (should honor OPENFOLD_PORT)"
  fi
  echo "PASS: molecular_viewer.py does not hard-code -p 8000:8000"
fi

echo "PASS: TICKET 6376168A port-8000 conflict regression checks all passed"
exit 0
