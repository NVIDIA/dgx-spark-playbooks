#!/usr/bin/env bash
# Regression test for QA ticket 6376156
# Older OpenFold3 NIM images crash-loop with NIMProfileIDNotFound when their
# PCI-gated model manifest does not recognize the GPU. The playbook must use the
# exact OpenFold3 1.6.0 image validated on GB300 and must not recommend rewriting
# the signed model manifest as a workaround.
#
# Usage: bash test_openfold3-pci-id.sh [playbook_dir]
# Exit 0 = pass, non-zero = fail.

set -u

# Script lives 3 levels below the playbook root:
#   <pb>/assets/automated-tests/qa-regression/test_openfold3-pci-id.sh
PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

TS="$PB_DIR/troubleshooting.md"
COMPOSE="$PB_DIR/assets/docker-compose.yml"
EXPECTED_IMAGE='nvcr.io/nim/openfold/openfold3:1.6.0@sha256:f2d4a3f2755d8aa7cc4be31853a98688357e0647707aed30d24dfae943592317'

fail() {
  echo "FAIL: $1: $2"
  exit 1
}

# has PATTERN [FILE]  -- grep -E, case-insensitive, fixed via -F-like escaping
# We use grep -E for extended regex; caller escapes regex-special chars.
has() {
  grep -Eiq -- "$1" "$2"
}

[ -f "$TS" ] || fail "troubleshooting.md" "file not found at $TS"
[ -f "$COMPOSE" ] || fail "docker-compose.yml" "file not found at $COMPOSE"

# ---------------------------------------------------------------------------
# Group 1: symptom string documented
# ---------------------------------------------------------------------------
has "NIMProfileIDNotFound" "$TS" \
  || fail "troubleshooting.md" "does not document the symptom 'NIMProfileIDNotFound'"
echo "PASS: symptom 'NIMProfileIDNotFound' is documented"

# ---------------------------------------------------------------------------
# Group 2: the exact validated NIM image is immutable in Compose
# ---------------------------------------------------------------------------
grep -Fq -- "image: $EXPECTED_IMAGE" "$COMPOSE" \
  || fail "docker-compose.yml" "does not pin the validated OpenFold3 1.6.0 digest"
! grep -Eq 'image:[[:space:]]+nvcr\.io/nim/openfold/openfold3:latest([[:space:]]|$)' "$COMPOSE" \
  || fail "docker-compose.yml" "still uses the floating OpenFold3 latest tag"
grep -Fq -- "$EXPECTED_IMAGE" "$PB_DIR/assets/scripts/molecular_viewer.py" \
  || fail "molecular_viewer.py" "standalone example does not use the validated image digest"
! grep -Rq --exclude='test_openfold3-pci-id.sh' 'NIM_OPTIMIZED_BACKEND\|torch_baseline' "$PB_DIR" \
  || fail "OpenFold3 configuration" "still selects the removed PyTorch-only backend"

BAD_IMAGE_REFS=$(grep -RIn --exclude='test_openfold3-pci-id.sh' \
  'nvcr\.io/nim/openfold/openfold3:' "$PB_DIR" 2>/dev/null \
  | grep -Fv -- "$EXPECTED_IMAGE" || true)
[ -z "$BAD_IMAGE_REFS" ] \
  || fail "OpenFold3 image references" "found an unpinned or mismatched image reference: $BAD_IMAGE_REFS"
echo "PASS: OpenFold3 1.6.0 image is pinned by digest and uses its supported optimized backend"

# ---------------------------------------------------------------------------
# Group 3: recovery uses the pinned Compose image
# ---------------------------------------------------------------------------
has "docker compose pull openfold3" "$TS" \
  || fail "troubleshooting.md" "does not tell users to pull the pinned OpenFold3 image"
has "force-recreate openfold3" "$TS" \
  || fail "troubleshooting.md" "does not tell users to replace the stale container"
echo "PASS: recovery flow pulls and recreates the pinned service"

# ---------------------------------------------------------------------------
# Group 4: unsafe manifest rewriting is explicitly prohibited
# ---------------------------------------------------------------------------
has "Do not rewrite or mount over the signed image manifest" "$TS" \
  || fail "troubleshooting.md" "does not prohibit the unvalidated manifest workaround"

MANIFEST_REWRITES=$(grep -RInE --exclude='test_openfold3-pci-id.sh' \
  'docker[[:space:]]+cp.*model_manifest|(^|[[:space:]])(sed|perl|yq)([[:space:]]|$).*model_manifest|model_manifest\.yaml:[^[:space:]]*/opt/nim/[^[:space:]]*model_manifest\.yaml' \
  "$PB_DIR" 2>/dev/null || true)
[ -z "$MANIFEST_REWRITES" ] \
  || fail "OpenFold3 manifest" "found an actionable manifest monkeypatch: $MANIFEST_REWRITES"

BUNDLED_MANIFEST=$(find "$PB_DIR" -type f -name 'model_manifest.yaml' -print -quit)
[ -z "$BUNDLED_MANIFEST" ] \
  || fail "OpenFold3 manifest" "replacement manifest is bundled at $BUNDLED_MANIFEST"
echo "PASS: unvalidated model-manifest rewriting is prohibited"

# ---------------------------------------------------------------------------
# Group 5: readiness and runtime-version verification are documented
# ---------------------------------------------------------------------------
has "/v1/health/ready" "$TS" \
  || fail "troubleshooting.md" "does not verify the OpenFold3 readiness endpoint"
has "/v1/version" "$TS" \
  || fail "troubleshooting.md" "does not verify the running NIM version"
echo "PASS: recovery verifies readiness and the running NIM version"

echo "PASS: ticket 6376156 uses the validated pinned NIM without manifest monkeypatching"
exit 0
