#!/usr/bin/env bash
# Regression test for QA ticket 6376156
# OpenFold3 NIM crash-loops with NIMProfileIDNotFound when the GPU's PCI id is
# absent from its bundled model_manifest.yaml. GB300 board SKUs vary: some report
# 31c2:10de (listed -> native match) while others report 31c3:10de (absent ->
# crash); the RTX PRO 6000 2bb4:10de is absent too. (Confirmed on a real aarch64
# GB300 whose manifest lists 31c2 but not 31c3/2bb4.)
# Docs-only fix: troubleshooting.md must document the symptom, the PCI-ID cause,
# and a coherent id-agnostic manifest-patch workaround, and name it an upstream
# NIM issue. This test asserts that documentation exists and is correct.
#
# Usage: bash test_openfold3-pci-id.sh [playbook_dir]
# Exit 0 = pass, non-zero = fail.

set -u

# Script lives 3 levels below the playbook root:
#   <pb>/assets/automated-tests/qa-regression/test_openfold3-pci-id.sh
PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

TS="$PB_DIR/troubleshooting.md"

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

# ---------------------------------------------------------------------------
# Group 1: symptom string documented
# ---------------------------------------------------------------------------
has "NIMProfileIDNotFound" "$TS" \
  || fail "troubleshooting.md" "does not document the symptom 'NIMProfileIDNotFound'"
echo "PASS: symptom 'NIMProfileIDNotFound' is documented"

# ---------------------------------------------------------------------------
# Group 2: both PCI IDs documented (31c2:10de manifest/native, 31c3:10de affected)
# ---------------------------------------------------------------------------
has "31c2:10de" "$TS" \
  || fail "troubleshooting.md" "does not mention the manifest/native PCI ID '31c2:10de'"
has "31c3:10de" "$TS" \
  || fail "troubleshooting.md" "does not mention the correct DGX Station GB300 PCI ID '31c3:10de'"
has "2bb4:10de" "$TS" \
  || fail "troubleshooting.md" "does not mention the RTX PRO 6000 PCI ID '2bb4:10de' (also absent from the manifest)"
echo "PASS: PCI IDs documented — 31c2:10de (manifest/native), 31c3:10de (affected GB300), 2bb4:10de (RTX PRO 6000)"

# ---------------------------------------------------------------------------
# Group 3: manifest filename referenced
# ---------------------------------------------------------------------------
has "model_manifest\.yaml" "$TS" \
  || fail "troubleshooting.md" "does not reference the manifest file 'model_manifest.yaml'"
echo "PASS: manifest filename 'model_manifest.yaml' is referenced"

# ---------------------------------------------------------------------------
# Group 4: id-agnostic patch flow. The doc must instruct patching ONLY when the
# GPU's id is absent (with a guard against patching an already-listed id), NOT a
# blind single-id remap. A concrete sed example may still be shown.
# ---------------------------------------------------------------------------
has "only if .*not listed|only patch if .*absent" "$TS" \
  || fail "troubleshooting.md" "does not document the id-agnostic rule (patch ONLY if your id is absent)"
has "sed .*:10de" "$TS" \
  || fail "troubleshooting.md" "does not show a manifest sed remap example"
echo "PASS: id-agnostic manifest patch flow (patch only if absent) is documented"

# ---------------------------------------------------------------------------
# Group 5: names this an upstream NIM issue (durable fix is upstream)
# ---------------------------------------------------------------------------
has "upstream" "$TS" \
  || fail "troubleshooting.md" "does not identify the durable fix as upstream"
# The upstream mention must be tied to the NIM (image/manifest), not something else.
has "upstream.*NIM|NIM.*upstream|manifest gap in the .{0,20}NIM" "$TS" \
  || has "upstream.*(manifest|OpenFold3 NIM|NIM image|NIM team)" "$TS" \
  || fail "troubleshooting.md" "'upstream' is present but not tied to the NIM manifest/image"
echo "PASS: identifies the durable fix as an upstream NIM manifest issue"

# ---------------------------------------------------------------------------
# Group 6: workaround is internally coherent --
#   docker cp of the manifest + mounting it back + real NGC key required
# ---------------------------------------------------------------------------
has "docker cp.*model_manifest\.yaml" "$TS" \
  || fail "troubleshooting.md" "workaround does not docker cp the model_manifest.yaml out of the image"

# Mount the patched manifest back into the container: require the actual mount
# mapping (host manifest -> the image's /opt/nim manifest path), not a bare
# 'volumes:' line which could match any unrelated compose snippet.
has "model_manifest\.yaml:/opt/nim.*model_manifest\.yaml" "$TS" \
  || fail "troubleshooting.md" "workaround does not mount the patched manifest into /opt/nim/.../model_manifest.yaml"

# A real NGC key (not the .env placeholder) is required.
has "real NGC_API_KEY|NGC_API_KEY.*(real|not the .{0,12}placeholder)|real .{0,6}NGC" "$TS" \
  || fail "troubleshooting.md" "workaround does not state a real NGC key is required"
echo "PASS: workaround is internally coherent (docker cp manifest, mount it back, real NGC key)"

echo "PASS: ticket 6376156 OpenFold3 PCI-ID regression documentation present and correct"
exit 0
