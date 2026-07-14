#!/usr/bin/env bash
# Regression test for QA ticket 6376168B
#
# Bug: the healthcare-agent playbook wrongly required completing the full
# NemoClaw playbook first for missing tooling. Fix: instructions.md and
# assets/SETUP-GUIDE.md now install OpenShell via the official standalone
# installer (curl ... OpenShell/main/install.sh | sh) and state NemoClaw is
# not required. SETUP-GUIDE Step 2 no longer uses `pip install openshell`
# with the urm.nvidia.com index.
#
# Exit 0 = pass, non-zero = fail.
# macOS bash 3.2 and Linux bash compatible. No mapfile/declare -A/grep -P.

set -u

# Playbook dir under test. Script lives 3 levels below playbook root:
#   <root>/assets/automated-tests/qa-regression/test_*.sh
PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"

INSTRUCTIONS="$PB_DIR/instructions.md"
SETUP_GUIDE="$PB_DIR/assets/SETUP-GUIDE.md"
OVERVIEW="$PB_DIR/overview.md"
RUNBOOK="$PB_DIR/assets/RUNBOOK.md"

INSTALLER_HOST="raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh"

fail() { echo "FAIL: $1: $2"; exit 1; }

# ---------------------------------------------------------------------------
# Group 1: instructions.md
# ---------------------------------------------------------------------------
[ -f "$INSTRUCTIONS" ] || fail "$INSTRUCTIONS" "file not found"

# 1a. Must NOT tell users to complete the NemoClaw playbook (Steps 1-4) as a
#     prerequisite for missing tooling. Old text:
#     "complete the NemoClaw playbook (...) Steps 1-4 first"
# Match robustly: "complete the NemoClaw playbook" ... "Steps 1" "first",
# tolerant of whitespace/punctuation between the phrase and the step range.
if grep -Eiq 'complete the NemoClaw playbook' "$INSTRUCTIONS"; then
  fail "$INSTRUCTIONS" "still tells users to 'complete the NemoClaw playbook' as a prerequisite (old pre-fix text)"
fi
# Also guard against the specific "Steps 1-4 first" instruction reappearing
# anywhere tied to NemoClaw. Any dash style between 1 and 4.
if grep -Eiq 'Steps 1[[:space:]]*[-–—][[:space:]]*4 first' "$INSTRUCTIONS"; then
  fail "$INSTRUCTIONS" "still contains 'Steps 1-4 first' prerequisite instruction (old pre-fix text)"
fi

# 1b. Must contain the standalone OpenShell installer URL.
grep -Fq "$INSTALLER_HOST" "$INSTRUCTIONS" \
  || fail "$INSTRUCTIONS" "missing standalone OpenShell installer URL '$INSTALLER_HOST'"

# 1c. Must state NemoClaw is not required.
#     Fixed text: "no NemoClaw required" / "do not need the NemoClaw playbook".
if grep -Eiq 'no NemoClaw required' "$INSTRUCTIONS" \
   || grep -Eiq 'do not need the NemoClaw' "$INSTRUCTIONS" \
   || grep -Eiq "don't need the NemoClaw" "$INSTRUCTIONS" \
   || grep -Eiq 'NemoClaw is not required' "$INSTRUCTIONS"; then
  :
else
  fail "$INSTRUCTIONS" "does not state that NemoClaw is not required"
fi

echo "PASS: instructions.md drops the NemoClaw-playbook prerequisite, uses the standalone installer, and states NemoClaw is not required"

# ---------------------------------------------------------------------------
# Group 2: assets/SETUP-GUIDE.md
# ---------------------------------------------------------------------------
[ -f "$SETUP_GUIDE" ] || fail "$SETUP_GUIDE" "file not found"

# 2a. Must NOT install openshell via pip against the urm.nvidia.com index.
#     Old text: pip install openshell ... --index-url https://urm.nvidia.com/...
if grep -Eiq 'pip install[[:space:]]+openshell' "$SETUP_GUIDE"; then
  fail "$SETUP_GUIDE" "still uses 'pip install openshell' (old pre-fix install method)"
fi
if grep -Fq 'urm.nvidia.com' "$SETUP_GUIDE"; then
  fail "$SETUP_GUIDE" "still references the urm.nvidia.com pip index (old pre-fix install method)"
fi

# 2b. Must use the standalone OpenShell installer instead.
grep -Fq "$INSTALLER_HOST" "$SETUP_GUIDE" \
  || fail "$SETUP_GUIDE" "missing standalone OpenShell installer URL '$INSTALLER_HOST'"

echo "PASS: SETUP-GUIDE.md installs OpenShell via the official installer, not 'pip install openshell' from urm.nvidia.com"

# ---------------------------------------------------------------------------
# Group 3: overview.md (a published tab — must not restate the dependency)
# ---------------------------------------------------------------------------
[ -f "$OVERVIEW" ] || fail "$OVERVIEW" "file not found"

# 3a. Must NOT tell users to complete the NemoClaw playbook.
if grep -Eiq 'complete the NemoClaw playbook' "$OVERVIEW"; then
  fail "$OVERVIEW" "still tells users to 'complete the NemoClaw playbook' (old pre-fix text)"
fi
# 3b. Must point to the standalone installer.
grep -Fq "$INSTALLER_HOST" "$OVERVIEW" \
  || fail "$OVERVIEW" "missing standalone OpenShell installer URL '$INSTALLER_HOST'"
echo "PASS: overview.md drops the NemoClaw-playbook prerequisite and uses the standalone installer"

# ---------------------------------------------------------------------------
# Group 4: assets/RUNBOOK.md (must not use the retired pip/urm install method)
# ---------------------------------------------------------------------------
[ -f "$RUNBOOK" ] || fail "$RUNBOOK" "file not found"

# 4a. Must NOT install openshell via pip / the urm.nvidia.com index.
if grep -Eiq 'pip install[[:space:]]+openshell' "$RUNBOOK"; then
  fail "$RUNBOOK" "still uses 'pip install openshell' (old pre-fix install method)"
fi
if grep -Fq 'urm.nvidia.com' "$RUNBOOK"; then
  fail "$RUNBOOK" "still references the urm.nvidia.com pip index (old pre-fix install method)"
fi
# 4b. Must use the standalone OpenShell installer.
grep -Fq "$INSTALLER_HOST" "$RUNBOOK" \
  || fail "$RUNBOOK" "missing standalone OpenShell installer URL '$INSTALLER_HOST'"
echo "PASS: RUNBOOK.md installs OpenShell via the official installer, not pip from urm.nvidia.com"

echo "PASS: ticket 6376168B — NemoClaw dependency removed; OpenShell installed standalone"
exit 0
