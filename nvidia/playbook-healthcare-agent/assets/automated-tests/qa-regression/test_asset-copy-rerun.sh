#!/usr/bin/env bash
# Regression test: Step 2 asset installation must be safe to re-run.
#
# The old `cp -r assets DEST` command changed behavior when DEST already
# existed: it created DEST/assets instead of refreshing files at DEST's root.
# Step 2 must also preserve an existing .env containing the user's NGC key.
#
# Usage: bash test_asset-copy-rerun.sh [PLAYBOOK_DIR]
# Exit 0 = pass, non-zero = fail. Portable to macOS bash 3.2 and Linux bash.

set -u

PB_DIR="${1:-$(cd "$(dirname "$0")/../../.." && pwd)}"
INSTRUCTIONS="$PB_DIR/instructions.md"
FAILURES=0

fail() {
    echo "FAIL: $1"
    FAILURES=$((FAILURES + 1))
}

if [ ! -f "$INSTRUCTIONS" ]; then
    echo "FAIL: instructions.md not found under $PB_DIR"
    exit 1
fi

# Guard the documented implementation so the behavioral fixtures below cannot
# silently pass after the instructions drift back to the unsafe form.
grep -F 'INSTALL_DIR="$HOME/clinical-intelligence"' "$INSTRUCTIONS" >/dev/null \
    || fail "Step 2 does not define a stable INSTALL_DIR"
grep -F 'cp -a "$PLAYBOOK_DIR/assets/." "$INSTALL_DIR/"' "$INSTRUCTIONS" >/dev/null \
    || fail "Step 2 does not copy assets/. into INSTALL_DIR"
grep -F 'if [ ! -f .env ]; then' "$INSTRUCTIONS" >/dev/null \
    || fail "Step 2 does not guard creation of .env"
grep -F 'cp .env.example .env' "$INSTRUCTIONS" >/dev/null \
    || fail "Step 2 does not create .env from the example"
grep -F '~/client-hardware-playbooks/${MODEL}' "$INSTRUCTIONS" >/dev/null \
    || fail "Step 2 no longer uses the converged MODEL path"

if grep -F 'cp -r "$PLAYBOOK_DIR/assets"' "$INSTRUCTIONS" >/dev/null; then
    fail "unsafe cp -r assets command is still present"
fi
if grep -F '~/client-hardware-playbooks/nvidia/${MODEL}' "$INSTRUCTIONS" >/dev/null; then
    fail "MODEL path duplicates the nvidia repository component"
fi

TMP_ROOT=$(mktemp -d "${TMPDIR:-/tmp}/healthcare-asset-copy.XXXXXX") || exit 1
trap 'rm -rf "$TMP_ROOT"' EXIT HUP INT TERM

MODEL="nvidia/playbook-healthcare-agent"
TEST_HOME="$TMP_ROOT/home"
PLAYBOOK_DIR="$TEST_HOME/client-hardware-playbooks/$MODEL"
SOURCE_ASSETS="$PLAYBOOK_DIR/assets"
INSTALL_DIR="$TEST_HOME/clinical-intelligence"
STEP2_BLOCK="$TMP_ROOT/step2.sh"
MOCK_BIN="$TMP_ROOT/bin"

# Execute the exact first Bash fence under Step 2, rather than a duplicate of
# its commands, so the test fails if the documented workflow regresses.
awk '
    /^# Step 2\./ { in_step = 1; next }
    in_step && /^```bash[[:space:]]*$/ { in_fence = 1; next }
    in_fence && /^```[[:space:]]*$/ { exit }
    in_fence { print }
' "$INSTRUCTIONS" > "$STEP2_BLOCK"

if [ ! -s "$STEP2_BLOCK" ]; then
    echo "FAIL: could not extract the Step 2 Bash block"
    exit 1
fi

mkdir -p "$MOCK_BIN" || exit 1
printf '%s\n' '#!/usr/bin/env bash' 'exit 0' > "$MOCK_BIN/nano"
chmod +x "$MOCK_BIN/nano" || exit 1

mkdir -p "$SOURCE_ASSETS/subdir" || exit 1
printf '%s\n' 'version-one' > "$SOURCE_ASSETS/visible.txt"
printf '%s\n' 'NGC_API_KEY=replace-me' > "$SOURCE_ASSETS/.env.example"
printf '%s\n' 'test-results/' > "$SOURCE_ASSETS/.gitignore"
printf '%s\n' 'nested-file' > "$SOURCE_ASSETS/subdir/data.txt"

run_step2() {
    RUN_HOME=$1
    HOME="$RUN_HOME" MODEL="$MODEL" PATH="$MOCK_BIN:$PATH" bash "$STEP2_BLOCK"
}

# The repository clone root plus the converged MODEL must resolve directly to
# the playbook; another nvidia/ or repository component would be incorrect.
if [ ! -d "$PLAYBOOK_DIR/assets" ]; then
    fail "converged path client-hardware-playbooks/$MODEL/assets does not resolve"
fi

# Fresh destination: files and dotfiles land at the install root.
if ! run_step2 "$TEST_HOME"; then
    fail "fresh Step 2 execution failed"
else
    [ -f "$INSTALL_DIR/visible.txt" ] || fail "fresh install missed a regular file"
    [ -f "$INSTALL_DIR/.gitignore" ] || fail "fresh install missed a dotfile"
    [ -f "$INSTALL_DIR/subdir/data.txt" ] || fail "fresh install missed a nested file"
    [ -f "$INSTALL_DIR/.env" ] || fail "fresh install did not create .env"
    [ ! -d "$INSTALL_DIR/assets" ] || fail "fresh install created a nested assets directory"
fi

# Existing configured destination: managed files refresh, while user config and
# unrelated files remain in place.
printf '%s\n' 'NGC_API_KEY=keep-this-key' > "$INSTALL_DIR/.env"
printf '%s\n' 'keep-user-file' > "$INSTALL_DIR/user-owned.txt"
printf '%s\n' 'version-two' > "$SOURCE_ASSETS/visible.txt"
printf '%s\n' 'new-dotfile' > "$SOURCE_ASSETS/.dockerignore"

if ! run_step2 "$TEST_HOME"; then
    fail "repeat Step 2 execution failed"
else
    grep -Fxq 'version-two' "$INSTALL_DIR/visible.txt" \
        || fail "repeat install did not refresh a managed file"
    grep -Fxq 'NGC_API_KEY=keep-this-key' "$INSTALL_DIR/.env" \
        || fail "repeat install overwrote the configured .env"
    [ -f "$INSTALL_DIR/.dockerignore" ] || fail "repeat install missed a new dotfile"
    [ -f "$INSTALL_DIR/user-owned.txt" ] || fail "repeat install removed an unrelated file"
    [ ! -d "$INSTALL_DIR/assets" ] || fail "repeat install created a nested assets directory"
fi

# A destination that exists but is still empty must behave like a fresh install.
EMPTY_HOME="$TMP_ROOT/existing-empty-home"
EMPTY_PLAYBOOK_DIR="$EMPTY_HOME/client-hardware-playbooks/$MODEL"
EMPTY_INSTALL_DIR="$EMPTY_HOME/clinical-intelligence"
mkdir -p "$EMPTY_PLAYBOOK_DIR/assets" "$EMPTY_INSTALL_DIR" || exit 1
cp -a "$SOURCE_ASSETS/." "$EMPTY_PLAYBOOK_DIR/assets/" || exit 1

if ! run_step2 "$EMPTY_HOME"; then
    fail "Step 2 failed with a pre-existing empty install directory"
else
    [ -f "$EMPTY_INSTALL_DIR/visible.txt" ] \
        || fail "pre-existing empty destination missed a regular file"
    [ -f "$EMPTY_INSTALL_DIR/.gitignore" ] \
        || fail "pre-existing empty destination missed a dotfile"
    [ -f "$EMPTY_INSTALL_DIR/.env" ] \
        || fail "pre-existing empty destination did not create .env"
    [ ! -d "$EMPTY_INSTALL_DIR/assets" ] \
        || fail "pre-existing empty destination created a nested assets directory"
fi

if [ "$FAILURES" -ne 0 ]; then
    echo "RESULT: FAIL ($FAILURES assertion(s))"
    exit 1
fi

echo "PASS: converged MODEL path resolves without duplication"
echo "PASS: the exact Step 2 block handles fresh and pre-existing empty destinations"
echo "PASS: dotfiles are copied and repeat installs preserve an existing .env"
echo "RESULT: PASS"
