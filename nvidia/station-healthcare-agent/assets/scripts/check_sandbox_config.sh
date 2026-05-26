#!/usr/bin/env bash
#
# Verify sandbox config matches the repo.
# Catches stale skills, missing files, broken imports, and memory errors.
#
# Usage:
#   bash scripts/check_sandbox_config.sh [sandbox-name]
#
# Returns exit code 0 if all checks pass, 1 if any fail.

set -uo pipefail

SANDBOX="${1:-clinical-sandbox}"
REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
FAIL=0
WARN=0

green() { printf "\033[32m✓ %s\033[0m\n" "$1"; }
red()   { printf "\033[31m✗ %s\033[0m\n" "$1"; FAIL=$((FAIL+1)); }
yellow(){ printf "\033[33m⚠ %s\033[0m\n" "$1"; WARN=$((WARN+1)); }

sandbox_exec() {
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -o LogLevel=ERROR -o ConnectTimeout=15 \
        -o "ProxyCommand=openshell ssh-proxy --gateway-name openshell --name ${SANDBOX}" \
        "sandbox@openshell-${SANDBOX}" "$1" 2>/dev/null
}

echo "=== Sandbox Config Check: ${SANDBOX} ==="
echo ""

# ── 1. Sandbox exists and is ready ─────────────────────────────
echo "--- Sandbox Status ---"
PHASE=$(openshell sandbox list 2>/dev/null | grep "$SANDBOX" | awk '{print $NF}' | sed 's/\x1b\[[0-9;]*m//g')
if [ "$PHASE" = "Ready" ]; then
    green "Sandbox $SANDBOX is Ready"
else
    red "Sandbox $SANDBOX is not Ready (phase: ${PHASE:-not found})"
    echo "  Run: make setup"
    exit 1
fi

# ── 2. Collect all hashes from sandbox in one SSH call ─────────
echo ""
echo "--- Collecting sandbox state (one SSH call) ---"
SANDBOX_STATE=$(sandbox_exec '
echo "=== HASHES ==="
for f in \
    ~/.openclaw/agents/main/agent/IDENTITY.md \
    ~/.openclaw/workspace/IDENTITY.md \
    ~/.openclaw/workspace/skills/fhir-basics/SKILL.md \
    ~/.openclaw/workspace/skills/analysis-methods/SKILL.md \
    ~/.openclaw/workspace/skills/clinical-knowledge/SKILL.md \
    ~/.openclaw/workspace/skills/cohort-compare/SKILL.md \
    ~/.openclaw/workspace/skills/case-summary/SKILL.md \
    ~/.openclaw/workspace/skills/clinical-delegation/SKILL.md \
    ~/.openclaw/workspace/skills/molecular-viz/SKILL.md; do
    if [ -f "$f" ]; then
        md5sum "$f"
    else
        echo "MISSING $f"
    fi
done
echo "=== HELPERS ==="
if [ -f /sandbox/clinical-intelligence/skills/analysis-methods/scripts/fhir_helpers.py ]; then
    echo "HELPERS_EXISTS"
    python3 -c "import sys; sys.path.insert(0, \"/sandbox/clinical-intelligence/skills/analysis-methods/scripts\"); from fhir_helpers import fhir_get; print(\"IMPORT_OK\")" 2>&1
else
    echo "HELPERS_MISSING"
fi
echo "=== MEMORY ==="
[ -d ~/.openclaw/workspace/memory ] && echo "MEMORY_DIR_EXISTS" || echo "MEMORY_DIR_MISSING"
[ -f ~/.openclaw/workspace/MEMORY.md ] && echo "MEMORY_FILE_EXISTS" || echo "MEMORY_FILE_MISSING"
echo "=== GATEWAY ==="
curl -sf http://127.0.0.1:18789/ >/dev/null 2>&1 && echo "GATEWAY_UP" || echo "GATEWAY_DOWN"
echo "=== AGENTS ==="
openclaw agents list 2>/dev/null | grep -c "^\- " || echo "0"
echo "=== INFERENCE ==="
curl -sf https://inference.local/v1/models 2>/dev/null | python3 -c "import sys,json; d=json.load(sys.stdin); print(d[\"data\"][0][\"id\"])" 2>/dev/null || echo "INFERENCE_FAIL"
echo "=== NESTED ==="
[ -d /sandbox/clinical-intelligence/skills/analysis-methods/analysis-methods ] && echo "NESTED_YES" || echo "NESTED_NO"
')

if [ -z "$SANDBOX_STATE" ]; then
    red "Could not connect to sandbox"
    exit 1
fi

# ── 3. Parse and check IDENTITY.md ─────────────────────────────
echo ""
echo "--- IDENTITY.md ---"
REPO_ID_HASH=$(md5sum "$REPO_DIR/IDENTITY.md" 2>/dev/null | awk '{print $1}')
AGENT_ID_HASH=$(echo "$SANDBOX_STATE" | grep "agents/main/agent/IDENTITY.md" | awk '{print $1}')
WORKSPACE_ID_HASH=$(echo "$SANDBOX_STATE" | grep "workspace/IDENTITY.md" | head -1 | awk '{print $1}')

if [ "$REPO_ID_HASH" = "$AGENT_ID_HASH" ]; then
    green "Agent IDENTITY.md matches repo"
else
    red "Agent IDENTITY.md is STALE (repo: ${REPO_ID_HASH:0:8} sandbox: ${AGENT_ID_HASH:0:8})"
fi

if [ "$REPO_ID_HASH" = "$WORKSPACE_ID_HASH" ]; then
    green "Workspace IDENTITY.md matches repo"
else
    red "Workspace IDENTITY.md is STALE (repo: ${REPO_ID_HASH:0:8} workspace: ${WORKSPACE_ID_HASH:0:8})"
fi

# ── 4. Check each skill ───────────────────────────────────────
echo ""
echo "--- Skills ---"
for skill_dir in "$REPO_DIR"/skills/*/; do
    skill_name=$(basename "$skill_dir")
    REPO_HASH=$(md5sum "$skill_dir/SKILL.md" 2>/dev/null | awk '{print $1}')
    SANDBOX_LINE=$(echo "$SANDBOX_STATE" | grep "workspace/skills/$skill_name/SKILL.md")

    if echo "$SANDBOX_LINE" | grep -q "MISSING"; then
        red "Skill $skill_name: NOT FOUND in workspace"
    else
        SANDBOX_HASH=$(echo "$SANDBOX_LINE" | awk '{print $1}')
        if [ "$REPO_HASH" = "$SANDBOX_HASH" ]; then
            green "Skill $skill_name matches repo"
        else
            red "Skill $skill_name is STALE (repo: ${REPO_HASH:0:8} workspace: ${SANDBOX_HASH:0:8})"
        fi
    fi
done

# ── 5. Helpers ─────────────────────────────────────────────────
echo ""
echo "--- Helper Scripts ---"
if echo "$SANDBOX_STATE" | grep -q "HELPERS_EXISTS"; then
    green "fhir_helpers.py exists"
else
    red "fhir_helpers.py MISSING"
fi

if echo "$SANDBOX_STATE" | grep -q "IMPORT_OK"; then
    green "fhir_helpers.py imports OK"
else
    red "fhir_helpers.py import FAILED"
fi

# ── 6. Memory ──────────────────────────────────────────────────
echo ""
echo "--- Memory ---"
if echo "$SANDBOX_STATE" | grep -q "MEMORY_DIR_EXISTS"; then
    green "memory/ directory exists"
else
    red "memory/ directory MISSING (causes ENOENT on session start)"
fi

if echo "$SANDBOX_STATE" | grep -q "MEMORY_FILE_EXISTS"; then
    green "MEMORY.md exists"
else
    red "MEMORY.md MISSING"
fi

# ── 7. Gateway ─────────────────────────────────────────────────
echo ""
echo "--- OpenClaw Gateway ---"
if echo "$SANDBOX_STATE" | grep -q "GATEWAY_UP"; then
    green "OpenClaw gateway responding"
else
    red "OpenClaw gateway DOWN"
    echo "  Fix: make restart"
fi

# ── 8. Agents ──────────────────────────────────────────────────
echo ""
echo "--- Agents ---"
AGENT_COUNT=$(echo "$SANDBOX_STATE" | sed -n '/=== AGENTS ===/,/=== INFERENCE ===/p' | grep -v '===' | head -1 | tr -d '[:space:]')
if [ "${AGENT_COUNT:-0}" -ge 6 ]; then
    green "$AGENT_COUNT agents registered"
else
    red "Only ${AGENT_COUNT:-0} agents (expected 6)"
fi

# ── 9. Inference ───────────────────────────────────────────────
echo ""
echo "--- Inference ---"
INFERENCE_MODEL=$(echo "$SANDBOX_STATE" | sed -n '/=== INFERENCE ===/,/=== NESTED ===/p' | grep -v '===' | head -1)
if echo "$INFERENCE_MODEL" | grep -q "nemotron"; then
    green "Inference: $INFERENCE_MODEL"
else
    red "Inference FAILED (got: ${INFERENCE_MODEL:-nothing})"
fi

# ── 10. Port forward ──────────────────────────────────────────
echo ""
echo "--- Port Forward ---"
FWD_STATUS=$(openshell forward list 2>/dev/null | grep "$SANDBOX" | sed 's/\x1b\[[0-9;]*m//g' | grep -o 'running\|dead')
if [ "$FWD_STATUS" = "running" ]; then
    green "Port forward 18789 running"
else
    red "Port forward 18789 ${FWD_STATUS:-not found}"
fi

# ── 11. Nested directories ────────────────────────────────────
echo ""
echo "--- Directory Structure ---"
if echo "$SANDBOX_STATE" | grep -q "NESTED_YES"; then
    red "Nested analysis-methods/analysis-methods/ directory found"
else
    green "No nested directory issues"
fi

# ── Summary ────────────────────────────────────────────────────
echo ""
echo "=== Summary ==="
if [ $FAIL -eq 0 ]; then
    green "All checks passed ($WARN warning(s))"
    exit 0
else
    red "$FAIL check(s) FAILED, $WARN warning(s)"
    exit 1
fi
