#!/usr/bin/env bash
# test-all.sh -- Comprehensive CLI test suite for clinical-intelligence.
#
# Usage:
#   bash scripts/test-all.sh                  # default: levels 1-3 (~3 min)
#   bash scripts/test-all.sh --level 1        # infrastructure only (~30s)
#   bash scripts/test-all.sh --level 4        # includes agent tests (~30 min)
#   bash scripts/test-all.sh --level 5        # full e2e (~45 min)
#   bash scripts/test-all.sh --test T3.8      # single test
#   bash scripts/test-all.sh --verbose        # show full output per test
#
# Runs from the DGX host. Requires: openshell CLI on PATH, sandbox running.

set -uo pipefail

# OpenShell installs to ~/.local/bin, not on the default non-interactive PATH
# (e.g. when invoked via `make test` over a non-login SSH). Ensure it resolves.
export PATH="$HOME/.local/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
source "$SCRIPT_DIR/test-lib.sh"

# Source .env so OLLAMA_PORT/OPENFOLD_PORT overrides reach the curl URLs below.
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_DIR/.env"
    set +a
fi

MAX_LEVEL=3
SINGLE_TEST=""

while [[ $# -gt 0 ]]; do
    case "$1" in
        --level) MAX_LEVEL="$2"; shift 2 ;;
        --test) SINGLE_TEST="$2"; shift 2 ;;
        --verbose) VERBOSE=true; shift ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

init_test_run

BRIDGE_IP=$(_bridge_ip)

# Helper: run only if test matches single-test filter or no filter set
should_run() {
    [[ -z "$SINGLE_TEST" ]] || [[ "$1" == "$SINGLE_TEST" ]]
}

# ═══════════════════════════════════════════════════════════════════════
# Level 1: Infrastructure Health (host-side, ~30 seconds)
# ═══════════════════════════════════════════════════════════════════════

run_level1() {
    echo ""
    echo "═══ Level 1: Infrastructure Health ═══"
    echo ""

    should_run "T1.1" && run_test "T1.1" "Ollama alive" \
        "curl -sf http://localhost:${OLLAMA_PORT:-11434}/" \
        assert_exit_0 \
        "Ollama not running. Docker (default): make up. Host Ollama alternative: OLLAMA_HOST=0.0.0.0 ollama serve."

    should_run "T1.2" && run_test "T1.2" "Model available (nemotron-3-super)" \
        "curl -s http://localhost:${OLLAMA_PORT:-11434}/api/tags | python3 -c \"import sys,json; names=[m['name'] for m in json.load(sys.stdin)['models']]; print('FOUND' if any('nemotron-3-super' in n for n in names) else 'MISSING')\"" \
        assert_contains \
        "Model not pulled. Run: ollama pull nemotron-3-super" \
        "FOUND"

    should_run "T1.3" && run_test "T1.3" "Ollama generates text (direct)" \
        "curl -sf -m 30 -X POST http://localhost:${OLLAMA_PORT:-11434}/v1/chat/completions -H 'Content-Type: application/json' -d '{\"model\":\"nemotron-3-super:120b-a12b\",\"messages\":[{\"role\":\"user\",\"content\":\"Say OK\"}],\"max_tokens\":5}' | python3 -c \"import sys,json; c=json.load(sys.stdin).get('choices',[{}])[0].get('message',{}).get('content',''); print(c if c else 'EMPTY')\"" \
        assert_output_not_empty \
        "Ollama can't generate. Check: curl localhost:${OLLAMA_PORT:-11434}/api/ps"

    should_run "T1.5" && run_test "T1.5" "OpenFold3 NIM ready" \
        "curl -sf http://localhost:${OPENFOLD_PORT:-8000}/v1/health/ready" \
        assert_contains \
        "OpenFold3 not ready. Check: docker ps | grep openfold" \
        "ready"

    should_run "T1.6" && run_test "T1.6" "GPU accessible" \
        "nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader 2>&1" \
        assert_contains \
        "GPU not accessible. Check NVIDIA driver." \
        "MiB"

    should_run "T1.7" && run_test "T1.7" "FHIR server reachable" \
        "curl -sf -o /dev/null -w '%{http_code}' https://r4.smarthealthit.org/metadata" \
        assert_equals \
        "FHIR unreachable. Check network/DNS." \
        "200"
}

# ═══════════════════════════════════════════════════════════════════════
# Level 2: OpenShell + Sandbox Health (~1 minute)
# ═══════════════════════════════════════════════════════════════════════

run_level2() {
    echo ""
    echo "═══ Level 2: OpenShell + Sandbox Health ═══"
    echo ""

    should_run "T2.1" && run_test "T2.1" "Gateway connected" \
        "openshell status 2>&1" \
        assert_contains \
        "Gateway down. Run: nohup openshell-gateway --disable-tls --drivers docker --bind-address 127.0.0.1 --port 17670 >/tmp/openshell-gateway.log 2>&1 & openshell gateway add http://127.0.0.1:17670 --name openshell" \
        "Connected"

    should_run "T2.2" && run_test "T2.2" "Sandbox exists and ready" \
        "openshell sandbox list 2>&1" \
        assert_contains \
        "Sandbox not found. Run: bash scripts/setup_sandbox.sh" \
        "Ready"

    should_run "T2.3" && run_test "T2.3" "Forward running on 18789" \
        "openshell forward list 2>&1" \
        assert_contains \
        "Forward dead. Run: openshell forward stop 18789 clinical-sandbox; openshell forward start -d 18789 clinical-sandbox" \
        "running"

    should_run "T2.3b" && run_test "T2.3b" "Gateway HTTP responding on 18789" \
        "curl -sf -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:18789/__openclaw__/health 2>&1 || curl -sf -m 5 -o /dev/null -w %{http_code} http://127.0.0.1:18789/ 2>&1" \
        assert_contains \
        "Gateway HTTP not responding. Forward exists but no listener — re-run scripts/restart_sandbox.sh inside the sandbox or check /tmp/gw.log for the os.networkInterfaces() crash (needs openclaw-os-shim.js loaded via NODE_OPTIONS=--require)." \
        "200"

    should_run "T2.4" && run_test "T2.4" "FHIR from sandbox (curl)" \
        "_sandbox 'curl -sf https://r4.smarthealthit.org/Patient?_count=1 -o /dev/null -w %{http_code}'" \
        assert_equals \
        "FHIR blocked by sandbox policy. Check fhir section + python binary wildcards." \
        "200"

    should_run "T2.5" && run_test "T2.5" "Inference from sandbox" \
        "_sandbox 'curl -sk https://inference.local/v1/models'" \
        assert_contains \
        "Inference not routed. Check: openshell inference get" \
        "nemotron"

    should_run "T2.6" && run_test "T2.6" "OpenFold3 health from sandbox" \
        "_sandbox 'curl -sf http://${BRIDGE_IP}:8000/v1/health/ready'" \
        assert_contains \
        "OpenFold3 health check blocked by sandbox. Check openfold3 policy." \
        "ready"

    should_run "T2.6b" && run_test "T2.6b" "OpenFold3 predict endpoint reachable from sandbox" \
        "_sandbox 'curl -s -o /dev/null -w %{http_code} -X POST -H \"Content-Type: application/json\" -d \"{\\\"dummy\\\": true}\" http://${BRIDGE_IP}:8000/biology/openfold/openfold3/predict'" \
        assert_not_contains \
        "OpenFold3 predict blocked (HTTP 403). Sandbox policy may have L7 rules that break plain HTTP." \
        "403"

    should_run "T2.6c" && run_test "T2.6c" "OpenFold3 predict accepts POST from sandbox" \
        "_sandbox 'curl -s -w \"\n%{http_code}\" -X POST -H \"Content-Type: application/json\" -d \"{\\\"inputs\\\":[{\\\"input_id\\\":\\\"test\\\",\\\"molecules\\\":[{\\\"type\\\":\\\"protein\\\",\\\"id\\\":\\\"A\\\",\\\"sequence\\\":\\\"MKTVRQERLKSIVRI\\\",\\\"msa\\\":{\\\"main\\\":{\\\"a3m\\\":{\\\"alignment\\\":\\\">q\\\\nMKTVRQERLKSIVRI\\\",\\\"format\\\":\\\"a3m\\\"}}}}],\\\"output_format\\\":\\\"pdb\\\"}]}\" http://${BRIDGE_IP}:8000/biology/openfold/openfold3/predict 2>&1 | tail -1'" \
        assert_contains \
        "OpenFold3 predict endpoint rejected POST from sandbox. Check sandbox policy and OpenFold3 NIM status." \
        "200"

    should_run "T2.7" && run_test "T2.7" "Outbound traffic blocked (security)" \
        "_sandbox 'curl --max-time 3 https://google.com 2>&1; echo EXIT_CODE=\$?'" \
        assert_not_contains \
        "SECURITY FAILURE: Outbound traffic NOT blocked!" \
        "EXIT_CODE=0"

    should_run "T2.8" && run_test "T2.8" "Python packages available" \
        "_sandbox 'python -c \"import subprocess, json, pandas, matplotlib; print(\\\"OK\\\")\"'" \
        assert_contains \
        "Python packages missing. Sandbox may need recreation." \
        "OK"
}

# ═══════════════════════════════════════════════════════════════════════
# Level 3: OpenClaw Configuration Correctness (~2 minutes)
# ═══════════════════════════════════════════════════════════════════════

run_level3() {
    echo ""
    echo "═══ Level 3: OpenClaw Configuration ═══"
    echo ""

    # -- Gateway process and logs --
    echo "  --- Gateway ---"

    should_run "T3.1" && run_test "T3.1" "Gateway process alive" \
        "_sandbox 'pgrep -f openclaw-gateway > /dev/null && echo ALIVE || echo DEAD'" \
        assert_contains \
        "OpenClaw gateway not running. Restart it." \
        "ALIVE"

    should_run "T3.2" && run_test "T3.2" "Gateway model correct" \
        "_sandbox 'grep \"agent model\" /tmp/gw.log 2>/dev/null | tail -1'" \
        warn:assert_contains \
        "Wrong model. Check ~/.openclaw/openclaw.json" \
        "local-ollama/nemotron-3-super"

    should_run "T3.3" && run_test "T3.3" "Gateway no errors" \
        "_sandbox 'head -50 /tmp/gw.log 2>/dev/null | grep -iE \"\\[error\\]|\\[fatal\\]|crashed|segfault\" | grep -cv apply_patch'" \
        assert_equals \
        "Gateway has startup errors. Run: _sandbox head -50 /tmp/gw.log" \
        "0"

    # -- Model and auth --
    echo "  --- Model & Auth ---"

    should_run "T3.4" && run_test "T3.4" "Model auth OK (not missing)" \
        "_sandbox 'openclaw models list 2>&1 | grep nemotron | grep -c missing; true'" \
        assert_equals \
        "Auth profile missing. Recreate auth-profiles.json for all agents." \
        "0"

    local agents="main patient-data labs-vitals medications analyst molecular"
    local auth_suffix=a
    for agent in $agents; do
        should_run "T3.5${auth_suffix}" && run_test "T3.5${auth_suffix}" "Auth profile exists: $agent" \
            "_sandbox 'test -f ~/.openclaw/agents/${agent}/agent/auth-profiles.json && echo EXISTS || echo MISSING'" \
            assert_contains \
            "Auth profile missing for $agent. Rerun setup step 10." \
            "EXISTS"
        auth_suffix=$(echo "$auth_suffix" | tr 'a-e' 'b-f')
    done

    should_run "T3.6" && run_test "T3.6" "Auth profile content valid" \
        "_sandbox 'cat ~/.openclaw/agents/main/agent/auth-profiles.json 2>/dev/null'" \
        assert_contains \
        "Auth profile malformed. Should contain version:1 and provider:local-ollama." \
        "local-ollama"

    # -- Skills --
    echo "  --- Skills ---"

    should_run "T3.7" && run_test "T3.7" "Skills count (expect 7)" \
        "_sandbox 'openclaw skills list 2>&1 | grep -c openclaw-workspace'" \
        assert_equals \
        "Not all skills loaded. Redeploy to ~/.openclaw/workspace/skills/" \
        "7"

    local skills="analysis-methods case-summary clinical-delegation clinical-knowledge cohort-compare fhir-basics molecular-viz"
    local skill_suffix=a
    for skill in $skills; do
        should_run "T3.8${skill_suffix}" && run_test "T3.8${skill_suffix}" "Skill loaded: $skill" \
            "_sandbox 'openclaw skills list 2>&1 | grep ${skill} | grep -c ready'" \
            assert_numeric_gt \
            "Skill $skill not loaded. Check ~/.openclaw/workspace/skills/${skill}/SKILL.md" \
            "0"
        skill_suffix=$(echo "$skill_suffix" | tr 'a-f' 'b-g')
    done

    should_run "T3.9" && run_test "T3.9" "analysis-methods uses subprocess" \
        "_sandbox 'grep -c subprocess ~/.openclaw/workspace/skills/analysis-methods/SKILL.md 2>/dev/null'" \
        assert_numeric_gt \
        "analysis-methods skill still uses requests. Redeploy updated version." \
        "0"

    should_run "T3.10" && run_test "T3.10" "fhir-basics uses subprocess" \
        "_sandbox 'grep -c subprocess ~/.openclaw/workspace/skills/fhir-basics/SKILL.md 2>/dev/null'" \
        assert_numeric_gt \
        "fhir-basics skill still uses requests. Redeploy updated version." \
        "0"

    # -- Agents --
    echo "  --- Agents ---"

    should_run "T3.11" && run_test "T3.11" "Agents count (expect >= 5)" \
        "_sandbox 'openclaw agents list 2>&1 | grep -c Workspace:'" \
        assert_numeric_gt \
        "Not all agents registered. Rerun setup step 9." \
        "4"

    local agent_suffix=a
    for agent in patient-data labs-vitals medications analyst molecular; do
        should_run "T3.12${agent_suffix}" && run_test "T3.12${agent_suffix}" "Agent registered: $agent" \
            "_sandbox 'openclaw agents list 2>&1 | grep -c ${agent}'" \
            assert_numeric_gt \
            "Agent $agent not registered." \
            "0"
        agent_suffix=$(echo "$agent_suffix" | tr 'a-d' 'b-e')
    done

    # -- IDENTITY.md --
    echo "  --- IDENTITY.md ---"

    should_run "T3.13" && run_test "T3.13" "IDENTITY.md exists" \
        "_sandbox 'test -f ~/.openclaw/workspace/IDENTITY.md && echo EXISTS || echo MISSING'" \
        assert_contains \
        "IDENTITY.md not deployed." \
        "EXISTS"

    should_run "T3.14" && run_test "T3.14" "IDENTITY.md header correct" \
        "_sandbox 'head -1 ~/.openclaw/workspace/IDENTITY.md'" \
        assert_contains \
        "IDENTITY.md has wrong header." \
        "Clinical Intelligence"

    should_run "T3.15" && run_test "T3.15" "IDENTITY.md has molecular delegation" \
        "_sandbox 'grep -c molecular ~/.openclaw/workspace/IDENTITY.md 2>/dev/null'" \
        assert_numeric_gt \
        "IDENTITY.md missing molecular agent delegation." \
        "0"

    should_run "T3.16" && run_test "T3.16" "IDENTITY.md has how-to-work section" \
        "_sandbox 'grep -c 'How to work' ~/.openclaw/workspace/IDENTITY.md 2>/dev/null'" \
        assert_numeric_gt \
        "IDENTITY.md missing How to work section." \
        "0"

    should_run "T3.17" && run_test "T3.17" "IDENTITY.md has principles" \
        "_sandbox 'grep -c Principles ~/.openclaw/workspace/IDENTITY.md 2>/dev/null'" \
        assert_numeric_gt \
        "IDENTITY.md missing Principles section." \
        "0"

    # -- openclaw.json --
    echo "  --- openclaw.json ---"

    should_run "T3.18" && run_test "T3.18" "Model in openclaw.json" \
        "_sandbox 'python3 -c \"import json,os; d=json.load(open(os.path.expanduser(\\\"~/.openclaw/openclaw.json\\\"))); print(d[\\\"agents\\\"][\\\"defaults\\\"][\\\"model\\\"])\"'" \
        assert_contains \
        "Wrong model in openclaw.json." \
        "local-ollama/nemotron-3-super"

    should_run "T3.19" && run_test "T3.19" "allowAgents includes molecular" \
        "_sandbox 'python3 -c \"import json,os; d=json.load(open(os.path.expanduser(\\\"~/.openclaw/openclaw.json\\\"))); a=d[\\\"agents\\\"][\\\"list\\\"][0][\\\"subagents\\\"][\\\"allowAgents\\\"]; print(\\\"OK\\\" if \\\"molecular\\\" in a else \\\"MISSING\\\")\"'" \
        assert_contains \
        "molecular not in allowAgents. Update openclaw.json." \
        "OK"

    # -- Scripts --
    echo "  --- Scripts ---"

    should_run "T3.20" && run_test "T3.20" "build_viewer.py exists in sandbox" \
        "_sandbox 'test -f /sandbox/clinical-intelligence/scripts/build_viewer.py && echo EXISTS || echo MISSING'" \
        assert_contains \
        "build_viewer.py not uploaded to sandbox." \
        "EXISTS"

    should_run "T3.21" && run_test "T3.21" "build_viewer.py uses subprocess" \
        "_sandbox 'grep -c subprocess.run /sandbox/clinical-intelligence/scripts/build_viewer.py 2>/dev/null'" \
        assert_numeric_gt \
        "build_viewer.py still uses urllib.request. Deploy updated version." \
        "0"

    should_run "T3.22" && run_test "T3.22" "validate_and_run.py exists" \
        "_sandbox 'test -f /sandbox/clinical-intelligence/scripts/validate_and_run.py && echo EXISTS || echo MISSING'" \
        assert_contains \
        "validate_and_run.py not uploaded to sandbox." \
        "EXISTS"

    # -- Smoke test --
    echo "  --- Smoke Test ---"

    should_run "T3.23" && run_test "T3.23" "Agent responds to prompt" \
        "_sandbox 'cd /sandbox/clinical-intelligence && openclaw agent --local --session-id smoke-\$\$ --thinking off --message \"Say OK\" --timeout 60 2>&1 | tail -5'" \
        assert_contains \
        "Agent cannot respond. Check all Level 3 tests above first." \
        "OK"
}

# ═══════════════════════════════════════════════════════════════════════
# Level 4: Agent Functional Tests (~20-30 minutes)
# ═══════════════════════════════════════════════════════════════════════

run_level4() {
    echo ""
    echo "═══ Level 4: Agent Functional Tests ═══"
    echo ""

    should_run "T4.1" && run_test "T4.1" "Cohort count (expect 47)" \
        "_sandbox 'cd /sandbox/clinical-intelligence && openclaw agent --local --session-id t41-\$\$ --thinking off --timeout 300 --message \"Find all diabetic patients and count them\" 2>&1 | tail -20'" \
        assert_contains \
        "Agent failed cohort query. Check FHIR access + analysis-methods skill." \
        "47"

    should_run "T4.2" && run_test "T4.2" "Lab retrieval (HbA1c value)" \
        "_sandbox 'cd /sandbox/clinical-intelligence && openclaw agent --local --session-id t42-\$\$ --thinking off --timeout 300 --message \"Get the latest HbA1c for patient 9eb43ac3-7c1e-4e25-94cd-4b2c43f7234e\" 2>&1 | tail -20'" \
        assert_output_not_empty \
        "Agent failed lab retrieval."

    should_run "T4.3" && run_test "T4.3" "Code execution (print 42)" \
        "_sandbox 'cd /sandbox/clinical-intelligence && openclaw agent --local --session-id t43-\$\$ --thinking off --timeout 120 --message \"Write a Python script that prints 42 and execute it\" 2>&1 | tail -10'" \
        assert_contains \
        "Agent cannot execute code." \
        "42"

    should_run "T4.4" && run_test "T4.4" "Molecular visualization" \
        "_sandbox 'cd /sandbox/clinical-intelligence && rm -f ~/.openclaw/canvas/atorvastatin*.html && openclaw agent --local --session-id t44-\$\$ --thinking off --timeout 300 --message \"Show me the 3D structure of atorvastatin bound to its target HMG-CoA reductase\" 2>&1 | tail -10; ls -la ~/.openclaw/canvas/atorvastatin*.html 2>/dev/null | wc -l'" \
        assert_numeric_gt \
        "Molecular viz failed. Check OpenFold3 access + build_viewer.py." \
        "0"

    # -- OpenFold3 / molecular-viz tests --
    echo "  --- OpenFold3 / Molecular Viz ---"

    should_run "T4.5" && run_test "T4.5" "OpenFold3 prediction response schema" \
        "_sandbox 'curl -sf --max-time 300 -X POST -H \"Content-Type: application/json\" -d \"{\\\"inputs\\\":[{\\\"input_id\\\":\\\"schema-test\\\",\\\"molecules\\\":[{\\\"type\\\":\\\"protein\\\",\\\"id\\\":\\\"A\\\",\\\"sequence\\\":\\\"FVNQHLCGSHLVEALYLVCGERGFFYTPKT\\\",\\\"msa\\\":{\\\"main\\\":{\\\"a3m\\\":{\\\"alignment\\\":\\\">q\\\\nFVNQHLCGSHLVEALYLVCGERGFFYTPKT\\\",\\\"format\\\":\\\"a3m\\\"}}}}],\\\"output_format\\\":\\\"pdb\\\"}]}\" http://${BRIDGE_IP}:8000/biology/openfold/openfold3/predict | python3 -c \"import sys,json; r=json.load(sys.stdin); o=r[\\\"outputs\\\"][0][\\\"structures_with_scores\\\"][0]; assert \\\"structure\\\" in o, \\\"missing structure\\\"; assert \\\"confidence_score\\\" in o, \\\"missing confidence_score\\\"; assert \\\"complex_plddt_score\\\" in o, \\\"missing plddt\\\"; assert \\\"ptm_score\\\" in o, \\\"missing ptm\\\"; print(\\\"SCHEMA_OK\\\")\"'" \
        assert_contains \
        "OpenFold3 response missing expected fields (structure, confidence_score, plddt, ptm). Check NIM version." \
        "SCHEMA_OK"

    should_run "T4.6" && run_test "T4.6" "OpenFold3 confidence scores are numeric" \
        "_sandbox 'curl -sf --max-time 300 -X POST -H \"Content-Type: application/json\" -d \"{\\\"inputs\\\":[{\\\"input_id\\\":\\\"score-test\\\",\\\"molecules\\\":[{\\\"type\\\":\\\"protein\\\",\\\"id\\\":\\\"A\\\",\\\"sequence\\\":\\\"FVNQHLCGSHLVEALYLVCGERGFFYTPKT\\\",\\\"msa\\\":{\\\"main\\\":{\\\"a3m\\\":{\\\"alignment\\\":\\\">q\\\\nFVNQHLCGSHLVEALYLVCGERGFFYTPKT\\\",\\\"format\\\":\\\"a3m\\\"}}}}],\\\"output_format\\\":\\\"pdb\\\"}]}\" http://${BRIDGE_IP}:8000/biology/openfold/openfold3/predict | python3 -c \"import sys,json; r=json.load(sys.stdin); o=r[\\\"outputs\\\"][0][\\\"structures_with_scores\\\"][0]; plddt=float(o[\\\"complex_plddt_score\\\"]); ptm=float(o[\\\"ptm_score\\\"]); conf=float(o[\\\"confidence_score\\\"]); iptm=float(o.get(\\\"iptm_score\\\",0)); print(f\\\"pLDDT={plddt:.1f} pTM={ptm:.2f} ipTM={iptm:.2f} conf={conf:.2f}\\\"); assert plddt > 0, \\\"pLDDT not positive\\\"; assert ptm >= 0, \\\"pTM negative\\\"; print(\\\"SCORES_OK\\\")\"'" \
        assert_contains \
        "Confidence scores not numeric or out of range. Check OpenFold3 prediction output." \
        "SCORES_OK"

    should_run "T4.7" && run_test "T4.7" "build_viewer.py HTML output valid" \
        "_sandbox 'cd /sandbox/clinical-intelligence && rm -f ~/.openclaw/canvas/metformin_complex.html ~/.openclaw/canvas/metformin_complex.pdb && python3 scripts/build_viewer.py --drug metformin --openfold-host ${BRIDGE_IP} 2>&1; cat ~/.openclaw/canvas/metformin_complex.html 2>/dev/null | python3 -c \"import sys; html=sys.stdin.read(); checks=[\\\"3Dmol\\\" in html, \\\"ATOM\\\" in html or \\\"HETATM\\\" in html, \\\"pLDDT\\\" in html, \\\"confidence\\\" in html.lower()]; print(f\\\"3Dmol={checks[0]} PDB={checks[1]} pLDDT={checks[2]} conf={checks[3]}\\\"); print(\\\"HTML_OK\\\" if all(checks) else \\\"HTML_FAIL\\\")\"'" \
        assert_contains \
        "build_viewer.py HTML missing 3Dmol.js, PDB structure data, or confidence scores. Check script output." \
        "HTML_OK"

    should_run "T4.8" && run_test "T4.8" "build_viewer.py creates file in canvas" \
        "_sandbox 'test -f ~/.openclaw/canvas/metformin_complex.html && stat -c %s ~/.openclaw/canvas/metformin_complex.html || echo 0'" \
        assert_numeric_gt \
        "build_viewer.py did not create output file in canvas dir. Run T4.7 first." \
        "1000"
}

# ═══════════════════════════════════════════════════════════════════════
# Level 5: End-to-End Integration (~15 minutes)
# ═══════════════════════════════════════════════════════════════════════

run_level5() {
    echo ""
    echo "═══ Level 5: End-to-End Integration ═══"
    echo ""

    should_run "T5.1" && run_test "T5.1" "Cross-condition analysis (diabetes + hypertension + eGFR)" \
        "_sandbox 'cd /sandbox/clinical-intelligence && openclaw agent --local --session-id t51-\$\$ --thinking off --timeout 600 --message \"Find all diabetic patients that also have hypertension. For the overlap, get their eGFR. Flag anyone with eGFR below 60 as kidney disease risk.\" 2>&1 | tail -30'" \
        assert_contains \
        "Cross-condition query failed. Run Level 4 tests individually to isolate." \
        "24"
}

# ═══════════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════════

echo ""
echo "╔════════════════════════════════════════════════╗"
echo "║  Clinical Intelligence Test Suite              ║"
echo "║  Max level: $MAX_LEVEL                                  ║"
echo "╚════════════════════════════════════════════════╝"

(( MAX_LEVEL >= 1 )) && run_level1
(( MAX_LEVEL >= 2 )) && run_level2
(( MAX_LEVEL >= 3 )) && run_level3
(( MAX_LEVEL >= 4 )) && run_level4
(( MAX_LEVEL >= 5 )) && run_level5

print_summary
