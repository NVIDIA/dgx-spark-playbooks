#!/usr/bin/env bash
# Automated sandbox setup for Clinical Intelligence.
#
# Creates an OpenShell sandbox, installs Python packages, deploys skills,
# registers agents, configures OpenClaw, and starts the gateway.
# Recreates the sandbox from scratch each time it runs.
#
# Prerequisites (must be done before running this script):
#   - OpenShell installed and gateway started
#   - Ollama installed with nemotron-3-super pulled
#   - Repo cloned to ~/clinical-intelligence
#
# The script will create the provider and set inference automatically if needed.
#
# Usage:
#   bash scripts/setup_sandbox.sh [--local]
#
# Options:
#   --local    Bind gateway to 0.0.0.0 for local browser access (no SSH tunnel needed)
#              Default: loopback only (requires SSH tunnel from remote machine)
#
# The Docker bridge IP is auto-detected via 'ip -4 addr show docker0' below.
set -euo pipefail

# OpenShell installs to ~/.local/bin, which is not on the default non-interactive
# PATH (e.g. when this script runs via `make setup` over a non-login SSH). Ensure
# it is reachable so `openshell` resolves regardless of how we were invoked.
export PATH="$HOME/.local/bin:$PATH"

BIND_MODE="loopback"
for arg in "$@"; do
    case "$arg" in
        --local) BIND_MODE="all" ;;
        --*) echo "Unknown option: $arg"; exit 1 ;;
    esac
done

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# Source .env so OLLAMA_PORT, OPENFOLD_PORT, SANDBOX_NAME, OLLAMA_MODEL, etc.
# overrides (e.g. moving Ollama off the host-conflicting port 11434) propagate
# to openshell provider creation and downstream commands. Without this, .env
# values are docker-compose-only and the sandbox provider would point at the
# wrong port.
if [ -f "$REPO_DIR/.env" ]; then
    set -a
    # shellcheck disable=SC1091
    . "$REPO_DIR/.env"
    set +a
fi

SANDBOX_NAME="${SANDBOX_NAME:-clinical-sandbox}"
MODEL="${OLLAMA_MODEL:-nemotron-3-super:120b-a12b}"
PORT="${GATEWAY_PORT:-18789}"

echo "=== Clinical Intelligence Sandbox Setup ==="
echo "Repo:    $REPO_DIR"
echo "Sandbox: $SANDBOX_NAME"
echo "Model:   $MODEL"
echo "Bind:    $BIND_MODE"
echo ""

# --- Pre-flight: verify OpenShell gateway is connected ---
echo "--- Pre-flight: Verify OpenShell gateway ---"
if ! openshell status 2>&1 | grep -q "Connected"; then
    echo "ERROR: OpenShell gateway is not connected." >&2
    echo "Start it with (see instructions.md Step 4):" >&2
    echo "  nohup openshell-gateway --disable-tls --drivers docker --bind-address 127.0.0.1 --port 17670 >/tmp/openshell-gateway.log 2>&1 &" >&2
    echo "  openshell gateway add http://127.0.0.1:17670 --name openshell" >&2
    exit 1
fi
echo "Gateway: Connected"
echo ""

# --- Step 1: Generate sandbox policy with correct Docker bridge IP ---
echo "--- Step 1: Generate sandbox policy ---"
BRIDGE_IP=$(ip -4 addr show docker0 2>/dev/null | grep -oP 'inet \K[\d.]+' || true)
if [ -z "$BRIDGE_IP" ]; then
    echo "WARN: Could not auto-detect docker0 IP, trying ip route..."
    BRIDGE_IP=$(ip route show default | grep -oP 'via \K[\d.]+' || true)
fi
if [ -z "$BRIDGE_IP" ]; then
    echo "ERROR: Cannot detect Docker bridge IP. Set DOCKER_BRIDGE_IP and re-run." >&2
    exit 1
fi
echo "Docker bridge IP: $BRIDGE_IP"

POLICY_FILE="$REPO_DIR/sandbox-policy-local.yaml"
bash "$REPO_DIR/scripts/gen_sandbox_policy.sh" "$POLICY_FILE"
echo ""

# --- Step 1b: Ensure provider and inference are configured ---
echo "--- Step 1b: Configure provider and inference ---"
# Current OpenShell releases require --config OPENAI_BASE_URL=... for the
# openai provider (the older --base-url shortcut is no longer accepted).
# Sourcing .env above lets users override OLLAMA_PORT here without breaking
# the provider URL.
OLLAMA_PORT_VAL="${OLLAMA_PORT:-11434}"
PROVIDER_BASE_URL="http://${BRIDGE_IP}:${OLLAMA_PORT_VAL}/v1"
if openshell provider list 2>/dev/null | grep -q "ollama-local"; then
    echo "Provider ollama-local already exists, skipping creation."
    echo "  (To rotate base URL/port, run: openshell provider delete ollama-local && re-run make setup)"
else
    echo "Creating provider ollama-local -> $PROVIDER_BASE_URL"
    openshell provider create \
        --name ollama-local \
        --type openai \
        --credential "OPENAI_API_KEY=ollama" \
        --config "OPENAI_BASE_URL=${PROVIDER_BASE_URL}"
fi

# Pre-warm the model before `inference set` runs its endpoint verifier.
# Nemotron-3-Super is ~86 GB and takes 60-120 s to map into VRAM on the
# first /v1/chat/completions call. The verifier's internal timeout is
# shorter than that, so the very first run of `make setup` against a
# cold Ollama always failed with "request to ... timed out". Sending a
# tiny chat completion here forces the load while we have a generous
# 240 s timeout, so the subsequent verifier call returns instantly.
echo "Pre-warming $MODEL (first request loads ~86 GB into VRAM)..."
curl -sf -m 240 -X POST "$PROVIDER_BASE_URL/chat/completions" \
    -H "Content-Type: application/json" \
    -d "{\"model\":\"$MODEL\",\"messages\":[{\"role\":\"user\",\"content\":\"hi\"}],\"max_tokens\":1}" \
    >/dev/null && echo "  Model warm." || echo "  WARN: pre-warm failed (verifier may still time out)"

openshell inference set --provider ollama-local --model "$MODEL"
echo "Inference set to ollama-local/$MODEL"
echo ""

# --- Step 2: Delete old sandbox if it exists ---
echo "--- Step 2: Clean up old sandbox ---"
if openshell sandbox list 2>/dev/null | grep -q "$SANDBOX_NAME"; then
    echo "Deleting existing sandbox: $SANDBOX_NAME"
    openshell sandbox delete "$SANDBOX_NAME" 2>/dev/null || true
    sleep 3
fi

# Stop any host-level service that owns $PORT (e.g. openclaw-gateway.service
# installed by the NemoClaw playbook as a systemd --user service). systemd
# will respawn the process if only the PID is killed, so stop the unit first.
if ss -tlnp 2>/dev/null | grep -qE "[: ]${PORT}[^0-9]"; then
    echo "Detected listener on host :$PORT — stopping before forwarding..."
    systemctl --user stop  openclaw-gateway.service 2>/dev/null || true
    systemctl --user disable openclaw-gateway.service 2>/dev/null || true
    # Kill any remaining listener not managed by systemd (e.g. stale PID)
    if ss -tlnp 2>/dev/null | grep -qE "[: ]${PORT}[^0-9]"; then
        fuser -k "${PORT}/tcp" 2>/dev/null || true
        sleep 1
    fi
fi

# Stop any stale port forwards on $PORT from prior (possibly deleted) sandboxes.
# Stale forwards block re-creation with a cryptic error like
# "× Port 18789 is already forwarded to sandbox 'dgx-demo'."
#
# We can't whitelist sandbox names (e.g. /sandbox|clinical/) — any prior
# playbook may have claimed the port. Strategy: parse `openshell forward
# list`, find the line that mentions :$PORT, and stop the forward for
# whichever sandbox owns it. Falls back to a broad sweep across all
# listed sandbox names if the line format is unfamiliar.
if openshell forward list 2>/dev/null | grep -q "[: ]$PORT[ \t]"; then
    echo "Cleaning up stale port forwards on :$PORT ..."
    # Capture every token on lines containing the port; the sandbox name
    # is whatever non-empty, non-numeric token follows the port column.
    OWNERS=$(openshell forward list 2>/dev/null \
        | awk -v p="$PORT" '$0 ~ ("(:|[ \\t])"p"([ \\t]|$)") {
            for (i=1;i<=NF;i++) {
                t=$i
                gsub(/[^A-Za-z0-9_.-]/,"",t)
                if (t != "" && t !~ /^[0-9]+$/ && t !~ /^(NAME|PORT|SANDBOX|TYPE|STATUS|running|stopped|loopback|tcp|udp)$/) print t
            }
        }' | sort -u)
    for FWD_SBOX in $OWNERS; do
        [ -n "$FWD_SBOX" ] || continue
        echo "  openshell forward stop $PORT $FWD_SBOX"
        openshell forward stop "$PORT" "$FWD_SBOX" 2>/dev/null || true
    done
fi
echo ""

# --- Step 3: Create sandbox ---
echo "--- Step 3: Create sandbox ---"
# The --no-tty SSH session can hang after sandbox creation completes
# (the SSH proxy doesn't cleanly terminate over non-interactive pipes).
# Wrap with timeout and verify the sandbox was actually created.
timeout 120 openshell sandbox create \
    --from openclaw \
    --name "$SANDBOX_NAME" \
    --policy "$POLICY_FILE" \
    --provider ollama-local \
    --forward "$PORT" \
    --keep \
    --no-tty \
    -- echo "sandbox-ok" || true

# Verify the sandbox was created regardless of timeout
if ! openshell sandbox list 2>/dev/null | grep -q "$SANDBOX_NAME"; then
    echo "ERROR: Sandbox '$SANDBOX_NAME' was not created." >&2
    exit 1
fi

# Wait for the sandbox to reach phase=Ready before uploading. The
# `sandbox create` call returns as soon as Kubernetes accepts the spec,
# but the OpenClaw image still has to pull and the pod has to start.
# Calling `sandbox upload` against a not-yet-Ready pod fails with
# "× status: FailedPrecondition, message: \"sandbox is not ready\"".
echo "Waiting for sandbox to become Ready..."
for i in $(seq 1 60); do
    PHASE=$(openshell sandbox list 2>/dev/null \
                | sed "s/$(printf '\033')\[[0-9;]*[a-zA-Z]//g" \
                | awk -v n="$SANDBOX_NAME" 'NR>1 { for (i=1;i<=NF;i++) if ($i==n) { print $NF; exit } }')
    if [ "$PHASE" = "Ready" ]; then
        echo "Sandbox Ready (after ${i} polls)."
        break
    fi
    if [ $i -eq 60 ]; then
        echo "ERROR: Sandbox '$SANDBOX_NAME' did not reach Ready in 5 min." >&2
        exit 1
    fi
    sleep 5
done
echo ""

# --- Step 4: Upload repo into sandbox ---
# Note: openshell sandbox upload (>= 0.0.44) copies the source *directory itself*
# (like `cp -r src/ dest/` creates dest/src/), not just its contents. We therefore
# upload to /sandbox/ so that the source directory `clinical-intelligence` lands at
# /sandbox/clinical-intelligence/ rather than /sandbox/clinical-intelligence/clinical-intelligence/.
echo "--- Step 4: Upload repo ---"
openshell sandbox upload "$SANDBOX_NAME" "$REPO_DIR" /sandbox/

# Fix nested directories caused by upload (analysis-methods/analysis-methods/)

# Resolve the active gateway name for the ssh-proxy ProxyCommand.
# Precedence: OPENSHELL_GATEWAY env var (set by the CLI for all subcommands) →
# active gateway from `openshell status` → fallback to 'openshell'.
# This prevents a failure when the user previously ran the NemoClaw playbook
# (which registers its gateway as 'nemoclaw' instead of 'openshell').
_gw_name() {
    if [ -n "${OPENSHELL_GATEWAY:-}" ]; then
        printf '%s' "$OPENSHELL_GATEWAY"
        return
    fi
    local name
    # Strip ANSI color codes first: `openshell status` prints e.g.
    # "Gateway:<ESC>[0m nemoclaw", and the reset code between "Gateway:" and the
    # name defeats the grep, silently falling back to 'openshell' (which then
    # fails with "Unknown gateway 'openshell'" when the active gateway differs).
    name=$(openshell status 2>/dev/null \
        | sed "s/$(printf '\033')\[[0-9;]*[a-zA-Z]//g" \
        | grep -oE 'Gateway:[[:space:]]+[A-Za-z0-9_-]+' \
        | awk '{print $NF}' | head -1)
    printf '%s' "${name:-openshell}"
}
GW_NAME="$(_gw_name)"

_sandbox() {
    ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o LogLevel=ERROR \
        -o ConnectTimeout=10 \
        -o "ProxyCommand=openshell ssh-proxy --gateway-name $GW_NAME --name $SANDBOX_NAME" \
        "sandbox@openshell-$SANDBOX_NAME" "$@"
}

_sandbox bash -s <<'FIX_NESTING'
for skill_dir in /sandbox/clinical-intelligence/skills/*/; do
    skill_name=$(basename "$skill_dir")
    nested="$skill_dir/$skill_name"
    if [ -d "$nested" ]; then
        cp -r "$nested"/* "$skill_dir/" 2>/dev/null || true
        rm -rf "$nested"
    fi
done
FIX_NESTING
echo ""

# --- Step 5: Install Python packages ---
echo "--- Step 5: Install Python packages ---"
_sandbox bash -s <<'REMOTE_SETUP'
set -euo pipefail

echo "  Creating venv..."
python3 -m venv /sandbox/.venv 2>/dev/null || true
uv pip install --python /sandbox/.venv/bin/python3 pandas matplotlib -q 2>/dev/null || \
    /sandbox/.venv/bin/pip install --timeout 120 --retries 10 pandas matplotlib

mkdir -p /sandbox/.local/lib/python3.12/site-packages
cp -r /sandbox/.venv/lib/python3.12/site-packages/* /sandbox/.local/lib/python3.12/site-packages/ 2>/dev/null || true

grep -q '/sandbox/.venv/bin' ~/.bashrc 2>/dev/null || \
    echo 'export PATH="/sandbox/.venv/bin:$PATH"' >> ~/.bashrc

/usr/bin/python3 -c "import pandas, matplotlib; print('  Python packages: OK')" 2>/dev/null || \
    echo "  WARN: system python3 cannot find packages (venv OK, this is fine)"
/sandbox/.venv/bin/python3 -c "import pandas, matplotlib; print('  Venv packages: OK')"
REMOTE_SETUP
echo ""

# --- Step 6: Deploy skills ---
echo "--- Step 6: Deploy skills ---"
_sandbox bash -s <<'SKILLS'
set -euo pipefail
mkdir -p ~/.openclaw/workspace/skills
for skill in fhir-basics clinical-knowledge analysis-methods case-summary cohort-compare molecular-viz clinical-delegation; do
    src="/sandbox/clinical-intelligence/skills/$skill"
    dst="$HOME/.openclaw/workspace/skills/$skill"
    if [ -d "$src" ]; then
        # Remove existing to avoid nested directories from cp -r
        rm -rf "$dst"
        cp -r "$src" "$dst"
        echo "  Deployed: $skill"
    fi
done
SKILLS
echo ""

# --- Step 7: Write IDENTITY.md and create memory ---
echo "--- Step 7: Write IDENTITY.md + memory ---"
_sandbox bash -s <<'IDENTITY'
# Deploy IDENTITY.md to both workspace and agent dir
mkdir -p ~/.openclaw/workspace
cp /sandbox/clinical-intelligence/IDENTITY.md ~/.openclaw/workspace/IDENTITY.md
mkdir -p ~/.openclaw/agents/main/agent
cp /sandbox/clinical-intelligence/IDENTITY.md ~/.openclaw/agents/main/agent/IDENTITY.md
echo "  IDENTITY.md deployed (workspace + agent)"

# Create memory directory to prevent ENOENT errors on session start
mkdir -p ~/.openclaw/workspace/memory
[ -f ~/.openclaw/workspace/MEMORY.md ] || echo "# Memory" > ~/.openclaw/workspace/MEMORY.md
echo "  Memory directory created"
IDENTITY
echo ""

# --- Step 8: Configure OpenClaw ---
echo "--- Step 8: Configure OpenClaw ---"
_sandbox bash -s <<'OPENCLAW_CFG'
mkdir -p ~/.openclaw
cp /sandbox/clinical-intelligence/openclaw.json ~/.openclaw/openclaw.json
OPENCLAW_CFG
echo "  openclaw.json deployed from repo"
echo ""

# --- Step 9: Register agents ---
echo "--- Step 9: Register agents ---"
_sandbox bash -s <<'AGENTS'
set -euo pipefail
for agent in patient-data labs-vitals medications analyst molecular; do
    mkdir -p ~/.openclaw/workspaces/$agent
    if [ -f "/sandbox/clinical-intelligence/agents/${agent}-agent.md" ]; then
        cp "/sandbox/clinical-intelligence/agents/${agent}-agent.md" ~/.openclaw/workspaces/$agent/AGENTS.md
    fi
    openclaw agents add $agent \
        --workspace ~/.openclaw/workspaces/$agent \
        --model local-ollama/nemotron-3-super \
        --non-interactive 2>/dev/null || true
    echo "  Registered: $agent"
done
AGENTS
echo ""

# --- Step 10: Create auth profiles ---
echo "--- Step 10: Auth profiles ---"
_sandbox bash -s <<'AUTH'
set -euo pipefail
AUTH='{"version":1,"profiles":{"ollama":{"type":"api_key","provider":"local-ollama","key":"ollama"}}}'
for agent in main patient-data labs-vitals medications analyst molecular; do
    mkdir -p ~/.openclaw/agents/$agent/agent
    echo "$AUTH" > ~/.openclaw/agents/$agent/agent/auth-profiles.json
done
echo "  Auth profiles created for all agents"
AUTH
echo ""

# --- Step 11: Start gateway ---
echo "--- Step 11: Start gateway ---"
# BIND_MODE is passed verbatim to the inner shell; the inner script then
# decides whether to add the `--bind loopback` flag. Passing an empty string
# previously triggered: "option '--bind <mode>' argument missing" because
# bash word-split a quoted empty arg into the openclaw arg vector.
if [ "$BIND_MODE" = "all" ]; then
    echo "  Binding to 0.0.0.0 (local GUI access, no tunnel needed)"
else
    echo "  Binding to loopback (SSH tunnel required for remote access)"
fi

_sandbox bash -s -- "$BIND_MODE" "$PORT" <<'GATEWAY'
set -u
# --require shim: works around `uv_interface_addresses returned Unknown
# system error 1` from os.networkInterfaces() inside the OpenShell
# sandbox kernel. Without it, OpenClaw 2026.3.x crashes during
# pickPrimaryLanIPv4 -> initSelfPresence and never binds the port.
SHIM="/sandbox/clinical-intelligence/scripts/openclaw-os-shim.js"
if [ -f "$SHIM" ]; then
    export NODE_OPTIONS="--require $SHIM --use-env-proxy"
else
    echo "  WARN: $SHIM not found; gateway may crash on networkInterfaces()" >&2
    export NODE_OPTIONS="--use-env-proxy"
fi
export NODE_TLS_REJECT_UNAUTHORIZED=0
export PATH="/sandbox/.venv/bin:$PATH"
BIND_MODE="$1"
GW_PORT="$2"
openclaw gateway stop 2>/dev/null || true
sleep 2
if [ "$BIND_MODE" = "all" ]; then
    nohup openclaw gateway run --port "$GW_PORT" --allow-unconfigured --auth none \
        > /tmp/gw.log 2>&1 &
else
    nohup openclaw gateway run --port "$GW_PORT" --allow-unconfigured --auth none \
        --bind loopback > /tmp/gw.log 2>&1 &
fi
# Poll for the gateway HTTP port instead of a fixed sleep — Node startup
# under --require is variable, and a hard 5 s sleep often missed it.
for i in $(seq 1 30); do
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:${GW_PORT}/" \
       || curl -sf -m 2 -o /dev/null "http://127.0.0.1:${GW_PORT}/__openclaw__/health"; then
        break
    fi
    sleep 1
done
tail -10 /tmp/gw.log
GW_PID=$(pgrep -f 'openclaw.*gateway' | head -1)
if [ -n "$GW_PID" ] && curl -sf -m 2 -o /dev/null "http://127.0.0.1:${GW_PORT}/__openclaw__/health"; then
    echo "  Gateway PID: $GW_PID (port ${GW_PORT} responding)"
else
    echo "  ERROR: Gateway failed to bind port ${GW_PORT}." >&2
    echo "  See /tmp/gw.log inside the sandbox for the full stack trace." >&2
    exit 1
fi
GATEWAY
echo ""

# --- Step 12: Start port forwarding ---
echo "--- Step 12: Port forwarding ---"
openshell forward start -d "$PORT" "$SANDBOX_NAME" 2>/dev/null || true
echo ""

# --- Step 13: Verify ---
echo "--- Step 13: Verify ---"
_sandbox bash -s <<'VERIFY'
echo "  Inference:"
curl -sk https://inference.local/v1/models 2>/dev/null | head -c 100 && echo "" || echo "  FAIL"

echo "  FHIR:"
curl -sk https://r4.smarthealthit.org/Patient?_count=1 2>/dev/null | head -c 100 && echo "" || echo "  FAIL"

echo "  Blocked (should fail):"
curl --max-time 3 https://google.com 2>&1 | head -c 80 && echo "" || echo "  BLOCKED (good)"

echo "  Smoke test:"
openclaw agent --local --session-id smoke --thinking off --message "Say OK" --timeout 60 2>&1 | tail -5
VERIFY
echo ""

echo "=== Setup Complete ==="
if [ "$BIND_MODE" = "all" ]; then
    echo "Open in browser: http://localhost:$PORT/"
else
    echo "SSH tunnel from your machine:"
    echo "  ssh -f -N -L $PORT:localhost:$PORT <user>@<dgx-ip>"
    echo "Then open: http://localhost:$PORT/"
fi
echo ""
echo "Canvas URL: http://localhost:$PORT/__openclaw__/canvas/"
echo ""
echo "To restart gateway later:"
echo "  openshell sandbox connect $SANDBOX_NAME"
echo "  bash /sandbox/clinical-intelligence/scripts/restart_sandbox.sh [--local] [model]"
