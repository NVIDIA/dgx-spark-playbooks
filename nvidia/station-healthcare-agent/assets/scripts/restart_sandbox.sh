#!/bin/bash
# Restart the OpenClaw gateway inside the clinical-sandbox.
#
# Usage:
#   bash restart_sandbox.sh [--local] [model]
#
# Options:
#   --local    Bind gateway to 0.0.0.0 (accessible from local browser without SSH tunnel)
#              Default: loopback only (requires SSH tunnel for remote access)
#   model      Ollama model name (default: nemotron-3-super)
#
# Examples:
#   bash restart_sandbox.sh                          # loopback, nemotron-3-super
#   bash restart_sandbox.sh --local                  # local GUI, nemotron-3-super
#   bash restart_sandbox.sh --local nemotron-3-super # local GUI, nemotron-3-super

BIND_MODE="loopback"
MODEL="nemotron-3-super:120b-a12b"

for arg in "$@"; do
    case "$arg" in
        --local) BIND_MODE="all" ;;
        --*) echo "Unknown option: $arg"; exit 1 ;;
        *) MODEL="$arg" ;;
    esac
done

if [ "$BIND_MODE" = "all" ]; then
    echo "--- bind: 0.0.0.0 (local GUI access) ---"
else
    echo "--- bind: loopback (SSH tunnel required) ---"
fi

# --require shim: prevents the gateway from crashing when
# os.networkInterfaces() throws ERR_SYSTEM_ERROR inside the sandbox.
# See assets/scripts/openclaw-os-shim.js for the full rationale.
SHIM="/sandbox/clinical-intelligence/scripts/openclaw-os-shim.js"
if [ -f "$SHIM" ]; then
    export NODE_OPTIONS="--require $SHIM --use-env-proxy"
else
    echo "WARN: $SHIM not found; gateway may crash on networkInterfaces()" >&2
    export NODE_OPTIONS="--use-env-proxy"
fi
export NODE_TLS_REJECT_UNAUTHORIZED=0
export PATH="/sandbox/.venv/bin:$PATH"

echo "--- stop ---"
openclaw gateway stop 2>/dev/null; sleep 2
kill $(pgrep -f openclaw-agent) 2>/dev/null; sleep 1
find ~/.openclaw -name '*.lock' -delete 2>/dev/null

echo "--- model: $MODEL ---"
# `agents.defaults.model` is a STRING in the current openclaw.json schema,
# not a dict. The previous `[..]['primary']=...` form crashed with
# "TypeError: 'str' object does not support item assignment". Detect the
# type at runtime so this script keeps working if openclaw ever revives
# the legacy {primary, fallback} shape.
python3 - "$MODEL" <<'PY'
import json, sys
from pathlib import Path
m = sys.argv[1]
p = Path.home() / '.openclaw' / 'openclaw.json'
d = json.loads(p.read_text())

provs = d.get('models', {}).get('providers', {})
if 'local-ollama' in provs and provs['local-ollama'].get('models'):
    provs['local-ollama']['models'][0]['id'] = m

defaults = d.get('agents', {}).get('defaults', {})
existing = defaults.get('model')
if isinstance(existing, dict):
    existing['primary'] = f'local-ollama/{m}'
else:
    defaults['model'] = f'local-ollama/{m}'

for a in d.get('agents', {}).get('list', []):
    if a.get('model'):
        a['model'] = f'local-ollama/{m}'

p.write_text(json.dumps(d, indent=2))
print(f'  set to {m}')
PY

echo "--- gateway ---"
# Branch instead of interpolating an empty $BIND_FLAG into the openclaw argv —
# bash word-splitting an empty quoted arg into commander.js triggers
# "option '--bind <mode>' argument missing".
if [ "$BIND_MODE" = "all" ]; then
    nohup openclaw gateway run --port 18789 --allow-unconfigured --auth none \
        > /tmp/gw.log 2>&1 &
else
    nohup openclaw gateway run --port 18789 --allow-unconfigured --auth none \
        --bind loopback > /tmp/gw.log 2>&1 &
fi
# Poll for the gateway HTTP port instead of a fixed sleep — Node startup
# under --require is variable.
for i in $(seq 1 30); do
    if curl -sf -m 2 -o /dev/null "http://127.0.0.1:18789/__openclaw__/health" \
       || curl -sf -m 2 -o /dev/null "http://127.0.0.1:18789/"; then
        break
    fi
    sleep 1
done
tail -10 /tmp/gw.log
pgrep -la openclaw
if curl -sf -m 2 -o /dev/null "http://127.0.0.1:18789/__openclaw__/health" \
   || curl -sf -m 2 -o /dev/null "http://127.0.0.1:18789/"; then
    echo "--- gateway: UP ---"
else
    echo "--- gateway: DOWN (see /tmp/gw.log) ---"
    exit 1
fi

echo "--- smoke ---"
openclaw agent --local --session-id smoke --thinking off --message "Say OK" --timeout 60 2>&1
echo "--- done ---"
