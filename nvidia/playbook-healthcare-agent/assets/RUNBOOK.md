# Technical Runbook

Installation, configuration, troubleshooting, and operational details for the Clinical Intelligence Playbook.

> [!IMPORTANT]
> The supported install path is **Docker Ollama via `make up`** (see `instructions.md` Step 3 of the parent playbook, or `SETUP-GUIDE.md`). The manual steps below are for advanced users who want to wire components by hand or substitute a host Ollama install. If host Ollama is already running on port 11434, stop it (`sudo systemctl stop ollama`) or override `OLLAMA_PORT` in `.env` before starting Docker Ollama.

---

## Quick Start (OpenShell Sandbox on GB300)

This sets up an isolated OpenShell sandbox with OpenClaw and a local Ollama inference backend. Everything the agent touches — filesystem, network, processes — is confined to the sandbox.

**Requirements:** DGX Station GB300, Python 3.10+, `uv`, Docker + NVIDIA Container Toolkit, OpenShell CLI ≥ 0.0.44, Node.js 22+ LTS.

### 1. Install OpenShell

The official installer provides both the `openshell` CLI and the `openshell-gateway` daemon — all this playbook needs (no NemoClaw required):

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
# Open a new shell (or `source ~/.bashrc`) so ~/.local/bin is on PATH, then:
openshell --version   # >= 0.0.44
```

### 2. Start Ollama (Docker, recommended)

The bundled `docker-compose.yml` runs Ollama in a container, binds it to `0.0.0.0:${OLLAMA_PORT:-11434}`, and pins it to a single GPU (`LLM_GPU` in `.env`):

```bash
make up        # docker compose up -d ollama openfold3 + model-pull
make status    # confirm Ollama and OpenFold3 are healthy
```

> [!TIP]
> **Host Ollama alternative.** If you prefer running Ollama on the host (e.g., to share with another playbook), `sudo systemctl stop ollama` first or set `OLLAMA_PORT` in `.env` to a free port. Then `OLLAMA_HOST=0.0.0.0 ollama serve` and `ollama pull nemotron-3-super:120b-a12b`. The host's systemd unit binds to `127.0.0.1` by default — override with `Environment="OLLAMA_HOST=0.0.0.0"` in `/etc/systemd/system/ollama.service.d/override.conf` so the sandbox can reach Ollama via the Docker bridge.

### 3. Pull Model (Docker path runs this automatically)

`make up` invokes the `model-pull` service, which pulls `${OLLAMA_MODEL:-nemotron-3-super:120b-a12b}` (~86 GB) on first run. Subsequent runs skip if the model is cached.

If you opted into host Ollama, run `ollama pull nemotron-3-super:120b-a12b` manually.

### 4. Configure OpenShell Inference Provider

`make setup` runs this for you (and detects the Docker bridge IP automatically). To do it manually, replace `<HOST_IP>` with your Docker bridge IP (`ip -4 addr show docker0 | grep -oP 'inet \K[\d.]+'`):

```bash
openshell provider create \
  --name ollama-local \
  --type openai \
  --credential OPENAI_API_KEY=ollama \
  --config OPENAI_BASE_URL=http://<HOST_IP>:${OLLAMA_PORT:-11434}/v1

openshell inference set \
  --provider ollama-local \
  --model nemotron-3-super:120b-a12b
```

> [!NOTE]
> Current OpenShell releases do not accept the `--base-url` shorthand for `provider create`. Use `--config OPENAI_BASE_URL=...` as shown above. The `setup_sandbox.sh` script uses this form.

### 5. Generate the Sandbox Policy

The repo's `sandbox-policy.yaml` is a **template**: it contains a `__DOCKER_BRIDGE_IP__` placeholder that must be substituted with the host's `docker0` IP before the policy is valid. The helper `scripts/gen_sandbox_policy.sh` does this automatically and writes `sandbox-policy-local.yaml`:

```bash
bash scripts/gen_sandbox_policy.sh   # writes sandbox-policy-local.yaml
```

`make setup` invokes this generator. If you are running steps by hand, always pass the **generated** file (`sandbox-policy-local.yaml`) to `openshell sandbox create`, not the template.

Verify the policy is working after sandbox creation:

```bash
# These should succeed:
curl -sk https://r4.smarthealthit.org/Patient?_count=1 | head -3    # FHIR
curl -sk https://inference.local/v1/models | head -3                  # LLM

# These should fail (blocked):
curl -sk --max-time 5 https://google.com                             # denied
curl -sk --max-time 5 https://api.openai.com                        # denied
ping 8.8.8.8 -c 1                                                    # Operation not permitted
```

### 6. Create the Sandbox

```bash
# Stop any stale forwards from a prior sandbox using the same port — these
# block re-creation with a "port already forwarded" error.
for s in $(openshell forward list 2>/dev/null | awk '/:18789 /{print $NF}' | sort -u); do
  openshell forward stop 18789 "$s" 2>/dev/null || true
done

openshell sandbox create \
  --policy sandbox-policy-local.yaml \
  --provider ollama-local \
  --forward 18789 \
  --keep
```

`--keep` prevents the sandbox from being torn down on exit so you can re-enter it later.

### 7. Inside the Sandbox: Set Up OpenClaw

> **Note:** These manual steps (7-9) are automated by `make setup`. Use them only if you need to customize individual steps.

Everything from here runs **inside** the sandbox shell that OpenShell drops you into.

```bash
git clone https://github.com/jaival-nvidia/clinical-intelligence.git
cd clinical-intelligence
uv pip install pandas matplotlib

curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.1/install.sh | bash
export NVM_DIR="$HOME/.nvm" && [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
nvm install --lts
npm install -g openclaw@latest
```

### 8. Inside the Sandbox: Configure Inference Provider

OpenShell maps the host Ollama to `inference.local` inside the sandbox. Save this as `configure_inference_local.py` and run it to register a custom OpenClaw provider:

```python
#!/usr/bin/env python3
"""configure_inference_local.py — patches OpenClaw config for OpenShell sandbox."""
import json
from pathlib import Path

config_path = Path.home() / ".openclaw" / "openclaw.json"
config_path.parent.mkdir(parents=True, exist_ok=True)

config = json.loads(config_path.read_text()) if config_path.exists() else {}

config.setdefault("providers", {})
config["providers"]["local-ollama"] = {
    "type": "openai-compatible",
    "baseUrl": "https://inference.local/v1",
    "apiKey": "ollama",
    "models": ["nemotron-3-super:120b-a12b"]
}

defaults = config.setdefault("agents", {}).setdefault("defaults", {})
defaults["model"] = "local-ollama/nemotron-3-super:120b-a12b"
defaults["timeoutSeconds"] = 600
sub = defaults.setdefault("subagents", {})
sub.update({"maxSpawnDepth": 2, "maxConcurrent": 4, "runTimeoutSeconds": 600})

config_path.write_text(json.dumps(config, indent=2))
print(f"Wrote {config_path}")
print("Provider: local-ollama -> https://inference.local/v1")
print("Model:    nemotron-3-super:120b-a12b")
```

```bash
python3 configure_inference_local.py
```

### 9. Inside the Sandbox: Install Skills and Agents

```bash
# Deploy all 7 skills to ~/.openclaw/workspace/skills/ (matching setup_sandbox.sh)
mkdir -p ~/.openclaw/workspace/skills
cp -r skills/fhir-basics skills/clinical-knowledge skills/analysis-methods \
      skills/case-summary skills/cohort-compare skills/molecular-viz \
      skills/clinical-delegation ~/.openclaw/workspace/skills/

# Register sub-agents
for agent in patient-data labs-vitals medications analyst molecular; do
  mkdir -p ~/.openclaw/workspaces/$agent
  cp agents/${agent}-agent.md ~/.openclaw/workspaces/$agent/AGENTS.md
  openclaw agents add $agent \
    --workspace ~/.openclaw/workspaces/$agent \
    --model local-ollama/nemotron-3-super:120b-a12b \
    --non-interactive
done

# TOOLS.md for sub-agents
for agent in patient-data labs-vitals medications analyst molecular; do
  cat > ~/.openclaw/workspaces/$agent/TOOLS.md << 'TOOLSEOF'
# Tools
Use the `exec` tool to run Python scripts against the FHIR server.
Call exec with: cat > /tmp/script.py << 'PYEOF'
import subprocess, json
r = subprocess.run(["curl", "-sf", "--max-time", "30", "https://r4.smarthealthit.org/Patient?_count=1"],
                   capture_output=True, text=True, timeout=35)
data = json.loads(r.stdout)
# your code here
PYEOF
python3 /tmp/script.py
Always use `exec` to run code. Do NOT use `write` for scripts.
Use subprocess.run(["curl", ...]) + json.loads() for HTTP requests. Do NOT use the requests library.
Print all results to stdout.
TOOLSEOF
done

# Auth profiles (provider name must match: local-ollama)
AUTH='{"version":1,"profiles":{"ollama":{"type":"api_key","provider":"local-ollama","key":"ollama"}}}'
for agent in main patient-data labs-vitals medications analyst molecular; do
  mkdir -p ~/.openclaw/agents/$agent/agent
  echo "$AUTH" > ~/.openclaw/agents/$agent/agent/auth-profiles.json
done

# Workspace stubs
for agent in patient-data labs-vitals medications analyst molecular; do
  echo "# Identity" > ~/.openclaw/workspaces/$agent/IDENTITY.md
  echo "# User" > ~/.openclaw/workspaces/$agent/USER.md
done
echo -e "# Identity\nClinical Intelligence Coordinator" > ~/.openclaw/workspace/IDENTITY.md
echo "# User" > ~/.openclaw/workspace/USER.md

# Allow coordinator to spawn sub-agents + disable unused skills
python3 -c "
import json
with open('$HOME/.openclaw/openclaw.json') as f: d = json.load(f)
for a in d.get('agents', {}).get('list', []):
    if a['id'] == 'main': a['subagents'] = {'allowAgents': ['*']}
d.setdefault('tools', {})['sessions'] = {'visibility': 'all'}
d.setdefault('skills', {})['entries'] = {
    'weather': {'enabled': False}, 'tmux': {'enabled': False},
    'healthcheck': {'enabled': False}, 'gh-issues': {'enabled': False},
    'skill-creator': {'enabled': False}, 'github': {'enabled': False}
}
with open('$HOME/.openclaw/openclaw.json', 'w') as f: json.dump(d, f, indent=2)
"
```

### 10. Verify Inside the Sandbox

```bash
# Verify inference routing
curl -sk https://inference.local/v1/models | head -3

# Verify FHIR access
curl -sk https://r4.smarthealthit.org/Patient?_count=1 | head -3

# Verify deny (should timeout or fail)
curl -sk --max-time 5 https://google.com

# Run a test
openclaw agent --local --session-id smoke --thinking off \
  --message "Say hello in one sentence" --timeout 120
```

---

## Compatible Models

Any Ollama model with native tool calling works. Tested:

- **nemotron-3-super:120b-a12b** — recommended, MoE, fast inference
- **qwen3:235b** — larger, slower, reliable
- **qwen3-coder:480b** — best for code generation
- **qwen3:32b** — for constrained environments
- **qwen2.5:72b** — proven stable, good tool calling

To switch: `ollama pull <model>`, then `openshell cluster inference set --provider ollama-local --model <model>`, then inside the sandbox: `openclaw config set agents.defaults.model local-ollama/<model>`.

---

## OpenShell Sandbox Security

OpenShell sandboxes use an **implicit-deny** model. All network egress, filesystem writes, and resource usage are blocked by default. The `sandbox-policy.yaml` explicitly whitelists what the sandbox can reach.

**In practice:**

- The agent **cannot** reach the internet except for the FHIR endpoint in the policy.
- The agent **cannot** exfiltrate data — no outbound HTTP, no DNS to arbitrary hosts.
- LLM inference routes through `inference.local`, a OpenShell-managed bridge to the host Ollama. This never leaves the machine.
- Filesystem writes are confined to `/tmp`, `/home/sandbox`, and `~/.openclaw`.
- Resource caps prevent runaway processes from consuming the host.

### Modifying the Policy

To allow additional FHIR endpoints (e.g., a hospital server), add entries under `network_policies` in `sandbox-policy.yaml`:

```yaml
  hospital_fhir:
    name: hospital_fhir
    endpoints:
      - host: fhir.your-hospital.org
        port: 443
        protocol: https
        description: Hospital FHIR R4 endpoint
```

Then recreate the sandbox:

```bash
openshell sandbox delete <sandbox-name>
openshell sandbox create --policy sandbox-policy.yaml --provider ollama-local --forward 18789 --keep
```

### Audit

OpenShell logs all network connections attempted by the sandbox, including blocked ones:

```bash
openshell sandbox logs <sandbox-name> --filter network
```

---

## Running Workflows

All `openclaw` commands below run **inside the OpenShell sandbox**.

```bash
openshell sandbox connect <sandbox-name>
```

### CMS122 Diabetes Quality Gap

```bash
openclaw agent --local --session-id test-cms122 \
  --thinking off \
  --message "Find all diabetic patients, get their latest A1c and medications. Identify gap patients with A1c above 9% not on insulin or GLP-1. Show the A1c distribution as a histogram." \
  --timeout 600
```

Validate: `python3 scripts/validate_and_run.py --validate-only /tmp/cms122.py`

### Patient Case Summary

```bash
openclaw agent --local --session-id test-case \
  --thinking off \
  --message "Look up the first patient. Compile a case summary: demographics, conditions, recent labs (flag abnormal), and medications." \
  --timeout 600
```

### Ad-Hoc Cross-Reference

```bash
openclaw agent --local --session-id test-adhoc \
  --thinking off \
  --message "Which patients have both diabetes and hypertension? For the overlap, get their latest HbA1c and blood pressure." \
  --timeout 600
```

### Conversational Follow-Up Test

Three turns in the same session (tests context persistence):

```bash
openclaw agent --local --session-id follow-up-test --thinking off \
  --message "Find all diabetic patients. Print the count and list their IDs." \
  --timeout 600

openclaw agent --local --session-id follow-up-test --thinking off \
  --message "From those diabetic patients, which ones also have hypertension? Intersect the two groups." \
  --timeout 600

openclaw agent --local --session-id follow-up-test --thinking off \
  --message "For the patients with both diabetes and hypertension, get their latest HbA1c and eGFR. Flag anyone with eGFR below 60 as CKD risk." \
  --timeout 600
```

### Bulk Validation

```bash
for f in /tmp/cms122.py /tmp/case.py /tmp/adhoc.py /tmp/step1.py /tmp/step2.py /tmp/step3.py; do
  echo "--- $f ---"
  python3 scripts/validate_and_run.py --validate-only "$f"
done
```

---

## Connecting a Hospital FHIR Endpoint

1. Get OAuth2 credentials from your integration team (client_id, token_endpoint, scopes `patient/*.read`).
2. Add the hospital host to `sandbox-policy.yaml` under `network_policies` and recreate the sandbox.
3. Specify the FHIR URL directly in your prompt instead of `https://r4.smarthealthit.org`.
4. Add an auth helper to `skills/fhir-basics/SKILL.md` for token-based requests.
5. Test: `python3 scripts/test-fhir.py --url https://your-hospital.org/fhir --token YOUR_TOKEN`

---

## Troubleshooting

### Sandbox Provisioning Failures

| Symptom | Fix |
|---------|-----|
| `sandbox create` hangs | Check host network; `openshell sandbox create --verbose` for pull progress |
| `failed to pull image` | GHCR rate limit — `docker login ghcr.io` with a PAT, or wait and retry |
| `policy validation error` | Check YAML syntax in `sandbox-policy.yaml`; recreate the sandbox |
| `provider not found` | `openshell provider list` — confirm `ollama-local` exists |

### inference.local Not Resolving

| Check | Fix |
|-------|-----|
| Ollama listening on 0.0.0.0? | `ss -tlnp \| grep 11434` on host. Docker Ollama (default): `make down && make up`. Host Ollama: add `Environment="OLLAMA_HOST=0.0.0.0"` to `/etc/systemd/system/ollama.service.d/override.conf` and `sudo systemctl restart ollama`. |
| Provider base URL correct? | `openshell provider show ollama-local` — must use host IP, not `127.0.0.1` |
| DNS working inside sandbox? | `nslookup inference.local` inside sandbox; recreate sandbox if broken |
| Firewall blocking bridge? | `curl -sk https://inference.local/v1/models` inside sandbox; check host firewall |

### LLM Connection Failures

| Check | Fix |
|-------|-----|
| Ollama running? | `curl -sk https://inference.local/v1/models` — start Ollama on host if down |
| Model pulled? | `ollama list` on host — `ollama pull nemotron-3-super:120b-a12b` if missing |
| Model loaded? | `ollama ps` on host — warmup: `curl -sk https://inference.local/v1/chat/completions -d '{"model":"nemotron-3-super:120b-a12b","messages":[{"role":"user","content":"ping"}]}'` |

### OpenClaw Auth Errors

"No API key found for provider" — Create auth profiles as shown in step 9. Provider name must match `local-ollama`.

"Profile ollama timed out" — Model unloaded. Warm it up from inside the sandbox: `curl -sk https://inference.local/v1/chat/completions -d '{"model":"nemotron-3-super:120b-a12b","messages":[{"role":"user","content":"ping"}]}' > /dev/null`

### FHIR Empty Results

| Check | Fix |
|-------|-----|
| Reachable from sandbox? | `curl -s https://r4.smarthealthit.org/metadata \| head -20` inside sandbox |
| Blocked by policy? | `openshell sandbox logs <sandbox-name> --filter network` — look for denied connections |
| SNOMED code format? | Use bare codes: `code=44054006`. The test server returns empty results with system-qualified URIs. |
| Page size? | Ensure `_count=200` in cohort queries |

### BP Values Always Null

Blood pressure is stored as a component Observation (LOINC 85354-9). The `fhir-basics` skill includes `get_bp()` that handles both panel and standalone formats. If the LLM queries `8480-6` directly without checking the panel, regenerate.

### Slow Performance

| Bottleneck | Fix |
|-----------|-----|
| LLM generation > 120s | Check GPU utilization with `nvidia-smi` on host; model may be on CPU |
| FHIR queries > 2s/patient | Test server can be slow; use offline mode for demos |
| Model keeps unloading | Set `keep_alive` to 120m in warmup curl |
| First sandbox command slow | One-time init cost; subsequent commands are fast |

---

## Managing Sandboxes

```bash
openshell sandbox list                    # list running sandboxes
openshell sandbox connect <sandbox-name>  # re-enter existing sandbox
openshell sandbox logs <sandbox-name>     # view sandbox logs
openshell sandbox delete <sandbox-name>   # tear down (after policy changes, recreate)
```

---

## Performance Notes

Performance depends on the model, hardware, and FHIR server responsiveness. LLM generation time typically dominates over FHIR query time. Larger cohorts take longer due to per-patient REST queries; production deployments should use Bulk FHIR for populations over 200.
