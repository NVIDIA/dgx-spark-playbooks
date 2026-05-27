# Local Healthcare Agent on DGX Station

> Run healthcare AI agents that analyze patient data and predict protein structures in an OpenShell sandbox on DGX Station


## Table of Contents

- [Overview](#overview)
  - [How it works](#how-it-works)
  - [Notice and disclaimers](#notice-and-disclaimers)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

This playbook deploys a healthcare AI agent system on your DGX Station. Six agents (one coordinator and five specialists) query patient records, identify clinical care gaps, and predict 3D protein structures. LLM inference (Nemotron 3 Super) and protein-structure prediction (OpenFold3) run on the local GPU, and patient data never passes through a hosted LLM, OpenFold3, or PubChem. An OpenShell sandbox enforces implicit-deny networking, so only a small whitelist of external endpoints — the SMART Health IT FHIR test server, PubChem reference lookups, and viewer CDNs — is reachable for read-only metadata and front-end assets. See the **Security** table below for the full allowed-endpoint list.

Clinical knowledge lives in editable Markdown skill files. Change a lab threshold, add a drug to a classification list, or update a quality measure definition — it takes effect on the next query, no retraining required.

### How it works

The system has four layers.

**Inference** — Nemotron 3 Super (120B MoE) runs locally via Ollama in a Docker container on the DGX Station GPU. No cloud APIs, no data transfer. Inside the sandbox, agents call `inference.local`, a virtual hostname that OpenShell routes to Ollama over the Docker bridge network.

**Orchestration** — OpenClaw coordinates five specialist agents. The coordinator receives the user's question, writes and executes Python scripts directly, and delegates to specialists when the query spans multiple domains.

| Agent | Role | Example |
|-------|------|---------|
| **Coordinator** | Receives questions, writes Python, executes analysis | "Find all diabetic patients and get their latest HbA1c" |
| patient-data | Finds patients, retrieves demographics and conditions | "Look up patient Aaron697" |
| labs-vitals | Lab results, vitals, blood pressure (component observations) | "Get their latest eGFR and potassium" |
| medications | Active prescriptions, drug class matching | "Which patients are on an ACE inhibitor?" |
| analyst | Python analysis, care gaps, CMS quality measures, charts | "Generate a histogram of A1c values" |
| molecular | 3D protein-ligand visualization via OpenFold3 + PubChem | "Show atorvastatin bound to its target" |

**Knowledge** — Editable Markdown skill files provide clinical context that agents read at query time. For example, from `skills/clinical-knowledge/SKILL.md`:

| Lab | Normal | Concerning | Notes |
|-----|--------|------------|-------|
| HbA1c | < 7.0% (diabetic target) | > 9.0% = poor control | ADA 2024 guidelines |
| eGFR | > 90 | < 60 = moderate CKD | CKD-EPI 2021 equation |
| BP | < 120/80 | ≥ 140/90 = uncontrolled HTN | ACC/AHA 2024 |

Change `9.0%` to `8.5%` and the next care gap query uses the stricter threshold. Other editable items include LOINC lab codes (`fhir-basics`), SNOMED condition codes, drug classification lists, and CMS quality measure definitions (`clinical-knowledge`).

**Security** — OpenShell enforces an implicit-deny sandbox. Only these endpoints are reachable:

| Rule | Target | Purpose |
|------|--------|---------|
| LLM inference | `https://inference.local` (port 443) | Routed to Ollama (never leaves the machine). HTTPS only — plain `http://inference.local` is denied. |
| FHIR data | `r4.smarthealthit.org` | Patient data queries (read-only) |
| PubChem | `pubchem.ncbi.nlm.nih.gov` | Drug SMILES lookup (read-only) |
| OpenFold3 | Docker bridge IP, port 8000 | Protein structure prediction |
| CDN | `code.jquery.com`, `3dmol.org`, `unpkg.com` | JavaScript for 3D viewers (read-only) |
| Everything else | `*` | **Denied** |

> [!NOTE]
> Additional rules for GitHub, npm, and PyPI are included for build dependencies during sandbox setup. These are setup-only and not used at runtime.

Patient data flows from FHIR → sandbox → Python execution. It never passes through the LLM, OpenFold3, or PubChem.

## What you'll accomplish

By the end of this playbook you will have six healthcare agents running inside a sandboxed environment on your DGX Station, with local inference, editable clinical knowledge, and verified network isolation.

- Serve Nemotron 3 Super (120B MoE) locally via Ollama
- Deploy six agents (coordinator + five specialists) with OpenClaw inside an OpenShell sandbox
- Query FHIR patient data, identify care gaps, and generate charts
- Predict 3D protein structures using OpenFold3 NIM
- Edit a skill file and see the change take effect immediately
- Verify implicit-deny networking — confirm unauthorized endpoints are blocked

## What to know before starting

- Basic use of the Linux terminal and SSH
- Familiarity with Docker (`docker run`, `docker compose`)
- Domain knowledge is not required — the skill files provide clinical context so the LLM does not need medical fine-tuning

## Prerequisites

**Hardware Requirements:**

- NVIDIA Grace Blackwell GB300 Ultra Superchip System (DGX Station)
- A single GPU with **at least 150 GB free GPU memory** to host Nemotron 3 Super (~94 GB resident) plus OpenFold3 (~40–80 GB on-demand). On dual-GPU stations (e.g., RTX PRO 6000 + GB300), target the GB300; the RTX PRO 6000's 98 GB is too small to load Nemotron 3 Super safely.
- **At least 200 GB available storage** on `/` for model downloads and containers (86 GB Ollama model + ~10 GB Docker images + working space). Verify with `df -h /` before starting.

**Software Requirements:**

- Docker with NVIDIA Container Toolkit: `docker info --format '{{.ServerVersion}}'`
- Node.js v22+: `node --version` (DGX Station ships with v18 — see Step 1 of `instructions.md` for upgrade)
- OpenShell CLI >= 0.0.33: `openshell --version` (binary installs to `~/.local/bin/openshell` — add to PATH; see Step 1 of `instructions.md`)
- NVIDIA NGC API key from [ngc.nvidia.com](https://ngc.nvidia.com/setup/api-key) (free) **and** Docker authentication for `nvcr.io` (`docker login nvcr.io`) so the OpenFold3 NIM image pull succeeds — see Step 2 of `instructions.md`
- Network access to `nvcr.io` (NGC registry), `ollama.com` (model downloads), and `r4.smarthealthit.org` (FHIR data server)
- Web browser access to `http://<STATION_IP>:18789`

> [!NOTE]
> This playbook runs Ollama as a Docker container; you do **not** need to install Ollama on the host. If host Ollama is already running (e.g., from the NemoClaw playbook), stop it before Step 3 of `instructions.md` to free port 11434, or override `OLLAMA_PORT` in `.env`.

If Docker, the NVIDIA runtime, or OpenShell are not yet installed, complete the NemoClaw playbook (`nvidia/station-nemoclaw/instructions.md`) Steps 1–4 first (~30–45 minutes).

## Ancillary files

All assets are bundled in the `assets/` directory of this playbook, copied to the DGX Station in Step 2.

- `Makefile` — One-command operations: `make up`, `make setup`, `make check`, `make test`
- `sandbox-policy.yaml` — OpenShell network policy (L7 endpoint whitelist)
- `skills/` — Editable Markdown skill files the agents read at query time
- `agents/` — Specialist agent definitions (one `.md` per agent)
- `docker-compose.yml` — Ollama and OpenFold3 NIM services

Supporting scripts (`setup_sandbox.sh`, `check_sandbox_config.sh`, `build_viewer.py`) are called by the Makefile.

## Time & risk

* **Estimated time:** ~60 minutes on first run (dominated by the ~86 GB Nemotron 3 Super model download). Under 5 minutes on subsequent runs with the model cached. Active hands-on time is ~15 minutes.
* **Risk level:** Medium — agents execute Python code inside an OpenShell sandbox. Filesystem, network, and process access are restricted. Use a clean environment for the demo.
  * Large model downloads (~86 GB) may fail on slow or unstable connections
  * OpenFold3 NIM takes ~3 minutes to load — the healthcheck waits automatically
* **Rollback:** `openshell sandbox delete clinical-sandbox`, `make down`, `make clean` (see Cleanup in Instructions).
* **Last Updated:** 05/12/2026
  * First Publication

### Notice and disclaimers

#### Quick start safety check

**Use only a clean environment.** Run this demo on a fresh device or VM with no personal data, confidential information, or sensitive credentials. Keep it isolated like a sandbox.

By installing this demo, you accept responsibility for all third-party components, including reviewing their licenses, terms, and security posture. Read and accept before you install or use.

#### What you're getting

This experience is provided "AS IS" for demonstration purposes only — no warranties, no guarantees. This is a demo, not a production-ready solution. It is not a regulated medical device. Test data is synthetic (Synthea). All clinical decisions must be made by qualified clinicians.

#### Key risks with AI agents

- **Data leakage** — Any materials the agent accesses could be exposed, leaked, or stolen.
- **Malicious code execution** — The agent or its connected tools could expose your system to malicious code or cyber-attacks.
- **Unintended actions** — The agent might modify or delete files, send messages, or access services without explicit approval.
- **Prompt injection and manipulation** — External inputs or connected content could hijack the agent's behavior in unexpected ways.

#### Participant acknowledgement

By participating in this demo, you acknowledge that you are solely responsible for your configuration and for any data, accounts, and tools you connect. To the maximum extent permitted by law, NVIDIA is not responsible for any loss of data, device damage, security incidents, or other harm arising from your configuration or use of these demo materials, including OpenClaw or any connected tools or services.

## Instructions

> [!IMPORTANT]
> This playbook requires Docker (with NVIDIA runtime), Node.js v22, and OpenShell CLI >= 0.0.33. If Docker, the NVIDIA runtime, or OpenShell are missing, complete the NemoClaw playbook (`nvidia/station-nemoclaw/instructions.md`) Steps 1–4 first — that takes about 30–45 minutes. Ollama runs as a Docker container in this playbook (host Ollama is not required and will conflict with port 11434, see Step 3).

> [!NOTE]
> Steps 1–3 are prerequisites. Steps 4–5 configure infrastructure and deploy the agent. Steps 6–9 are the demo. Steps 10–11 are cleanup and next steps.

## Step 1. Verify your environment

Confirm your DGX Station has the required software and free disk space:

```bash
## OpenShell installs to ~/.local/bin, which is not on the default
## non-interactive PATH. Add it before running anything below.
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
export PATH="$HOME/.local/bin:$PATH"

nvidia-smi
docker info --format '{{.ServerVersion}}'
node --version
ollama --version 2>/dev/null || echo "ollama not installed (OK — Docker provides it)"
openshell --version
df -h /
```

Expected: Blackwell Ultra GPU, Docker >= 23.0.1, **Node.js v22.x** (the DGX Station ships with v18 — see below), OpenShell >= 0.0.33, and **at least 200 GB free** on `/` (86 GB model + Docker images + working space).

> [!WARNING]
> If `openshell --version` says `command not found`, the binary is at `~/.local/bin/openshell` but isn't on PATH. Run the `export PATH=...` line above and re-source `~/.bashrc`. Without this, every `openshell` and `make` command in later steps fails.

> [!TIP]
> `make prereq` (run from `~/clinical-intelligence` after Step 2) bundles all of the checks below — Docker, Node version, OpenShell, disk space, GPU, port 11434, and NGC auth — into one command.

**If `node --version` reports v18.x or older**, install Node.js v22 before continuing:

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
node --version   # should now show v22.x
```

**Host Ollama must not be on port 11434.** This playbook runs Ollama in a Docker container that binds 11434 on the host. Run **both** of these:

```bash
## Always run — succeeds silently if no service exists, stops the host
## Ollama daemon if NemoClaw or another playbook left one running.
sudo systemctl stop ollama 2>/dev/null || true
sudo systemctl disable ollama 2>/dev/null || true

## Verify nothing else owns 11434.
ss -tlnp 2>/dev/null | grep 11434 || echo 'port 11434 free'
```

Expected: `port 11434 free`. If the line still shows a listener, something else (an old `ollama serve`, another container, etc.) owns the port — stop it, or change `OLLAMA_PORT` in `.env` (Step 2) to a free port such as `11435`. `make setup` sources `.env` and configures the sandbox provider against the override.

**Stale OpenShell gateway?** If you previously ran the NemoClaw playbook, an existing gateway will be silently reused under the new name. To start clean:

```bash
openshell gateway destroy 2>/dev/null || true
```

## Step 2. Copy the assets and configure

The playbook assets include the Docker Compose file, agent definitions, skill files, and setup scripts. Locate the `assets/` directory shipped with this playbook (DGX Station catalog or your local clone) and copy it to `~/clinical-intelligence`:

```bash
## Find the playbook directory first. Try the common locations:
PLAYBOOK_DIR=""
for d in ~/dgx-spark-playbooks/nvidia/station-healthcare-agent \
         /opt/dgx-spark-playbooks/nvidia/station-healthcare-agent \
         /usr/local/share/dgx-spark-playbooks/nvidia/station-healthcare-agent; do
  if [ -d "$d/assets" ]; then PLAYBOOK_DIR="$d"; break; fi
done

if [ -z "$PLAYBOOK_DIR" ]; then
  echo "ERROR: assets/ not found. Locate it manually:"
  echo "  find / -type d -path '*station-healthcare-agent/assets' 2>/dev/null"
  echo "Then re-run with PLAYBOOK_DIR set."
else
  cp -r "$PLAYBOOK_DIR/assets" ~/clinical-intelligence
  cd ~/clinical-intelligence
  cp .env.example .env
  nano .env
#  # Set: NGC_API_KEY=nvapi-...
fi
```

> [!IMPORTANT]
> If the catalog tarball did not include `assets/` (some early distributions stripped it), pull the playbook directly from the repo:
> ```bash
> git clone https://github.com/NVIDIA/dgx-spark-playbooks ~/dgx-spark-playbooks
> cp -r ~/dgx-spark-playbooks/nvidia/station-healthcare-agent/assets ~/clinical-intelligence
> ```

The NGC API key is required to **download** the OpenFold3 NIM image from `nvcr.io` and to **run** it at runtime. Get one for free at [ngc.nvidia.com](https://ngc.nvidia.com/setup/api-key).

Authenticate Docker against NGC so the OpenFold3 image pull succeeds (without this you get a raw HTML 401 from nginx):

```bash
make ngc-login
## or, equivalent manual command:
## echo "$NGC_API_KEY" | docker login nvcr.io --username '$oauthtoken' --password-stdin
```

## Step 3. Start Nemotron 3 Super and OpenFold3

Ollama serves Nemotron 3 Super for LLM inference. OpenFold3 NIM handles protein structure prediction. Both run as Docker containers with GPU access.

If you are connected via SSH, start a `tmux` or `screen` session first so the download survives a disconnection:

```bash
tmux new -s clinical
```

Start the services:

```bash
docker compose up -d ollama openfold3
```

> [!NOTE]
> First run downloads Nemotron 3 Super (~86 GB). This takes 15–25 minutes on a fast connection, up to an hour on slower links. Monitor with `docker compose logs -f ollama`. If interrupted, re-run — it resumes where it left off.

> [!TIP]
> **Multi-GPU stations:** docker-compose pins both Ollama and OpenFold3 to GPU 0 by default (set `LLM_GPU` and `OPENFOLD_GPU` in `.env` to override). On the dual-GPU DGX Station (RTX PRO 6000 + GB300), set `LLM_GPU` and `OPENFOLD_GPU` to the index of the **GB300** because Nemotron-3-Super (~94 GB resident) does not fit safely on the RTX PRO 6000 (98 GB). Find the GB300 index with: `nvidia-smi --query-gpu=index,name --format=csv,noheader | awk -F', ' '/GB300/{print $1; exit}'`

Pull the model into Ollama:

```bash
docker compose up model-pull
```

Wait for all services to report healthy:

```bash
make status
```

Expected:

```
  Ollama:    ✓ healthy
  OpenFold3: ✓ healthy
```

OpenFold3 takes ~3 minutes to load model weights on startup. If it shows "down (may still be loading)", wait and check again.

> [!TIP]
> `make up` runs all of the above (container start + model pull) in one command but does not block on health. Run `make status` separately to verify services are healthy.

## Step 4. Start the OpenShell gateway

The OpenShell gateway runs a lightweight k3s Kubernetes cluster inside Docker to manage sandboxes. On DGX Station, the kernel uses cgroup v2 with the systemd driver, but k3s defaults to cgroupfs. The flag below tells k3s to match the host:

```bash
OPENSHELL_K3S_ARGS='--kubelet-arg=cgroup-driver=systemd' openshell gateway start
```

Wait for the gateway's embedded k3s cluster to finish initializing (10–15 seconds after `gateway start` returns), then verify:

```bash
## Wait until the gateway accepts connections, fail after 60s
for i in $(seq 1 30); do
    if openshell status 2>/dev/null | grep -q "Connected"; then
        echo "Gateway: Connected"; break
    fi
    sleep 2
done
openshell status
```

Expected: `Status: Connected`. If the first `openshell status` immediately after `gateway start` reports `Connection reset by peer`, that is normal — k3s is still warming up. The loop above polls until it is ready.

> [!NOTE]
> Step 4 configures OpenShell infrastructure (gateway). Step 5 deploys the healthcare agent into this infrastructure.

## Step 5. Deploy the healthcare agent

`make setup` automates six operations:
1. Creates the `clinical-sandbox` OpenShell sandbox with the network policy from `sandbox-policy.yaml`
2. Creates the inference provider and configures routing to Ollama (previously a manual step)
3. Installs Python packages (requests, pandas, matplotlib) inside the sandbox
4. Copies agent definitions from `agents/` and skill files from `skills/` into the sandbox workspace
5. Deploys the OpenClaw configuration (`openclaw.json`, `IDENTITY.md`) and registers specialist agents
6. Starts the OpenClaw gateway and runs a smoke test

```bash
make setup
```

When complete, you will see `=== Setup Complete ===`. If setup fails, re-run `make setup` — it recreates the sandbox from scratch, so all config is fresh.

Verify the sandbox config matches the repo:

```bash
make check
```

All checks should pass. If any skills or config files are stale, `make check` tells you what to fix.

Run the quick test suite:

```bash
make test
```

> [!TIP]
> `make setup` runs `bash scripts/setup_sandbox.sh` (loopback bind — use the SSH tunnel from Step 6 for remote browsers). For direct local-browser access on the Station itself, run `make setup-local` instead, which calls `bash scripts/setup_sandbox.sh --local`. See `RUNBOOK.md` in the assets for the manual step-by-step equivalent.

## Step 6. Open the dashboard

**Remote access** (run this from your machine, not the DGX Station). This forwards port 18789 so you can open the dashboard in a local browser:

```bash
ssh -f -N -L 18789:localhost:18789 your-user@your-dgx-station-ip
```

**Cursor / VS Code:** Open the **Ports** tab in the bottom panel, click **Forward a Port**, enter **18789**.

Then open `http://localhost:18789/` in your browser.

**Local** (keyboard and monitor on Station): Open `http://localhost:18789/`.

You should see the OpenClaw dashboard with **Health: OK**. Click **Chat**.

Setup is done. The next steps are the demo.

## Step 7. Run healthcare queries

All queries execute inside the OpenShell sandbox — only whitelisted endpoints are reachable. You will verify this in Step 9.

Paste the first query. The first response takes 30–60 seconds while the 120B model loads into GPU memory — this is normal and only happens once per session.

```
Find all diabetic patients and get their latest HbA1c. Generate a histogram with a red dashed line at 9%. Use dark background with green bars.
```

The agent reads its skill files (`fhir-basics`, `clinical-knowledge`, `analysis-methods`), imports the FHIR helpers library, writes a Python script, queries the FHIR server, and generates the chart — all inside the sandbox.

You should see ~48 diabetic patients and an A1c histogram with a 9% threshold line. The agent includes a clickable link to the chart in its response. You can also browse all visualizations at `http://localhost:18789/__openclaw__/canvas/`.

Stay in the same session for follow-ups:

```
Which of those diabetic patients also have hypertension? For the overlap, get their eGFR. Flag anyone with eGFR below 60 as kidney disease risk.
```

You should see ~24 with both conditions, ~12 flagged as kidney disease risk.

```
Of those kidney disease risk patients, which ones are not on an ACE inhibitor or ARB?
```

You should see ~12 patients missing guideline-recommended therapy (100% care gap in the synthetic data).

## Step 8. Visualize a drug target

```
Show me the 3D protein structure of atorvastatin bound to its target
```

The molecular agent looks up atorvastatin's target protein (HMG-CoA reductase), fetches the drug's SMILES from PubChem, sends the protein sequence to OpenFold3 for structure prediction, and generates an interactive 3D viewer with confidence scores (pLDDT, pTM, ipTM).

You should see an interactive 3D viewer with the protein ribbon structure and atorvastatin ligand. The agent includes a clickable link. Confidence scores appear in the viewer header — pLDDT > 70 indicates a good prediction.

## Step 9. Understand sandbox isolation

The sandbox policy (`sandbox-policy.yaml`) enforces implicit-deny networking at Layer 7. Every outbound connection is blocked unless an explicit rule allows it. The policy also restricts HTTP methods — FHIR and PubChem are limited to read-only (GET/HEAD), OpenFold3 accepts only POST to specific prediction paths.

Confirm that unauthorized endpoints are blocked. From inside the sandbox (connect with `openshell sandbox connect clinical-sandbox`), run:

```bash
curl --max-time 5 https://google.com
```

Expected: connection refused or `CONNECT tunnel failed, response 403`. The allowed endpoints are:

| Endpoint | Purpose | Allowed methods |
|----------|---------|----------------|
| `https://inference.local` (port 443 only) | LLM calls to Ollama | All (OpenAI protocol) |
| `r4.smarthealthit.org` | FHIR patient data | GET, HEAD |
| `pubchem.ncbi.nlm.nih.gov` | Drug SMILES lookup | GET, HEAD |
| OpenFold3 NIM (Docker bridge) | Structure prediction | POST `/biology/openfold/**`, GET `/v1/health/*` |
| CDN (jquery, 3dmol, unpkg) | JavaScript for 3D viewers | GET |

Everything else is denied. Additional rules for GitHub, npm, and PyPI are included in `sandbox-policy.yaml` for build dependencies during sandbox setup — these are setup-only and not used at runtime.

> [!NOTE]
> `inference.local` is HTTPS-only. Plain `http://inference.local/...` returns `policy_denied` because the OpenShell L7 proxy enforces the TLS-terminated CONNECT path. All skills, helper scripts, and `_sandbox` curls in this repo use `https://inference.local` with `-k` (the proxy presents a self-signed cert).

Patient data flows from FHIR → sandbox Python execution. It never passes through the LLM, OpenFold3, or PubChem.

Inspect a skill file to see the editable clinical knowledge:

```bash
head -30 skills/clinical-knowledge/SKILL.md
```

Skill files are Markdown. Edit a threshold or drug classification — it takes effect on the next query, no retraining. Try changing the HbA1c threshold from `9.0%` to `7.0%` and re-running the diabetes query to see the difference.

## Step 10. Cleanup

> [!WARNING]
> This removes the sandbox and stops all services.

```bash
openshell sandbox delete clinical-sandbox
make down
openshell gateway destroy
```

To also remove downloaded models and volumes:

```bash
make clean
```

## Step 11. Next steps

1. **Edit a skill** — change a lab reference range in `skills/clinical-knowledge/SKILL.md` and re-run the same prompt to see the effect.
2. **Add an agent** — create a `.md` in `agents/`, register in `openclaw.json`, redeploy with `make setup`.
3. **Connect a real FHIR server** — replace the test server URL, add OAuth2 authentication, and update `sandbox-policy.yaml` with explicit path-level rules for the FHIR resources the agent needs.
4. **Swap models** — try `qwen2.5:72b` or another Ollama model by editing `.env` and `openclaw.json`.
5. **Verify after changes** — run `make check` after any config or skill file change to catch stale sandbox copies.
6. **Full agent validation** — `make test-full` runs all test levels including end-to-end agent queries (~20 min).
7. **Monitor GPU** — `nvidia-smi` shows memory allocation across Nemotron 3 Super and OpenFold3.

## Troubleshooting

#### Docker and infrastructure

| Symptom | Cause | Fix |
|---------|-------|-----|
| `make up` hangs on model pull | Nemotron-3-Super is ~86 GB and takes 15–25 min on first download (longer on slow links) | Wait. Check progress with `docker compose logs -f ollama`. If interrupted, re-run — it resumes where it left off. |
| `OpenFold3: ✗ down` in `make status` | OpenFold3 takes ~3 minutes to load model weights on startup | Wait and re-run `make status`. Check logs with `docker compose logs -f openfold3`. |
| `failed to bind host port for 0.0.0.0:11434` on `docker compose up ollama` | Host Ollama is already listening on 11434 (common after the NemoClaw playbook) | Stop host Ollama: `sudo systemctl stop ollama && sudo systemctl disable ollama`. Or override in `.env`: `OLLAMA_PORT=11435` — `make setup` and `setup_sandbox.sh` source `.env` and configure the sandbox provider against the new port. |
| `unauthorized: <html><head><title>401 Authorization Required` when pulling `nvcr.io/nim/openfold/openfold3` | Docker is not authenticated against NGC; `NGC_API_KEY` in `.env` is the runtime credential, not the pull credential | Run `make ngc-login` (reads `NGC_API_KEY` from `.env`). Manual equivalent: `echo "$NGC_API_KEY" \| docker login nvcr.io -u '$oauthtoken' --password-stdin`. |
| OpenFold3 crashes with `device >= 0 && device < num_gpus INTERNAL ASSERT FAILED` | OpenFold3's PyTorch backend rejects multi-GPU containers; `count: all` exposes both GPUs on a dual-GPU station | `docker-compose.yml` pins to `LLM_GPU`/`OPENFOLD_GPU` (default `0`). On dual-GPU stations, set both to the **GB300** index in `.env` and `docker compose up -d --force-recreate openfold3`. |
| `NGC_API_KEY not set` error | `.env` file missing or NGC key not configured | Run `cp .env.example .env` and edit to add your NGC API key from [ngc.nvidia.com](https://ngc.nvidia.com/setup/api-key). |
| `exec format error` when pulling containers | Container architecture mismatch (x86 container on ARM64) | Ensure you're using ARM64-compatible containers. OpenFold3 (v1.3.0+) and Ollama support ARM64. Check with `docker inspect --format '{{.Architecture}}' <image>`. |
| Sandbox policy validation fails on startup | `landlock: hard_requirement` aborts if filesystem paths can't be enforced | Check that all paths in `sandbox-policy.yaml` exist on the system. If running on non-standard DGX OS, try `compatibility: best_effort` temporarily to diagnose. |
| `node: command not found` or OpenShell rejects v18 | DGX Station ships with Node.js v18.19.1; OpenShell/OpenClaw need v22+ | `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt-get install -y nodejs`. `make prereq` validates the version automatically. |

#### Gateway and sandbox

| Symptom | Cause | Fix |
|---------|-------|-----|
| Gateway fails with "ContainerManager" error | DGX Station uses cgroup v2 and needs the systemd driver flag | Start gateway with: `OPENSHELL_K3S_ARGS='--kubelet-arg=cgroup-driver=systemd' openshell gateway start` |
| `openshell status` returns "Connection reset by peer" right after `gateway start` | k3s inside the gateway container takes 10–15s to accept connections | Wait. Use the polling loop from `instructions.md` Step 4: `for i in $(seq 1 30); do openshell status 2>/dev/null \| grep -q Connected && break; sleep 2; done`. |
| `openshell status` shows "Not Connected" after 30s | Gateway not started or crashed | Run `openshell gateway start` (with the cgroup flag above). Check `docker ps` for the gateway container. |
| `openshell sandbox create` fails with "port already forwarded" or hangs on `--forward 18789` | Stale port forward from a previously deleted sandbox is still registered | List forwards: `openshell forward list`. Stop each one bound to `:18789`: `openshell forward stop 18789 <sandbox-name>`. `setup_sandbox.sh` does this automatically before re-creating the sandbox. |
| Existing OpenShell gateway from another playbook silently reused with new name | `openshell gateway start` resumes any existing gateway in stopped state | Acceptable, but to start clean: `openshell gateway destroy` before running `openshell gateway start`. |
| Port 18789 not accessible remotely | SSH tunnel not active or port forward dead inside sandbox | Check with `openshell forward list`. If dead: `openshell forward stop 18789 clinical-sandbox && openshell forward start -d 18789 clinical-sandbox`. Then re-establish SSH tunnel from your machine. |
| `requests` library doesn't work in sandbox | Sandbox Python uses curl subprocess for HTTP, not the requests library | This is by design. All HTTP calls in agent scripts must use `subprocess.run(["curl", ...])` and `json.loads()`. The `fhir_helpers.py` library handles this automatically. |

#### Inference and model

| Symptom | Cause | Fix |
|---------|-------|-----|
| Agent returns empty response or timeout | Model unloaded from GPU memory after idle timeout | Send a warmup message first. Check `OLLAMA_KEEP_ALIVE` is set to `4h` in docker-compose.yml. |
| `curl: (7) Failed to connect` to inference.local | OpenShell inference provider not configured or Ollama not running | Verify Ollama: `curl -sf http://localhost:${OLLAMA_PORT:-11434}/`. Re-run `make setup` — it configures the inference provider automatically. |
| Sandbox cannot reach host Ollama (only Docker bridge IP times out) | Host Ollama's systemd unit binds to `127.0.0.1` by default | Add a systemd override binding to all interfaces: `sudo systemctl edit ollama` and insert `[Service]` then `Environment="OLLAMA_HOST=0.0.0.0"`, then `sudo systemctl daemon-reload && sudo systemctl restart ollama`. Docker Ollama (the default in this playbook) already binds to `0.0.0.0`. |
| OpenFold3 returns error for molecular visualization | Protein sequence too long or malformed input | OpenFold3 supports sequences up to 4096 amino acids (PyTorch backend) or 2048 (TensorRT). Check the protein sequence in `build_viewer.py`'s drug-target table. |

#### Agent and skills

| Symptom | Cause | Fix |
|---------|-------|-----|
| `make setup` fails | Setup did not complete successfully | Re-run `make setup` — the script recreates the sandbox from scratch with fresh config. Ensure you're on OpenShell >= 0.0.33. |
| `make check` shows stale skills | Workspace skill copies don't match the repo after an update | The check output tells you which skills are stale. Re-run `make setup` or manually copy from `/sandbox/clinical-intelligence/skills/` to `~/.openclaw/workspace/skills/` inside the sandbox. |
| ENOENT errors for memory files in logs | OpenClaw tries to read daily memory files that don't exist | Create the memory directory: `mkdir -p ~/.openclaw/workspace/memory && touch ~/.openclaw/workspace/MEMORY.md` inside the sandbox. `make check` detects this. |
| Agent writes code from scratch instead of using helpers | Stale IDENTITY.md or analysis-methods skill in workspace | Run `make check` to verify. If stale, the workspace IDENTITY.md doesn't have the `fhir_helpers` import instruction. |
| Agent uses wrong LOINC code for eGFR | Agent used its own training knowledge instead of reading the skill file | Run `make check` to verify skills are synced. The fhir-basics skill lists `33914-3` for eGFR. If the workspace copy is stale, the model uses its own (often wrong) LOINC codes. |

#### Demo and queries

| Symptom | Cause | Fix |
|---------|-------|-----|
| FHIR queries return 0 patients | Wrong SNOMED code format | Use bare codes: `code=44054006`, not `code=http://snomed.info/sct\|44054006`. The skill files contain the correct patterns. |
| Charts not visible in dashboard | Canvas directory not accessible or file not saved to correct path | Charts must be saved to `~/.openclaw/canvas/`. View canvas at `http://localhost:18789/__openclaw__/canvas/`. |
| `make test-full` fails on L4/L5 agent tests | Agent query timed out, FHIR server unreachable from sandbox, or Ollama model unloaded | Check step by step: (1) `make status` — are Ollama and OpenFold3 healthy? (2) `make check` — are skills and config synced? (3) Send a warmup message in the dashboard to reload the model. (4) Run `make test --level 3` first to isolate whether the issue is infrastructure, config, or agent-level. |
