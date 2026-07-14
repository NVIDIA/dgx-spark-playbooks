# Local Healthcare Agent — Setup Guide

From-scratch setup for a DGX Station. Produces a working multi-agent clinical analysis system with OpenFold3 protein structure prediction.

**Hardware**: DGX Station (284 GB VRAM, aarch64)

---

## Prerequisites

- Docker + NVIDIA Container Toolkit (`docker info --format '{{.ServerVersion}}'` ≥ 23.0.1)
- **Node.js v22+** (DGX OS 7.5.0 ships v22; if an older image reports v18 or Node is missing, download the setup script first, then run it: `curl -fsSL https://deb.nodesource.com/setup_22.x -o /tmp/nodesource_setup.sh && sudo bash /tmp/nodesource_setup.sh && sudo apt-get install -y nodejs`)
- OpenShell CLI ≥ 0.0.44
- **At least 200 GB free** on `/` (86 GB Ollama model + Docker images + working space; verify with `df -h /`)
- A single GPU with **≥150 GB free VRAM** (target the GB300, not the RTX PRO 6000, on dual-GPU stations)
- Network access to `r4.smarthealthit.org` (FHIR test server) and `nvcr.io` (NGC registry)
- NVIDIA NGC API key ([get one here](https://ngc.nvidia.com/setup/api-key)) **and** `docker login nvcr.io` (run `make ngc-login` after Step 1)

> [!TIP]
> `make prereq` checks all of the above (Docker, Node.js v22, OpenShell, disk, GPU, port 11434, NGC docker login) in one shot.

## 1. Clone the repo

```bash
git clone https://github.com/jaival-nvidia/local-healthcare-agent.git
cd local-healthcare-agent
cp .env.example .env
# Edit .env: set NGC_API_KEY=nvapi-...
make ngc-login   # docker login nvcr.io with NGC_API_KEY (required to pull OpenFold3)
```

## 2. Install OpenShell

The official installer installs both the `openshell` CLI and the `openshell-gateway` daemon. This is all the Healthcare Agent playbook needs — you do **not** need the full NemoClaw stack.

```bash
curl -LsSf https://raw.githubusercontent.com/NVIDIA/OpenShell/main/install.sh | sh
# Open a new shell (or `source ~/.bashrc`) so ~/.local/bin is on PATH, then:
openshell --version   # >= 0.0.44
```

> [!NOTE]
> Ollama runs as a Docker container in this playbook (Step 3) — you do **not** need a host Ollama install. If host Ollama is already running on port 11434 (e.g., from the NemoClaw playbook), stop it first (`sudo systemctl stop ollama`) or override `OLLAMA_PORT` in `.env`. Both `docker-compose.yml` and `setup_sandbox.sh` honor the override.

## 3. Start infrastructure

```bash
make up
# Starts Ollama and OpenFold3 via Docker Compose.
# Auto-pulls nemotron-3-super:120b-a12b (~86 GB) if not cached.
# Wait for both services to report healthy:
make status
```

This starts:
- **Ollama** (port `${OLLAMA_PORT:-11434}`) — LLM inference with nemotron-3-super:120b-a12b
- **OpenFold3 NIM** (port `${OPENFOLD_PORT:-8000}`) — protein structure prediction (~3 min startup)

> [!TIP]
> On dual-GPU stations, set `LLM_GPU` and `OPENFOLD_GPU` in `.env` to the **GB300** index (find with `nvidia-smi --query-gpu=index,name --format=csv,noheader`). Nemotron 3 Super (~94 GB resident) does not fit safely on the RTX PRO 6000 (98 GB), and OpenFold3 crashes on multi-GPU containers.

## 4. Start OpenShell gateway

```bash
# OpenShell >= 0.0.44: start the standalone gateway server with the Docker
# driver, then register it with the CLI.
nohup openshell-gateway \
    --disable-tls \
    --drivers docker \
    --bind-address 127.0.0.1 \
    --port 17670 \
    > /tmp/openshell-gateway.log 2>&1 &

openshell gateway add http://127.0.0.1:17670 --name openshell

# The gateway typically starts in under 1 second.
openshell status   # Should show: Status: Connected
```

If `openshell status` does not show `Connected`, check `/tmp/openshell-gateway.log` for errors.

## 5. (Optional) Configure inference provider manually

`make setup` (Step 6) creates the inference provider for you. To do it by hand:

```bash
BRIDGE_IP=$(ip -4 addr show docker0 | grep -oP 'inet \K[\d.]+')

openshell provider create \
  --name ollama-local \
  --type openai \
  --credential OPENAI_API_KEY=ollama \
  --config OPENAI_BASE_URL=http://${BRIDGE_IP}:${OLLAMA_PORT:-11434}/v1

openshell inference set \
  --provider ollama-local \
  --model nemotron-3-super:120b-a12b
```

> [!NOTE]
> Current OpenShell releases do not accept the `--base-url` shorthand for `provider create` — use `--config OPENAI_BASE_URL=...` as shown above.

## 6. Create sandbox and deploy everything

```bash
make setup
# Creates sandbox, installs Python packages, deploys skills/agents/config,
# starts OpenClaw gateway, runs smoke test.
# Takes ~5-15 min (PyPI downloads are slow through the sandbox proxy).
```

Or for local access without SSH tunnel:
```bash
make setup-local
```

## 7. Verify

```bash
make test           # L1-3: infrastructure + OpenShell + config (~1 min)
make test-full      # L1-5: includes agent functional + E2E (~20 min)
```

## 8. Access the demo

**Remote (SSH tunnel):**
```bash
ssh -f -N -L 18789:localhost:18789 <user>@<dgx-ip>
open http://localhost:18789/
```

**Local:**
```bash
open http://localhost:18789/
```

Canvas (charts + molecular viewers): `http://localhost:18789/__openclaw__/canvas/`

---

## Day-to-day commands

| Command | What |
|---------|------|
| `make up` | Start Ollama + OpenFold3 |
| `make down` | Stop all Docker services |
| `make status` | Health dashboard |
| `make restart` | Restart OpenClaw gateway in sandbox |
| `make test` | Quick validation (L1-3, ~1 min) |
| `make test-full` | Full validation (L1-5, ~20 min) |
| `make logs` | Tail all service logs |
| `make clean` | Remove test results + volumes |

---

## Demo Prompts

**Clinical cohort analysis:**
```
Find all diabetic patients, get their latest A1c and medications. Identify gap patients with A1c above 9% not on insulin or GLP-1. Show the A1c distribution as a histogram.
```

**Patient case summary:**
```
Look up the first patient. Compile a case summary: demographics, conditions, recent labs (flag abnormal), and medications.
```

**Molecular visualization:**
```
Show me the 3D protein structure of atorvastatin bound to its target
```

**End-to-end investigation:**
```
Find all diabetic patients with poorly controlled A1c, identify what medications they are on, show me the distribution, and visualize the molecular target of the most common therapy.
```

---

## Known Issues

**PyPI slow through sandbox proxy**: Package installs take 10-15 minutes due to the privacy router throttling. The setup script retries automatically.

**Canvas proxy strips `<script src=...>`**: The `build_viewer.py` script inlines JS libraries to work around this.

**Model misspellings**: nemotron-3-super:120b-a12b (120B total, 12B active MoE) occasionally misspells drug names. The actual code execution is correct.

**Ollama model unloading**: The model unloads after the `keep_alive` timeout. First request after unloading takes ~30s to reload. For demos, send a warmup message first.

**OpenFold3 startup**: Takes ~3 minutes to load the model. The healthcheck waits for it automatically.
