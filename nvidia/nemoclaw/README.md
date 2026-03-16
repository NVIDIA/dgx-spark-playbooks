# NemoClaw

> Run OpenClaw in an OpenShell sandbox on DGX Spark with Ollama (Nemotron)

## Table of Contents

- [Overview](#overview)
  - [Quick start safety check](#quick-start-safety-check)
  - [What you're getting](#what-youre-getting)
  - [Key risks with AI agents](#key-risks-with-ai-agents)
  - [Participant acknowledgement](#participant-acknowledgement)
- [Instructions](#instructions)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

**NVIDIA OpenShell** is an open-source runtime for running autonomous AI agents in sandboxed environments with kernel-level isolation. **NVIDIA NemoClaw** is an OpenClaw plugin that packages OpenShell with an AI agent: it includes the `nemoclaw onboard` wizard to automate setup so you can get a browser-based chat interface running locally on your DGX Spark using Ollama (e.g. NVIDIA Nemotron 3 Super).

By the end of this playbook you will have a working AI agent inside an OpenShell sandbox, accessible via a dashboard URL, with inference routed to a local model on your Spark—all without exposing your host filesystem or network to the agent.

## What you'll accomplish

- Install and configure Docker for OpenShell (including cgroup fix for DGX Spark)
- Install Node.js, Ollama, the OpenShell CLI, and the NemoClaw plugin
- Run the NemoClaw onboard wizard to create a sandbox and configure inference
- Start the OpenClaw web UI inside the sandbox and chat with Nemotron 3 Super (or another Ollama model) locally

## Notice and disclaimers

The following sections describe safety, risks, and your responsibilities when running this demo.

### Quick start safety check

**Use only a clean environment.** Run this demo on a fresh device or VM with no personal data, confidential information, or sensitive credentials. Keep it isolated like a sandbox.

By installing this demo, you accept responsibility for all third-party components, including reviewing their licenses, terms, and security posture. Read and accept before you install or use.

### What you're getting

This experience is provided "AS IS" for demonstration purposes only—no warranties, no guarantees. This is a demo, not a production-ready solution. You will need to implement appropriate security controls for your environment and use case.

### Key risks with AI agents

- **Data leakage** — Any materials the agent accesses could be exposed, leaked, or stolen.
- **Malicious code execution** — The agent or its connected tools could expose your system to malicious code or cyber-attacks.
- **Unintended actions** — The agent might modify or delete files, send messages, or access services without explicit approval.
- **Prompt injection and manipulation** — External inputs or connected content could hijack the agent's behavior in unexpected ways.

### Participant acknowledgement

By participating in this demo, you acknowledge that you are solely responsible for your configuration and for any data, accounts, and tools you connect. To the maximum extent permitted by law, NVIDIA is not responsible for any loss of data, device damage, security incidents, or other harm arising from your configuration or use of NemoClaw demo materials, including OpenClaw or any connected tools or services.

## Isolation layers (OpenShell)

| Layer      | What it protects                                   | When it applies             |
|------------|----------------------------------------------------|-----------------------------|
| Filesystem | Prevents reads/writes outside allowed paths.       | Locked at sandbox creation.  |
| Network    | Blocks unauthorized outbound connections.          | Hot-reloadable at runtime.  |
| Process    | Blocks privilege escalation and dangerous syscalls.| Locked at sandbox creation.  |
| Inference  | Reroutes model API calls to controlled backends.   | Hot-reloadable at runtime.  |

## What to know before starting

- Basic use of the Linux terminal and SSH
- Familiarity with Docker (permissions, `docker run`)
- Awareness of the security and risk sections above

## Prerequisites

**Hardware and access:**

- A DGX Spark (GB10) with keyboard and monitor, or SSH access
- An **NVIDIA API key** from [build.nvidia.com](https://build.nvidia.com) (free; the onboard wizard will prompt for it)
- A GitHub account with access to the NVIDIA organization (for installing the OpenShell CLI from GitHub releases)

**Software:**

- Fresh install of DGX OS with latest updates

Verify your system before starting:

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
python3 --version
```

Expected: Ubuntu 24.04, NVIDIA GB10 GPU, Docker server version, Python 3.12+.

## Ancillary files

All required assets are in the [openshell-openclaw-plugin repository](https://github.com/NVIDIA/openshell-openclaw-plugin). You will clone it during the instructions to install NemoClaw.

## Time and risk

- **Estimated time:** 45–90 minutes (including first-time gateway and sandbox build, and Nemotron 3 Super download of ~87GB).
- **Risk level:** Medium — you are running an AI agent in a sandbox; risks are reduced by isolation but not eliminated. Use a clean environment and do not connect sensitive data or production accounts.
- **Rollback:** Remove the sandbox with `openshell sandbox delete <name>`, destroy the gateway with `openshell gateway destroy -g nemoclaw`, and uninstall NemoClaw with `sudo npm uninstall -g nemoclaw` and `rm -rf ~/.nemoclaw` (see Cleanup in Instructions).
- **Last Updated:** 03/13/2026
  - First publication

## Instructions

## Step 1. Docker configuration

Verify Docker permissions and configure the NVIDIA runtime. OpenShell's gateway runs k3s inside Docker and on DGX Spark requires a cgroup setting so the gateway can start correctly.

Verify Docker:

```bash
docker ps
```

If you get a permission denied error, add your user to the Docker group:

```bash
sudo usermod -aG docker $USER
```

Log out and back in for the group to take effect.

Configure Docker for the NVIDIA runtime and set cgroup namespace mode for OpenShell on DGX Spark:

```bash
sudo nvidia-ctk runtime configure --runtime=docker

sudo python3 -c "
import json, os
path = '/etc/docker/daemon.json'
d = json.load(open(path)) if os.path.exists(path) else {}
d['default-cgroupns-mode'] = 'host'
json.dump(d, open(path, 'w'), indent=2)
"

sudo systemctl restart docker
```

Verify:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

> [!NOTE]
> DGX Spark uses cgroup v2. OpenShell's gateway embeds k3s inside Docker and needs host cgroup namespace access. Without `default-cgroupns-mode: host`, the gateway can fail with "Failed to start ContainerManager" errors.

## Step 2. Install Node.js

NemoClaw is installed via npm and requires Node.js.

```bash
curl -fsSL https://deb.nodesource.com/setup_22.x | sudo -E bash -
sudo apt-get install -y nodejs
```

Verify: `node --version` should show v22.x.x.

## Step 3. Install Ollama and download a model

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Verify it is running:

```bash
curl http://localhost:11434
```

Expected: `Ollama is running`. If not, start it: `ollama serve &`

Download Nemotron 3 Super 120B (~87GB; may take several minutes):

```bash
ollama pull nemotron-3-super:120b
```

Run it briefly to pre-load weights (type `/bye` to exit):

```bash
ollama run nemotron-3-super:120b
```

Configure Ollama to listen on all interfaces so the sandbox container can reach it:

```bash
sudo systemctl edit ollama.service
```

Add the following on the **third line** of the file (above "Edits below this comment will be discarded"):

```ini
[Service]
Environment="OLLAMA_HOST=0.0.0.0"
```

Save (Ctrl+X, then Y), then restart:

```bash
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

## Step 4. Install the OpenShell CLI

The OpenShell binary is distributed via GitHub releases. You need the GitHub CLI and access to the NVIDIA organization.

```bash
sudo apt-get install -y gh
gh auth login
```

When using SSH, `gh` will show a one-time code. Visit https://github.com/login/device in a browser, enter the code, and authorize for the NVIDIA org.

Configure git for NVIDIA SAML SSO and download OpenShell:

```bash
gh auth setup-git

ARCH=$(uname -m)
case "$ARCH" in
  x86_64|amd64) ARCH="x86_64" ;;
  aarch64|arm64) ARCH="aarch64" ;;
esac
gh release download --repo NVIDIA/OpenShell \
  --pattern "openshell-${ARCH}-unknown-linux-musl.tar.gz"
tar xzf openshell-${ARCH}-unknown-linux-musl.tar.gz
sudo install -m 755 openshell /usr/local/bin/openshell
rm -f openshell openshell-${ARCH}-unknown-linux-musl.tar.gz
```

Verify: `openshell --version`

## Step 5. Install NemoClaw

Clone the NemoClaw plugin and install it globally:

```bash
git clone https://github.com/NVIDIA/NemoClaw
cd NemoClaw
sudo npm install -g .
```

Verify: `nemoclaw --help`

> [!NOTE]
> OpenClaw (the AI agent) is installed **automatically inside the sandbox** during onboarding. You do not install it on the host.

## Step 6. Run the NemoClaw onboard wizard

Ensure Ollama is running (`curl http://localhost:11434` should return "Ollama is running"). From the directory where you cloned the plugin in Step 5 (e.g. `~/openshell-openclaw-plugin`), or that directory in a new terminal, run:

```bash
cd ~/openshell-openclaw-plugin
nemoclaw onboard
```

The wizard walks you through seven steps:

1. **NVIDIA API key** — Paste your key from [build.nvidia.com](https://build.nvidia.com) (starts with `nvapi-`). Only needed once.
2. **Preflight** — Checks Docker and OpenShell. "No GPU detected" is normal on DGX Spark (GB10 reports unified memory differently).
3. **Gateway** — Starts the OpenShell gateway (30–60 seconds on first run).
4. **Sandbox** — Enter a name or press Enter for the default. First build takes 2–5 minutes.
5. **Inference** — The wizard auto-detects Ollama (e.g. "Ollama detected on localhost:11434 — using it").
6. **OpenClaw** — Configured on first connect.
7. **Policies** — Press Enter or Y to accept suggested presets (pypi, npm).

When complete you will see something like:

```text
  Dashboard    http://localhost:18789/
  Sandbox      my-assistant (Landlock + seccomp + netns)
  Model        nemotron-3-nano (ollama-local)
```

## Step 7. Configure inference for Nemotron 3 Super

The onboard wizard defaults to `nemotron-3-nano`. Switch the inference route to the Super model you downloaded in Step 3:

```bash
openshell inference set --provider ollama-local --model nemotron-3-super:120b
```

Verify:

```bash
openshell inference get
```

Expected: `provider: ollama-local` and `model: nemotron-3-super:120b`.

## Step 8. Start the OpenClaw web UI

Connect to the sandbox (use the name you chose in Step 6, e.g. `my-assistant`):

```bash
openshell sandbox connect my-assistant
```

You are now inside the sandbox. Run these commands in order.

Set the API key environment variables (required for the gateway). For local Ollama, use the value `local-ollama` — no real API key is required. If you use a different inference provider later, replace with your API key:

```bash
export NVIDIA_API_KEY=local-ollama
export ANTHROPIC_API_KEY=local-ollama
```

Initialize NemoClaw (this may drop you into a new shell when done):

```bash
nemoclaw-start
```

After the "NemoClaw ready" banner, re-export the environment variables:

```bash
export NVIDIA_API_KEY=local-ollama
export ANTHROPIC_API_KEY=local-ollama
```

Create memory files and start the web UI:

```bash
mkdir -p /sandbox/.openclaw/workspace/memory
echo "# Memory" > /sandbox/.openclaw/workspace/MEMORY.md

openclaw config set gateway.controlUi.dangerouslyAllowHostHeaderOriginFallback true

nohup openclaw gateway run \
  --allow-unconfigured --dev \
  --bind loopback --port 18789 \
  > /tmp/gateway.log 2>&1 &
```

Wait a few seconds, then get your dashboard URL:

```bash
openclaw dashboard
```

This prints something like:

```text
Dashboard URL: http://127.0.0.1:18789/#token=YOUR_UNIQUE_TOKEN
```

**Save this URL.** Type `exit` to leave the sandbox (the gateway keeps running).

## Step 9. Open the chat interface

Open the dashboard URL from Step 8 in your Spark's web browser:

```text
http://127.0.0.1:18789/#token=YOUR_UNIQUE_TOKEN
```

> [!IMPORTANT]
> The token is in the URL as a hash fragment (`#token=...`), not a query parameter (`?token=`). Paste the full URL including `#token=...` into the address bar.

You should see the OpenClaw dashboard with **Version** and **Health: OK**. Click **Chat** in the left sidebar and send a message to your agent.

Try: *"Hello! What can you help me with?"* or *"How many rs are there in the word strawberry?"*

> [!NOTE]
> Nemotron 3 Super 120B responses may take 30–90 seconds. This is normal for a 120B parameter model running locally.

## Step 10. Using the agent from the command line

Connect to the sandbox:

```bash
openshell sandbox connect my-assistant
```

Run a prompt:

```bash
export NVIDIA_API_KEY=local-ollama
export ANTHROPIC_API_KEY=local-ollama
openclaw agent --agent main --local -m "How many rs are there in strawberry?" --session-id s1
```

Test sandbox isolation (this should be blocked by the network policy):

```bash
curl -sI https://httpbin.org/get
```

Type `exit` to leave the sandbox.

## Step 11. Monitoring with the OpenShell TUI

In a separate terminal on the host:

```bash
openshell term
```

Press `f` to follow live output, `s` to filter by source, `q` to quit.

## Step 12. Cleanup

Remove the sandbox and destroy the NemoClaw gateway:

```bash
openshell sandbox delete my-assistant
openshell gateway destroy -g nemoclaw
```

To fully uninstall NemoClaw:

```bash
sudo npm uninstall -g nemoclaw
rm -rf ~/.nemoclaw
```

## Step 13. Clean slate (start over)

To remove everything and start again from Step 5:

```bash
cd ~
openshell sandbox delete my-assistant 2>/dev/null
openshell gateway destroy -g nemoclaw 2>/dev/null
sudo npm uninstall -g nemoclaw
rm -rf ~/openshell-openclaw-plugin ~/.nemoclaw
```

Verify:

```bash
which nemoclaw        # Should report "not found"
openshell status      # Should report "No gateway configured"
```

Then restart from Step 5 (Install NemoClaw).

## Step 14. Optional: Remote access via SSH

If you access the Spark remotely, forward port 18789 to your machine.

**SSH tunnel** (from your local machine, not the Spark):

```bash
ssh -L 18789:127.0.0.1:18789 your-user@your-spark-ip
```

Then open the dashboard URL in your local browser.

**Cursor / VS Code:** Open the **Ports** tab in the bottom panel, click **Forward a Port**, enter **18789**, then open the dashboard URL in your browser.

## Useful commands

| Command | Description |
|---------|-------------|
| `openshell status` | Check gateway health |
| `openshell sandbox list` | List all running sandboxes |
| `openshell sandbox connect my-assistant` | Shell into the sandbox |
| `openshell term` | Open the monitoring TUI |
| `openshell inference get` | Show current inference routing |
| `openshell forward list` | List active port forwards |
| `nemoclaw my-assistant connect` | Connect to sandbox (alternate) |
| `nemoclaw my-assistant status` | Show sandbox status |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| Gateway fails with cgroup / "Failed to start ContainerManager" errors | Docker not configured for host cgroup namespace on DGX Spark | Run the cgroup fix: `sudo python3 -c "import json, os; path='/etc/docker/daemon.json'; d=json.load(open(path)) if os.path.exists(path) else {}; d['default-cgroupns-mode']='host'; json.dump(d, open(path,'w'), indent=2)"` then `sudo systemctl restart docker` |
| "No GPU detected" during onboard | DGX Spark GB10 reports unified memory differently | Expected on DGX Spark. The wizard still works and will use Ollama for inference. |
| "unauthorized: gateway token missing" | Dashboard URL used without token or wrong format | Paste the **full URL** including `#token=...` (hash fragment, not `?token=`). Run `openclaw dashboard` inside the sandbox to get the URL again. |
| "No API key found for provider anthropic" | API key env vars not set when starting gateway in sandbox | Inside the sandbox, set both before running the gateway: `export NVIDIA_API_KEY=local-ollama` and `export ANTHROPIC_API_KEY=local-ollama` |
| Agent gives no response | Model not loaded or Nemotron 3 Super is slow | Nemotron 3 Super can take 30–90 seconds per response. Verify Ollama: `curl http://localhost:11434`. Ensure inference is set: `openshell inference get` |
| Port forward dies or dashboard unreachable | Forward not active or wrong port | List forwards: `openshell forward list`. Restart: `openshell forward stop 18789 my-assistant` then `openshell forward start --background 18789 my-assistant` |
| Docker permission denied | User not in docker group | `sudo usermod -aG docker $USER`, then log out and back in. |
| Ollama not reachable from sandbox (503 / timeout) | Ollama bound to localhost only or firewall blocking 11434 | Ensure Ollama listens on all interfaces: add `Environment="OLLAMA_HOST=0.0.0.0"` in `sudo systemctl edit ollama.service`, then `sudo systemctl daemon-reload` and `sudo systemctl restart ollama`. If using UFW: `sudo ufw allow 11434/tcp comment 'Ollama for NemoClaw'` and `sudo ufw reload` |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory issues even when within the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
