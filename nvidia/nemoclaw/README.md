# NemoClaw with Nemotron 3 Super and Telegram on DGX Spark

> Install NemoClaw on DGX Spark with local Ollama inference and Telegram bot integration


## Table of Contents

- [Overview](#overview)
  - [Overview](#overview)
  - [Basic idea](#basic-idea)
  - [What you'll accomplish](#what-youll-accomplish)
  - [Notice and disclaimers](#notice-and-disclaimers)
  - [Isolation layers (OpenShell)](#isolation-layers-openshell)
  - [What to know before starting](#what-to-know-before-starting)
  - [Prerequisites](#prerequisites)
  - [Have ready before you begin](#have-ready-before-you-begin)
  - [Ancillary files](#ancillary-files)
  - [Time and risk](#time-and-risk)
- [Instructions](#instructions)
  - [Step 1. Configure Docker and the NVIDIA container runtime](#step-1-configure-docker-and-the-nvidia-container-runtime)
  - [Step 2. Install Ollama](#step-2-install-ollama)
  - [Step 3. Pull the Nemotron 3 Super model](#step-3-pull-the-nemotron-3-super-model)
  - [Step 4. Install NemoClaw](#step-4-install-nemoclaw)
  - [Step 5. Connect to the sandbox and verify inference](#step-5-connect-to-the-sandbox-and-verify-inference)
  - [Step 6. Talk to the agent (CLI)](#step-6-talk-to-the-agent-cli)
  - [Step 7. Interactive TUI](#step-7-interactive-tui)
  - [Step 8. Exit the sandbox and access the Web UI](#step-8-exit-the-sandbox-and-access-the-web-ui)
  - [Step 9. Create a Telegram bot](#step-9-create-a-telegram-bot)
  - [Step 10. Install cloudflared and start the Telegram bridge](#step-10-install-cloudflared-and-start-the-telegram-bridge)
  - [Step 11. Stop services](#step-11-stop-services)
  - [Step 12. Uninstall NemoClaw](#step-12-uninstall-nemoclaw)
- [Troubleshooting](#troubleshooting)

---

## Overview

### Overview

### Basic idea

**NVIDIA NemoClaw** is an open-source reference stack that simplifies running OpenClaw always-on assistants more safely. It installs the **NVIDIA OpenShell** runtime -- an environment designed for executing agents with additional security -- and open-source models like NVIDIA Nemotron. A single installer command handles Node.js, OpenShell, and the NemoClaw CLI, then walks you through an onboard wizard to create a sandboxed agent on your DGX Spark using Ollama with Nemotron 3 Super.

By the end of this playbook you will have a working AI agent inside an OpenShell sandbox, accessible via a web dashboard and a Telegram bot, with inference routed to a local Nemotron 3 Super 120B model on your Spark -- all without exposing your host filesystem or network to the agent.

### What you'll accomplish

- Configure Docker and the NVIDIA container runtime for OpenShell on DGX Spark
- Install Ollama, pull Nemotron 3 Super 120B, and configure it for sandbox access
- Install NemoClaw with a single command (handles Node.js, OpenShell, and the CLI)
- Run the onboard wizard to create a sandbox and configure local inference
- Chat with the agent via the CLI, TUI, and web UI
- Set up a Telegram bot that forwards messages to your sandboxed agent

### Notice and disclaimers

The following sections describe safety, risks, and your responsibilities when running this demo.

#### Quick start safety check

**Use only a clean environment.** Run this demo on a fresh device or VM with no personal data, confidential information, or sensitive credentials. Keep it isolated like a sandbox.

By installing this demo, you accept responsibility for all third-party components, including reviewing their licenses, terms, and security posture. Read and accept before you install or use.

#### What you're getting

This experience is provided "AS IS" for demonstration purposes only -- no warranties, no guarantees. This is a demo, not a production-ready solution. You will need to implement appropriate security controls for your environment and use case.

#### Key risks with AI agents

- **Data leakage** -- Any materials the agent accesses could be exposed, leaked, or stolen.
- **Malicious code execution** -- The agent or its connected tools could expose your system to malicious code or cyber-attacks.
- **Unintended actions** -- The agent might modify or delete files, send messages, or access services without explicit approval.
- **Prompt injection and manipulation** -- External inputs or connected content could hijack the agent's behavior in unexpected ways.

#### Participant acknowledgement

By participating in this demo, you acknowledge that you are solely responsible for your configuration and for any data, accounts, and tools you connect. To the maximum extent permitted by law, NVIDIA is not responsible for any loss of data, device damage, security incidents, or other harm arising from your configuration or use of NemoClaw demo materials, including OpenClaw or any connected tools or services.

### Isolation layers (OpenShell)

| Layer      | What it protects                                   | When it applies             |
|------------|----------------------------------------------------|-----------------------------|
| Filesystem | Prevents reads/writes outside allowed paths.       | Locked at sandbox creation.  |
| Network    | Blocks unauthorized outbound connections.          | Hot-reloadable at runtime.  |
| Process    | Blocks privilege escalation and dangerous syscalls.| Locked at sandbox creation.  |
| Inference  | Reroutes model API calls to controlled backends.   | Hot-reloadable at runtime.  |

### What to know before starting

- Basic use of the Linux terminal and SSH
- Familiarity with Docker (permissions, `docker run`)
- Awareness of the security and risk sections above

### Prerequisites

**Hardware and access:**

- A DGX Spark (GB10) with keyboard and monitor, or SSH access
- A **Telegram bot token** from [@BotFather](https://t.me/BotFather) (create one with `/newbot`) -- only needed if you want the Telegram bot. Have it ready *before* running the installer; the onboard wizard prompts for it.

**Software:**

- Fresh install of DGX OS with latest updates

Verify your system before starting:

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

Expected: Ubuntu 24.04, NVIDIA GB10 GPU, Docker 28.x+.

### Have ready before you begin

| Item | Where to get it |
|------|----------------|
| Telegram bot token (optional) | [@BotFather](https://t.me/BotFather) on Telegram -- create with `/newbot`. Required only for the Telegram bot; have it ready before running the installer. |

### Ancillary files

All required assets are handled by the NemoClaw installer. No manual cloning is needed.

### Time and risk

- **Estimated time:** 20--30 minutes (with Ollama and model already downloaded). First-time model download adds ~15--30 minutes depending on network speed.
- **Risk level:** Medium -- you are running an AI agent in a sandbox; risks are reduced by isolation but not eliminated. Use a clean environment and do not connect sensitive data or production accounts.
- **Last Updated:** 04/28/2026
  * Updated for NemoClaw v0.0.22+: revised Telegram setup, renamed tunnel commands, refreshed uninstall instructions.

## Instructions

## Phase 1: Prerequisites

These steps prepare a fresh DGX Spark for NemoClaw. If Docker, the NVIDIA runtime, and Ollama are already configured, skip to Phase 2.

### Step 1. Configure Docker and the NVIDIA container runtime

OpenShell's gateway runs k3s inside Docker. On DGX Spark (Ubuntu 24.04, cgroup v2), Docker must be configured with the NVIDIA runtime and host cgroup namespace mode.

Configure the NVIDIA container runtime for Docker:

```bash
sudo nvidia-ctk runtime configure --runtime=docker
```

Set the cgroup namespace mode required by OpenShell on DGX Spark:

```bash
sudo python3 -c "
import json, os
path = '/etc/docker/daemon.json'
d = json.load(open(path)) if os.path.exists(path) else {}
d['default-cgroupns-mode'] = 'host'
json.dump(d, open(path, 'w'), indent=2)
"
```

Restart Docker:

```bash
sudo systemctl restart docker
```

Verify the NVIDIA runtime works:

```bash
docker run --rm --runtime=nvidia --gpus all ubuntu nvidia-smi
```

If you get a permission denied error on `docker`, add your user to the Docker group and activate the new group in your current session:

```bash
sudo usermod -aG docker $USER
newgrp docker
```

This applies the group change immediately. Alternatively, you can log out and back in instead of running `newgrp docker`.

> [!NOTE]
> DGX Spark uses cgroup v2. OpenShell's gateway embeds k3s inside Docker and needs host cgroup namespace access. Without `default-cgroupns-mode: host`, the gateway can fail with "Failed to start ContainerManager" errors.

### Step 2. Install Ollama

Install Ollama:

```bash
curl -fsSL https://ollama.com/install.sh | sh
```

Configure Ollama to listen on all interfaces so the sandbox container can reach it:

```bash
sudo mkdir -p /etc/systemd/system/ollama.service.d
printf '[Service]\nEnvironment="OLLAMA_HOST=0.0.0.0"\n' | sudo tee /etc/systemd/system/ollama.service.d/override.conf
sudo systemctl daemon-reload
sudo systemctl restart ollama
```

Verify it is running and reachable on all interfaces:

```bash
curl http://0.0.0.0:11434
```

Expected: `Ollama is running`. If not, start it with `sudo systemctl start ollama`.

> [!IMPORTANT]
> Always start Ollama via systemd (`sudo systemctl restart ollama`) — do not use `ollama serve &`. A manually started Ollama process does not pick up the `OLLAMA_HOST=0.0.0.0` setting above, and the NemoClaw sandbox will not be able to reach the inference server.

### Step 3. Pull the Nemotron 3 Super model

Download Nemotron 3 Super 120B (~87 GB; may take 15--30 minutes depending on network speed):

```bash
ollama pull nemotron-3-super:120b
```

Run it briefly to pre-load weights into memory (type `/bye` to exit):

```bash
ollama run nemotron-3-super:120b
```

Verify the model is available:

```bash
ollama list
```

You should see `nemotron-3-super:120b` in the output.

---

## Phase 2: Install and Run NemoClaw

### Step 4. Install NemoClaw

This single command handles everything: installs Node.js (if needed), installs OpenShell, clones the latest stable NemoClaw release, builds the CLI, and runs the onboard wizard to create a sandbox.

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

The onboard wizard walks you through setup:

1. **Sandbox name** -- Pick a name (e.g. `my-assistant`). Names must be lowercase alphanumeric with hyphens only.
2. **Inference provider** -- Select **Local Ollama**.
3. **Model** -- Select **nemotron-3-super:120b**.
4. **Messaging channels** -- If you want a Telegram bot, select `telegram` here and paste your bot token when prompted. Create the bot first via [@BotFather](https://t.me/BotFather) in Telegram (see Step 9). If you skip this, you can re-run the installer later to recreate the sandbox with Telegram enabled.
5. **Policy presets** -- Accept the suggested presets when prompted (hit **Y**).

> [!IMPORTANT]
> Telegram must be configured at this step. The channel plugin and bot token are wired into the sandbox container during onboarding — they cannot be added to an existing sandbox by exporting environment variables on the host.

When complete you will see output like:

```text
──────────────────────────────────────────────────
Dashboard    http://localhost:18789/
Sandbox      my-assistant (Landlock + seccomp + netns)
Model        nemotron-3-super:120b (Local Ollama)
──────────────────────────────────────────────────
Run:         nemoclaw my-assistant connect
Status:      nemoclaw my-assistant status
Logs:        nemoclaw my-assistant logs --follow
──────────────────────────────────────────────────
```

> [!IMPORTANT]
> Save the tokenized Web UI URL printed at the end -- you will need it in Step 8. It looks like:
> `http://127.0.0.1:18789/#token=<long-token-here>`

> [!NOTE]
> If `nemoclaw` is not found after install, run `source ~/.bashrc` to reload your shell path.

### Step 5. Connect to the sandbox and verify inference

Connect to the sandbox:

```bash
nemoclaw my-assistant connect
```

You will see `sandbox@my-assistant:~$` -- you are now inside the sandboxed environment.

Verify that the inference route is working:

```bash
curl -sf https://inference.local/v1/models
```

Expected: JSON listing `nemotron-3-super:120b`.

### Step 6. Talk to the agent (CLI)

Still inside the sandbox, send a test message:

```bash
openclaw agent --agent main -m "hello" --session-id test
```

The agent will respond using Nemotron 3 Super. First responses may take 30--90 seconds for a 120B parameter model running locally.

### Step 7. Interactive TUI

Launch the terminal UI for an interactive chat session:

```bash
openclaw tui
```

Press **Ctrl+C** to exit the TUI.

### Step 8. Exit the sandbox and access the Web UI

Exit the sandbox to return to the host:

```bash
exit
```

**If accessing the Web UI directly on the Spark** (keyboard and monitor attached), open a browser and navigate to the tokenized URL from Step 4:

```text
http://127.0.0.1:18789/#token=<long-token-here>
```

**If accessing the Web UI from a remote machine**, you need to set up an SSH tunnel. The NemoClaw onboard wizard already created the port 18789 forward on the Spark, so you only need to tunnel from your remote machine.

First, find your Spark's IP address. On the Spark, run:

```bash
hostname -I | awk '{print $1}'
```

This prints the primary IP address (e.g. `192.168.1.42`). You can also find it in **Settings > Wi-Fi** or **Settings > Network** on the Spark's desktop, or check your router's connected-devices list.

From your remote machine, create an SSH tunnel to the Spark (replace `<your-spark-ip>` with the IP address from above):

```bash
ssh -L 18789:127.0.0.1:18789 <your-user>@<your-spark-ip>
```

Now open the tokenized URL in your remote machine's browser:

```text
http://127.0.0.1:18789/#token=<long-token-here>
```

> [!IMPORTANT]
> Use `127.0.0.1`, not `localhost` -- the gateway origin check requires an exact match.

> [!NOTE]
> If the Web UI fails to load and the port forward may be stale, reset it on the Spark host:
> ```bash
> openshell forward stop 18789 my-assistant || true
> openshell forward start 18789 my-assistant --background
> ```

---

## Phase 3: Telegram Bot

> [!IMPORTANT]
> Telegram must be enabled in the **NemoClaw onboard wizard** (Step 4 → Messaging channels). The channel plugin and bot token are wired into the sandbox container at sandbox creation time — `policy-add` only opens network egress and is not enough on its own. If you skipped Telegram during onboard, re-run the installer to recreate the sandbox with Telegram enabled.

### Step 9. Create a Telegram bot

Do this **before** running the NemoClaw installer in Step 4 so you have your bot token ready when the wizard prompts for it.

Open Telegram, find [@BotFather](https://t.me/BotFather), send `/newbot`, and follow the prompts. Copy the bot token it gives you and paste it into the wizard when you reach the **Messaging channels** step.

### Step 10. Install cloudflared and start the Telegram bridge

The Telegram bridge needs a public webhook URL so Telegram can deliver messages to your bot. NemoClaw uses [cloudflared](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/) to create a free `trycloudflare.com` tunnel.

Make sure you are on the **host** (not inside the sandbox). If you are inside the sandbox, run `exit` first.

Install cloudflared (DGX Spark is arm64):

```bash
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
```

Start the tunnel:

```bash
nemoclaw tunnel start
```

Verify the public URL is live:

```bash
nemoclaw status
```

You should see `● cloudflared` with a `trycloudflare.com` public URL (e.g. `https://assembled-peer-persian-kitty.trycloudflare.com`).

Open Telegram, find your bot, and send it a message. The bot forwards it to the agent and replies.

> [!NOTE]
> If `nemoclaw tunnel start` prints `cloudflared not found — no public URL`, the cloudflared install above did not complete successfully. Re-run the install, then restart the tunnel:
> ```bash
> nemoclaw tunnel stop && nemoclaw tunnel start
> ```

> [!NOTE]
> The first response may take 30--90 seconds for a 120B parameter model running locally.

> [!NOTE]
> If sending a message returns `Error: Channel is unavailable: telegram`, the channel was not enabled during onboard. Re-run the installer to recreate the sandbox with Telegram selected at the **Messaging channels** step.

> [!NOTE]
> For details on restricting which Telegram chats can interact with the agent, see the [NemoClaw Telegram bridge documentation](https://docs.nvidia.com/nemoclaw/latest/deployment/set-up-telegram-bridge.html).

---

## Phase 4: Cleanup and Uninstall

### Step 11. Stop services

Stop the cloudflared tunnel:

```bash
nemoclaw tunnel stop
```

Stop the port forward:

```bash
openshell forward list          # find active forwards
openshell forward stop 18789    # stop the dashboard forward
```

### Step 12. Uninstall NemoClaw

Run the uninstaller via curl (matches the [NemoClaw README](https://github.com/NVIDIA/NemoClaw)). It removes all sandboxes, the OpenShell gateway, Docker containers/images/volumes, the CLI, and all state files. Docker, Node.js, npm, and Ollama are preserved.

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NemoClaw/refs/heads/main/uninstall.sh | bash
```

**Uninstaller flags** (pass via `bash -s -- <flags>`):

| Flag | Effect |
|------|--------|
| `--yes` | Skip the confirmation prompt |
| `--keep-openshell` | Leave the `openshell` binary in place |
| `--delete-models` | Also remove the Ollama models pulled by NemoClaw |

To remove everything including the Ollama model, non-interactively:

```bash
curl -fsSL https://raw.githubusercontent.com/NVIDIA/NemoClaw/refs/heads/main/uninstall.sh | bash -s -- --yes --delete-models
```

The uninstaller runs 6 steps:
1. Stop NemoClaw helper services and port-forward processes
2. Delete all OpenShell sandboxes, the NemoClaw gateway, and providers
3. Remove the global `nemoclaw` npm package
4. Remove NemoClaw/OpenShell Docker containers, images, and volumes
5. Remove Ollama models (only with `--delete-models`)
6. Remove state directories (`~/.nemoclaw`, `~/.config/openshell`, `~/.config/nemoclaw`) and the OpenShell binary

> [!NOTE]
> If you have a local clone at `~/.nemoclaw/source` you want to keep, move or back it up before running the uninstaller — it is removed as part of state cleanup in step 6.

## Useful commands

| Command | Description |
|---------|-------------|
| `nemoclaw my-assistant connect` | Shell into the sandbox |
| `nemoclaw my-assistant status` | Show sandbox status and inference config |
| `nemoclaw my-assistant logs --follow` | Stream sandbox logs in real time |
| `nemoclaw list` | List all registered sandboxes |
| `nemoclaw tunnel start` | Start cloudflared tunnel (public URL for Telegram webhooks) |
| `nemoclaw tunnel stop` | Stop the cloudflared tunnel |
| `openshell term` | Open the monitoring TUI on the host |
| `openshell forward list` | List active port forwards |
| `openshell forward start 18789 my-assistant --background` | Restart port forwarding for Web UI |
| `curl -fsSL https://raw.githubusercontent.com/NVIDIA/NemoClaw/refs/heads/main/uninstall.sh \| bash` | Remove NemoClaw (preserves Docker, Node.js, Ollama) |
| `curl -fsSL https://raw.githubusercontent.com/NVIDIA/NemoClaw/refs/heads/main/uninstall.sh \| bash -s -- --delete-models` | Remove NemoClaw and Ollama models |

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nemoclaw: command not found` after install | Shell PATH not updated | Run `source ~/.bashrc` (or `source ~/.zshrc` for zsh), or open a new terminal window. |
| Installer fails with Node.js version error | Node.js version below 20 | Install Node.js 20+: `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt-get install -y nodejs` then re-run the installer. |
| npm install fails with `EACCES` permission error | npm global directory not writable | `mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global && export PATH=~/.npm-global/bin:$PATH` then re-run the installer. Add the `export` line to `~/.bashrc` to make it permanent. |
| Docker permission denied | User not in docker group | `sudo usermod -aG docker $USER`, then log out and back in. |
| Gateway fails with cgroup / "Failed to start ContainerManager" errors | Docker not configured for host cgroup namespace on DGX Spark | Run the cgroup fix: `sudo python3 -c "import json, os; path='/etc/docker/daemon.json'; d=json.load(open(path)) if os.path.exists(path) else {}; d['default-cgroupns-mode']='host'; json.dump(d, open(path,'w'), indent=2)"` then `sudo systemctl restart docker`. Alternatively, run `sudo nemoclaw setup-spark` which applies this fix automatically. |
| Gateway fails with "port 8080 is held by container..." | Another OpenShell gateway or container is using port 8080 | Stop the conflicting container: `openshell gateway destroy -g <old-gateway-name>` or `docker stop <container-name> && docker rm <container-name>`, then retry `nemoclaw onboard`. |
| Sandbox creation fails | Stale gateway state or DNS not propagated | Run `openshell gateway destroy && openshell gateway start`, then re-run the installer or `nemoclaw onboard`. |
| CoreDNS crash loop | Known issue on some DGX Spark configurations | Run `sudo ./scripts/fix-coredns.sh` from the NemoClaw repo directory. |
| "No GPU detected" during onboard | DGX Spark GB10 reports unified memory differently | Expected on DGX Spark. The wizard still works and uses Ollama for inference. |
| Inference timeout or hangs | Ollama not running or not reachable | Check Ollama: `curl http://localhost:11434`. If not running: `ollama serve &`. If running but unreachable from sandbox, ensure Ollama is configured to listen on `0.0.0.0` (see Step 2 in Instructions). |
| Agent gives no response or is very slow | Normal for 120B model running locally | Nemotron 3 Super 120B can take 30--90 seconds per response. Verify inference route: `nemoclaw my-assistant status`. |
| Port 18789 already in use | Another process is bound to the port | `lsof -i :18789` then `kill <PID>`. If needed, `kill -9 <PID>` to force-terminate. |
| Web UI port forward dies or dashboard unreachable | Port forward not active | `openshell forward stop 18789 my-assistant` then `openshell forward start 18789 my-assistant --background`. |
| Web UI shows `origin not allowed` | Accessing via `localhost` instead of `127.0.0.1` | Use `http://127.0.0.1:18789/#token=...` in the browser. The gateway origin check requires `127.0.0.1` exactly. |
| Telegram bridge does not start | Missing environment variables | Ensure `TELEGRAM_BOT_TOKEN` and `SANDBOX_NAME` are set on the host. `SANDBOX_NAME` must match the sandbox name from onboarding. |
| Telegram bridge needs restart but `nemoclaw stop` does not work | Known bug in `nemoclaw stop` | Find the PID from the `nemoclaw start` output, force-kill with `kill -9 <PID>`, then run `nemoclaw start` again. |
| Telegram bot receives messages but does not reply | Telegram policy not added to sandbox | Run `nemoclaw my-assistant policy-add`, type `telegram`, hit Y. Then restart the bridge with `nemoclaw start`. |

> [!NOTE]
> DGX Spark uses a Unified Memory Architecture (UMA), which enables dynamic memory sharing between the GPU and CPU. With many applications still updating to take advantage of UMA, you may encounter memory issues even when within the memory capacity of DGX Spark. If that happens, manually flush the buffer cache with:

```bash
sudo sh -c 'sync; echo 3 > /proc/sys/vm/drop_caches'
```

For the latest known issues, please review the [DGX Spark User Guide](https://docs.nvidia.com/dgx/dgx-spark/known-issues.html).
