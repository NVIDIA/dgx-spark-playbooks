# Run NemoClaw with a Local LLM

> Build your first local AI assistant on DGX Station using NemoClaw in a secure sandbox, with optional Telegram.


## Table of Contents

- [Overview](#overview)
  - [What you'll accomplish](#what-youll-accomplish)
  - [Notice and disclaimers](#notice-and-disclaimers)
  - [Isolation layers (OpenShell)](#isolation-layers-openshell)
  - [What to know before starting](#what-to-know-before-starting)
  - [Prerequisites](#prerequisites)
  - [Have ready before you begin](#have-ready-before-you-begin)
  - [Ancillary files](#ancillary-files)
  - [Time and risk](#time-and-risk)
- [Instructions](#instructions)
  - [Interaction Rules](#interaction-rules)
  - [Goal](#goal)
  - [Agent Selection](#agent-selection)
  - [Hardware and Readiness](#hardware-and-readiness)
  - [Administrator Access](#administrator-access)
  - [DGX Express Install](#dgx-express-install)
  - [Windows WSL Express Install](#windows-wsl-express-install)
  - [Runtime and Provider Selection](#runtime-and-provider-selection)
  - [Local Models](#local-models)
  - [Avoid Interactive Menus](#avoid-interactive-menus)
  - [Handle Tokens Securely and Visually](#handle-tokens-securely-and-visually)
  - [Credential Form and SSH](#credential-form-and-ssh)
  - [Messaging During Initial Onboarding](#messaging-during-initial-onboarding)
  - [Policy, Approval, and Verification](#policy-approval-and-verification)
  - [Use Docs for Information](#use-docs-for-information)
  - [Step 1. Install NemoClaw](#step-1-install-nemoclaw)
  - [Step 2. NemoClaw Onboarding](#step-2-nemoclaw-onboarding)
  - [Step 3. Interact with OpenClaw](#step-3-interact-with-openclaw)
  - [Step 4. Enable Brave Search in sandbox](#step-4-enable-brave-search-in-sandbox)
  - [Step 5. Set up Messaging Channel (Telegram Bot as an example)](#step-5-set-up-messaging-channel-telegram-bot-as-an-example)
  - [Step 6. Set Up NemoClaw Agents](#step-6-set-up-nemoclaw-agents)
  - [Step 7. Update NemoClaw](#step-7-update-nemoclaw)
  - [Step 8. Stop services](#step-8-stop-services)
  - [Step 9. Uninstall NemoClaw](#step-9-uninstall-nemoclaw)
- [On Dual DGX Station](#on-dual-dgx-station)
  - [Step 1. Prerequisites](#step-1-prerequisites)
  - [Step 2. Verify the CX8 fabric](#step-2-verify-the-cx8-fabric)
  - [Step 3. Download and copy the model cache](#step-3-download-and-copy-the-model-cache)
  - [Step 4. Start the Nemotron 3 Ultra on station-1](#step-4-start-the-nemotron-3-ultra-on-station-1)
  - [Step 5. Start the Nemotron 3 Ultra on station-2](#step-5-start-the-nemotron-3-ultra-on-station-2)
  - [Step 6. Monitor startup without interrupting it](#step-6-monitor-startup-without-interrupting-it)
  - [Step 7. Validate the API, reasoning, and tool calls](#step-7-validate-the-api-reasoning-and-tool-calls)
  - [Step 8. Perform routine health checks](#step-8-perform-routine-health-checks)
  - [Step 9. Troubleshoot common failures](#step-9-troubleshoot-common-failures)
  - [Step 10. Stop or remove the deployment](#step-10-stop-or-remove-the-deployment)
  - [Related resources](#related-resources)
- [Troubleshooting](#troubleshooting)

---

## Overview

## Basic idea

**NVIDIA NemoClaw** is an open-source reference stack that simplifies the safe deployment of specialized autonomous agents. It installs the **NVIDIA OpenShell** runtime — an environment designed for executing agents with additional security — and connects them to **local vLLM** inference on your DGX Station. A single installer command (`nemoclaw.sh`) detects the DGX Station and offers an **Express Install** that selects the recommended model and policy, installs Node.js, OpenShell, and the NemoClaw CLI, and creates the sandbox.

By the end of this playbook you will have a working AI agent inside an OpenShell sandbox, reachable through the **Web UI** or **terminal TUI**, with inference routed to **local vLLM** on the DGX Station. You can optionally add **Telegram** and optional **web search** — all without exposing your host filesystem or network beyond what you explicitly allow in policy.

### What you'll accomplish

- Install **NemoClaw** with one command (`nemoclaw.sh`), which pulls Node.js, OpenShell, and the CLI as needed
- Accept the DGX Station **Express Install** for managed vLLM, the `nemotron-3-ultra-550b-a55b` model, the `my-assistant` sandbox, and Balanced policy, or choose custom onboarding
- Open the **Web UI** to interact with agent
- Optionally enable **Brave Search** or **Telegram** after onboarding
- **Cleanup and uninstall** with the documented `uninstall.sh` flags when finished

### Notice and disclaimers

The following sections describe safety, risks, and your responsibilities when running this demo.

#### Quick start safety check

**Use only a clean environment.** Run this demo on a fresh device or VM with no personal data, confidential information, or sensitive credentials. Keep it isolated like a sandbox.

By installing this demo, you accept responsibility for all third-party components, including reviewing their licenses, terms, and security posture. Read and accept before you install or use.

#### What you're getting

This experience is provided "AS IS" for demonstration purposes only — no warranties, no guarantees. This is a demo, not a production-ready solution. You will need to implement appropriate security controls for your environment and use case.

#### Key risks with AI agents

- **Data leakage** — Any materials the agent accesses could be exposed, leaked, or stolen.
- **Malicious code execution** — The agent or its connected tools could expose your system to malicious code or cyber-attacks.
- **Unintended actions** — The agent might modify or delete files, send messages, or access services without explicit approval.
- **Prompt injection and manipulation** — External inputs or connected content could hijack the agent's behavior in unexpected ways.

#### Participant acknowledgement

By participating in this demo, you acknowledge that you are solely responsible for your configuration and for any data, accounts, and tools you connect. To the maximum extent permitted by law, NVIDIA is not responsible for any loss of data, device damage, security incidents, or other harm arising from your configuration or use of NemoClaw demo materials, including OpenClaw or any connected tools or services.

### Isolation layers (OpenShell)

| Layer      | What it protects                                   | When it applies             |
|------------|----------------------------------------------------|-----------------------------|
| Filesystem | Prevents reads/writes outside allowed paths.       | Locked at sandbox creation. |
| Network    | Blocks unauthorized outbound connections.          | Hot-reloadable at runtime.  |
| Process    | Blocks privilege escalation and dangerous syscalls.| Locked at sandbox creation. |
| Inference  | Reroutes model API calls to controlled backends.   | Hot-reloadable at runtime.  |

### What to know before starting

- Basic use of the Linux terminal and SSH
- Familiarity with Docker (permissions, `docker run`, optional `docker` group membership)
- Awareness of the security and risk sections above
- **`sudo` (passwordless or interactive) access on the DGX Station** -- the installer and several troubleshooting steps in this playbook require root privileges (e.g. installing Node.js, adding your user to the `docker` group, installing `cloudflared` via `dpkg`, editing `/etc/docker/daemon.json`, restarting the `docker` service).

### Prerequisites

**Hardware:**

- A DGX Station (GB300) with keyboard and monitor, or SSH access

**Software:**

- Fresh install of DGX OS with latest updates

**Storage:**

- Capacity for the approximately **352 GB** Express model download, the vLLM container, and temporary download space

Verify your system before starting:

```bash
head -n 2 /etc/os-release
nvidia-smi
docker info --format '{{.ServerVersion}}'
```

Expected: Ubuntu 24.04, NVIDIA GB300 GPU, Docker 28.x+.

### Have ready before you begin

| Item | When you need it |
|------|------------------|
| **Telegram bot token** (optional) | Create with [@BotFather](https://t.me/BotFather) (`/newbot`). You can paste it during **onboarding** (Step 3) **or** when you run **`nemoclaw <sandbox> channels add telegram`** later. |
| **Brave Search API key** (optional) | From [Brave Search API](https://brave.com/search/api/) if you enable web search during onboarding, or to add it later by re-running onboarding with `BRAVE_API_KEY` set (see Step 4). |

### Ancillary files

All required assets are handled by the NemoClaw installer. No manual cloning is needed.

### Time and risk

- **Estimated time:** Installation and onboarding take about 30–60 minutes, plus the approximately 352 GB model download. Total time depends heavily on network and storage performance. Optional Brave, Telegram, and cloudflared steps add time if you do them in a second session.
- **Risk level:** Medium — you are running an AI agent in a sandbox; risks are reduced by isolation but not eliminated. Use a clean environment and do not connect sensitive data or production accounts.
- **Last Updated:** 07/16/2026
  - Make the DGX Station Express Install the primary setup path

## Instructions

## Set Up from Your Coding Agent

You can ask a local coding agent such as Cursor, Claude Code, Codex, or Copilot to guide this setup and run approved commands for you. When the agent asks which computer you are using, select **Linux**.

Copy the starter prompt below and paste it into your coding agent. Use the code block's copy control when available, or select the text manually.

<!-- BEGIN NEMOCLAW STARTER PROMPT
Canonical source: https://github.com/NVIDIA/NemoClaw/blob/main/docs/resources/starter-prompt.md
Keep the prompt text unchanged when updating this synchronized block.
-->

````markdown
## NemoClaw Instructions for a Non-Technical User

Help me install and run NVIDIA NemoClaw from this coding-agent UI.
I may use Cursor, Claude Code, Codex, Copilot, or another local coding agent.
I do not know how to use a terminal.

### Interaction Rules

- Ask exactly one question at a time.
- Use clickable choices when supported; otherwise show one short numbered list and wait.
- Start by asking: "What computer are you using?" Choices: macOS, Windows, Linux.
- Next ask which agent I want: OpenClaw, Hermes, or LangChain Deep Agents Code.
- Never ask me to run commands myself, except the one workstation-side `ssh -N -L` command needed to open a remote credential form securely.
- Explain each command in plain language, ask permission, then run it for me.
- Pause before installs, system changes, administrator access, large downloads, credentials, sandbox creation, and long-running processes.
- Summarize command output instead of asking me to copy it into chat.
- Explain errors and unfamiliar terms such as Docker, container, model, API key, port, and SSH.
- Never ask me to paste passwords, API keys, tokens, or private credentials into chat.
- Use redacted placeholders such as `<PASTE_YOUR_API_KEY_HERE>` in examples.
- During long operations, give a short update at least once per minute.
- Do not start duplicate installers, downloads, or model servers.
- Verify results after important commands; do not rely only on exit codes.

### Goal

Install NemoClaw, collect onboarding choices before execution, include messaging in the first sandbox build when the selected agent supports it, launch the selected agent, and verify that it responds.

### Agent Selection

Ask: "Which NemoClaw agent would you like?"
Choices:

1. OpenClaw, the default.
2. Hermes.
3. LangChain Deep Agents Code.

Use `NEMOCLAW_AGENT=hermes` or `nemohermes onboard` for Hermes.
Use `NEMOCLAW_AGENT=langchain-deepagents-code` or `nemo-deepagents onboard` for Deep Agents.

### Hardware and Readiness

- On Linux, ask permission to run a read-only readiness check before provider selection.
- Check distribution, architecture, product and firmware identity, GPU and memory, NVIDIA driver, Container Toolkit, Docker, Node.js, disk space, existing NemoClaw, Ollama, vLLM, relevant ports, and administrator access.
- Classify the computer as DGX Spark, DGX Station, NVIDIA GB300, another NVIDIA computer, ordinary macOS/Linux, or unknown.
- Do not identify DGX Spark or DGX Station from the GPU name alone; combine product, firmware, architecture, and GPU evidence.
- A confirmed NVIDIA GB300 can independently qualify for expanded local-runtime choices.
- If uncertain, explain that and let NemoClaw's official preflight make the final platform decision.

### Administrator Access

- Check administrator availability without waiting for input, such as with a non-interactive sudo check.
- If passwordless sudo works, continue without prompt mode.
- If passwordless sudo is unavailable but the coding-agent UI provides a secure visible password prompt, explain why access is needed, ask permission, and set `NEMOCLAW_NON_INTERACTIVE_SUDO_MODE=prompt`.
- Let the real `sudo` program collect the password; never use chat or the API-key form for the computer password.
- If neither passwordless sudo nor a secure password prompt is available, stop before the affected install or system change.
- Never pipe a password, store it in a file, generate a password helper, or put it in command arguments.
- Offer a user-local alternative only when official documentation supports it for that exact operation.
- Do not silently use user-local Ollama for a system Ollama upgrade when the old system service would remain active.

### DGX Express Install

If DGX Spark or DGX Station is detected, ask: "Do you want the recommended Express Install?"
Choices:

1. Yes, use the platform's Express model and required Balanced policy.
2. No, let me choose the runtime and model.

If DGX Spark Express is selected:

- Use managed vLLM and set `NEMOCLAW_PROVIDER=install-vllm`.
- Leave `NEMOCLAW_VLLM_MODEL` unset so the installed maintained release selects its current Spark Express model.
- Explain container and model download sizes before asking permission.
- Report the model selected by the installed release.

If DGX Station Express is selected:

- Use managed vLLM.
- Explicitly select `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`.
- Do not leave the model unset; the ordinary managed-vLLM default can select DeepSeek and would not reproduce Express.
- Set `NEMOCLAW_PROVIDER=install-vllm`.
- Set `NEMOCLAW_VLLM_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4`.
- Disclose that the model download is approximately 352 GB, in addition to the vLLM container and temporary download space.
- Verify the model-cache filesystem and Docker storage have sufficient capacity.
- Warn that DGX Station managed deployment has deferred end-to-end physical-hardware validation.
- Describe it as an evaluation path, not a validated production deployment.
- Explain that startup may fail despite passing initial checks.
- Ask separately for approval of the approximately 352 GB download.

For both Express paths:

- Balanced policy is required for Express; set `NEMOCLAW_POLICY_TIER=balanced`, `NEMOCLAW_NON_INTERACTIVE=1`, and the selected `NEMOCLAW_AGENT`.
- Set `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` only after explaining the notice and receiving approval.
- Set `NEMOCLAW_YES=1` only after both the separate download approval and final install approval.
- Set `NEMOCLAW_NON_INTERACTIVE_SUDO_MODE=prompt` only when required and a secure sudo prompt is available.
- Ask separately for sandbox name, web search, messaging when the selected agent supports it, download approval, and final install approval.

### Windows WSL Express Install

If official detection identifies Windows WSL, offer the maintained Windows Express path before the normal provider menu.
Explain that it uses Windows-host Ollama through Docker Desktop WSL integration.
If selected, set `NEMOCLAW_PROVIDER=install-windows-ollama`, collect the same separate approvals, and let the installed release choose its maintained Ollama model.
Do not start a second Ollama service on the same port.

### Runtime and Provider Selection

If Express is declined on DGX Spark, DGX Station, or GB300, ask: "Which inference runtime or provider would you like?"
Choices:

1. Existing vLLM, only when a ready server is detected on `localhost:8000`.
2. Managed vLLM, optimized local inference with a large download.
3. Local Ollama, only when the selected agent and platform support it.
4. NVIDIA Endpoints, which requires an NVIDIA API key.
5. OpenRouter, which requires an OpenRouter API key.
6. OpenAI, which requires an OpenAI API key.
7. Anthropic, which requires an Anthropic API key.
8. Google Gemini, which requires a Gemini API key.
9. Model Router, which requires an NVIDIA API key.
10. Other OpenAI-compatible endpoint, which requires an endpoint, model, and usually a key.
11. Other Anthropic-compatible endpoint, which requires an endpoint, model, and usually a key.
12. Hermes Provider, only when Hermes is selected.

On ordinary supported macOS or Linux:

- Offer Local Ollama for OpenClaw or Hermes when it is installed, running, or officially installable.
- Do not offer Local Ollama for Deep Agents unless current official documentation adds support.
- Offer an existing ready vLLM server when detected.
- Also show all applicable hosted and compatible providers.
- Do not hide Ollama merely because the computer is not DGX or GB300.
- Omit managed vLLM unless current official support permits it for the detected hardware.

On other platforms, show every provider supported by the selected agent and platform.
Renumber choices after filtering and do not hide hosted providers behind another menu.
Ask required model, endpoint, credential, and download questions one at a time.

### Local Models

- Fetch current model choices from the selected agent's official Markdown documentation.
- The selected maintained NemoClaw release is authoritative for supported slugs and arguments.
- Managed-vLLM examples include `qwen3.6-27b`, `qwen3.6-35b-a3b-nvfp4`, `nemotron-3-nano-4b`, `deepseek-v4-flash`, and gated `deepseek-r1-distill-70b`.
- For Ollama, ask permission to inspect installed models and offer NemoClaw's memory-aware recommendation first.
- Current Ollama starter examples include `qwen3.6:35b`, `nemotron-3-nano:30b`, and `qwen3.5:9b`.
- Explain download size and storage requirements, then ask separately for permission.
- Do not request an NGC or Hugging Face credential unless the selected operation actually requires it.

### Avoid Interactive Menus

- Collect every choice before running the installer.
- Ask one question at a time for model, endpoint, sandbox name, web search, messaging when the selected agent supports it, policy when Express is not selected, credentials, administrator access, and downloads.
- Use non-interactive environment variables whenever supported.
- Never leave a command waiting at `Choose [1]:`.
- If a choice cannot be supplied non-interactively, stop before starting and explain the supported alternative.

### Handle Tokens Securely and Visually

Before collecting secrets, determine the exact environment-variable names and exact command argv, explain them, and ask permission.
Do not generate, rewrite, or redesign the helper or form.
Use this reviewed pair without modification:

- Helper: `https://raw.githubusercontent.com/NVIDIA/NemoClaw/dd61a307d7ddf7be99de8ff1e2678fb8ef42f8e6/scripts/local-credential-helper.mts` (SHA-256 `1a42bbe8dbc9003cb79d4e641b53760571aacd85293671aee97c09c0746fef33`).
- Form: `https://raw.githubusercontent.com/NVIDIA/NemoClaw/dd61a307d7ddf7be99de8ff1e2678fb8ef42f8e6/docs/resources/local-credential-form.html` (SHA-256 `5512a256e0ad7c63a26ab82cf4f5924e98652097172ab8a5dc9d9358dd4f6ae8`).

- Treat the two immutable URL and digest pairs as one reviewed trust boundary; before executing the helper, compute the SHA-256 digest of both downloaded files and compare each result with its pinned digest.
- If either digest differs, do not execute the helper; delete both temporary files and stop.
- Store them in a private temporary directory and delete them afterward.
- The helper requires Node.js 22.19 or newer.
- If Node is unavailable, use an existing secure local application prompt or secure terminal prompt; never use chat or generated credential code.
- Keep the helper bound to `http://127.0.0.1`, accept only one valid submission, and run only the already-approved command.
- Use `:secret` for secrets and `:text` only for non-secret values.
- Use `--execution-profile isolated` for stateless commands.
- For persistent install or onboarding, use `--execution-profile account-home --cwd <approved-absolute-directory>` and ask permission for both.
- Pass every `--field NAME:type`, then a literal `--`, an absolute executable path, and the exact approved argv.
- Never omit the literal `--`.
- Never use a relative, alias-only, or PATH-only approved executable.
- Never put credentials in argv.
- Command shape: `node --experimental-strip-types <helper> --execution-profile <profile> --form <form> --field NAME:secret -- <absolute-executable> <approved-args...>`.
- Use **Preview Credentials**, **Edit**, then **Confirm and Run Approved Command**.
- If the outcome is unknown, check whether the command ran; do not retry or resubmit blindly.
- Keep secrets in memory only long enough to start the command.
- Treat deletion as exposure minimization, not guaranteed erasure.
- Prefer letting an account-persistent command use its own reviewed secure credential prompt when available.
- For credential-bearing installation, use the reviewed helper only with an already-downloaded and verified installer.
- Do not hand-assemble a `curl | bash` wrapper around credentials.
- Never print, log, commit, cache, or paste secrets.

Use this provider mapping for non-interactive setup:

- NVIDIA Endpoints: `NEMOCLAW_PROVIDER=build`, `NVIDIA_INFERENCE_API_KEY`.
- OpenRouter: `NEMOCLAW_PROVIDER=openrouter`, `OPENROUTER_API_KEY`.
- OpenAI: `NEMOCLAW_PROVIDER=openai`, `OPENAI_API_KEY`.
- Anthropic: `NEMOCLAW_PROVIDER=anthropic`, `ANTHROPIC_API_KEY`.
- Gemini: `NEMOCLAW_PROVIDER=gemini`, `GEMINI_API_KEY`.
- Hermes Provider: `NEMOCLAW_PROVIDER=hermes-provider`; Hermes only.
- Model Router: `NEMOCLAW_PROVIDER=routed`, `NVIDIA_INFERENCE_API_KEY`.
- OpenAI-compatible: `NEMOCLAW_PROVIDER=custom`, endpoint, model, `COMPATIBLE_API_KEY`.
- Anthropic-compatible: `NEMOCLAW_PROVIDER=anthropicCompatible`, endpoint, model, `COMPATIBLE_ANTHROPIC_API_KEY`.
- Ollama: `NEMOCLAW_PROVIDER=ollama`, optional `NEMOCLAW_MODEL`.
- Existing vLLM: `NEMOCLAW_PROVIDER=vllm`.
- Managed vLLM: `NEMOCLAW_PROVIDER=install-vllm`; leave `NEMOCLAW_VLLM_MODEL` unset for DGX Spark Express, set it to `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` for DGX Station Express, or use an approved optional override for non-Express setup.
- Windows WSL Express: `NEMOCLAW_PROVIDER=install-windows-ollama`.

Do not offer Hermes Provider for OpenClaw or Deep Agents.

### Credential Form and SSH

Ask whether I use SSH only after the helper starts and prints its complete one-time URL: "Are you connected to this computer through SSH?"
Choices:

1. No, I am using it directly.
2. Yes, this is a remote SSH computer.
3. I am not sure.

- Treat the helper's complete URL as an opaque, sensitive, one-time capability.
- Preserve its scheme, host, port, `/local-credential-form.html` path, complete `field=` query string, and `#cap=` fragment exactly.
- Never replace it with a reconstructed bare `http://127.0.0.1:<port>` URL.
- If local, give me the complete original URL unchanged.
- If remote, read its port and ask me to run: `ssh -N -L <port>:127.0.0.1:<port> <username>@<host>`.
- Fill in the actual port, username, and host when known.
- Explain that it runs on my workstation, normally prints nothing, and must remain open until credential entry finishes.
- After the tunnel starts, give me the helper's original complete URL unchanged.
- Require the same port on both sides; do not remap the helper to another local port.
- If that local port is occupied, stop the unused helper safely, resolve the conflict or start a fresh helper session, and use only the new complete URL.
- Never reuse an old URL or expose the form through `0.0.0.0`, LAN, public URL, shared tunnel, or unauthenticated proxy.
- Tell me when it is safe to stop the forwarding command.

### Messaging During Initial Onboarding

For OpenClaw or Hermes, ask before the first sandbox build: "Do you want to configure a messaging channel during onboarding?"
Choices: No, Telegram, Discord, Slack, WhatsApp, WeChat (experimental).
Skip messaging for Deep Agents.
Configure one channel at a time, then ask whether to add another.
Collect messaging before policy selection so the first image includes channel configuration and matching network presets.

- Telegram requires `TELEGRAM_BOT_TOKEN`; optional settings include allowed IDs, mention mode, and OpenClaw group policy.
- Discord requires `DISCORD_BOT_TOKEN`; optional settings include server ID, user ID, and mention mode.
- Slack requires `SLACK_BOT_TOKEN` and `SLACK_APP_TOKEN`; optional settings include allowed users and channels.
- WhatsApp uses documented allowed IDs for non-interactive selection, followed by QR pairing after startup.
- WeChat requires an interactive QR handshake; explain the limitation before installation and never leave an unsupported UI waiting.

Collect messaging secrets through the reviewed helper and exact-URL SSH flow.
Do not manually set `NEMOCLAW_MESSAGING_CHANNELS_B64`; let NemoClaw generate it.
Use `channels add` and rebuild only for channels omitted from initial onboarding or changed later.

### Policy, Approval, and Verification

- For Express, state that Balanced policy is required, keep `NEMOCLAW_POLICY_TIER=balanced`, and skip the policy-tier question.
- For non-Express installation, ask for Balanced, Restricted, or Open policy.
- Explain that messaging and web-search selections add required endpoints.
- Before installation, summarize platform, administrator access, agent, Express choice, provider, exact model, validation warning, downloads, storage, sandbox, web search, messaging, policy, credential names without their values, and system changes.
- Ask for final permission.
- Set `NEMOCLAW_ACCEPT_THIRD_PARTY_SOFTWARE=1` and `NEMOCLAW_YES=1` only after their approvals.
- Keep credentials in the approved environment and never display them.
- Verify the command and version, sandbox status, provider, model, `inference.local`, GPU access when applicable, messaging bridges when configured, and dashboard route when available.
- If `curl | bash` returns no output, verify installation; if absent, ask permission to download and inspect the official installer before retrying.
- For remote dashboards, use private loopback SSH forwarding, preserve authenticated URLs exactly, and treat them as secrets.
- Ask permission before sending a live channel test or harmless first agent prompt.
- Declare success only after the sandbox is ready and the agent responds.
- Summarize what was installed, how to reconnect, what starts after reboot, and anything skipped.

### Use Docs for Information

- Use clean `.md` pages for searching more information in the selected agent's documentation. Example URLs:
  - [Documentation index for AI clients](https://docs.nvidia.com/nemoclaw/llms.txt)
  - [OpenClaw quickstart](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/get-started/quickstart.md)
  - [Hermes quickstart](https://docs.nvidia.com/nemoclaw/latest/user-guide/hermes/get-started/quickstart.md)
  - [Deep Agents quickstart](https://docs.nvidia.com/nemoclaw/latest/user-guide/deepagents/get-started/quickstart.md)
- Suggest to add the docs MCP server `https://docs.nvidia.com/nemoclaw/_mcp/server` if the coding agent supports MCP.
````

Prompt source: [NVIDIA NemoClaw starter prompt](https://github.com/NVIDIA/NemoClaw/blob/main/docs/resources/starter-prompt.md)

<!-- END NEMOCLAW STARTER PROMPT -->

## Set Up from Your Terminal

If you prefer to perform the setup directly in a terminal, continue with Phase 1.

## Phase 1: Install and Run NemoClaw

### Step 1. Install NemoClaw

This single command handles everything: installs Node.js (if needed), installs OpenShell, clones the last known good (LKG) NemoClaw release automatically, builds the CLI, and creates a sandbox.

```bash
curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash
```

After you accept the third-party software notice, the installer detects the DGX Station and immediately offers the recommended Express Install:

```text
Detected DGX Station.
...
Run express install with these settings? [Y/n]:
```

Press **Enter** or enter `Y` to accept. Express Install:

- selects managed local vLLM with `nemotron-3-ultra-550b-a55b`
- discloses the approximately **352 GB** model download before confirmation
- uses `my-assistant` as the sandbox name
- applies policy in `suggested` mode with the `balanced` tier
- runs onboarding non-interactively while still allowing secure `sudo` password prompts for required host changes
- downloads the configured vLLM container and model, starts local inference, and creates the sandbox

> [!NOTE]
> Express Install deliberately uses Nemotron 3 Ultra instead of the DGX Station managed-vLLM profile default, `deepseek-v4-flash`. To keep the one-confirmation Express flow but use DeepSeek V4 Flash, run `curl -fsSL https://www.nvidia.com/nemoclaw.sh | bash -s -- --station-deepseek`.

> [!NOTE]
> If you want to choose the provider, model, sandbox name, or policy yourself, enter `n` and continue with Step 2. To skip the prompt before running the installer, set `NEMOCLAW_NO_EXPRESS=1`; setting `NEMOCLAW_PROVIDER` also bypasses Express Install and uses that provider.

> [!WARNING]
> Express Install automates configuration but does not change DGX Station's **Deferred** support status. Full onboarding with this recipe has not completed end-to-end validation on physical DGX Station hardware.

The installer requires **Node.js 22.16+** (installed automatically if missing). Allow capacity for the approximately 352 GB model, the vLLM container, and temporary download space. When Express Install finishes, skip to Step 3.

### Step 2. NemoClaw Onboarding

> [!NOTE]
> This section is for custom setup. If you accepted **Express Install** in Step 1, onboarding already ran with the recommended DGX Station settings; skip to Step 3. Follow this section if you declined Express Install or later run `nemoclaw onboard` to customize the configuration.

During custom setup, the onboard wizard walks you through:

1. **Select your agent** -- Choose which agent to run in the sandbox. Enter `1` for **OpenClaw**.
2. **Configuring inference** -- Choose a local inference option to run models on your DGX Station.
3. **Inference models** -- Select a model from the installer prompts. NemoClaw will prepare any required local model artifacts when needed.
4. **Sandbox name** -- Pick a name (e.g. my-assistant). Each sandbox requires a unique name.
5. **Apply this configuration** -- Enter `Y` to confirm setting up local inference.
6. **Enable Brave Web Search** -- Optional. If you enable it, paste a [Brave Search API](https://brave.com/search/api/) key when prompted.
7. **Messaging channels** -- Optional. If you enable it, choose your desired bot (`telegram`, `discord` or `slack`) and paste your bot token when prompted.
8. **Resource profiles** -- Choose how much CPU and RAM the sandbox may use (`creator`, `gamer`, `game-developer`, `developer`, `custom`, or `No profile`). Press **Enter** to accept the default (`No profile`).
9. **Policy presets** -- Choose desired Policy tier (`Balanced` recommended) and accept/edit the suggested presets when prompted (confirm with **Enter**).

When complete you will see output like:

```text
 ──────────────────────────────────────────────────
  OpenClaw is ready

  Sandbox:  my-assistant
  Model:    nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4 (Local vLLM)

  Start chatting

    Browser:
      http://127.0.0.1:18789/

    Terminal:
      nemoclaw my-assistant connect
      then run: openclaw tui

  Authenticated dashboard URL, if needed:
    nemoclaw my-assistant dashboard-url --quiet

  Remote access (SSH session detected):
    On your workstation, run:
      ssh -L 18789:127.0.0.1:18789 lab@<host>
    Then open the dashboard URL above in your local browser.

  Manage later

    Status:      nemoclaw my-assistant status
    Logs:        nemoclaw my-assistant logs --follow
    Model:       nemoclaw inference set --model <model> --provider <provider> --sandbox my-assistant
    Policies:    nemoclaw my-assistant policy-add
    Credentials: nemoclaw credentials reset <KEY> && nemoclaw onboard
  ──────────────────────────────────────────────────
```

> [!NOTE]
> - If `nemoclaw` is not found after install, run `source ~/.bashrc` to reload your shell path.
> - Time to finish **Onboarding** can vary, depending on the model choice and internet speed.

NemoClaw Onboarding can be run repeatedly to create multiple sandboxes for independent use cases. Use `--name <new-name>` to create an additional sandbox alongside any existing ones:

```bash
nemoclaw onboard --gpu --name <new-name>
```

> [!IMPORTANT]
> Use `--name <new-name>` to create an additional sandbox without affecting existing ones. The `--fresh` flag is a destructive option reserved for starting a completely new onboard session — if a sandbox with the same name already exists, `--fresh` will **destroy and recreate it**. Only use `--fresh` when you intend to wipe and re-onboard — for example, to recover from a failed or interrupted onboarding session. To add a feature such as Brave Search to an existing sandbox, use `--recreate-sandbox` instead (see Step 4).

### Step 3. Interact with OpenClaw

There are two ways to interact with your OpenClaw, Web UI or terminal UI. 

#### Option 1. Web UI

Get the full dashboard URL (includes the auto-assigned port and token):

```bash
nemoclaw my-assistant dashboard-url --quiet
```

This prints a URL like `http://127.0.0.1:18790/#token=<token>`. The port is auto-assigned (commonly 18789 or 18790) and may differ between installs.

**If accessing the Web UI directly on the DGX Station** (keyboard and monitor attached), open the dashboard URL in a browser.

**If accessing the Web UI from a remote machine**, you need to set up an SSH tunnel.

First, note the port number from the dashboard URL above (e.g. `18790`).

Find your DGX Station's IP address:

```bash
hostname -I | awk '{print $1}'
```

This prints the primary IP address (e.g. `192.168.1.42`). You can also find it in **Settings > Wi-Fi** or **Settings > Network** on the DGX Station's desktop, or check your router's connected-devices list.

From your remote machine, create an SSH tunnel using the port from above (replace `<port>` and `<your-station-ip>`):

```bash
ssh -L <port>:127.0.0.1:<port> <your-user>@<your-station-ip>
```

Now open the dashboard URL in your remote machine's browser.

> [!IMPORTANT]
> Use `127.0.0.1`, not `localhost` -- the gateway origin check requires an exact match.

> [!NOTE]
> If the Web UI fails to load and the port forward may be stale, get the port from `nemoclaw my-assistant dashboard-url --quiet` and reset:
> ```bash
> openshell forward stop <port> my-assistant || true
> openshell forward start <port> my-assistant --background
> ```

#### Option 2. Terminal UI

Connect to the sandbox:

```bash
nemoclaw my-assistant connect
```

Then launch the terminal UI inside the sandbox:

```bash
openclaw tui
```

You can start chatting with OpenClaw. Press **Ctrl+C** to exit the terminal UI.

To exit the sandbox:

```bash
exit
```

---

## Phase 2: Modify NemoClaw Policy

### Step 4. Enable Brave Search in sandbox

To add Brave Web Search to an existing sandbox, re-run onboarding with `BRAVE_API_KEY` set. Get a key from the [Brave Search API](https://brave.com/search/api/) console:

```bash
BRAVE_API_KEY=<your-brave-search-api-key> nemoclaw onboard --name <sandbox-name>
```

If the sandbox already exists without web search enabled, NemoClaw needs to **recreate** the sandbox so the Brave configuration is baked into the agent runtime. Accept the recreate prompt when it appears.

For scripted (non-interactive) runs, pass `--recreate-sandbox` to perform the rebuild without prompting:

```bash
BRAVE_API_KEY=<your-brave-search-api-key> \
  nemoclaw onboard --name <sandbox-name> --recreate-sandbox --non-interactive
```

> [!NOTE]
> `--recreate-sandbox` clearly describes the intentional rebuild needed to add web search. Reserve `--fresh` for recovery from a failed or interrupted onboarding session — it discards the wizard state and starts over, which is not the right tool for adding a feature to an already-created sandbox.

To confirm web search is enabled, relaunch your OpenClaw WebUI or terminal UI. Ask the agent for something that needs **live web search**. If requests still fail, recheck **`policy-list`** and re-read the onboard output for Brave/API errors.

### Step 5. Set up Messaging Channel (Telegram Bot as an example)

These steps apply when your sandbox exists but **Telegram was never configured** (you skipped **Messaging channels** in Step 2, or the sandbox policy tier never included Telegram-related egress). Replace `<sandbox-name>` with your sandbox (for example `my-assistant`).

> [!IMPORTANT]
> Telegram does **not** require cloudflared. The bot uses long-polling to pull messages from Telegram servers, so no public URL or tunnel is needed. cloudflared is only for exposing the dashboard/Web UI remotely (see sub-step 5 below) and is unrelated to messaging.

#### 1. Create a Telegram bot

In Telegram, open [@BotFather](https://t.me/BotFather), send `/newbot`, and complete the prompts. Copy the **bot token** BotFather returns and keep it ready for the next step.

#### 2. Register Telegram with NemoClaw and rebuild the sandbox

```bash
nemoclaw <sandbox-name> channels add telegram
```

Paste the token when prompted. NemoClaw persists credentials and **rebuilds** the sandbox so OpenClaw can use Telegram as a messaging channel.

#### 3. (If needed) Allow Telegram egress in the sandbox policy

If messages fail with network or policy errors after the channel is registered, inspect presets and add Telegram-related egress if your tier omitted it:

```bash
nemoclaw <sandbox-name> policy-list
nemoclaw <sandbox-name> policy-add telegram
```

Preset names follow your selected tier; confirm against [Network policies](https://docs.nvidia.com/nemoclaw/latest/reference/network-policies.html).

> [!NOTE]
> To approve blocked network requests for new endpoints at runtime, use the OpenShell TUI. See [Approve or Deny Agent Network Requests](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/network-policy/approve-network-requests).

#### 4. Verify Telegram

Telegram uses long-polling (`getUpdates`) — the sandbox actively pulls messages from Telegram servers. **No public URL or cloudflared tunnel is required for Telegram to work.**

Open Telegram, find your bot, and send a message. The bot should forward traffic to the agent in your NemoClaw sandbox and reply.

> [!NOTE]
> The first response may take longer depending on model size (30B models respond in a few seconds; larger models may take longer on first inference).

> [!NOTE]
> If the bot does not respond:
> - Run `nemoclaw <sandbox-name> status` to confirm the sandbox is running and inference is healthy.
> - Run `nemoclaw <sandbox-name> logs --follow` and look for Telegram-related errors.
> - If Telegram egress is missing, run `nemoclaw <sandbox-name> policy-add` and select `telegram`.
> - If the channel was never registered, run `nemoclaw <sandbox-name> channels add telegram`.

> [!NOTE]
> The `channels add telegram` wizard also prompts for an optional **Telegram User ID** to restrict who can DM the bot. Send `/start` to [@userinfobot](https://t.me/userinfobot) on Telegram to get your numeric user ID. If you skip this, the bot will require device pairing (a terminal-based code confirmation) before responding to messages.

> [!NOTE]
> For details on restricting which Telegram chats can interact with the agent, see the [NemoClaw Telegram bridge documentation](https://docs.nvidia.com/nemoclaw/latest/deployment/set-up-telegram-bridge.html).

#### 5. (Optional) Install cloudflared for remote Web UI access

The cloudflared tunnel provides a **public URL for the Web UI dashboard** — it is not related to Telegram messaging.

Install cloudflared (DGX Station is arm64):

```bash
curl -L --output cloudflared.deb \
  https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-arm64.deb
sudo dpkg -i cloudflared.deb
```

Start the tunnel:

```bash
nemoclaw tunnel start
```

Verify:

```bash
nemoclaw status
```

You should see `● cloudflared` with a `trycloudflare.com` public URL.

---

## Phase 3: Set Up NemoClaw Agent

### Step 6. Set Up NemoClaw Agents

Set up NemoClaw Agents in general require three steps: Configure NemoClaw security policy, Run Agent Workflow Prompt, Personalize the Workflow for your own use case.

Checkout these [Example NemoClaw Agents](https://build.nvidia.com/spark/nemoclaw-applications) for reference. Consider sharing your NemoClaw agent setup with the community at [DGX Station Developer Forum](https://forums.developer.nvidia.com/c/accelerated-computing/dgx-station-gb300-gb200)

---

## Phase 4: Update NemoClaw

### Step 7. Update NemoClaw

To check whether a newer NemoClaw LKG release is available, run:

```bash
nemoclaw update --check
```

To update the host-side NemoClaw CLI to the current LKG release without prompts, run:

```bash
nemoclaw update --yes
```

This updates the host CLI only; it does not rebuild existing sandboxes. After updating, check whether any sandboxes need to be rebuilt:

```bash
nemoclaw upgrade-sandboxes --check
```

If stale sandboxes are reported, follow the command output to rebuild them.

For details, see the official NemoClaw [command reference](https://docs.nvidia.com/nemoclaw/latest/reference/commands.html).

---

## Phase 5: Cleanup and Uninstall

### Step 8. Stop services

Stop the cloudflared tunnel:

```bash
nemoclaw tunnel stop
```

Stop the port forward:

```bash
openshell forward list          # find active forwards and their ports
openshell forward stop <port>   # stop the dashboard forward (use the port shown above)
```

### Step 9. Uninstall NemoClaw

The NemoClaw CLI includes a built-in uninstaller. It removes all sandboxes, the OpenShell gateway, Docker containers/images/volumes, the CLI, and state directories. Docker, Node.js, npm, and the vLLM container image are preserved. Your `~/.nemoclaw/` user data (`rebuild-backups/`, `backups/`, `sandboxes.json`) is also preserved unless you pass `--destroy-user-data`.

```bash
nemoclaw uninstall --yes
```

To remove everything including the downloaded Ollama models:

```bash
nemoclaw uninstall --yes --delete-models
```

**Uninstaller flags:**

| Flag | Effect |
|------|--------|
| `--yes` | Skip the confirmation prompt |
| `--keep-openshell` | Leave the `openshell` binary in place |
| `--delete-models` | Also remove Ollama models pulled by NemoClaw |
| `--destroy-user-data` | Also remove preserved user data under `~/.nemoclaw/` (`rebuild-backups/`, `backups/`, `sandboxes.json`) |

> [!NOTE]
> If the `nemoclaw` CLI is not available (e.g. install failed partway), use the remote uninstaller as a fallback:
> ```bash
> curl -fsSL https://raw.githubusercontent.com/NVIDIA/NemoClaw/refs/heads/main/uninstall.sh | bash -s -- --yes
> ```

The uninstaller runs up to 7 steps:
1. Stop NemoClaw helper services and port-forward processes
2. Delete all OpenShell sandboxes, the NemoClaw gateway, and providers
3. Remove the global `nemoclaw` npm package
4. Remove NemoClaw/OpenShell Docker containers, images, and volumes
5. Remove downloaded Ollama models (only with `--delete-models`)
6. Remove config/state directories (`~/.config/openshell`, `~/.config/nemoclaw`) and the OpenShell binary
7. Remove preserved user data under `~/.nemoclaw/` (`rebuild-backups/`, `backups/`, `sandboxes.json`) — only with `--destroy-user-data`

> [!NOTE]
> `~/.nemoclaw/` user data is preserved by default and only removed in step 7 with `--destroy-user-data`. If you have a local clone at `~/.nemoclaw/source` you want to keep, move or back it up before running the uninstaller with that flag.

## Useful commands

| Command | Description |
|---------|-------------|
| `nemoclaw my-assistant connect` | Shell into the sandbox |
| `nemoclaw my-assistant status` | Show sandbox status and inference config |
| `nemoclaw my-assistant logs --follow` | Stream sandbox logs in real time |
| `nemoclaw list` | List all registered sandboxes |
| `nemoclaw tunnel start` | Start cloudflared tunnel (public URL for remote Web UI access) |
| `nemoclaw tunnel stop` | Stop the cloudflared tunnel |
| `nemoclaw my-assistant dashboard-url --quiet` | Print the full tokenized Web UI URL (includes auto-assigned port) |
| `openshell term` | Open the monitoring TUI on the host |
| `openshell forward list` | List active port forwards |
| `nemoclaw uninstall --yes` | Remove NemoClaw (preserves Docker, Node.js, vLLM image) |
| `nemoclaw uninstall --yes --delete-models` | Remove NemoClaw and downloaded Ollama models |

## On Dual DGX Station

## Deploy NVIDIA Nemotron 3 Ultra on Two DGX Stations

## Phase 1: Prepare Both Stations

Below steps deploy **NVIDIA Nemotron 3 Ultra** across 2 DGX Station with GB300 GPUs.

### Step 1. Prerequisites
- Complete [Connect Two DGX Stations for Distributed Workloads](https://build.nvidia.com/station/connect-two-stations/instructions) before starting this tab. Both CX8 RoCE rails must be configured and validated.
- Huggging Face [Access Token](https://huggingface.co/docs/hub/en/security-tokens) for downloading the model.
- Docker and [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/latest/install-guide.html) installed on both DGX Stations.
- Both DGX Stations must be connected to each other via CX8 RoCE rails.

> [!NOTE]
> The initial model download, weight loading, kernel compilation, autotuning, and CUDA graph capture can take more than an hour. Later starts are faster when the model cache and the existing containers are retained. Recreating the containers can rebuild compilation artifacts.

### Step 2. Verify the CX8 fabric

On both stations, confirm that the two CX8 interfaces are active:

```bash
ibdev2netdev
ip -br link show
ip -br address show
show_gids
```

The examples in this guide use the following direct-attach network. Use your configured addresses if they differ.

| Rail | `station-1` | `station-2` | HCA |
| --- | --- | --- | --- |
| 0 | `192.168.240.1/30` | `192.168.240.2/30` | `mlx5_0` |
| 1 | `192.168.240.5/30` | `192.168.240.6/30` | `mlx5_1` |

From `station-1`, verify both peers with jumbo packets:

```bash
ping -c 4 -M do -s 8972 192.168.240.2
ping -c 4 -M do -s 8972 192.168.240.6
```

Do not continue until both pings succeed and the two-station fabric playbook's RDMA and NCCL checks pass.

### Step 3. Download and copy the model cache

>[!NOTE]
> This step takes more than an hour depending on the internet speed and requires minimum storage of 350GB on both stations.

Authenticate to Hugging Face on `station-1`. The model is gated, so the account must have access before the download starts.

```bash
ssh station-1
hf auth login
hf auth whoami

export HF_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4

hf download "${HF_MODEL}" --cache-dir "${HOME}/.cache/huggingface"
```

Copy the complete Hugging Face cache layout to `station-2`. Copying only the model shards is not sufficient.

```bash
ssh station-2 'mkdir -p ~/.cache/huggingface'

rsync -aH --info=progress2 "${HOME}/.cache/huggingface/" station-2:.cache/huggingface/
```

Confirm that the model snapshot exists on both stations:

```bash
find "${HOME}/.cache/huggingface" -maxdepth 1 -type d \
  -name 'models--nvidia--NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4'
```

## Phase 2: Launch Nemotron 3 Ultra

### Step 4. Start the Nemotron 3 Ultra on station-1

Connect to `station-1` and set the deployment variables:

```bash
ssh station-1

export HEAD_IFACE=$(ibdev2netdev | awk '$1 == "mlx5_0" {print $5; exit}')
export HEAD_IP=$(ip -4 -o address show dev "${HEAD_IFACE}" \
  | awk '{split($4, address, "/"); print address[1]; exit}')
export GPU_UUID_HEAD=$(nvidia-smi \
  --query-gpu=name,uuid --format=csv,noheader \
  | awk -F', ' '/GB300|B300/ {print $2; exit}')
export IMG=vllm/vllm-openai:v0.25.1-aarch64
export HF_MODEL=nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4
export SERVED_MODEL=nemotron-ultra
```

Review the discovered values before starting the container:

```bash
printf 'HEAD_IFACE=%s\nHEAD_IP=%s\nGPU_UUID_HEAD=%s\n' \
  "${HEAD_IFACE}" "${HEAD_IP}" "${GPU_UUID_HEAD}"
```

Start the head container. It creates the Ray cluster and waits up to one hour for the worker GPU before launching vLLM.

```bash
sudo docker rm -f nemotron-ultra-head 2>/dev/null || true

sudo docker run -d --name nemotron-ultra-head \
  --restart unless-stopped --init \
  --network host --shm-size 16g \
  --gpus "device=${GPU_UUID_HEAD}" \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --ulimit memlock=-1 \
  -e HEAD_IP="${HEAD_IP}" \
  -e MODEL="${HF_MODEL}" \
  -e SERVED_MODEL="${SERVED_MODEL}" \
  -e VLLM_HOST_IP="${HEAD_IP}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=7200 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_TC=106 \
  -e NCCL_NET_GDR_LEVEL=PHB \
  -e NCCL_SOCKET_IFNAME="${HEAD_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${HEAD_IFACE}" \
  -e TP_SOCKET_IFNAME="${HEAD_IFACE}" \
  -e NCCL_IB_QPS_PER_CONNECTION=4 \
  -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
  -e UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 \
  -e HF_HOME=/models/huggingface \
  -v "${HOME}/.cache/huggingface:/models/huggingface" \
  --entrypoint bash "${IMG}" -lc '
    set -euo pipefail
    python3 -m pip install --break-system-packages "ray[cgraph]"
    ray start --head --node-ip-address="${HEAD_IP}" --port=6379 --num-gpus=1

    python3 - <<"PY"
import time
import ray

ray.init(address="auto")
deadline = time.time() + 3600
while ray.cluster_resources().get("GPU", 0) < 2:
    if time.time() >= deadline:
        raise TimeoutError("station-2 GPU did not join Ray within 3600 seconds")
    time.sleep(5)
print(ray.cluster_resources())
PY

    exec vllm serve "${HF_MODEL}" \
      --served-model-name "${SERVED_MODEL}" \
      --host 0.0.0.0 --port 8000 \
      --trust-remote-code \
      --tensor-parallel-size 1 \
      --pipeline-parallel-size 2 \
      --distributed-executor-backend ray \
      --kv-cache-dtype fp8 \
      --max-model-len 262144 \
      --gpu-memory-utilization 0.9 \
      --max-num-seqs 256 \
      --distributed-timeout-seconds 7200 \
      --enable-prefix-caching \
      --enable-auto-tool-choice \
      --tool-call-parser qwen3_coder \
      --reasoning-parser nemotron_v3
  '
```

### Step 5. Start the Nemotron 3 Ultra on station-2

Connect to `station-2` and set the worker variables:

>[!NOTE]
> HEAD_IP value should be fetched from station-1 [Step 4](#step-4-start-the-nemotron-3-ultra-on-station-1)

```bash
ssh station-2

export WORKER_IFACE=$(ibdev2netdev \
  | awk '$1 == "mlx5_0" {print $5; exit}')
export WORKER_IP=$(ip -4 -o address show dev "${WORKER_IFACE}" \
  | awk '{split($4, address, "/"); print address[1]; exit}')
export GPU_UUID_WORKER=$(nvidia-smi \
  --query-gpu=name,uuid --format=csv,noheader \
  | awk -F', ' '/GB300|B300/ {print $2; exit}')

export HEAD_IP="<station-1 CX8 IP>" # e.g. 192.168.240.1
export IMG=vllm/vllm-openai:v0.25.1-aarch64
```

Review the values and then start the worker:

```bash
printf 'WORKER_IFACE=%s\nWORKER_IP=%s\nGPU_UUID_WORKER=%s\nHEAD_IP=%s\n' \
  "${WORKER_IFACE}" "${WORKER_IP}" "${GPU_UUID_WORKER}" "${HEAD_IP}"

sudo docker rm -f nemotron-ultra-worker 2>/dev/null || true

sudo docker run -d --name nemotron-ultra-worker \
  --restart unless-stopped --init \
  --network host --shm-size 16g \
  --gpus "device=${GPU_UUID_WORKER}" \
  --device=/dev/infiniband/uverbs0 \
  --device=/dev/infiniband/uverbs1 \
  --ulimit memlock=-1 \
  -e HEAD_IP="${HEAD_IP}" \
  -e WORKER_IP="${WORKER_IP}" \
  -e VLLM_HOST_IP="${WORKER_IP}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS=3600 \
  -e VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 \
  -e NCCL_IB_HCA=mlx5_0,mlx5_1 \
  -e NCCL_IB_DISABLE=0 \
  -e NCCL_IB_ADDR_FAMILY=AF_INET \
  -e NCCL_IB_ROCE_VERSION_NUM=2 \
  -e NCCL_IB_TC=106 \
  -e NCCL_NET_GDR_LEVEL=PHB \
  -e NCCL_SOCKET_IFNAME="${WORKER_IFACE}" \
  -e GLOO_SOCKET_IFNAME="${WORKER_IFACE}" \
  -e TP_SOCKET_IFNAME="${WORKER_IFACE}" \
  -e NCCL_IB_QPS_PER_CONNECTION=4 \
  -e NCCL_IB_PCI_RELAXED_ORDERING=1 \
  -e UCX_NET_DEVICES=mlx5_0:1,mlx5_1:1 \
  -e HF_HOME=/models/huggingface \
  -v "${HOME}/.cache/huggingface:/models/huggingface" \
  --entrypoint bash "${IMG}" -lc '
    set -euo pipefail

    python3 -m pip install --break-system-packages "ray[cgraph]"

    exec ray start --address="${HEAD_IP}:6379" --node-ip-address="${WORKER_IP}" --num-gpus=1 --block
  '
```

## Phase 3: Verify the Nemotron 3 Ultra inference server

### Step 6. Monitor startup without interrupting it

Open one terminal for each station to monitor the startup:

```bash
## station-1
sudo docker logs -f nemotron-ultra-head
```

```bash
## station-2
sudo docker logs -f nemotron-ultra-worker
```

During a first start, vLLM loads model shards, compiles kernels, builds the KV cache, autotunes FlashInfer/TRT-LLM kernels, and captures CUDA graphs. The containers can remain `Up` while port 8000 returns HTTP `000`. This is expected while the logs continue to advance.

Do not restart merely because the API is not yet listening. Investigate when a container exits, its restart count increases, an explicit error appears, or the worker logs stop advancing for an extended period.

### Step 7. Validate the API, reasoning, and tool calls

Wait for this message in the head logs:

```text
Application startup complete.
```

Then run all API checks on `station-1`.

Confirm the model alias:

```bash
curl -fsS http://127.0.0.1:8000/v1/models | jq
```

Send a short chat request:

```bash
curl -fsS http://127.0.0.1:8000/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{
    "model": "nemotron-ultra",
    "messages": [
      {"role": "user", "content": "Reply with exactly: READY"}
    ],
    "max_tokens": 16,
    "temperature": 0
  }' | jq
```

### Step 8. Perform routine health checks

Run these checks from `station-1`:

```bash
sudo docker inspect \
  --format 'head={{.State.Status}} restarts={{.RestartCount}}' \
  nemotron-ultra-head

ssh station-2 sudo docker inspect \
  --format 'worker={{.State.Status}} restarts={{.RestartCount}}' \
  nemotron-ultra-worker

curl -fsS http://127.0.0.1:8000/v1/models \
  | jq -e '.data[] | select(.id == "nemotron-ultra")'
```

All three checks must succeed before treating the agent service as healthy.

Recommended to Proceed to [Install NeMoClaw](https://build.nvidia.com/station/nemoclaw/instructions) on `station-1` with **NVIDIA Nemotron 3 Ultra** as inference backend.

## Phase 4: Troubleshoot or Remove the Deployment

### Step 9. Troubleshoot common failures

| Symptom | Likely cause | Corrective action |
| --- | --- | --- |
| API responds, but tool calls appear as text | Tool parser flags are missing or NeMoClaw is using the Responses API | Confirm `--enable-auto-tool-choice`, `--tool-call-parser qwen3_coder`, `--reasoning-parser nemotron_v3`, and Chat Completions. |
| Direct `curl` works, but NeMoClaw inference is unhealthy | OpenShell cannot reach the host endpoint | Confirm vLLM listens on `0.0.0.0`, inspect the OpenShell subnet, review firewall rules, and rerun onboarding. |
| NeMoClaw reports the wrong model | The provider route or served-model alias is stale | Verify `/v1/models`, then rerun onboarding or use `nemoclaw inference set` with `nemotron-ultra`. |

Useful diagnostics:

```bash
## station-1
sudo docker logs --tail 200 nemotron-ultra-head
sudo docker exec nemotron-ultra-head \
  ray status --address=${HEAD_IP}:6379
ss -ltnp | grep ':8000' || true
```

```bash
## station-2
sudo docker logs --tail 200 nemotron-ultra-worker
nvidia-smi
```

### Step 10. Stop or remove the deployment

Remove the inference containers in this order:

```bash
## station-1
sudo docker rm -f nemotron-ultra-head
```

```bash
## station-2
sudo docker rm -f nemotron-ultra-worker
```

Keep the Hugging Face caches unless reclaiming disk is intentional. Retaining the model cache makes a controlled relaunch substantially faster. Reusing the existing containers also preserves their compilation artifacts.

### Related resources

- [NeMoClaw documentation](https://docs.nvidia.com/nemoclaw/latest/index.html)
- [Set up vLLM for NeMoClaw](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/inference/local-inference/set-up-vllm)
- [Verify the sandbox inference route](https://docs.nvidia.com/nemoclaw/latest/user-guide/openclaw/inference/validate-inference/verify-inference-route)
- [NVIDIA Nemotron](https://github.com/NVIDIA-NeMo/Nemotron)

## Troubleshooting

| Symptom | Cause | Fix |
|---------|-------|-----|
| `nemoclaw: command not found` after install | Shell PATH not updated | Run `source ~/.bashrc` (or `source ~/.zshrc` for zsh), or open a new terminal window. |
| Installer fails with Node.js version error | Node.js version below 22.16 | Install Node.js 22.16+: `curl -fsSL https://deb.nodesource.com/setup_22.x \| sudo -E bash - && sudo apt-get install -y nodejs` then re-run the installer. |
| npm install fails with `EACCES` permission error | npm global directory not writable | `mkdir -p ~/.npm-global && npm config set prefix ~/.npm-global && export PATH=~/.npm-global/bin:$PATH` then re-run the installer. Add the `export` line to `~/.bashrc` to make it permanent. |
| Docker permission denied | User not in docker group | `sudo usermod -aG docker $USER`, then log out and back in. |
| Gateway fails with cgroup / "Failed to start ContainerManager" errors | Older OpenShell or Docker still using a **private** cgroup namespace for the gateway so kubelet cannot see cgroup v2 controllers | First **upgrade OpenShell** (re-run the Phase 1 `nemoclaw.sh` install so you get a build that sets host cgroupns on the gateway container). If it still fails, force Docker's default to host mode by running the [daemon.json cgroup fix](#daemonjson-cgroup-fix) below, then run `sudo systemctl restart docker`. |
| Gateway fails with "port 8080 is held by container..." | Another OpenShell gateway or container is using port 8080 | Run `nemoclaw onboard` (or `nemoclaw onboard --resume`) again. NemoClaw probes the existing managed gateway, reuses it if healthy, and recreates stale gateway state when it can do so safely. See the [NemoClaw commands documentation](https://docs.nvidia.com/nemoclaw/latest/reference/commands.html). |
| Sandbox creation fails | Stale gateway state or DNS not propagated | Run `nemoclaw onboard` (or `nemoclaw onboard --resume`) again. NemoClaw probes the existing managed gateway, reuses it if healthy, and recreates stale gateway state when it can do so safely. See the [NemoClaw commands documentation](https://docs.nvidia.com/nemoclaw/latest/reference/commands.html). |
| CoreDNS crash loop | Known issue on some DGX Station configurations | Re-run the NemoClaw installer (`curl -fsSL https://www.nvidia.com/nemoclaw.sh \| bash`) which includes the CoreDNS fix. If the issue persists, see [NemoClaw troubleshooting](https://docs.nvidia.com/nemoclaw/latest/reference/troubleshooting.html). |
| "No GPU detected" during onboard | DGX Station GB300 reports unified memory differently | Expected on DGX Station. The wizard still works and uses vLLM for inference. |
| Inference timeout or hangs | vLLM not running, the large Express model is still loading, or the server is not reachable | Check the vLLM server: `curl http://127.0.0.1:8000/v1/models` should list `nvidia/NVIDIA-Nemotron-3-Ultra-550B-A55B-NVFP4` for the default Express Install. If you used `--station-deepseek` or custom onboarding, expect the model you selected instead. If the request hangs, wait for `Application startup complete`, then check `nemoclaw my-assistant status` for the Inference health line. |
| Agent gives no response or is very slow | First response can be slow, especially with larger models | Response time depends on model size (30B: a few seconds, 120B: 30–90 seconds). Verify inference route: `nemoclaw my-assistant status`. |
| Port 18789 already in use | Another process is bound to the port | `lsof -i :18789` then `kill <PID>`. If needed, `kill -9 <PID>` to force-terminate. |
| Web UI port forward dies or dashboard unreachable | Port forward not active | `openshell forward stop 18789 my-assistant` then `openshell forward start 18789 my-assistant --background`. |
| Web UI shows `origin not allowed` | Accessing via `localhost` instead of `127.0.0.1` | Use `http://127.0.0.1:18789/#token=...` in the browser. The gateway origin check requires `127.0.0.1` exactly. |
| Telegram bridge does not start | Telegram channel is not configured, the sandbox gateway is unhealthy, or Telegram startup/config failed | Run `nemoclaw <name> status` and `nemoclaw <name> logs` to confirm the failure. If the sandbox gateway is unhealthy, run `nemoclaw <name> recover`. If Telegram is not configured, rerun `nemoclaw onboard` and enable Telegram during onboarding. See the [NemoClaw Troubleshooting guide](https://docs.nvidia.com/nemoclaw/latest/reference/troubleshooting.html). |
| Telegram stops responding after sandbox rebuild | Duplicate bot-token consumer, missing DM allowlist, BotFather group privacy mode, inference failure, policy denial, or rebuilt channel config issue | Run `nemoclaw <name> status` and `nemoclaw <name> logs`. Look for Telegram 409 Conflict, allowlist warnings, privacy-mode issues, inference errors, or policy denials. If configuration needs to change, rerun `nemoclaw onboard`. See the [NemoClaw Troubleshooting guide](https://docs.nvidia.com/nemoclaw/latest/reference/troubleshooting.html). |
| Telegram bot receives messages but does not reply | Inbound Telegram delivery works, but the agent turn, inference call, policy check, allowlist/mention gate, or outbound reply failed | Run `nemoclaw <name> status` and `nemoclaw <name> logs`. Check for inbound Telegram update, outbound send, inference, and policy-denial messages. Fix the logged cause; run `nemoclaw <name> recover` only if the sandbox gateway is unhealthy. For Telegram configuration changes, rerun `nemoclaw onboard`. See the [NemoClaw Troubleshooting guide](https://docs.nvidia.com/nemoclaw/latest/reference/troubleshooting.html). |

#### daemon.json cgroup fix

Use this script as the fallback for the cgroup / "Failed to start ContainerManager" row above. It validates any existing `/etc/docker/daemon.json`, writes a `.bak` backup, sets `default-cgroupns-mode` to `host`, and atomically replaces the file. It exits non-zero with an error on stderr if anything fails, leaving the original `daemon.json` untouched.

```bash
sudo python3 - <<'PY'
import json, os, shutil, sys, tempfile

path = '/etc/docker/daemon.json'
try:
    if os.path.exists(path):
        with open(path) as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise ValueError(f'{path} is not a JSON object')
    else:
        data = {}
except (json.JSONDecodeError, ValueError, OSError) as e:
    print(f'error: failed to read {path}: {e}', file=sys.stderr)
    sys.exit(1)

if os.path.exists(path):
    try:
        shutil.copy2(path, path + '.bak')
    except OSError as e:
        print(f'error: failed to back up {path}: {e}', file=sys.stderr)
        sys.exit(1)

data['default-cgroupns-mode'] = 'host'

target_dir = os.path.dirname(path) or '/'
fd, tmp = tempfile.mkstemp(prefix='daemon.json.', dir=target_dir)
try:
    with os.fdopen(fd, 'w') as f:
        json.dump(data, f, indent=2)
        f.write('\n')
    os.chmod(tmp, 0o644)
    os.replace(tmp, path)
except OSError as e:
    if os.path.exists(tmp):
        try:
            os.unlink(tmp)
        except OSError:
            pass
    print(f'error: failed to write {path}: {e}', file=sys.stderr)
    sys.exit(1)
PY
```
